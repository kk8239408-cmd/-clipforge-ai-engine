#!/usr/bin/env python3
"""CLIPFORGE V5.1 live Twitch clipper: automatic face crop, split 9:16, word captions."""
import argparse, collections, csv, json, os, re, shutil, socket, subprocess, threading, time
import cv2
import numpy as np

def log(s):
    print(s, flush=True)

def probe(path):
    p=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration:stream=codec_type,width,height,r_frame_rate","-of","json",path],capture_output=True,text=True)
    if p.returncode: return {}
    j=json.loads(p.stdout or "{}")
    v=next((x for x in j.get("streams",[]) if x.get("codec_type")=="video"),{})
    a=any(x.get("codec_type")=="audio" for x in j.get("streams",[]))
    n,d=(v.get("r_frame_rate","0/1").split("/") + ["1"])[:2]
    return {"duration":float(j.get("format",{}).get("duration",0) or 0),"w":int(v.get("width",0) or 0),"h":int(v.get("height",0) or 0),"fps":float(n)/max(1,float(d)),"audio":a}

class Chat(threading.Thread):
    def __init__(self, channel):
        super().__init__(daemon=True); self.channel=channel.lower(); self.messages=[]; self.lock=threading.Lock(); self.connected=False
    def run(self):
        while True:
            try:
                s=socket.create_connection(("irc.chat.twitch.tv",6667),timeout=25); s.settimeout(90)
                s.sendall(("NICK justinfan%d\r\nJOIN #%s\r\n"%(int(time.time())%90000+1000,self.channel)).encode())
                self.connected=True; log("[IRC] Connected and joined #"+self.channel)
                buf=b""
                while True:
                    data=s.recv(4096)
                    if not data: break
                    buf+=data
                    while b"\r\n" in buf:
                        line,buf=buf.split(b"\r\n",1)
                        if line.startswith(b"PING"): s.sendall(b"PONG :tmi.twitch.tv\r\n")
                        elif b" PRIVMSG " in line:
                            with self.lock: self.messages.append((time.time(),line.decode("utf-8","ignore")))
                self.connected=False
            except Exception as e:
                self.connected=False; log("[IRC] reconnect: "+str(e)[:150]); time.sleep(3)
    def count(self,start,end):
        with self.lock: return sum(start<=t<end for t,_ in self.messages)

def detect_face_crop(path):
    """Sample frames and return a normalized face-centered crop; None means explicit full-frame fallback."""
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): log("[face] Could not open sample; fallback=full-frame"); return None
    cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    candidates=[]
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps=cap.get(cv2.CAP_PROP_FPS) or 30
    for sec in np.linspace(0,max(0,(total/max(fps,1))-1),min(12,max(3,int(total/max(fps,1))//2+1))):
        cap.set(cv2.CAP_PROP_POS_MSEC,float(sec)*1000); ok,frame=cap.read()
        if not ok: continue
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(35,35))
        if len(faces):
            x,y,w,h=max(faces,key=lambda b:b[2]*b[3]); H,W=gray.shape
            # Expand around face to include head/shoulders; normalized box clamped to source.
            cw=min(W,max(w*2.5,h*2.1)); ch=min(H,cw/1.125)
            if ch>H: ch=H; cw=min(W,ch*1.125)
            cx=x+w/2; cy=y+h*0.45
            x0=max(0,min(W-cw,cx-cw/2)); y0=max(0,min(H-ch,cy-ch*0.42))
            candidates.append((x0/W,y0/H,cw/W,ch/H))
    cap.release()
    if not candidates:
        log("[face] NO FACE DETECTED; fallback=full-frame crop (logged)")
        return None
    # Median of detections reduces single-frame jumps.
    box=np.median(np.array(candidates),axis=0).tolist()
    log("[face] detected face in %d sampled frames; automatic crop=%s"%(len(candidates),",".join("%.3f"%v for v in box)))
    return box

def ass_time(t):
    return "%d:%02d:%05.2f"%(int(t//3600),int(t%3600//60),t%60)

def captions(path,ass_path):
    try:
        from faster_whisper import WhisperModel
        model=WhisperModel("base",device="cpu",compute_type="int8")
        segs,_=model.transcribe(path,word_timestamps=True,vad_filter=True)
        words=[(w.start,w.end,w.word.strip().upper().strip(",.!?;:")) for s in segs for w in (s.words or []) if w.word.strip()]
    except Exception as e:
        log("[captions] Faster-Whisper failed: "+str(e)[:180]); words=[]
    log("[captions] word timestamps=%d"%len(words))
    header="""[Script Info]
ScriptType: v4.00+
PlayResX: 720
PlayResY: 1280
[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Arial,68,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,7,3,5,20,20,0,1
[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    colors=["&H00FFFF00&","&H0000FFFF&","&H0000FF00&"] # cyan, yellow, green (ASS BGR)
    rows=[]
    for i,(st,en,w) in enumerate(words):
        end=max(st+0.12,min(words[i+1][0] if i+1<len(words) else en+0.28,en+0.28))
        txt="{\\c%s\\bord8\\shad3\\b1}%s"%(colors[i%3],w)
        rows.append("Dialogue: 0,%s,%s,Default,,0,0,0,,%s"%(ass_time(st),ass_time(end),txt))
    with open(ass_path,"w",encoding="utf-8") as f: f.write(header+"\n".join(rows)+"\n")
    return len(words)

def render_clip(recording,start,duration,out,chat_count):
    tmp=out+".src.mp4"
    p=subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error","-ss",str(start),"-i",recording,"-t",str(duration),"-map","0:v:0","-map","0:a:0?","-c","copy",tmp])
    if p.returncode or not os.path.exists(tmp): raise RuntimeError("could not extract candidate")
    info=probe(tmp)
    if info.get("duration",0)<14: raise RuntimeError("candidate shorter than 14s")
    face=detect_face_crop(tmp)
    ass=out+".ass"; word_count=captions(tmp,ass)
    if face:
        x,y,w,h=face
        top=("crop=w='trunc(iw*%.5f/2)*2':h='trunc(ih*%.5f/2)*2':x='trunc(iw*%.5f/2)*2':y='trunc(ih*%.5f/2)*2'"%(w,h,x,y))
    else:
        top="scale=720:640:force_original_aspect_ratio=increase,crop=720:640"
    # Each panel fills its half. No blur, letterboxing or black bars.
    graph=("[0:v]split=2[a][b];[a]%s,scale=720:640:force_original_aspect_ratio=increase,crop=720:640[top];[b]scale=720:640:force_original_aspect_ratio=increase,crop=720:640[bottom];[top][bottom]vstack,drawbox=x=0:y=637:w=720:h=6:color=black:t=fill,subtitles=%s,format=yuv420p[v]"%(top,ass))
    cmd=["ffmpeg","-y","-hide_banner","-loglevel","error","-i",tmp,"-filter_complex",graph,"-map","[v]"]
    if info.get("audio"): cmd+=["-map","0:a:0","-c:a","aac","-b:a","192k","-ar","48000"]
    cmd+=["-c:v","libx264","-preset","veryfast","-crf","21","-r","30","-fps_mode","cfr","-t",str(duration),"-pix_fmt","yuv420p","-movflags","+faststart",out]
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode: raise RuntimeError("render failed: "+p.stderr[-1200:])
    os.remove(tmp); os.remove(ass)
    outinfo=probe(out)
    if (outinfo.get("w"),outinfo.get("h"))!=(720,1280): raise RuntimeError("wrong output resolution")
    if abs(outinfo.get("fps",0)-30)>0.2: raise RuntimeError("output is not 30fps")
    if info.get("audio") and not outinfo.get("audio"): raise RuntimeError("audio missing")
    log("[clip] rendered %s | %.1fs | 720x1280 | %.2ffps | audio=%s | words=%d | chat_msgs=%d"%(out,outinfo.get("duration",0),outinfo.get("fps",0),outinfo.get("audio"),word_count,chat_count))
    return outinfo

def score_segments(recording):
    info=probe(recording); dur=info.get("duration",0); scored=[]
    if dur<15: return []
    for start in np.arange(0,max(0,dur-15),5):
        seg=recording+".score.mp4"
        subprocess.run(["ffmpeg","-y","-v","error","-ss",str(float(start)),"-i",recording,"-t","20","-an","-vf","fps=2,scale=96:54,format=gray","-f","rawvideo","-"],capture_output=True)
        # Score audio loudness and frame motion on the actual segment.
        p=subprocess.run(["ffmpeg","-y","-v","error","-ss",str(float(start)),"-i",recording,"-t","20","-c","copy",seg])
        if p.returncode: continue
        v=probe(seg)
        if v.get("duration",0)>=15:
            frames=cv2.VideoCapture(seg); prev=None; motion=0; n=0
            while n<30:
                ok,fr=frames.read()
                if not ok: break
                gray=cv2.cvtColor(cv2.resize(fr,(96,54)),cv2.COLOR_BGR2GRAY)
                if prev is not None: motion+=float(cv2.absdiff(gray,prev).mean())
                prev=gray;n+=1
            frames.release()
            aud=subprocess.run(["ffmpeg","-v","error","-i",seg,"-vn","-af","volumedetect","-f","null","-"],capture_output=True,text=True)
            import re as _re
            m=_re.search(r"mean_volume: (-?[0-9.]+) dB",aud.stderr)
            loud=float(m.group(1)) if m else -60
            scored.append((motion/max(1,n-1)+max(0,loud+60)*0.25,float(start),min(20,v["duration"])))
        if os.path.exists(seg): os.remove(seg)
    return sorted(scored,reverse=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("channel")
    ap.add_argument("--max-minutes",type=float,default=4)
    ap.add_argument("--layout",choices=["split"],default="split")
    ap.add_argument("--pre",type=int,default=12); ap.add_argument("--post",type=int,default=8)
    ap.add_argument("--max-clips",type=int,default=3); ap.add_argument("--out",default="clips")
    ap.add_argument("--no-effects",action="store_true")
    a=ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+",a.channel): raise SystemExit("Invalid Twitch channel")
    if not 3<=a.max_minutes<=5: raise SystemExit("Live run must be 3-5 minutes")
    os.makedirs(a.out,exist_ok=True)
    rec=os.path.join(a.out,"_recording_"+a.channel+".ts")
    chat=Chat(a.channel); chat.start()
    url="twitch.tv/"+a.channel
    log("[streamlink] checking live access: "+url)
    check=subprocess.run(["streamlink","--stream-url",url,"best"],capture_output=True,text=True,timeout=60)
    if check.returncode: raise SystemExit("[streamlink] access failed: "+(check.stderr or check.stdout)[-500:])
    log("[streamlink] access OK; starting recording")
    with open(os.path.join(a.out,"streamlink.log"),"w") as sl_log:
        sl=subprocess.Popen(["streamlink","--retry-streams","3","--retry-max","20",url,"best","-o",rec,"--force"],stdout=sl_log,stderr=subprocess.STDOUT)
        start_time=time.time(); spike_windows=[]; last_count=0
        while time.time()-start_time<a.max_minutes*60:
            time.sleep(10)
            now=time.time(); count=chat.count(now-10,now)
            log("[IRC] interval_messages=%d total_messages=%d connected=%s"%(count,len(chat.messages),chat.connected))
            if count>=8: spike_windows.append((now,count))
            if sl.poll() is not None: log("[streamlink] process exited early code=%s"%sl.returncode); break
        sl.terminate()
        try: sl.wait(timeout=10)
        except subprocess.TimeoutExpired: sl.kill(); sl.wait()
    if not os.path.isfile(rec) or os.path.getsize(rec)==0: raise SystemExit("[ERROR] No recording file")
    size=os.path.getsize(rec); log("[recording] file=%s bytes=%d duration=%.1fs"%(rec,size,probe(rec).get("duration",0)))
    log("[IRC] total_messages=%d"%len(chat.messages))
    candidates=[]
    for ts,c in spike_windows:
        candidates.append((c, max(0,probe(rec).get("duration",0)-(time.time()-ts)-a.pre), min(25,a.pre+a.post),c))
    candidates=sorted(candidates,reverse=True)
    made=0; used=[]
    for score,st,dur,cnt in candidates:
        if made>=a.max_clips: break
        if any(abs(st-u)<12 for u in used): continue
        dur=max(15,min(25,dur))
        if st+dur>probe(rec).get("duration",0): st=max(0,probe(rec)["duration"]-dur)
        if dur<15: continue
        out=os.path.join(a.out,"clip_%02d_chat.mp4"%(made+1))
        try:
            render_clip(rec,st,dur,out,cnt); made+=1; used.append(st)
        except Exception as e: log("[clip] chat candidate rejected: "+str(e))
    if made==0:
        log("[fallback] no usable chat-spike clip; ranking audio/motion windows")
        for score,st,dur in score_segments(rec):
            if made>=a.max_clips: break
            if any(abs(st-u)<12 for u in used): continue
            out=os.path.join(a.out,"clip_%02d_fallback.mp4"%(made+1))
            try:
                render_clip(rec,st,max(15,min(25,dur)),out,0); made+=1; used.append(st)
            except Exception as e: log("[fallback] candidate rejected: "+str(e))
    with open(os.path.join(a.out,"index.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["clips_created","recording_bytes","irc_messages"]); w.writerow([made,size,len(chat.messages)])
    log("[summary] candidates=%d clips=%d recording_bytes=%d"%(len(spike_windows),made,size))
    if made<1: raise SystemExit("[ERROR] No clip was created")
if __name__=="__main__": main()
