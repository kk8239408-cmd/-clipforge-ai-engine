#!/usr/bin/env python3
"""V5.1 Twitch live clipper with a bounded rolling TS buffer and continuous scoring."""
import argparse, csv, json, os, re, shutil, socket, subprocess, threading, time, glob
from collections import deque
import cv2
import numpy as np

SEGMENT_SECONDS = 10
BUFFER_SECONDS = 180
COOLDOWN_SECONDS = 45
PRE_SECONDS = 20
POST_SECONDS = 10
CHAT_SPIKE_MULTIPLIER = 1.35
MIN_CHAT_SCORE = 5
LOUDNESS_JUMP_DB = 3.0

def log(msg):
    print(msg, flush=True)

def run(cmd, **kw):
    return subprocess.run(cmd, **kw)

def probe(path):
    p=run(["ffprobe","-v","error","-show_entries","format=duration:stream=codec_type,width,height,r_frame_rate","-of","json",path],capture_output=True,text=True)
    if p.returncode: return {}
    j=json.loads(p.stdout or "{}")
    v=next((s for s in j.get("streams",[]) if s.get("codec_type")=="video"),{})
    audio=any(s.get("codec_type")=="audio" for s in j.get("streams",[]))
    rate=v.get("r_frame_rate","0/1").split("/")
    fps=float(rate[0])/max(1.0,float(rate[1])) if len(rate)>1 else 0
    return {"duration":float(j.get("format",{}).get("duration",0) or 0),"width":int(v.get("width",0) or 0),"height":int(v.get("height",0) or 0),"fps":fps,"audio":audio}

class TwitchChat(threading.Thread):
    def __init__(self, channel):
        super().__init__(daemon=True); self.channel=channel.lower(); self.items=deque(maxlen=50000); self.lock=threading.Lock(); self.connected=False; self.stop_flag=False
    def run(self):
        while not self.stop_flag:
            try:
                s=socket.create_connection(("irc.chat.twitch.tv",6667),timeout=20); s.settimeout(60)
                s.sendall(("NICK justinfan%d\r\nJOIN #%s\r\n"%(int(time.time())%80000+10000,self.channel)).encode())
                self.connected=True; log("[IRC] connected JOIN #"+self.channel)
                buf=b""
                while not self.stop_flag:
                    data=s.recv(4096)
                    if not data: break
                    buf+=data
                    while b"\r\n" in buf:
                        line,buf=buf.split(b"\r\n",1)
                        if line.startswith(b"PING"): s.sendall(b"PONG :tmi.twitch.tv\r\n")
                        elif b" PRIVMSG " in line:
                            with self.lock: self.items.append((time.time(),line))
                s.close()
            except Exception as e:
                if not self.stop_flag: log("[IRC] reconnect: "+str(e)[:120])
            self.connected=False
            if not self.stop_flag: time.sleep(2)
    def count(self, since):
        now=time.time()
        with self.lock: return sum(1 for t,_ in self.items if since<=t<=now)
    def total(self):
        with self.lock: return len(self.items)

def audio_loudness(path):
    p=run(["ffmpeg","-hide_banner","-nostats","-i",path,"-vn","-af","volumedetect","-f","null","-"],capture_output=True,text=True)
    m=re.search(r"mean_volume:\s*(-?[0-9.]+)\s*dB",p.stderr or "")
    return float(m.group(1)) if m else -60.0

def motion_score(path):
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): return 0.0
    prev=None; vals=[]
    for _ in range(8):
        ok,frame=cap.read()
        if not ok: break
        gray=cv2.cvtColor(cv2.resize(frame,(96,54)),cv2.COLOR_BGR2GRAY)
        if prev is not None: vals.append(float(cv2.absdiff(gray,prev).mean()))
        prev=gray
    cap.release()
    return float(np.mean(vals)) if vals else 0.0

def detect_face_crop(path):
    cap=cv2.VideoCapture(path)
    if not cap.isOpened():
        log("[face] sample open failed; full-frame fallback")
        return None
    cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    found=[]; frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    for pos in np.linspace(0,max(0,frames-1),min(10,max(2,frames//20+1))):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(pos)); ok,frame=cap.read()
        if not ok: continue
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4,minSize=(30,30))
        if len(faces):
            x,y,w,h=max(faces,key=lambda b:b[2]*b[3]); H,W=gray.shape
            cw=min(W,max(w*2.4,h*2.1)); ch=min(H,cw/1.125)
            if ch>H: ch=H; cw=min(W,ch*1.125)
            cx=x+w/2; cy=y+h*.45
            x0=max(0,min(W-cw,cx-cw/2)); y0=max(0,min(H-ch,cy-ch*.4))
            found.append((x0/W,y0/H,cw/W,ch/H))
    cap.release()
    if not found:
        log("[face] no face detected; explicit full-frame crop fallback")
        return None
    box=np.median(np.asarray(found),axis=0).tolist()
    log("[face] Haar face detections=%d; auto crop=%s"%(len(found),",".join("%.3f"%v for v in box)))
    return box

def ass_time(t):
    return "%d:%02d:%05.2f"%(int(t//3600),int(t%3600//60),t%60)

def make_captions(src, ass):
    words=[]
    try:
        from faster_whisper import WhisperModel
        model=WhisperModel("base",device="cpu",compute_type="int8")
        segs,_=model.transcribe(src,word_timestamps=True,vad_filter=True)
        words=[(w.start,w.end,w.word.strip().upper().strip(",.!?;:")) for s in segs for w in (s.words or []) if w.word.strip()]
    except Exception as e: log("[captions] transcription failed: "+str(e)[:160])
    log("[captions] word-level timestamps=%d"%len(words))
    head="""[Script Info]
ScriptType: v4.00+
PlayResX: 720
PlayResY: 1280
[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Arial,64,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,7,3,5,10,10,0,1
[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    colors=["&H00FFFF00&","&H0000FFFF&","&H0000FF00&"]
    rows=[]
    for i,(st,en,word) in enumerate(words):
        end=max(st+.12,min(words[i+1][0] if i+1<len(words) else en+.25,en+.25))
        rows.append("Dialogue: 0,%s,%s,Default,,0,0,0,,{\\c%s\\bord8\\shad3\\b1}%s"%(ass_time(st),ass_time(end),colors[i%3],word))
    with open(ass,"w",encoding="utf-8") as f: f.write(head+"\n".join(rows)+"\n")
    return len(words)

def render_clip(segment_paths, out, clip_no):
    concat=os.path.join(os.path.dirname(out),"render_concat.txt")
    with open(concat,"w",encoding="utf-8") as f:
        for p in segment_paths: f.write("file '%s'\n"%os.path.abspath(p).replace("'","'\\''"))
    joined=out+".joined.ts"
    p=run(["ffmpeg","-y","-hide_banner","-loglevel","error","-f","concat","-safe","0","-i",concat,"-c","copy",joined],capture_output=True,text=True)
    if p.returncode: raise RuntimeError("segment concat failed: "+p.stderr[-700:])
    src=out+".src.mp4"
    p=run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",joined,"-t","30","-map","0:v:0","-map","0:a:0?","-c","copy",src],capture_output=True,text=True)
    if p.returncode: raise RuntimeError("clip extraction failed: "+p.stderr[-700:])
    info=probe(src)
    if info.get("duration",0)<24: raise RuntimeError("assembled event shorter than 24 seconds")
    face=detect_face_crop(src)
    ass=out+".ass"; words=make_captions(src,ass)
    if face:
        x,y,w,h=face
        top="crop=w='trunc(iw*%.6f/2)*2':h='trunc(ih*%.6f/2)*2':x='trunc(iw*%.6f/2)*2':y='trunc(ih*%.6f/2)*2'"%(w,h,x,y)
    else: top="scale=720:640:force_original_aspect_ratio=increase,crop=720:640"
    graph="[0:v]split=2[a][b];[a]%s,scale=720:640:force_original_aspect_ratio=increase,crop=720:640[top];[b]scale=720:640:force_original_aspect_ratio=increase,crop=720:640[bottom];[top][bottom]vstack,drawbox=x=0:y=637:w=720:h=6:color=black:t=fill,subtitles=%s,format=yuv420p[v]"%(top,ass)
    cmd=["ffmpeg","-y","-hide_banner","-loglevel","error","-i",src,"-filter_complex",graph,"-map","[v]"]
    if info.get("audio"): cmd += ["-map","0:a:0","-c:a","aac","-b:a","128k","-ar","48000"]
    cmd += ["-c:v","libx264","-preset","veryfast","-crf","24","-r","30","-fps_mode","cfr","-t","30","-pix_fmt","yuv420p","-movflags","+faststart",out]
    p=run(cmd,capture_output=True,text=True)
    if p.returncode: raise RuntimeError("render failed: "+p.stderr[-1200:])
    result=probe(out); size=os.path.getsize(out)
    if (result.get("width"),result.get("height"))!=(720,1280): raise RuntimeError("ffprobe wrong dimensions: %s"%result)
    if not result.get("audio"): raise RuntimeError("ffprobe says audio is missing")
    if abs(result.get("fps",0)-30)>.2: raise RuntimeError("ffprobe says output is not 30fps")
    log("[render] clip-%d | bytes=%d (%.2f MB) | ffprobe=%s | captions=%d"%(clip_no,size,size/1048576,result,words))
    if size>40*1024*1024: log("[WARN] clip-%d exceeds 40 MB"%(clip_no))
    for pth in (concat,joined,src,ass):
        try: os.remove(pth)
        except OSError: pass

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("channel"); ap.add_argument("--max-minutes",type=int,default=4); ap.add_argument("--max-clips",type=int,default=3); ap.add_argument("--out",default="clips"); ap.add_argument("--no-effects",action="store_true")
    a=ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+",a.channel): raise SystemExit("invalid channel")
    if not 3<=a.max_minutes<=5: raise SystemExit("duration must be 3-5 minutes")
    os.makedirs(a.out,exist_ok=True); buf=os.path.join(a.out,"buffer"); os.makedirs(buf,exist_ok=True)
    log("[thresholds] chat_spike_multiplier=%.2f minimum_chat_score=%d loudness_jump_db=%.1f cooldown=%ds"%(CHAT_SPIKE_MULTIPLIER,MIN_CHAT_SCORE,LOUDNESS_JUMP_DB,COOLDOWN_SECONDS))
    chat=TwitchChat(a.channel); chat.start()
    url="https://twitch.tv/"+a.channel
    check=run(["streamlink","--stream-url",url,"best"],capture_output=True,text=True,timeout=60)
    if check.returncode: raise SystemExit("[streamlink] access failed: "+(check.stderr or check.stdout)[-400:])
    log("[streamlink] access OK; starting 10-second segmented stream")
    streamlog=open(os.path.join(a.out,"streamlink.stderr.log"),"w")
    sl=subprocess.Popen(["streamlink","--stdout",url,"best"],stdout=subprocess.PIPE,stderr=streamlog)
    pattern=os.path.join(buf,"segment_%06d.ts")
    ff=subprocess.Popen(["ffmpeg","-y","-hide_banner","-loglevel","warning","-i","pipe:0","-map","0","-c","copy","-f","segment","-segment_time",str(SEGMENT_SECONDS),"-reset_timestamps","1","-segment_format","mpegts",pattern],stdin=sl.stdout,stdout=subprocess.DEVNULL,stderr=streamlog)
    sl.stdout.close()
    start=time.time(); deadline=start+a.max_minutes*60; processed=0; scores=[]; recent_chat=[]; loud_hist=[]; motion_hist=[]; triggered=[]; last_trigger=-999; made=0
    log("[buffer] rolling retention=%ds segment=%ds; clip window=20s pre +10s post"%(BUFFER_SECONDS,SEGMENT_SECONDS))
    try:
        while time.time()<deadline and ff.poll() is None:
            time.sleep(1)
            files=sorted(glob.glob(os.path.join(buf,"segment_*.ts")))
            while processed<len(files):
                seg=files[processed]; idx=processed; processed+=1
                # Wait for file close/stable size before analysis.
                time.sleep(.35)
                count=chat.count(time.time()-SEGMENT_SECONDS-1)
                recent_chat.append(count); baseline=float(np.mean(recent_chat[-6:-1])) if len(recent_chat)>1 else float(count)
                loud=audio_loudness(seg); motion=motion_score(seg)
                prev_loud=float(np.mean(loud_hist[-5:])) if loud_hist else loud
                loud_jump=loud-prev_loud
                loud_hist.append(loud); motion_hist.append(motion)
                chat_trigger=count>=max(MIN_CHAT_SCORE,baseline*CHAT_SPIKE_MULTIPLIER)
                audio_trigger=loud_jump>=LOUDNESS_JUMP_DB
                motion_base=float(np.mean(motion_hist[-5:])) if len(motion_hist)>1 else motion
                motion_trigger=motion>=max(2.0,motion_base*1.35)
                score=(count/max(1,baseline+1))+max(0,loud_jump)*.6+max(0,motion-motion_base)*.15
                scores.append({"idx":idx,"score":score,"chat":count,"loud":loud,"loud_jump":loud_jump,"motion":motion,"chat_trigger":chat_trigger,"audio_trigger":audio_trigger,"motion_trigger":motion_trigger,"file":seg})
                log("[score] t=%ds chat=%d baseline=%.1f loud=%.1fdB jump=%.1fdB motion=%.2f score=%.2f trigger=%s"%(idx*10,count,baseline,loud,loud_jump,motion,score,chat_trigger or audio_trigger or motion_trigger))
                # Keep rolling 3-minute buffer; preserve segments referenced by pending events.
                cutoff=max(0,idx-(BUFFER_SECONDS//SEGMENT_SECONDS)+1)
                for old in glob.glob(os.path.join(buf,"segment_*.ts")):
                    m=re.search(r"segment_(\d+)\.ts$",old)
                    if m and int(m.group(1))<cutoff and all(int(m.group(1)) not in e["indices"] for e in triggered): 
                        try: os.remove(old)
                        except OSError: pass
                fired=chat_trigger or audio_trigger or motion_trigger
                if fired and idx*10-last_trigger>=COOLDOWN_SECONDS and made+len(triggered)<a.max_clips:
                    prior=[s for s in scores if s["idx"]<=idx][-3:]
                    if len(prior)>=2:
                        event={"indices":[s["idx"] for s in prior],"score":score,"trigger_idx":idx,"chat":count}
                        triggered.append(event); last_trigger=idx*10
                        log("[trigger] idx=%d score=%.2f captured_segments=%s; waiting for 10s post-roll"%(idx,score,event["indices"]))
                # Once a full post-roll segment exists, assemble the selected 3 prior + 1 after.
                ready=[e for e in triggered if idx>=e["trigger_idx"]+1]
                for e in ready:
                    if made>=a.max_clips: break
                    paths=[os.path.join(buf,"segment_%06d.ts"%i) for i in e["indices"]+[e["trigger_idx"]+1]]
                    if all(os.path.isfile(p) and os.path.getsize(p)>1000 for p in paths):
                        made+=1; out=os.path.join(a.out,"clip-%d.mp4"%made)
                        try: render_clip(paths,out,made)
                        except Exception as ex: made-=1; log("[render-error] "+str(ex))
                    else: log("[trigger] skipped: required roll segments unavailable")
                    triggered.remove(e)
                if time.time()>=deadline: break
        # Let the segment muxer close the last chunk, then stop child processes.
    finally:
        for proc in (ff,sl):
            if proc.poll() is None: proc.terminate()
        for proc in (ff,sl):
            try: proc.wait(timeout=8)
            except subprocess.TimeoutExpired: proc.kill()
        streamlog.close()
        chat.stop_flag=True
    # If no trigger produced a clip, use the highest score from the rolling score log.
    if made==0 and scores:
        best=max(scores,key=lambda x:x["score"])
        log("[fallback] no trigger clip rendered; highest score idx=%d score=%.2f"% (best["idx"],best["score"]))
        available=sorted(glob.glob(os.path.join(buf,"segment_*.ts")))
        all_by_idx={int(re.search(r"segment_(\d+)\.ts$",p).group(1)):p for p in available}
        ids=[best["idx"]-2,best["idx"]-1,best["idx"],best["idx"]+1]
        if min(ids)<0 or any(i not in all_by_idx for i in ids):
            ids=sorted(all_by_idx)[-4:]
        paths=[all_by_idx[i] for i in ids if i in all_by_idx]
        if len(paths)>=3:
            try: made=1; render_clip(paths,os.path.join(a.out,"clip-1.mp4"),1)
            except Exception as e: made=0; log("[fallback-render-error] "+str(e))
    total_bytes=sum(os.path.getsize(p) for p in glob.glob(os.path.join(buf,"*.ts")))
    log("[summary] scores=%d clips=%d IRC_messages=%d buffer_bytes=%d"%(len(scores),made,chat.total(),total_bytes))
    if made<1: raise SystemExit("[ERROR] no clips rendered")
if __name__=="__main__": main()
