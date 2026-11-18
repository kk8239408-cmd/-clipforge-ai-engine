#!/usr/bin/env python3
"""CLIPFORGE V3: guarded Twitch recorder + adaptive chat candidates + CLIP visual rating.

The live runner records Twitch to MPEG-TS. Chat only proposes timestamps; it does not
create a clip unless the recording is healthy and the requested footage exists.
After rendering, OpenAI CLIP (openai/clip-vit-base-patch32) samples frames and rates
visual excitement. This is a visual ranking model, not a human-level event detector.
"""
import argparse, collections, json, os, queue, re, shutil, socket, subprocess, sys, threading, time, statistics, tempfile
from pathlib import Path

class ChatReader(threading.Thread):
    def __init__(self, channel):
        super().__init__(daemon=True); self.channel=channel.lower(); self.events=collections.deque(); self.lock=threading.Lock()
    def run(self):
        while True:
            try:
                with socket.create_connection(("irc.chat.twitch.tv",6667),timeout=20) as s:
                    s.settimeout(120); s.sendall((f"NICK justinfan{int(time.time())%90000+1000}\r\nJOIN #{self.channel}\r\n").encode())
                    buf=b""
                    while True:
                        data=s.recv(4096)
                        if not data: break
                        buf+=data
                        while b"\r\n" in buf:
                            line,buf=buf.split(b"\r\n",1)
                            if line.startswith(b"PING"): s.sendall(b"PONG :tmi.twitch.tv\r\n")
                            elif b" PRIVMSG " in line:
                                try:
                                    pre,payload=line.split(b" PRIVMSG ",1); user=pre.split(b"!",1)[0].lstrip(b":").decode(errors="ignore").lower()
                                    body=payload.split(b" :",1)[1].decode(errors="ignore") if b" :" in payload else ""
                                    with self.lock: self.events.append((time.monotonic(),user,body))
                                except Exception: pass
            except Exception as e:
                print(f"[chat] reconnecting: {e}",flush=True); time.sleep(3)
    def stats(self, start):
        with self.lock: rows=[r for r in self.events if r[0]>=start]
        texts=[r[2].lower() for r in rows]; users={r[1] for r in rows if r[1]}
        hype_words=("pog","pogchamp","omegalul","lul","lmao","no way","insane","clutch","wtf","crazy","lets go","let's go","holy","💀","😭","😂","🤣","🔥","LUL")
        hype=sum(any(w in t for w in hype_words) for t in texts)/max(1,len(texts))
        emote=sum(any(ch in t for ch in "💀😭😂🤣🔥") for t in texts)/max(1,len(texts))
        return len(rows),len(users),hype,emote

def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)
def probe(path):
    try:
        raw=subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",path],text=True,stderr=subprocess.DEVNULL,timeout=20).strip()
        streams=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries","stream=codec_type,width,height","-of","json",path],text=True,stderr=subprocess.DEVNULL,timeout=20)).get("streams",[])
        return float(raw or 0), any(s.get("codec_type")=="video" for s in streams), any(s.get("codec_type")=="audio" for s in streams)
    except Exception: return 0.0,False,False

def black_fraction(path, duration):
    # Detect a mostly black candidate using 1fps sampling; fail closed if ffmpeg fails.
    try:
        p=subprocess.run(["ffmpeg","-hide_banner","-i",path,"-vf","blackdetect=d=0.8:pix_th=0.10","-an","-f","null","-"],capture_output=True,text=True,timeout=90)
        spans=[]
        for m in re.finditer(r"black_start:([0-9.]+) black_end:([0-9.]+)",p.stderr): spans.append(max(0,float(m.group(2))-float(m.group(1))))
        return min(1.0,sum(spans)/max(duration,0.1))
    except Exception: return 1.0

def ass_time(t):
    return f"{int(t//3600)}:{int(t%3600//60):02d}:{t%60:05.2f}"
def make_ass(words,path,font):
    head=("[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n[V4+ Styles]\n"
    "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
    f"Style: Default,{font},84,&H00FFFFFF,&H0000FFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,520,1\n\n[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    lines=[]
    for i in range(0,len(words),3):
        g=words[i:i+3]
        for j,(st,en,_) in enumerate(g):
            end=g[j+1][0] if j+1<len(g) else en
            txt=" ".join((r"{\c&H00FFFF&}"+w.upper()+r"{\c&HFFFFFF&}") if k==j else w.upper() for k,(_,_,w) in enumerate(g))
            lines.append(f"Dialogue: 0,{ass_time(st)},{ass_time(end)},Default,,0,0,0,,{txt}")
    Path(path).write_text(head+"\n".join(lines)+"\n",encoding="utf-8")

def captions_for(path):
    try:
        from faster_whisper import WhisperModel
        if not hasattr(captions_for,"model"): captions_for.model=WhisperModel("base",device="cpu",compute_type="int8")
        segs,_=captions_for.model.transcribe(path,word_timestamps=True,vad_filter=True)
        return [(w.start,w.end,w.word.strip()) for s in segs for w in (s.words or []) if w.word.strip()]
    except Exception as e:
        print(f"[captions] skipped: {e}",flush=True); return []

VF=("[0:v]split=2[bgsrc][fgsrc];[bgsrc]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=30:5[bg];"
    "[fgsrc]scale=1080:1920:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]")
def render(rec,start,dur,out,font):
    raw=out+".raw.ts"; ass=out+".ass"
    run(["ffmpeg","-y","-hide_banner","-loglevel","error","-ss",f"{start:.3f}","-i",rec,"-t",f"{dur:.3f}","-map","0:v:0","-map","0:a?","-c","copy",raw],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=120)
    rd,hasv,hasa=probe(raw)
    if not hasv or rd < min(dur*0.65,20): raise RuntimeError(f"recording segment invalid (duration={rd:.1f}s, video={hasv})")
    words=captions_for(raw)
    if words: make_ass(words,ass,font)
    graph=VF+(f";[v]subtitles={ass}[o]" if words else ";[v]null[o]")
    run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",raw,"-filter_complex",graph,"-map","[o]","-map","0:a?","-c:v","libx264","-preset","veryfast","-crf","21","-r","60","-pix_fmt","yuv420p","-c:a","aac","-b:a","160k","-movflags","+faststart",out],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=240)
    for f in (raw,ass):
        if os.path.exists(f): os.remove(f)
    vd,v,a=probe(out)
    if not v or vd<20: raise RuntimeError("rendered clip failed video validation")
    black=black_fraction(out,vd)
    if black>0.55:
        os.remove(out); raise RuntimeError(f"rejected mostly-black clip ({black:.0%} black frames)")
    return {"duration_seconds":round(vd,2),"has_audio":a,"black_fraction":round(black,3)}

# OpenAI CLIP visual model: samples 8 frames, compares excitement vs calm/loading/black prompts.
def visual_rate(path):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    if not hasattr(visual_rate,"model"):
        name="openai/clip-vit-base-patch32"
        visual_rate.model=CLIPModel.from_pretrained(name).eval()
        visual_rate.processor=CLIPProcessor.from_pretrained(name)
    model,proc=visual_rate.model,visual_rate.processor
    duration,_,_=probe(path)
    prompts=[
      "an exciting high action video game moment with intense combat or clutch gameplay",
      "a streamer reacting with shock, laughter, excitement or celebration on facecam",
      "a dramatic winning moment or unexpected event during a livestream",
      "fast movement and chaotic action in a competitive video game",
      "ordinary calm gameplay with no special event",
      "a static game menu, lobby, loading screen or pause screen",
      "a black screen or missing video signal"
    ]
    frames=[]
    with tempfile.TemporaryDirectory() as td:
        for i,t in enumerate([duration*x/9 for x in range(1,9)]):
            p=os.path.join(td,f"f{i}.jpg")
            subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error","-ss",str(t),"-i",path,"-frames:v","1","-q:v","4",p],check=True,timeout=30)
            frames.append(Image.open(p).convert("RGB"))
        inputs=proc(text=prompts,images=frames,return_tensors="pt",padding=True)
        with torch.no_grad(): out=model(**inputs)
        probs=out.logits_per_image.softmax(dim=1).cpu().numpy()
        # Reward excitement/reaction/action, penalize menus/black; report transparent score.
        excitement=float(probs[:,:4].sum(axis=1).mean())
        boring=float(probs[:,4:7].sum(axis=1).mean())
        score=max(0,min(100,round(100*excitement/max(excitement+boring,1e-8))))
        labels=["action","streamer reaction","dramatic moment","fast gameplay","calm gameplay","menu/loading","black screen"]
        avg=probs.mean(axis=0)
        return {"visual_score":score,"top_visual_signal":labels[int(avg.argmax())],"sampled_frames":len(frames)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("channel"); p.add_argument("--max-minutes",type=float,default=5); p.add_argument("--window",type=int,default=10)
    p.add_argument("--mult",type=float,default=1.35); p.add_argument("--min-count",type=int,default=10); p.add_argument("--min-unique",type=int,default=4)
    p.add_argument("--pre",type=int,default=25); p.add_argument("--post",type=int,default=8); p.add_argument("--cooldown",type=int,default=45)
    p.add_argument("--out",default="clips"); p.add_argument("--font",default="Liberation Sans")
    a=p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+",a.channel): p.error("Invalid Twitch channel")
    if not 1<=a.max_minutes<=330: p.error("Duration must be 1-330 minutes")
    for tool in ("streamlink","ffmpeg","ffprobe"):
        if not shutil.which(tool): sys.exit(f"Missing {tool}")
    os.makedirs(a.out,exist_ok=True); rec=os.path.join(a.out,f"_recording_{a.channel}.ts")
    if os.path.exists(rec): os.remove(rec)
    start_wall=time.monotonic(); deadline=start_wall+a.max_minutes*60
    print(f"[recorder] starting Twitch capture: twitch.tv/{a.channel}",flush=True)
    sl=subprocess.Popen(["streamlink","--retry-streams","3","--retry-max","20",f"https://www.twitch.tv/{a.channel}","best","-o",rec,"--force"],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
    chat=ChatReader(a.channel); chat.start()
    jobs=queue.Queue(); reports=[]; report_lock=threading.Lock()
    def worker():
        idx=0
        while True:
            item=jobs.get()
            if item is None: jobs.task_done(); break
            event_elapsed,meta=item
            target=event_elapsed+a.post
            wait=target-(time.monotonic()-start_wall-(recorder_start_elapsed or 0.0))
            if wait>0: time.sleep(wait)
            try:
                rec_dur,hasv,hasa=probe(rec)
                clip_start=max(0,event_elapsed-a.pre)
                wanted=min(a.pre+a.post,rec_dur-clip_start)
                if not hasv or wanted<20: raise RuntimeError(f"recorder not ready: duration={rec_dur:.1f}s video={hasv}")
                idx+=1; out=os.path.join(a.out,f"clip_{time.strftime('%H%M%S')}_{idx}.mp4")
                print(f"[clip] candidate #{idx}: {meta}; start={clip_start:.1f}s",flush=True)
                quality=render(rec,clip_start,wanted,out,a.font)
                print(f"[clip] saved {out} ({quality})",flush=True)
                with report_lock: reports.append({"file":out,"chat":meta,"recording_validation":quality})
            except Exception as e: print(f"[clip] rejected: {e}",flush=True)
            finally: jobs.task_done()
    wt=threading.Thread(target=worker,daemon=True); wt.start()
    history=collections.deque(maxlen=24); last_event=-9999; recorder_start_elapsed=None
    try:
        while time.monotonic()<deadline and sl.poll() is None:
            bucket_start=time.monotonic(); time.sleep(max(0,a.window-(time.monotonic()-bucket_start)))
            elapsed=time.monotonic()-start_wall
            # Guard: verify recording file is present, growing, and contains a video stream.
            size=os.path.getsize(rec) if os.path.exists(rec) else 0
            if size>500000 and recorder_start_elapsed is None:
                recorder_start_elapsed=elapsed
                print(f"[recorder] video file is receiving data ({size/1e6:.1f} MB)",flush=True)
            if elapsed>25 and size<500000:
                raise RuntimeError("Twitch recorder has not produced a healthy video file; stopping to avoid black clips")
            c,u,hype,emotes=chat.stats(bucket_start)
            base=statistics.median(history) if len(history)>=4 else max(5.0,statistics.median(history) if history else 5.0)
            ratio=c/max(base,1.0)
            # Adaptive: accept moderate chat lift when accompanied by genuine hype/reactions.
            lift=(ratio>=a.mult and c>=a.min_count)
            hype_hit=(c>=a.min_count and u>=a.min_unique and ratio>=1.18 and (hype>=0.16 or emotes>=0.08))
            event=(u>=a.min_unique and c>=a.min_count and (lift or hype_hit) and elapsed-last_event>=a.cooldown)
            print(f"[chat] {c:3d}/{a.window}s unique={u:3d} base={base:5.1f} ratio={ratio:3.1f} hype={hype:.0%} emotes={emotes:.0%}"+("  < CANDIDATE" if event else ""),flush=True)
            if event and recorder_start_elapsed is not None:
                last_event=elapsed
                rec_elapsed=max(0.0,elapsed-recorder_start_elapsed)
                jobs.put((rec_elapsed,{"messages":c,"unique":u,"ratio":round(ratio,2),"hype":round(hype,2),"emotes":round(emotes,2)}))
            else:
                history.append(c)
            # A zero-size/stalled recorder should not quietly produce unusable clips.
            if elapsed>35 and os.path.exists(rec):
                age=time.time()-os.path.getmtime(rec)
                if age>30: raise RuntimeError("Twitch recording file stopped growing; capture disconnected")
        if sl.poll() is None and time.monotonic()>=deadline:
            print("[watch] Watch duration completed normally.",flush=True)
        elif sl.poll() is not None:
            print(f"[watch] Stream recorder exited early (code {sl.returncode}); capture ended.",flush=True)
    except KeyboardInterrupt: print("[watch] Stopping by request.",flush=True)
    except Exception as e: print(f"[ERROR] {e}",flush=True)
    finally:
        if sl.poll() is None: sl.terminate()
        try: sl.wait(timeout=10)
        except subprocess.TimeoutExpired: sl.kill(); sl.wait()
        jobs.join(); jobs.put(None); jobs.join(); wt.join(timeout=5)
    # AI visual review on generated clips. If model unavailable, clips remain available with explicit status.
    for item in reports:
        try:
            item["ai_visual_review"]=visual_rate(item["file"])
            print(f"[AI] {os.path.basename(item['file'])}: score={item['ai_visual_review']['visual_score']}/100; {item['ai_visual_review']['top_visual_signal']}",flush=True)
        except Exception as e:
            item["ai_visual_review"]={"status":"unavailable","reason":str(e)[:240]}
            print(f"[AI] visual rating unavailable: {e}",flush=True)
    Path(a.out,"clipforge_report.json").write_text(json.dumps({"channel":a.channel,"watch_minutes":a.max_minutes,"clips":reports},indent=2),encoding="utf-8")
    print(f"[done] {len(reports)} valid clips; report: {a.out}/clipforge_report.json",flush=True)

if __name__=="__main__": main()
