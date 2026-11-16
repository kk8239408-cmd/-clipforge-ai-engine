import os, re, uuid, subprocess, threading, shutil, json
from pathlib import Path
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

BASE=Path(os.getenv("DATA_DIR","/tmp/clipforge"))
INBOX=BASE/"inbox"; OUT=BASE/"outputs"
INBOX.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
app=FastAPI(title="CLIPFORGE Lightweight AI Engine",version="0.2.0")
app.add_middleware(CORSMiddleware,allow_origins=os.getenv("CORS_ORIGINS","*").split(","),allow_credentials=False,allow_methods=["*"],allow_headers=["*"])
JOBS={}; LOCK=threading.Lock()
@app.get("/")
def home():
    return {
        "name": "CLIPFORGE",
        "status": "online",
        "message": "AI clipping engine is running!"
    }
def cmd(args):
    p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if p.returncode: raise RuntimeError(p.stderr[-1800:])
    return p.stdout

def get_duration(p):
    return float(cmd(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(p)]).strip())

def transcribe_groq(video_path):
    key=os.getenv("GROQ_API_KEY")
    if not key: raise RuntimeError("Missing GROQ_API_KEY. Add it in Render → Environment after creating a Groq API key. Check the provider's current limits and pricing first.")
    audio=video_path.with_suffix(".mp3")
    # Mono low-bitrate audio keeps uploads small and avoids loading an ML model on Render.
    cmd(["ffmpeg","-y","-i",str(video_path),"-vn","-ac","1","-ar","16000","-b:a","48k",str(audio)])
    try:
        with audio.open("rb") as f:
            r=requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization":f"Bearer {key}"},
                data={"model":os.getenv("GROQ_WHISPER_MODEL","whisper-large-v3-turbo"),"response_format":"verbose_json","timestamp_granularities[]":["segment"]},
                files={"file":(audio.name,f,"audio/mpeg")},
                timeout=240
            )
        if r.status_code>=400: raise RuntimeError(f"Transcription provider error ({r.status_code}): {r.text[:800]}")
        data=r.json()
        return [{"start":float(s.get("start",0)),"end":float(s.get("end",0)),"text":s.get("text","").strip()} for s in data.get("segments",[]) if s.get("text","").strip()]
    finally:
        audio.unlink(missing_ok=True)

def rank(text, span):
    t=text.lower()
    hooks=["no way","oh my god","wait","actually","i can't believe","crazy","insane","secret","never","best","worst","why","how","what happened"]
    return min(len(text.split())/22,2.0)+sum(0.8 for h in hooks if h in t)+min(text.count("!")*.25,1)+(.5 if 12<=span<=55 else 0)

def pick(segments,total):
    # Score rolling windows and choose up to five non-overlapping speech-rich moments.
    windows=[]; start=0.0
    while start<total:
        end=min(total,start+38)
        ss=[s for s in segments if s["end"]>start and s["start"]<end]
        if ss:
            txt=" ".join(s["text"] for s in ss).strip()
            if len(txt.split())>=8:
                a=max(0,ss[0]["start"]-1); b=min(total,ss[-1]["end"]+1.5)
                windows.append({"start":a,"end":b,"text":txt,"score":rank(txt,b-a)})
        start+=19
    windows.sort(key=lambda x:x["score"],reverse=True)
    chosen=[]
    for w in windows:
        if all(w["end"]<=c["start"] or w["start"]>=c["end"] for c in chosen):
            chosen.append(w)
        if len(chosen)>=5: break
    return sorted(chosen,key=lambda x:x["start"])

def render(src,item,dest):
    dur=max(1,item["end"]-item["start"])
    # Conservative 720x1280 output and ultrafast encoding for small free instances.
    vf="scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,setsar=1"
    cmd(["ffmpeg","-y","-ss",str(item["start"]),"-i",str(src),"-t",str(dur),
         "-vf",vf,"-c:v","libx264","-preset","ultrafast","-threads","1","-crf","29",
         "-c:a","aac","-b:a","96k","-movflags","+faststart",str(dest)])

def process(jid,source):
    try:
        with LOCK: JOBS[jid].update(status="transcribing",progress=10)
        segs=transcribe_groq(source)
        total=get_duration(source)
        moments=pick(segs,total)
        if not moments: raise RuntimeError("No clear speech moments found. Try a video with spoken dialogue.")
        folder=OUT/jid; folder.mkdir(parents=True,exist_ok=True); clips=[]
        for i,m in enumerate(moments,1):
            p=folder/f"clip_{i}.mp4"; render(source,m,p)
            clips.append({"index":i,"start":round(m["start"],2),"end":round(m["end"],2),"score":round(m["score"],2),"transcript":m["text"],"download_url":f"/jobs/{jid}/clips/{p.name}"})
            with LOCK: JOBS[jid]["progress"]=10+int(i/len(moments)*85)
        with LOCK: JOBS[jid].update(status="completed",progress=100,clips=clips,note="Transcript-ranked candidates. Review each clip; ranking is heuristic, not a guarantee of virality.")
    except Exception as e:
        with LOCK: JOBS[jid].update(status="failed",error=str(e))
    finally: source.unlink(missing_ok=True)

@app.get("/health")
def health(): return {"ok":True,"service":"CLIPFORGE lightweight engine","version":"0.2.0","transcription":"hosted Groq API"}

@app.post("/jobs")
async def create_job(background:BackgroundTasks,video:UploadFile=File(...)):
    ext=Path(video.filename or "").suffix.lower()
    if ext not in {".mp4",".mov",".m4v"}: raise HTTPException(400,"Upload MP4 or MOV.")
    jid=str(uuid.uuid4()); src=INBOX/f"{jid}{ext}"; size=0
    with src.open("wb") as f:
        while chunk:=await video.read(1024*1024):
            size+=len(chunk)
            if size>int(os.getenv("MAX_UPLOAD_MB","150"))*1024*1024:
                src.unlink(missing_ok=True); raise HTTPException(413,"Video exceeds 150 MB upload limit.")
            f.write(chunk)
    with LOCK: JOBS[jid]={"id":jid,"status":"queued","progress":0,"clips":[]}
    background.add_task(process,jid,src)
    return {"job_id":jid,"status":"queued"}

@app.get("/jobs/{jid}")
def status(jid):
    with LOCK:
        if jid not in JOBS: raise HTTPException(404,"Job not found (free instances may restart and clear temporary jobs).")
        return JOBS[jid]
@app.get("/jobs/{jid}/clips/{filename}")
def clip(jid, filename):
    if not re.fullmatch(r"clip_[1-5]\.mp4", filename):
        raise HTTPException(400, "Invalid clip name")
    p = OUT / jid / filename
    if not p.exists():
        raise HTTPException(404, "Clip not ready")
    return FileResponse(p, media_type="video/mp4", filename=filename)
