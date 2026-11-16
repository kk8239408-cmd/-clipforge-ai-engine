import os, re, uuid, json, shutil, subprocess, threading
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

BASE = Path(os.getenv("DATA_DIR", "/tmp/clipforge"))
INBOX = BASE / "inbox"
OUT = BASE / "outputs"
for p in (INBOX, OUT): p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="CLIPFORGE AI Engine", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=False, allow_methods=["*"], allow_headers=["*"]
)

JOBS = {}
LOCK = threading.Lock()
MODEL = None

def whisper():
    global MODEL
    if MODEL is None:
        # CPU-friendly starter. On a small free server this can still be slow.
        MODEL = WhisperModel(os.getenv("WHISPER_MODEL", "base"),
                             device="cpu", compute_type="int8")
    return MODEL

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-3000:])
    return p.stdout

def duration(path):
    out = run(["ffprobe","-v","error","-show_entries","format=duration",
               "-of","default=noprint_wrappers=1:nokey=1",str(path)])
    return float(out.strip())

def score_segment(text, avg_words, duration_s):
    # Transparent starter ranking, not a trained virality predictor.
    t = text.lower()
    hooks = ["what happened", "no way", "oh my god", "wait", "actually",
             "i can't believe", "the truth", "crazy", "insane", "secret",
             "never", "best", "worst", "why", "how"]
    score = min(len(text.split()) / 24, 1.0) * 2
    score += sum(1.0 for h in hooks if h in t)
    score += min(sum(1 for c in "!?..." if c in text) * 0.25, 1.0)
    score += 0.5 if 12 <= duration_s <= 55 else 0
    score += min(avg_words / 4, 1.0)
    return score

def choose_moments(segments, total, wanted=5):
    # Merge speech into overlapping 20–55s windows, rank, then avoid overlap.
    windows = []
    start = 0.0
    while start < total:
        end = min(total, start + 42)
        selected = [s for s in segments if s["end"] > start and s["start"] < end]
        if selected:
            txt = " ".join(s["text"].strip() for s in selected).strip()
            if len(txt.split()) >= 12:
                avg = len(txt.split()) / max(end-start, 1)
                windows.append({"start": max(0, selected[0]["start"]-1.2),
                                "end": min(total, selected[-1]["end"]+1.8),
                                "text": txt,
                                "score": score_segment(txt, avg, end-start)})
        start += 24
    windows.sort(key=lambda x: x["score"], reverse=True)
    chosen = []
    for w in windows:
        if all(w["end"] <= c["start"] or w["start"] >= c["end"] for c in chosen):
            chosen.append(w)
        if len(chosen) == wanted: break
    return sorted(chosen, key=lambda x: x["start"])

def escape_ass(s):
    return s.replace("\\","\\\\").replace("{","\\{").replace("}","\\}").replace("\n"," ")

def render_clip(src, item, outpath, index):
    start, end = item["start"], item["end"]
    # One starter style: vertical 9:16, center crop, clean bold bottom captions.
    # Captions use the selected moment's transcript as a readable two-line block.
    caption = escape_ass(item["text"])
    ass = outpath.with_suffix(".ass")
    ass.write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n"
        "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,OutlineColour,"
        "BackColour,Bold,Italic,BorderStyle,Outline,Shadow,Alignment,MarginL,"
        "MarginR,MarginV,Encoding\n"
        "Style: Default,Arial,58,&H00FFFFFF,&H00101010,&H80000000,-1,0,1,4,2,2,70,70,180,1\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
        f"Dialogue: 0,0:00:00.00,0:00:{max(1,end-start):05.2f},Default,,0,0,0,,{caption}\n",
        encoding="utf-8")
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,subtitles=" + str(ass)
    run(["ffmpeg","-y","-ss",str(start),"-i",str(src),"-t",str(end-start),
         "-vf",vf,"-c:v","libx264","-preset","veryfast","-crf","24",
         "-c:a","aac","-b:a","128k","-movflags","+faststart",str(outpath)])
    ass.unlink(missing_ok=True)

def process(job_id, source):
    try:
        with LOCK: JOBS[job_id]["status"] = "transcribing"
        model = whisper()
        segs, info = model.transcribe(str(source), vad_filter=True, word_timestamps=False)
        segments = [{"start": float(s.start), "end": float(s.end), "text": s.text}
                    for s in segs if s.text.strip()]
        total = duration(source)
        moments = choose_moments(segments, total, 5)
        if not moments:
            raise RuntimeError("No speech-rich moments found. Try a video with clear dialogue.")
        jobout = OUT / job_id
        jobout.mkdir(parents=True, exist_ok=True)
        clips = []
        for i, item in enumerate(moments, 1):
            path = jobout / f"clip_{i}.mp4"
            render_clip(source, item, path, i)
            clips.append({"index": i, "start": round(item["start"],2),
                          "end": round(item["end"],2), "score": round(item["score"],2),
                          "file": f"/jobs/{job_id}/clips/{path.name}"})
            with LOCK:
                JOBS[job_id]["progress"] = int(i/len(moments)*100)
        with LOCK:
            JOBS[job_id].update(status="completed", progress=100, clips=clips,
                                note="Transcript-ranked highlights; review before posting.")
    except Exception as e:
        with LOCK: JOBS[job_id].update(status="failed", error=str(e))
    finally:
        # Keep source during the job; remove it after processing to limit disk use.
        source.unlink(missing_ok=True)

@app.get("/health")
def health(): return {"ok": True, "service": "CLIPFORGE AI Engine", "version": "0.1.0"}

@app.post("/jobs")
async def create_job(background: BackgroundTasks, video: UploadFile = File(...)):
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v"}:
        raise HTTPException(400, "Upload an MP4 or MOV video.")
    job_id = str(uuid.uuid4())
    source = INBOX / f"{job_id}{suffix}"
    size = 0
    with source.open("wb") as f:
        while chunk := await video.read(1024*1024):
            size += len(chunk)
            if size > int(os.getenv("MAX_UPLOAD_MB","500"))*1024*1024:
                source.unlink(missing_ok=True)
                raise HTTPException(413, "Video exceeds the upload limit.")
            f.write(chunk)
    with LOCK:
        JOBS[job_id] = {"id": job_id, "status":"queued", "progress":0, "clips":[]}
    background.add_task(process, job_id, source)
    return {"job_id": job_id, "status":"queued"}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with LOCK:
        if job_id not in JOBS: raise HTTPException(404, "Job not found")
        return JOBS[job_id]

@app.get("/jobs/{job_id}/clips/{filename}")
def get_clip(job_id: str, filename: str):
    from fastapi.responses import FileResponse
    if not re.fullmatch(r"clip_[1-5]\.mp4", filename): raise HTTPException(400,"Invalid filename")
    path = OUT / job_id / filename
    if not path.exists(): raise HTTPException(404,"Clip not ready")
    return FileResponse(path, media_type="video/mp4", filename=filename)
