import os, re, uuid, subprocess, threading
from pathlib import Path

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

BASE = Path(os.getenv("DATA_DIR", "/tmp/clipforge"))
INBOX, OUT = BASE / "inbox", BASE / "outputs"
INBOX.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

FONT_DIR = Path("/usr/local/share/fonts/clipforge")
app = FastAPI(title="CLIPFORGE Lightweight AI Engine", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
JOBS = {}
LOCK = threading.Lock()


@app.get("/")
def home():
    return {"name": "CLIPFORGE", "status": "online", "version": "0.4.0",
            "message": "AI clipping engine is running!"}


def cmd(args):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-1800:])
    return p.stdout


def get_duration(p):
    return float(cmd([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(p)
    ]).strip())


def transcribe_groq(video_path):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY. Add it in Render Environment. Never share the key publicly.")
    audio = video_path.with_suffix(".mp3")
    cmd(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1",
         "-ar", "16000", "-b:a", "48k", str(audio)])
    try:
        with audio.open("rb") as f:
            r = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                data={"model": os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo"),
                      "response_format": "verbose_json",
                      "timestamp_granularities[]": ["segment"]},
                files={"file": (audio.name, f, "audio/mpeg")},
                timeout=240,
            )
        if r.status_code >= 400:
            raise RuntimeError(f"Transcription provider error ({r.status_code}): {r.text[:800]}")
        data = r.json()
        return [{"start": float(s.get("start", 0)), "end": float(s.get("end", 0)),
                 "text": s.get("text", "").strip()}
                for s in data.get("segments", []) if s.get("text", "").strip()]
    finally:
        audio.unlink(missing_ok=True)


def rank(text, span):
    t = text.lower()
    hooks = ["no way", "oh my god", "wait", "actually", "i can't believe",
             "crazy", "insane", "secret", "never", "best", "worst", "why",
             "how", "what happened"]
    return (min(len(text.split()) / 22, 2.0)
            + sum(0.8 for h in hooks if h in t)
            + min(text.count("!") * 0.25, 1)
            + (0.5 if 12 <= span <= 55 else 0))


def pick(segments, total):
    windows, start = [], 0.0
    while start < total:
        end = min(total, start + 38)
        ss = [s for s in segments if s["end"] > start and s["start"] < end]
        if ss:
            txt = " ".join(s["text"] for s in ss).strip()
            if len(txt.split()) >= 8:
                a, b = max(0, ss[0]["start"] - 1), min(total, ss[-1]["end"] + 1.5)
                windows.append({"start": a, "end": b, "text": txt,
                                "segments": ss, "score": rank(txt, b - a)})
        start += 19
    windows.sort(key=lambda x: x["score"], reverse=True)
    chosen = []
    for w in windows:
        if all(w["end"] <= c["start"] or w["start"] >= c["end"] for c in chosen):
            chosen.append(w)
        if len(chosen) >= 5:
            break
    return sorted(chosen, key=lambda x: x["start"])


def font_family(filename, fallback):
    """Read the actual family name embedded in a font file."""
    p = FONT_DIR / filename
    if not p.exists():
        return fallback
    try:
        family = cmd(["fc-scan", "--format", "%{family}", str(p)]).strip()
        return family.split(",")[0].strip() if family else fallback
    except Exception:
        return fallback


def choose_font(text):
    """Heuristic style choice from transcript; not a trained visual classifier."""
    t = text.lower()
    if any(k in t for k in ["goal", "match", "champion", "fight", "knockout",
                            "basketball", "football", "sports", "vs.", "versus"]):
        return font_family("LeagueSpartan-Bold.ttf", "Montserrat")
    if any(k in t for k in ["bro", "bruh", "no way", "crazy", "insane",
                            "what", "wait", "plot twist", "meme"]):
        return font_family("ObelixPro-Bold.ttf", "Montserrat")
    if len(text) > 150:
        return font_family("ArialNarrow-Bold.ttf", "Montserrat")
    return font_family("Montserrat-Bold.ttf", "Montserrat")


def ass_time(seconds):
    seconds = max(0.0, seconds)
    whole = int(seconds)
    cs = int(round((seconds - whole) * 100))
    if cs >= 100:
        whole, cs = whole + 1, 0
    return f"{whole // 3600}:{(whole % 3600) // 60:02d}:{whole % 60:02d}.{cs:02d}"


def clean_caption_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[{}\r\n]+", " ", text)).strip()


def make_ass(item, subtitle_path, font_name):
    clip_start, clip_end = item["start"], item["end"]
    events = []
    for seg in item.get("segments", []):
        a, b = max(float(seg["start"]), clip_start), min(float(seg["end"]), clip_end)
        words = clean_caption_text(seg.get("text", "")).split()
        if not words or b <= a:
            continue
        chunks = [words[i:i + 6] for i in range(0, len(words), 6)]
        index, total_words, duration = 0, len(words), b - a
        for chunk in chunks:
            chunk_start = a + duration * index / total_words
            index += len(chunk)
            chunk_end = a + duration * index / total_words
            word_cs = max(1, int(round((duration / total_words) * 100)))
            line = " ".join(r"{\k" + str(word_cs) + "}" + w for w in chunk)
            events.append(
                "Dialogue: 0," + ass_time(chunk_start - clip_start) + ","
                + ass_time(chunk_end - clip_start)
                + ",Caption,,0,0,0,," + line
            )

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 720
PlayResY: 1280
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Caption,{font_name},54,&H00FFFFFF,&H0000D7FF,&H00101010,&H80000000,-1,0,0,0,100,100,0,0,1,4,1,2,48,48,235,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    subtitle_path.write_text(header + "\n".join(events) + ("\n" if events else ""),
                             encoding="utf-8")


def render(src, item, dest):
    dur = max(1, item["end"] - item["start"])
    subtitle_path = dest.with_suffix(".ass")
    font_name = choose_font(item["text"])
    make_ass(item, subtitle_path, font_name)
    vf = ("scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,"
          "setsar=1,eq=contrast=1.06:brightness=0.01:saturation=1.08,"
          "unsharp=5:5:0.65:3:3:0.25,ass=" + str(subtitle_path))
    try:
        cmd(["ffmpeg", "-y", "-ss", str(item["start"]), "-i", str(src),
             "-t", str(dur), "-vf", vf, "-c:v", "libx264",
             "-preset", "ultrafast", "-threads", "1", "-crf", "23",
             "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dest)])
    finally:
        subtitle_path.unlink(missing_ok=True)
    return font_name


def process(jid, source):
    try:
        with LOCK:
            JOBS[jid].update(status="transcribing", progress=10)
        segs = transcribe_groq(source)
        total = get_duration(source)
        moments = pick(segs, total)
        if not moments:
            raise RuntimeError("No clear speech moments found. Try a video with spoken dialogue.")
        folder = OUT / jid
        folder.mkdir(parents=True, exist_ok=True)
        clips = []
        for i, moment in enumerate(moments, 1):
            p = folder / f"clip_{i}.mp4"
            selected_font = render(source, moment, p)
            clips.append({"index": i, "start": round(moment["start"], 2),
                          "end": round(moment["end"], 2), "score": round(moment["score"], 2),
                          "transcript": moment["text"], "font": selected_font,
                          "download_url": f"/jobs/{jid}/clips/{p.name}"})
            with LOCK:
                JOBS[jid]["progress"] = 10 + int(i / len(moments) * 85)
        with LOCK:
            JOBS[jid].update(status="completed", progress=100, clips=clips,
                note=("Captions use heuristic automatic font selection and basic visual enhancement. "
                      "Font choice is based on transcript keywords, not AI visual understanding. "
                      "Music, sound effects, URL ingestion and reference editing presets are not included yet."))
    except Exception as e:
        with LOCK:
            JOBS[jid].update(status="failed", error=str(e))
    finally:
        source.unlink(missing_ok=True)


@app.get("/health")
def health():
    installed = []
    if FONT_DIR.exists():
        installed = sorted(p.name for p in FONT_DIR.glob("*") if p.suffix.lower() in {".ttf", ".otf"})
    return {"ok": True, "service": "CLIPFORGE lightweight engine", "version": "0.4.0",
            "transcription": "hosted Groq API", "automatic_font_selection": True,
            "installed_font_files": installed}


@app.post("/jobs")
async def create_job(background: BackgroundTasks, video: UploadFile = File(...)):
    ext = Path(video.filename or "").suffix.lower()
    if ext not in {".mp4", ".mov", ".m4v"}:
        raise HTTPException(400, "Upload MP4 or MOV.")
    jid, size = str(uuid.uuid4()), 0
    src = INBOX / f"{jid}{ext}"
    with src.open("wb") as f:
        while chunk := await video.read(1024 * 1024):
            size += len(chunk)
            if size > int(os.getenv("MAX_UPLOAD_MB", "150")) * 1024 * 1024:
                src.unlink(missing_ok=True)
                raise HTTPException(413, "Video exceeds 150 MB upload limit.")
            f.write(chunk)
    with LOCK:
        JOBS[jid] = {"id": jid, "status": "queued", "progress": 0, "clips": []}
    background.add_task(process, jid, src)
    return {"job_id": jid, "status": "queued"}


@app.get("/jobs/{jid}")
def status(jid):
    with LOCK:
        if jid not in JOBS:
            raise HTTPException(404, "Job not found (free instances may restart and clear temporary jobs).")
        return JOBS[jid]


@app.get("/jobs/{jid}/clips/{filename}")
def clip(jid, filename):
    if not re.fullmatch(r"clip_[1-5]\.mp4", filename):
        raise HTTPException(400, "Invalid clip name")
    p = OUT / jid / filename
    if not p.exists():
        raise HTTPException(404, "Clip not ready")
    return FileResponse(p, media_type="video/mp4", filename=filename)
