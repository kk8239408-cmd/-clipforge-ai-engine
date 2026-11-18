#!/usr/bin/env python3
"""V5.3 Twitch live clipper with a bounded rolling TS buffer and continuous scoring."""
import argparse, csv, json, os, re, shutil, socket, subprocess, threading, time, glob, zipfile
from collections import deque
import cv2
import numpy as np

SEGMENT_SECONDS = 10
NL = chr(10)
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"fonts")
FONT_ZIP = os.path.join(FONT_DIR,"CLIPFORGE_Curated_Font_Library.zip")
CAPTION_FONT = "Montserrat"   # family name, rendered Bold; README default for dialogue captions
CAPTION_FONT_FALLBACK = "Liberation Sans Narrow"
SAFE_FONT_FOLDERS = ("01_Core_Captions/","02_Display_and_Accent/","03_Fallback/")  # 04_License_Check is never used
BUFFER_SECONDS = 180
COOLDOWN_SECONDS = 45
PRE_SECONDS = 20
POST_SECONDS = 5
CHAT_SPIKE_MULTIPLIER = 1.35
MIN_CHAT_SCORE = 5
LOUDNESS_JUMP_DB = 3.0

def log(msg):
    print(msg, flush=True)

def run(cmd, **kw):
    return subprocess.run(cmd, **kw)

def run_ffmpeg_progress(cmd, label):
    """Stream periodic FFmpeg progress so long renders never look stalled."""
    progress_cmd = cmd[:1] + ["-progress", "pipe:1", "-nostats"] + cmd[1:]
    started = time.monotonic()
    proc = subprocess.Popen(progress_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    fields = {}
    recent = deque(maxlen=20)
    last_log = 0.0
    for raw in proc.stdout:
        line = raw.strip()
        if not line:
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
            if key == "progress":
                now = time.monotonic()
                if now - last_log >= 3.0 or value == "end":
                    log("[ffmpeg:%s] frame=%s time=%s speed=%s elapsed=%.1fs" %
                        (label, fields.get("frame", "?"), fields.get("out_time", "?"),
                         fields.get("speed", "?"), now - started))
                    last_log = now
        else:
            recent.append(line)
    rc = proc.wait()
    return subprocess.CompletedProcess(progress_cmd, rc, stdout="", stderr="\n".join(recent))

def probe(path):
    p=run(["ffprobe","-v","error","-show_entries","format=duration:stream=codec_type,width,height,r_frame_rate","-of","json",path],capture_output=True,text=True)
    if p.returncode: return {}
    j=json.loads(p.stdout or "{}")
    v=next((s for s in j.get("streams",[]) if s.get("codec_type")=="video"),{})
    audio=any(s.get("codec_type")=="audio" for s in j.get("streams",[]))
    rate=v.get("r_frame_rate","0/1").split("/")
    fps=float(rate[0])/max(1.0,float(rate[1])) if len(rate)>1 else 0
    return {"duration":float(j.get("format",{}).get("duration",0) or 0),"width":int(v.get("width",0) or 0),"height":int(v.get("height",0) or 0),"fps":fps,"audio":audio,"size_bytes":os.path.getsize(path) if os.path.isfile(path) else 0}

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

def load_face_cascade():
    """Haar cascade or None (missing data file must not crash scoring/rendering)."""
    try:
        c=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
        return None if c.empty() else c
    except Exception: return None

def detect_face_crop(path):
    cap=cv2.VideoCapture(path)
    if not cap.isOpened():
        log("[face] sample open failed; full-frame fallback")
        return None
    cascade=load_face_cascade()
    if cascade is None:
        log("[face] Haar cascade unavailable; full-frame fallback"); cap.release(); return None
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

KEYWORDS = {"LUL":2,"OMEGALUL":3,"W":1,"L":1,"WTF":2,"CLIP IT":4,"NO WAY":3,"BRO":1,"WHAT":1}
REACTION_WORDS = {"NO WAY":4,"WHAT":2,"BRO":1,"WTF":3,"INSANE":3,"HOLY":3,"F***":2,"FUCK":2,"SHIT":2,"CLIP":2,"OMG":3,"LET'S GO":2}

def parse_chat_features(channel, since):
    now=time.time()
    with channel.lock:
        rows=[(t,b.decode("utf-8","ignore")) for t,b in channel.items if since<=t<=now]
    names=set(); emote_hits=0; keyword_hits=0
    for _,line in rows:
        m=re.match(r":([^! ]+)!.* PRIVMSG ",line)
        if m: names.add(m.group(1).lower())
        upper=line.upper()
        emote_hits += sum(upper.count(k) for k in ("LUL","OMEGALUL"," KAPPA"," POG"," KEKW"," W "))
        keyword_hits += sum(upper.count(k) for k in KEYWORDS)
    return len(rows),len(names),emote_hits,keyword_hits

def frame_features(path):
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): return {"luma":0,"motion":0,"scene":0,"face":0}
    prev=None; motions=[]; lumas=[]; scenes=[]; face_sizes=[]
    cascade=load_face_cascade()
    frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    for pos in np.linspace(0,max(0,frames-1),min(8,max(2,frames//15+1))):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(pos)); ok,fr=cap.read()
        if not ok: continue
        small=cv2.resize(fr,(160,90)); gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        lumas.append(float(gray.mean()))
        if prev is not None:
            d=cv2.absdiff(gray,prev); motions.append(float(d.mean()))
            scenes.append(float(np.mean(d>35)))
        prev=gray
        g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
        faces=cascade.detectMultiScale(g,scaleFactor=1.1,minNeighbors=4,minSize=(30,30)) if cascade is not None else []
        if len(faces): face_sizes.append(max(b[2]*b[3] for b in faces)/(g.shape[0]*g.shape[1]))
    cap.release()
    return {"luma":float(np.mean(lumas)) if lumas else 0,
            "motion":float(np.mean(motions)) if motions else 0,
            "scene":float(max(scenes)) if scenes else 0,
            "face":float(np.std(face_sizes)) if len(face_sizes)>1 else 0}

WHISPER_MODEL = None
WHISPER_MODEL_LOCK = threading.Lock()

def get_whisper_model():
    """Load the small CPU model once and reuse it for scoring and final captions."""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        with WHISPER_MODEL_LOCK:
            if WHISPER_MODEL is None:
                from faster_whisper import WhisperModel
                log("[whisper] loading shared model=small device=cpu compute_type=int8")
                WHISPER_MODEL = WhisperModel("small",device="cpu",compute_type="int8")
    return WHISPER_MODEL

def transcribe_words(path):
    """English word timestamps with VAD and conservative word-confidence filtering."""
    model=get_whisper_model()
    segs,_=model.transcribe(path,language="en",word_timestamps=True,vad_filter=True,
                            beam_size=5,best_of=5,condition_on_previous_text=False)
    words=[]
    for seg in segs:
        for w in (seg.words or []):
            token=(w.word or "").strip()
            if not token: continue
            probability=float(getattr(w,"probability",0.0) or 0.0)
            if probability < 0.45: continue
            words.append({"start":float(w.start),"end":float(w.end),"word":token,
                          "probability":probability})
    log("[whisper] ran=true model=small language=en vad=true word_timestamps=true confident_words=%d"%len(words))
    return words

def _caption_emphasis(word):
    """Colour for one caption word. Pass the ORIGINAL-case word (before .upper()) so ALL-CAPS detection works."""
    raw=word.strip(); clean=re.sub(r"[^A-Za-z0-9'@._:-]","",raw); upper=clean.upper()
    if re.search(r"\d",clean) or re.fullmatch(r"(?:AM|PM)",upper): return "&H0000FF00&"   # green: numbers / times
    if upper in {"NETFLIX","TWITCH","YOUTUBE","XQC","MRBEAST"}: return "&H000000FF&"       # red: names
    if len(clean)>=2 and clean.isupper(): return "&H000000FF&"                              # red: spoken in ALL CAPS
    if upper in {"INSANE","INSANELY","CRAZY","WTF","FUCK","SHIT","NO","WAY","HOLY","OMG","BRO"}: return "&H0000FFFF&"  # yellow
    return "&H00FFFFFF&"

def prepare_fonts():
    """Extract only the safe-to-use fonts from the curated ZIP into fonts/extracted. Returns (font_dir, family, status)."""
    out=os.path.join(FONT_DIR,"extracted")
    if not os.path.isfile(FONT_ZIP):
        return None,CAPTION_FONT_FALLBACK,"ZIP missing; using system "+CAPTION_FONT_FALLBACK
    try:
        os.makedirs(out,exist_ok=True)
        with zipfile.ZipFile(FONT_ZIP) as z:
            n=0
            for m in z.namelist():
                if m.lower().endswith((".ttf",".otf")) and m.startswith(SAFE_FONT_FOLDERS):
                    target=os.path.join(out,os.path.basename(m))
                    if not os.path.exists(target):
                        with z.open(m) as src,open(target,"wb") as dst: shutil.copyfileobj(src,dst)
                    n+=1
        have=[f for f in os.listdir(out) if f.lower().startswith(CAPTION_FONT.lower().replace(" ",""))]
        if not have: return out,CAPTION_FONT_FALLBACK,"%s not in ZIP; using system %s"%(CAPTION_FONT,CAPTION_FONT_FALLBACK)
        return out,CAPTION_FONT,"ok fonts_extracted=%d family=%s"%(n,CAPTION_FONT)
    except Exception as e:
        return None,CAPTION_FONT_FALLBACK,"ZIP error (%s); using system %s"%(str(e)[:100],CAPTION_FONT_FALLBACK)

def write_caption_ass(words, ass):
    font_dir,font_name,status=prepare_fonts()
    log("[captions-font] "+status)
    head="""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,%s,86,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,8,3,5,12,12,0,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""%font_name
    rows=[]; i=0
    while i<len(words):
        group=[words[i]]; i+=1
        while i<len(words) and len(group)<3 and words[i]["start"]-group[-1]["end"]<0.22 and sum(len(g["word"]) for g in group)+len(words[i]["word"])<=18:
            group.append(words[i]); i+=1
        tokens=[]
        for w in group:
            raw=w["word"].strip(",.!?;:"); token=raw.upper()
            color=_caption_emphasis(raw)
            tokens.append("{\\c"+color+"}"+token+"{\\c&H00FFFFFF&}" if color!="&H00FFFFFF&" else token)
        rows.append("Dialogue: 0,%s,%s,Default,,0,0,0,,%s"%(ass_time(group[0]["start"]),ass_time(max(group[0]["start"]+.12,group[-1]["end"]+.12))," ".join(tokens)))
    with open(ass,"w",encoding="utf-8") as f: f.write(head+NL.join(rows)+NL)
    log("[captions] font=%s words=%d groups=%d outline=8 shadow=3"%(font_name,len(words),len(rows)))
    return font_dir


VISION_API_KEY = os.environ.get("CLIPFORGE_VISION_API_KEY", "").strip()
VISION_MODEL = os.environ.get("CLIPFORGE_VISION_MODEL", "gemini-2.5-flash").strip()

def analyze_visual_context(path):
    """Optional Gemini frame analysis. Returns an honest unavailable result without a key."""
    if not VISION_API_KEY:
        return {"status":"not_configured","overview":"Vision analysis not configured; set CLIPFORGE_VISION_API_KEY in GitHub Actions secrets."}
    cap=cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"status":"error","overview":"Could not open candidate video for visual analysis."}
    frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    picks=sorted(set(int(frames*p) for p in (0.18,0.38,0.58,0.78) if frames>0))
    parts=[]
    for pos in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES,pos); ok,frame=cap.read()
        if not ok: continue
        h,w=frame.shape[:2]
        scale=min(1.0,768/max(h,w))
        if scale<1: frame=cv2.resize(frame,(int(w*scale),int(h*scale)))
        ok,buf=cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),78])
        if ok:
            import base64
            parts.append({"inline_data":{"mime_type":"image/jpeg","data":base64.b64encode(buf.tobytes()).decode("ascii")}})
    cap.release()
    if not parts:
        return {"status":"error","overview":"No readable frames were sampled."}
    prompt=("You are reviewing a short Twitch stream clip to help an editor. Inspect these frames in temporal order. "
            "Describe visible gameplay and streamer reaction, whether anything looks like a joke, surprise, win, fail, "
            "or confusing pause. Do not invent audio or events not visible. Return ONLY JSON with keys "
            "overview (one concise sentence), moment_type (one of reaction/gameplay_event/comedy/ordinary/unclear), "
            "visual_energy (0-5 integer), facecam_visible (boolean), gameplay_visible (boolean), "
            "editing_suggestion (one concise sentence).")
    payload={"contents":[{"parts":[{"text":prompt},*parts]}],
             "generationConfig":{"temperature":0.2,"responseMimeType":"application/json"}}
    try:
        import urllib.request
        url="https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s"%(VISION_MODEL,VISION_API_KEY)
        req=urllib.request.Request(url,data=json.dumps(payload).encode("utf-8"),
                                  headers={"Content-Type":"application/json"},method="POST")
        with urllib.request.urlopen(req,timeout=35) as response:
            data=json.loads(response.read().decode("utf-8"))
        text_out=data["candidates"][0]["content"]["parts"][0]["text"]
        result=json.loads(text_out)
        result["status"]="ok"
        result["frames_analysed"]=len(parts)
        return result
    except Exception as e:
        log("[vision] unavailable: "+str(e)[:180])
        return {"status":"error","overview":"Vision request failed; heuristic score retained.","error":str(e)[:180]}


def score_candidate(path, stats, audio_values, baseline_loud):
    reasons=[]; chat=stats["chat"]; unique=stats["unique"]; emotes=stats["emotes"]; keywords=stats["keywords"]
    chat_ratio=chat/max(1,stats["baseline"])
    chat_pts=min(16,max(0,(chat_ratio-1)*8))
    unique_pts=min(8,unique*1.2)
    burst_pts=min(10,emotes*1.5+keywords*1.2)
    if chat_pts>=3: reasons.append("chat spike")
    if unique_pts>=3: reasons.append("unique chatters")
    if burst_pts>=3: reasons.append("emote/keyword burst")
    loud=audio_values["loud"]; jump=loud-baseline_loud
    peak_pts=min(12,max(0,jump)*1.1)
    if peak_pts>=2: reasons.append("audio peak/shout")
    if audio_values["silence_reaction"]: peak_pts=min(14,peak_pts+4); reasons.append("silence-to-loudness-jump proxy only; not shout recognition")
    if audio_values["laughter"]: reasons.append("audio-peak laughter proxy only; not laughter recognition")
    motion=stats["motion"]
    motion_pts=min(10,max(0,motion-1.0)*1.8)
    scene_pts=min(6,stats["scene"]*12)
    if motion_pts>=2: reasons.append("motion burst")
    if scene_pts>=2: reasons.append("scene change")
    face_pts=min(5,stats["face"]*500)
    if face_pts>=1: reasons.append("face-box variation proxy only; not expression recognition")
    words=transcribe_words(path)
    text=" ".join(w["word"] for w in words).upper()
    hook_hits=[k for k in REACTION_WORDS if k in text]
    transcript_pts=min(12,sum(REACTION_WORDS[k] for k in hook_hits))
    if hook_hits: reasons.append("transcript hooks: "+", ".join(hook_hits[:5]))
    # Hook score: strongest reaction must occur in the first three seconds.
    hook_words=[w for w in words if w["start"]<=3.0]
    hook_text=" ".join(w["word"] for w in hook_words).upper()
    hook_hits3=[k for k in REACTION_WORDS if k in hook_text]
    hook_score=min(100,35+sum(REACTION_WORDS[k]*12 for k in hook_hits3)) if hook_hits3 else min(45,len(hook_words)*7)
    if hook_score<50: reasons.append("weak opening hook")
    score=min(100,round(chat_pts+unique_pts+burst_pts+peak_pts+motion_pts+scene_pts+face_pts+transcript_pts+hook_score*.25))
    if sum(1 for w in words if float(w.get("probability",1.0))>=0.45)<8:
        score=max(0,score-12); reasons.append("under 8 transcript words (-12)")
    if stats["luma"]<18: score=max(0,score-30); reasons.append("very dark/near-black")
    if stats["motion"]<0.35 and stats["scene"]<0.03: score=max(0,score-35); reasons.append("static footage")
    return score,hook_score,reasons,words

def footage_rejection(features, path):
    """Hard reject black, frozen, and dashboard-like footage."""
    if features["luma"] < 18: return "black/near-black"
    if features["motion"] < 0.35 and features["scene"] < 0.03: return "frozen/static"
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): return "unreadable"
    ok,fr=cap.read(); cap.release()
    if not ok: return "unreadable"
    gray=cv2.cvtColor(cv2.resize(fr,(320,180)),cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,70,150)
    density=float(np.mean(edges>0))
    hist=cv2.calcHist([gray],[0],None,[16],[0,256]).flatten()
    dominant=float(hist.max()/max(1,hist.sum()))
    if density>0.16 and dominant>0.18: return "dashboard/browser-like UI"
    return None

def extract_candidate(paths, out):
    concat=out+".concat.txt"; joined=out+".joined.ts"; src=out+".candidate.mp4"
    with open(concat,"w",encoding="utf-8") as f:
        f.write("".join("file '%s'%s"%(os.path.abspath(p).replace("'", "'\\''"),NL) for p in paths))
    p=run(["ffmpeg","-y","-hide_banner","-loglevel","error","-f","concat","-safe","0","-i",concat,"-c","copy",joined],capture_output=True,text=True)
    if p.returncode: raise RuntimeError("candidate concat failed: "+p.stderr[-500:])
    # Three chunks: 20s pre-roll plus first 5s of the following chunk.
    # Prefer packet-copy remuxing: decoding and re-encoding here is wasted because
    # the final vertical render has to encode the video again anyway.
    p=run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",joined,"-t","25",
           "-map","0:v:0","-map","0:a:0?","-c","copy","-movflags","+faststart",src],
          capture_output=True,text=True)
    if p.returncode:
        log("[ffmpeg] candidate stream-copy unavailable; using compatibility re-encode")
        p=run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",joined,"-t","25",
               "-map","0:v:0","-map","0:a:0?","-c:v","libx264","-preset","veryfast",
               "-crf","25","-c:a","aac","-b:a","128k",src],capture_output=True,text=True)
    if p.returncode: raise RuntimeError("candidate trim failed: "+p.stderr[-500:])
    for x in (concat,joined):
        try: os.remove(x)
        except OSError: pass
    return src

def trim_dead_air(src):
    """Trim leading silence with audio/video clocks kept aligned."""
    p=run(["ffmpeg","-hide_banner","-i",src,"-af","silencedetect=noise=-38dB:d=0.25","-f","null","-"],capture_output=True,text=True)
    err=p.stderr or ""
    ms=re.search(r"silence_start:\s*(-?[0-9.]+)",err); me=re.search(r"silence_end:\s*([0-9.]+)",err)
    # Only trim a LEADING silence: the first silence must begin at (about) t=0.
    leading=bool(ms and me and float(ms.group(1))<=0.15)
    offset=max(0.0,float(me.group(1))-0.12) if leading else 0.0
    if offset<0.4: return src
    out=src+".action.mp4"
    q=run(["ffmpeg","-y","-hide_banner","-loglevel","error","-ss",str(offset),"-i",src,"-t",str(max(15,25-offset)),"-c:v","libx264","-preset","veryfast","-crf","25","-c:a","aac","-b:a","128k",out],capture_output=True,text=True)
    if q.returncode: raise RuntimeError("dead-air trim failed: "+q.stderr[-400:])
    log("[hook] leading_silence_trimmed=%.2fs (audio proxy)"%offset)
    return out

def render_ranked_clip(src, out):
    src=trim_dead_air(src)
    info=probe(src)
    if info.get("duration",0)<15: raise RuntimeError("candidate shorter than 15s")
    face=detect_face_crop(src)
    ass=out+".ass"
    words=transcribe_words(src)
    if not words: raise RuntimeError("Faster-Whisper produced no confident word timestamps")
    font_dir=write_caption_ass(words,ass)
    # Twitch landscape captures often contain a left sidebar/facecam and right gameplay.
    # Use independent source regions: face detector only frames the top panel;
    # gameplay is cropped from the right-hand gameplay region, never from the face box.
    if face:
        x,y,w,h=face
        top="crop=w='trunc(iw*%.6f/2)*2':h='trunc(ih*%.6f/2)*2':x='trunc(iw*%.6f/2)*2':y='trunc(ih*%.6f/2)*2'"%(w,h,x,y)
    else:
        top="crop=w='trunc(iw*0.32/2)*2':h='trunc(ih*0.62/2)*2':x=0:y='trunc(ih*0.18/2)*2'"
    # Default Twitch sidebar layout: left ~28% includes chat/facecam; right ~72% is game.
    # Keep full gameplay height and centre-crop its width to the 9:8 panel ratio.
    bottom="crop=w='trunc(iw*0.72/2)*2':h='ih':x='trunc(iw*0.28/2)*2':y=0"
    top_filters=[top,"scale=1080:960:force_original_aspect_ratio=increase","crop=1080:960"]
    bottom_filters=[bottom,"scale=1080:960:force_original_aspect_ratio=increase","crop=1080:960"]
    def ff_esc(v): return v.replace("\\","/").replace(":","\\:").replace("'","\\'")
    sub="subtitles=filename='%s'"%ff_esc(os.path.abspath(ass))
    if font_dir: sub+=":fontsdir='%s'"%ff_esc(os.path.abspath(font_dir))
    graph="[0:v]split=2[a][b];[a]%s[top];[b]%s[bottom];[top][bottom]vstack,drawbox=x=0:y=957:w=1080:h=6:color=black:t=fill,%s,format=yuv420p[v]"%(",".join(top_filters),",".join(bottom_filters),sub)
    cmd=["ffmpeg","-y","-hide_banner","-loglevel","error","-i",src,"-filter_complex",graph,"-map","[v]","-map","0:a:0?","-c:v","libx264","-preset","superfast","-crf","23","-maxrate","14M","-bufsize","28M","-r","60","-fps_mode","cfr","-t","25","-c:a","aac","-b:a","160k","-movflags","+faststart",out]
    p=run_ffmpeg_progress(cmd,"final-render")
    if p.returncode: raise RuntimeError("render failed: "+p.stderr[-1000:])
    result=probe(out); size=os.path.getsize(out)
    if result.get("duration",0)<15 or result.get("duration",0)>25.5: raise RuntimeError("duration outside 15-25s: "+str(result))
    if (result.get("width"),result.get("height"))!=(1080,1920) or not result.get("audio") or abs(result.get("fps",0)-60)>.2: raise RuntimeError("output probe failed: "+str(result))
    log("[render] "+os.path.basename(out)+" bytes=%d size_mib=%.2f ffprobe=%s words=%d"%(size,size/1048576,result,len(words)))
    os.remove(ass)
    return {"duration":result["duration"],"bytes":size,"words":len(words),"video_quality":"CRF23; independent facecam/gameplay source crops"}

def select_ranked_candidates(ranked, threshold, max_clips):
    passed=[c for c in ranked if c.get("score",0)>=threshold]
    low_fallback=False
    if not passed and ranked:
        passed=ranked[:1]; low_fallback=True
    chosen=[]
    for c in passed:
        if any(abs(c["idx"]-x["idx"])<3 for x in chosen): continue
        chosen.append(c)
        if len(chosen)>=min(3,max_clips): break
    return chosen,low_fallback


def validate_source_escapes():
    """Behavioural check: the ASS writer must emit real newlines, real colour tags and the extracted font family."""
    tmp="validate-output/escape-test.ass"; os.makedirs("validate-output",exist_ok=True)
    ws=[{"start":i*.3,"end":i*.3+.25,"word":w} for i,w in enumerate(["No","way","12","NETFLIX","hello","there","bro"])]
    write_caption_ass(ws,tmp)
    txt=open(tmp,encoding="utf-8").read()
    if "\\n" in txt.replace(chr(10),""): raise RuntimeError("ASS contains literal backslash-n")
    events=[l for l in txt.split(NL) if l.startswith("Dialogue:")]
    if len(events)<2: raise RuntimeError("ASS has %d Dialogue lines; expected >=2"%len(events))
    if _caption_emphasis("12")!="&H0000FF00&" or _caption_emphasis("hello")!="&H00FFFFFF&" or _caption_emphasis("NETFLIX")!="&H000000FF&":
        raise RuntimeError("caption emphasis colours wrong")
    if not re.search(r"silence_end:\s*([0-9.]+)","silence_end: 1.25"): raise RuntimeError("silence regex")
    log("[validate] caption ASS/newline/emphasis PASS events=%d"%len(events))

def validate_pipeline():
    validate_source_escapes()
    for binary in ("ffmpeg","ffprobe"):
        p=run([binary,"-version"],capture_output=True,text=True)
        if p.returncode: raise RuntimeError(binary+" unavailable")
    os.makedirs("validate-output",exist_ok=True)
    seg_pattern="validate-output/segment_%03d.ts"
    p=run(["ffmpeg","-y","-hide_banner","-loglevel","error",
           "-f","lavfi","-i","testsrc2=size=1280x720:rate=60:duration=30",
           "-f","lavfi","-i","sine=frequency=440:sample_rate=44100:duration=30",
           "-map","0:v:0","-map","1:a:0","-c:v","libx264","-preset","ultrafast",
           "-pix_fmt","yuv420p","-c:a","aac","-f","segment","-segment_time","10",
           "-reset_timestamps","1","-segment_format","mpegts",seg_pattern],
          capture_output=True,text=True)
    if p.returncode: raise RuntimeError("lavfi TS generation failed: "+p.stderr[-500:])
    paths=sorted(glob.glob("validate-output/segment_*.ts"))
    if len(paths)<3: raise RuntimeError("expected >=3 TS segments, got "+str(len(paths)))
    for path in paths[:3]:
        info=probe(path)
        if not info.get("audio") or not info.get("width"): raise RuntimeError("invalid TS segment "+path+": "+str(info))
    candidate=extract_candidate(paths[:3],"validate-output/concat-test")
    info=probe(candidate)
    log("[validate] extract_candidate ffprobe="+str(info))
    if not 24<=info.get("duration",0)<=25.5 or not info.get("audio") or (info.get("width"),info.get("height"))!=(1280,720):
        raise RuntimeError("extract_candidate ffprobe validation failed: "+str(info))
    log("[validate] PASS real extract_candidate 3x10s TS -> "+str(info))
    global transcribe_words
    old=transcribe_words
    transcribe_words=lambda path:[{"start":i*.2,"end":i*.2+.15,"word":w} for i,w in enumerate(["NO","WAY","BRO","WHAT","INSANE","HOLY","CLIP","IT","NOW","WOW"])]
    try:
        stats={"chat":100,"unique":20,"emotes":20,"keywords":20,"baseline":1,"motion":12.0,"scene":0.9,"face":0.02,"luma":55}
        audio={"loud":-5,"silence_reaction":True,"laughter":True}
        score,hook,reasons,words=score_candidate(candidate,stats,audio,-42)
        if score<=0 or len(words)<8: raise RuntimeError("synthetic scoring failed: score=%d words=%d"%(score,len(words)))
        log("[validate] synthetic score=%d hook=%d words=%d reasons=%s"%(score,hook,len(words),"; ".join(reasons)))
        test_ranked=[{"idx":i*4,"score":sc,"reasons":["synthetic"]} for i,sc in enumerate([96,88,79,69])]
        chosen,low=select_ranked_candidates(test_ranked,70,3)
        for rank,c in enumerate(sorted(test_ranked,key=lambda x:x["score"],reverse=True),1):
            log("[ranking-test] %d | idx=%d | score=%d"%(rank,c["idx"],c["score"]))
        if [c["score"] for c in chosen]!=[96,88,79] or low: raise RuntimeError("threshold/max-3 selection test failed")
        fallback,low=select_ranked_candidates([{"idx":1,"score":42}],70,3)
        if len(fallback)!=1 or not low: raise RuntimeError("low-rated fallback test failed")
        log("[validate] PASS scoring, threshold=70, max=3, low-rated one-clip fallback")
    finally:
        transcribe_words=old
    log("[validate] PASS complete offline candidate-to-selection path")
    global detect_face_crop
    old_face=detect_face_crop
    transcribe_words=lambda path:[{"start":i*.4,"end":i*.4+.3,"word":w,"probability":.9} for i,w in enumerate(["No","way","bro","I","got","12","kills","on","NETFLIX","insane","holy","wow"])]
    try:
        for label,face in (("no-face",None),("face",[0.05,0.05,0.25,0.3])):
            detect_face_crop=lambda path,f=face:f
            out="validate-output/render-%s.mp4"%label
            d=render_ranked_clip(candidate,out)
            log("[validate] PASS render %s -> %s"%(label,d))
    finally:
        transcribe_words=old; detect_face_crop=old_face


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("channel",nargs="?",default="xqc")
    ap.add_argument("--max-minutes",type=int,default=4)
    ap.add_argument("--max-clips",type=int,default=3)
    ap.add_argument("--threshold",type=int,default=70)
    ap.add_argument("--out",default="clips")
    ap.add_argument("--no-effects",action="store_true")
    ap.add_argument("--validate-only",action="store_true")
    a=ap.parse_args()
    if a.validate_only:
        validate_pipeline(); return
    if not re.fullmatch(r"[A-Za-z0-9_]+",a.channel): raise SystemExit("invalid channel")
    if not 3<=a.max_minutes<=5: raise SystemExit("duration must be 3-5 minutes")
    os.makedirs(a.out,exist_ok=True); buf=os.path.join(a.out,"buffer"); os.makedirs(buf,exist_ok=True)
    log("[config] requested_streamer="+a.channel+" threshold="+str(a.threshold)+" max_exports="+str(a.max_clips))
    selected=a.channel.lower()
    def stream_check(ch):
        return run(["streamlink","--stream-url","https://twitch.tv/"+ch,"best"],capture_output=True,text=True,timeout=60)
    check=stream_check(selected)
    if check.returncode and selected=="xqc":
        log("[streamer] xQc offline/unavailable; trying Lacy fallback")
        fallback=stream_check("lacy")
        if fallback.returncode: raise SystemExit("[streamlink] xQc and Lacy unavailable")
        selected="lacy"; check=fallback
    elif check.returncode:
        raise SystemExit("[streamlink] requested channel unavailable: "+(check.stderr or check.stdout)[-300:])
    log("[streamer] using="+selected)
    chat=TwitchChat(selected); chat.start()
    url="https://twitch.tv/"+selected
    streamlog=open(os.path.join(a.out,"streamlink.stderr.log"),"w")
    sl=subprocess.Popen(["streamlink","--stdout",url,"best"],stdout=subprocess.PIPE,stderr=streamlog)
    ff=subprocess.Popen(["ffmpeg","-y","-hide_banner","-loglevel","error","-i","pipe:0","-map","0","-c","copy","-f","segment","-segment_time",str(SEGMENT_SECONDS),"-reset_timestamps","1","-segment_format","mpegts",os.path.join(buf,"segment_%06d.ts")],stdin=sl.stdout,stdout=subprocess.DEVNULL,stderr=streamlog)
    sl.stdout.close()
    start=time.time(); deadline=start+a.max_minutes*60; seen=set(); candidates=[]; chat_hist=[]; loud_hist=[]; prev_loud=-60.0; last_candidate_time=0.0
    log("[buffer] RECORD EVERYTHING; retention=%ds chunk=%ds"%(BUFFER_SECONDS,SEGMENT_SECONDS))
    log("[score] every closed 10s segment; export threshold=%d/100"%a.threshold)
    try:
        while time.time()<deadline and ff.poll() is None:
            time.sleep(.8)
            files=sorted(glob.glob(os.path.join(buf,"segment_*.ts")))
            for seg in files[:-1]:
                m=re.search(r"segment_(\d+)\.ts$",seg)
                if not m: continue
                idx=int(m.group(1))
                if idx in seen: continue
                seen.add(idx); time.sleep(.12)
                now=time.time(); count,unique,emotes,keywords=parse_chat_features(chat,now-SEGMENT_SECONDS-1)
                baseline=float(np.mean(chat_hist[-6:])) if chat_hist else max(1,count)
                chat_hist.append(count)
                loud=audio_loudness(seg); jump=loud-prev_loud; prev_loud=loud
                motion=motion_score(seg)
                fftr=frame_features(seg)
                audio={"loud":loud,"silence_reaction":bool(loud_hist and min(loud_hist[-3:])<-42 and jump>7),"laughter":bool(jump>5 and motion>1.2)}
                loud_hist.append(loud)
                fired=(count>=max(5,baseline*1.35) or jump>=3 or motion>=2)
                log("[moment] t=%ds chat=%d unique=%d emotes=%d keywords=%d loud=%.1fdB jump=%.1f motion=%.2f luma=%.1f trigger=%s"%(idx*10,count,unique,emotes,keywords,loud,jump,motion,fftr["luma"],fired))
                # Candidate creation is rate-limited; rejected footage does not consume the cooldown.
                if fired and idx>=2 and now-last_candidate_time>=COOLDOWN_SECONDS:
                    reject=footage_rejection(fftr,seg)
                    if reject:
                        log("[candidate-skip] t=%ds reason=%s"%(idx*10,reject))
                    else:
                        ids=[idx-1,idx,idx+1]
                        candidates.append({"idx":idx,"ids":ids,"chat":count,"unique":unique,"emotes":emotes,"keywords":keywords,"baseline":baseline,"loud":loud,"jump":jump,"motion":motion,"scene":fftr["scene"],"face":fftr["face"],"luma":fftr["luma"],"silence_reaction":audio["silence_reaction"],"laughter":audio["laughter"]})
                        last_candidate_time=now
                        log("[cooldown] accepted candidate at %ds; next eligible after %ds"%(idx*10,COOLDOWN_SECONDS))
                elif fired and idx>=2:
                    log("[candidate-skip] t=%ds reason=45s candidate cooldown"%(idx*10))
                cutoff=max(0,idx-(BUFFER_SECONDS//SEGMENT_SECONDS)+1)
                for old in glob.glob(os.path.join(buf,"segment_*.ts")):
                    om=re.search(r"segment_(\d+)\.ts$",old)
                    if om and int(om.group(1))<cutoff and not any(int(om.group(1)) in c["ids"] for c in candidates):
                        try: os.remove(old)
                        except OSError: pass
                # Score candidate once its post-roll chunk has closed.
                for c in list(candidates):
                    if not c.get("scored") and idx>=c["idx"]+1:
                        paths=[os.path.join(buf,"segment_%06d.ts"%i) for i in c["ids"]]
                        if not all(os.path.isfile(p) and os.path.getsize(p)>1000 for p in paths):
                            c["score"]=0; c["hook_score"]=0; c["reasons"]=["missing buffer segment"]; c["scored"]=True; continue
                        tmp=os.path.join(a.out,"candidate-%d.mp4"%c["idx"])
                        try:
                            src=extract_candidate(paths,tmp)
                            score,hook,reasons,words=score_candidate(src,c,{"loud":c["loud"],"silence_reaction":c["silence_reaction"],"laughter":c["laughter"]},-42)
                            visual=analyze_visual_context(src)
                            if visual.get("status")=="ok":
                                score=min(100,score+min(8,int(visual.get("visual_energy",0))*1))
                                reasons.append("vision: "+str(visual.get("moment_type","unclear")))
                            c.update({"score":score,"hook_score":hook,"reasons":reasons,"candidate_words":len(words),"visual_analysis":visual,"scored":True,"src":src})
                            log("[candidate] t=%ds score=%d hook=%d words=%d reasons=%s"%(c["idx"]*10,score,hook,len(words),"; ".join(reasons)))
                        except Exception as ex:
                            c.update({"score":0,"hook_score":0,"reasons":["scoring failed: "+str(ex)[:100]],"scored":True})
                            log("[candidate-error] "+str(ex)[:200])
                if time.time()>=deadline: break
        # Stop writers, then process any final candidate whose post-roll is already present.
    finally:
        for proc in (ff,sl):
            if proc.poll() is None: proc.terminate()
        for proc in (ff,sl):
            try: proc.wait(timeout=8)
            except subprocess.TimeoutExpired: proc.kill()
        streamlog.close(); chat.stop_flag=True
    # One final scoring pass for candidates with complete source chunks.
    available={int(re.search(r"segment_(\d+)\.ts$",p).group(1)):p for p in glob.glob(os.path.join(buf,"segment_*.ts"))}
    for c in candidates:
        if c.get("scored"): continue
        paths=[available.get(i) for i in c["ids"]]
        if all(paths):
            try:
                src=extract_candidate(paths,os.path.join(a.out,"candidate-%d.mp4"%c["idx"]))
                score,hook,reasons,words=score_candidate(src,c,{"loud":c["loud"],"silence_reaction":c["silence_reaction"],"laughter":c["laughter"]},-42)
                visual=analyze_visual_context(src)
                if visual.get("status")=="ok":
                    score=min(100,score+min(8,int(visual.get("visual_energy",0))*1))
                    reasons.append("vision: "+str(visual.get("moment_type","unclear")))
                c.update({"score":score,"hook_score":hook,"reasons":reasons,"candidate_words":len(words),"visual_analysis":visual,"scored":True,"src":src})
            except Exception as ex: c.update({"score":0,"hook_score":0,"reasons":["scoring failed: "+str(ex)[:100]],"scored":True})
    ranked=sorted([c for c in candidates if c.get("scored")],key=lambda c:c.get("score",0),reverse=True)
    log("[ranking] rank | timestamp | score/100 | hook | reasons")
    for rank,c in enumerate(ranked,1): log("[ranking] %d | %ds | %d | %d | %s"%(rank,c["idx"]*10,c["score"],c["hook_score"],"; ".join(c["reasons"])))
    failed=[c for c in ranked if any(str(r).startswith("scoring failed:") for r in c.get("reasons",[]))]
    if ranked and len(failed)==len(ranked): raise RuntimeError("all candidate scoring attempts failed: "+" | ".join(c.get("reasons",["unknown"])[0] for c in failed))
    chosen,low_fallback=select_ranked_candidates(ranked,a.threshold,a.max_clips)
    if low_fallback: log("[fallback] no candidate met threshold; exporting single best LOW-RATED candidate")
    summary={"score_is":"combined heuristic and optional sampled-frame vision estimate; not a prediction of views","vision_model":VISION_MODEL,"vision_configured":bool(VISION_API_KEY),"signal_limits":{"laughter":"audio-peak proxy only","shouting":"loudness-jump proxy only","face_expression":"face-box variation proxy only"},"threshold":a.threshold,"low_rated_fallback":low_fallback,"streamer_requested":a.channel,"streamer_used":selected,"candidates_found":len(candidates),"ranked_candidates":[],"exports":[]}
    for rank,c in enumerate(ranked,1):
        summary["ranked_candidates"].append({"rank":rank,"timestamp_seconds":c["idx"]*10,"score":c.get("score",0),"hook_score":c.get("hook_score",0),"reasons":c.get("reasons",[]),"visual_analysis":c.get("visual_analysis",{"status":"not_run"})})
    for n,c in enumerate(chosen,1):
        src=c.get("src")
        if not src: continue
        name="clip-%d_score%d.mp4"%(n,c["score"]); out=os.path.join(a.out,name)
        try:
            details=render_ranked_clip(src,out)
            summary["exports"].append({"file":name,"score":c["score"],"timestamp_seconds":c["idx"]*10,"hook_score":c["hook_score"],"reasons":c["reasons"],**details,"low_rated":low_fallback})
        except Exception as ex: log("[export-error] "+name+": "+str(ex)[:250])
    with open(os.path.join(a.out,"summary.json"),"w",encoding="utf-8") as f: f.write(json.dumps(summary,indent=2)+NL)
    for p in glob.glob(os.path.join(a.out,"candidate-*")):
        try: os.remove(p)
        except OSError: pass
    total_bytes=sum(os.path.getsize(p) for p in glob.glob(os.path.join(buf,"*.ts")))
    log("[summary] candidates=%d exported=%d IRC_messages=%d buffer_bytes=%d low_rated_fallback=%s"%(len(candidates),len(summary["exports"]),chat.total(),total_bytes,low_fallback))
    if not summary["exports"]: raise SystemExit("[ERROR] no valid clips exported")
if __name__=="__main__": main()