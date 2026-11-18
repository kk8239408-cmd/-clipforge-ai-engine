#!/usr/bin/env python3
"""
CLIPFORGE clip_watcher_v5.py - free Twitch auto-clipper with real editing.

PIPELINE (per chat-spike candidate)
  1. cut the candidate window out of the live recording (stream copy, no quality loss)
  2. ANALYSE it BEFORE rendering: black / frozen footage, motion, loudness peak
     (+ optional CLIP zero-shot frame scoring) -> reject bad candidates, never render them
  3. EDIT in ONE re-encode: layout (blur|split|full), slow push-in + punch-in zoom on the
     loudest moment, brightness flash, pop captions, optional emoji stickers
  4. AUDIO: original stream kept at full level; optional synthetic riser/impact/whoosh timed to the
     peak; optional music (your own royalty-free file) auto-ducked under the streamer
  5. VERIFY the finished file (size, fps, duration, A/V sync, black/frozen, audio present, no clipping)

HONEST LIMITS
  * "AI" here = loudness/motion/chat signals + optional CLIP zero-shot frame scoring. It does NOT
    understand what happens in the clip, and it cannot write a voiceover that describes it.
  * Emoji stickers are keyword-triggered (from the transcript), not scene-aware.
"""
import argparse, collections, csv, glob, json, math, os, queue, random, re, shutil, socket
import subprocess, sys, tempfile, threading, time, wave

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install numpy")

BOTS = {"nightbot", "streamelements", "moobot", "streamlabs", "fossabot", "wizebot", "sery_bot", "commanderroot"}
HYPE = re.compile(r"(lul|kekw|omegalul|pog|pogchamp|poggers|omg|lmao|lmfao|lol|wtf|no way|nooo|clip|clutch|insane|crazy|"
                  r"holy|bruh|dead|rip|ratio|w\b|gg|huge|cracked|cooked|\?\?|!!)", re.I)
SR = 48000

# ============================================================== chat reader
class ChatCounter(threading.Thread):
    def __init__(self, channel):
        super().__init__(daemon=True)
        self.channel = channel.lower().lstrip("#")
        self.stamps = collections.deque()
        self.lock = threading.Lock()

    def run(self):
        while True:
            try:
                s = socket.create_connection(("irc.chat.twitch.tv", 6667), timeout=30)
                s.settimeout(300)
                s.sendall(b"NICK justinfan%d\r\n" % (int(time.time()) % 90000 + 1000))
                s.sendall(f"JOIN #{self.channel}\r\n".encode())
                buf = b""
                while True:
                    data = s.recv(4096)
                    if not data:
                        break
                    buf += data
                    while b"\r\n" in buf:
                        line, buf = buf.split(b"\r\n", 1)
                        if line.startswith(b"PING"):
                            s.sendall(b"PONG :tmi.twitch.tv\r\n")
                        elif b" PRIVMSG " in line:
                            try:
                                txt = line.decode("utf-8", "ignore")
                                user = txt[1:txt.index("!")].lower()
                                msg = txt.split(" PRIVMSG ", 1)[1].split(" :", 1)[1]
                            except Exception:
                                continue
                            with self.lock:
                                self.stamps.append((time.time(), user, msg.strip().lower()))
            except Exception as e:
                print(f"[chat] reconnecting ({e})", flush=True)
                time.sleep(5)

    def score_since(self, t0):
        """Spam-resistant bucket score -> (score, raw_msgs, unique_users, top_text)."""
        with self.lock:
            while self.stamps and self.stamps[0][0] < t0 - 600:
                self.stamps.popleft()
            msgs = [(u, m) for t, u, m in self.stamps if t >= t0 and u not in BOTS]
        if not msgs:
            return 0.0, 0, 0, ""
        users = len({u for u, _ in msgs})
        seen, eff, hype = set(), 0, 0
        cap = collections.Counter()
        for u, m in msgs:
            if (u, m) in seen or cap[u] >= 2:
                continue
            seen.add((u, m)); cap[u] += 1; eff += 1
            if HYPE.search(m):
                hype += 1
        top_text, top_n = collections.Counter(m for _, m in msgs).most_common(1)[0]
        score = eff + 0.5 * hype
        if top_n / len(msgs) > 0.7 and len(msgs) >= 8:
            score *= 0.6
        return score, len(msgs), users, top_text

# ============================================================== ffmpeg helpers
def run(cmd, timeout=600):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout or f"{cmd[0]} failed")[-1500:])
    return p

def fnum(v):
    try:
        if "/" in str(v):
            a, b = str(v).split("/"); return float(a) / float(b) if float(b) else 0.0
        return float(v)
    except Exception:
        return 0.0

def probe(path):
    """-> dict(dur, w, h, fps, has_v, has_a, vdur, adur)"""
    out = dict(dur=0.0, w=0, h=0, fps=0.0, has_v=False, has_a=False, vdur=0.0, adur=0.0)
    try:
        j = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,width,height,r_frame_rate,duration:format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=30).stdout or "{}")
        out["dur"] = fnum(j.get("format", {}).get("duration"))
        for s in j.get("streams", []):
            if s.get("codec_type") == "video" and not out["has_v"]:
                out.update(has_v=True, w=int(s.get("width", 0)), h=int(s.get("height", 0)),
                           fps=fnum(s.get("r_frame_rate")), vdur=fnum(s.get("duration")))
            elif s.get("codec_type") == "audio" and not out["has_a"]:
                out.update(has_a=True, adur=fnum(s.get("duration")))
    except Exception:
        pass
    return out

def gray_frames(path, rate=4, size=(96, 54)):
    w, h = size
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-an", "-vf",
                        f"fps={rate},scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
                       capture_output=True, timeout=180)
    n = len(p.stdout) // (w * h)
    if n == 0:
        return np.zeros((0, h, w), dtype=np.uint8)
    return np.frombuffer(p.stdout[: n * w * h], dtype=np.uint8).reshape(n, h, w)

def loudness(path):
    """-> (times, momentary LUFS array, integrated LUFS or None). Uses EBU R128 via ffmpeg."""
    p = subprocess.run(["ffmpeg", "-nostats", "-hide_banner", "-loglevel", "verbose", "-i", path, "-vn",
                        "-af", "ebur128=framelog=verbose", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=180)
    ts, ms = [], []
    for m in re.finditer(r"t:\s*([0-9.]+)\s+TARGET.*?M:\s*(-?[0-9.]+|-inf)", p.stderr):
        ts.append(float(m.group(1)))
        ms.append(-120.0 if "inf" in m.group(2) else max(-120.0, float(m.group(2))))
    integ = None
    mm = re.findall(r"\n\s+I:\s+(-?[0-9.]+) LUFS", p.stderr)
    if mm:
        integ = float(mm[-1])
    return np.array(ts), np.array(ms), integ

def analyse(path, pre_hint=None):
    """Signal analysis of a candidate (or a finished clip). No AI claims: just measurements."""
    info = probe(path)
    res = dict(info=info, black=1.0, frozen=0.0, motion=0.0, peak_t=None, lift=0.0, integrated=None)
    if not info["has_v"] or info["dur"] <= 0:
        return res
    fr = gray_frames(path).astype(np.float32)
    if len(fr) >= 2:
        res["black"] = float(np.mean(fr.mean(axis=(1, 2)) < 14))
        d = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))
        res["motion"] = float(d.mean())
        res["frozen"] = float(np.mean(d < 0.25))
    elif len(fr) == 1:
        res["black"] = float(fr.mean() < 14)
    if info["has_a"]:
        ts, ms, integ = loudness(path)
        res["integrated"] = integ
        if len(ms) > 10:
            k = 5
            sm = np.convolve(ms, np.ones(k) / k, mode="same")
            lo, hi = 4.0, max(5.0, info["dur"] - 4.0)
            mask = (ts >= lo) & (ts <= hi)
            if mask.any():
                idx = np.where(mask)[0]
                best = idx[np.argmax(sm[idx])]
                valid = ms[ms > -70]
                base = float(np.median(valid)) if len(valid) else -70.0
                res["lift"] = float(sm[best] - base)
                res["peak_t"] = float(ts[best])
    return res

# ============================================================== optional CLIP scoring
CLIP_GOOD = ["a streamer laughing or reacting with shock to something", "intense action in a video game",
             "a dramatic victory or kill in a video game", "a funny or unexpected moment on a livestream"]
CLIP_BAD = ["a video game menu or lobby screen", "a loading screen", "a black screen", "a static screen with nothing happening"]
_clip = {}

def clip_review(path, dur):
    """Zero-shot CLIP frame scoring. Returns dict or None when unavailable. NOT scene understanding."""
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
        if "m" not in _clip:
            name = "openai/clip-vit-base-patch32"
            _clip["m"] = CLIPModel.from_pretrained(name).eval()
            _clip["p"] = CLIPProcessor.from_pretrained(name)
        imgs = []
        with tempfile.TemporaryDirectory() as td:
            for i in range(8):
                f = os.path.join(td, f"{i}.jpg")
                run(["ffmpeg", "-y", "-v", "error", "-ss", f"{dur * (i + 1) / 9:.2f}", "-i", path,
                     "-frames:v", "1", "-vf", "scale=448:-2", f], timeout=30)
                imgs.append(Image.open(f).convert("RGB"))
            inp = _clip["p"](text=CLIP_GOOD + CLIP_BAD, images=imgs, return_tensors="pt", padding=True)
            with torch.no_grad():
                pr = _clip["m"](**inp).logits_per_image.softmax(dim=1).numpy()
        good, bad = float(pr[:, :len(CLIP_GOOD)].sum(1).mean()), float(pr[:, len(CLIP_GOOD):].sum(1).mean())
        top = (CLIP_GOOD + CLIP_BAD)[int(pr.mean(0).argmax())]
        return dict(good=good, bad=bad, score=round(100 * good), top=top)
    except Exception as e:
        print(f"[ai] CLIP unavailable, using signal scoring only ({str(e)[:90]})", flush=True)
        return None

# ============================================================== captions
def ass_time(t):
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"

def make_ass(words, path, font, style, layout, peak=None):
    if style == "pop":
        size, outline, align, mv = 118, 9, 5, 0
        if layout != "split":
            align, mv = 2, 620
    else:
        size, outline, align, mv = 84, 6, 2, (520 if layout != "full" else 420)
    head = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
            "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
            "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,"
            "MarginL,MarginR,MarginV,Encoding\n"
            f"Style: Default,{font},{size},&H00FFFFFF,&H0000FFFF,&H00000000,&H64000000,"
            f"-1,0,0,0,100,100,0,0,1,{outline},3,{align},60,60,{mv},1\n\n"
            "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    lines = []
    if style == "pop":
        cols = ["&H00FF00&", "&HFFFFFF&", "&H00FFFF&"]
        for i, (st, en, w) in enumerate(words):
            nxt = words[i + 1][0] if i + 1 < len(words) else en + 0.35
            end = max(st + 0.12, min(nxt, en + 0.35))
            hot = peak is not None and abs(st - peak) < 0.7          # words at the peak: red + bigger
            col, big = ("&H0000FF&", 128) if hot else (cols[i % 3], 100)
            txt = (r"{\fscx60\fscy60\t(0,110,\fscx%d\fscy%d)\1c%s}" % (big, big, col)) + w.upper().strip(",.!?;:")
            lines.append(f"Dialogue: 0,{ass_time(st)},{ass_time(end)},Default,,0,0,0,,{txt}")
    else:
        for i in range(0, len(words), 3):
            grp = words[i:i + 3]
            for j, (st, en, _) in enumerate(grp):
                end = grp[j + 1][0] if j + 1 < len(grp) else en
                txt = " ".join((r"{\c&H00FFFF&}" + w.upper() + r"{\c&HFFFFFF&}") if k == j else w.upper()
                               for k, (_, _, w) in enumerate(grp))
                lines.append(f"Dialogue: 0,{ass_time(st)},{ass_time(end)},Default,,0,0,0,,{txt}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(lines) + "\n")

_whisper = None
def transcribe(path):
    global _whisper
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return []
    if _whisper is None:
        _whisper = WhisperModel("base", device="cpu", compute_type="int8")
    segs, _ = _whisper.transcribe(path, word_timestamps=True, vad_filter=True)
    return [(w.start, w.end, w.word.strip()) for s in segs for w in (s.words or []) if w.word.strip()]

# ============================================================== stickers (keyword triggered, optional)
STICKER_WORDS = [("no way", "😱"), ("omg", "😱"), ("insane", "🤯"), ("crazy", "🤯"), ("what the", "😳"),
                 ("dead", "💀"), ("dying", "💀"), ("lol", "😂"), ("haha", "😂"), ("funny", "😂"),
                 ("clutch", "🔥"), ("let's go", "🔥"), ("lets go", "🔥"), ("win", "🏆"), ("won", "🏆")]
EMOJI_FONTS = ["/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", "/usr/share/fonts/noto/NotoColorEmoji.ttf"]

def pick_stickers(words, dur, max_n=2, min_gap=4.0):
    out, last = [], -99
    for i, (st, en, w) in enumerate(words):
        low = w.lower().strip(",.!?'\"")
        two = (low + " " + words[i + 1][2].lower().strip(",.!?'\"")) if i + 1 < len(words) else low
        for key, emo in STICKER_WORDS:
            if (" " in key and two.startswith(key)) or (" " not in key and low == key):
                if st - last >= min_gap and 1.0 < st < dur - 2.0:
                    out.append((emo, st)); last = st
                break
        if len(out) >= max_n:
            break
    return out

def make_sticker_png(emoji, path):
    """Needs NotoColorEmoji (apt: fonts-noto-color-emoji). Returns False when unavailable."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        fp = next((f for f in EMOJI_FONTS if os.path.exists(f)), None)
        if not fp:
            return False
        font = ImageFont.truetype(fp, 109)
        im = Image.new("RGBA", (136, 128), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((0, 0), emoji, font=font, embedded_color=True)
        im.resize((272, 256), Image.LANCZOS).save(path)
        return True
    except Exception:
        return False

# ============================================================== synthetic sound design (original, royalty-free)
def synth_sfx(dur, peak_t, hook=True, extra_ticks=()):
    """Returns float32 stereo array: hook whoosh, riser -> impact at peak_t, tick per sticker."""
    n = int(dur * SR)
    out = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(7)

    def add(sig, t0):
        i = int(t0 * SR)
        if i < 0 or i >= n:
            return
        sig = sig[: n - i]
        out[i:i + len(sig)] += sig.astype(np.float32)

    def lp(x, k):                                     # moving-average low-pass
        return np.convolve(x, np.ones(k) / k, mode="same")

    if hook:                                          # short whoosh at the start
        L = int(0.45 * SR); t = np.linspace(0, 1, L)
        w = lp(rng.standard_normal(L), 6) * np.sin(np.pi * t) ** 2
        add(0.5 * w / (np.abs(w).max() + 1e-9), 0.05)
    if peak_t is not None and peak_t > 1.2:
        L = int(0.9 * SR); t = np.linspace(0, 1, L)   # riser: noise + rising sweep, builds into the peak
        f = 300 + 2400 * t ** 2
        sweep = np.sin(2 * np.pi * np.cumsum(f) / SR)
        noise = lp(rng.standard_normal(L), 4)
        r = (0.6 * sweep + 0.5 * noise) * (t ** 2.2)
        add(0.5 * r / (np.abs(r).max() + 1e-9), peak_t - 0.9 + 0.12)
        L = int(0.7 * SR); t = np.arange(L) / SR      # impact: pitch-dropping sub + noise burst
        f = 48 + 90 * np.exp(-9 * t)
        sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-6 * t)
        burst = lp(rng.standard_normal(L), 12) * np.exp(-18 * t)
        im = sub + 0.6 * burst
        add(0.8 * im / (np.abs(im).max() + 1e-9), peak_t + 0.12)
    for tk in extra_ticks:                            # soft pop for stickers
        L = int(0.12 * SR); t = np.arange(L) / SR
        p = np.sin(2 * np.pi * (700 + 2500 * t) * t) * np.exp(-30 * t)
        add(0.35 * p, tk)
    out = np.clip(out, -1, 1) * 0.55
    return np.stack([out, out], axis=1)

def write_wav(path, stereo):
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(stereo, -1, 1) * 32767).astype("<i2").tobytes())

# ============================================================== video graph
def parse_box(s, name):
    try:
        x, y, w, h = [float(v) for v in s.split(",")]
        assert 0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1 and x + w <= 1.001 and y + h <= 1.001
        return x, y, w, h
    except Exception:
        sys.exit(f"--{name} must be 4 fractions 'x,y,w,h' between 0 and 1, e.g. 0.72,0.62,0.28,0.38")

def crop_expr(b):
    x, y, w, h = b
    return (f"crop=w='trunc(iw*{w}/2)*2':h='trunc(ih*{h}/2)*2':"
            f"x='trunc(iw*{x}/2)*2':y='trunc(ih*{y}/2)*2'")

def layout_graph(layout, cam, game, focus):
    """[vin] -> [v0] 1080x1920."""
    grade = "eq=saturation=1.2:contrast=1.05"
    if layout == "split" and cam:
        gc = (crop_expr(game) + ",") if game else ""
        return ("[vin]split=2[c][g];"
                f"[c]{crop_expr(cam)},scale=1080:960:force_original_aspect_ratio=increase,crop=1080:960[top];"
                f"[g]{gc}scale=-2:960,crop=1080:960:x='max(0,(iw-1080)*{focus})':y=0[bot];"
                f"[top][bot]vstack,drawbox=x=0:y=957:w=1080:h=6:color=black@1:t=fill,{grade},setsar=1[v0]")
    if layout == "full":
        return f"[vin]scale=-2:1920,crop=1080:1920:x='max(0,(iw-1080)*{focus})':y=0,{grade},setsar=1[v0]"
    return ("[vin]split[a][b];[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=30:5[bg];"
            "[b]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v0]")

def zoom_expr(peak, dur, strength=1.0):
    """slow push-in + quick punch-in around the peak + tiny snap at the start."""
    s = strength
    z = f"1+{0.035*s:.3f}*t/{dur:.2f}+{0.08*s:.3f}*max(0,1-t/0.35)"
    if peak is not None:
        z += f"+{0.13*s:.3f}*max(0,1-abs(t-{peak + 0.2:.2f})/0.55)"
    return f"({z})"

def build_filter(o, dur, peak, ass, stickers):
    g = "[0:v]setpts=PTS-STARTPTS[vin];" + layout_graph(o["layout"], o["cam"], o["game"], o["focus"])
    last = "v0"
    if o["effects"]:
        z = zoom_expr(peak, dur, o["zoom"])
        g += (f";[v0]split=2[zm][zz];[zz]scale=w='trunc(1080*{z}/2)*2':h='trunc(1920*{z}/2)*2':eval=frame[zs];"
              "[zm][zs]overlay=x='(W-w)/2':y='(H-h)/2':eval=frame[vz]")
        last = "vz"
        if peak is not None:
            g += f";[vz]eq=brightness='0.16*max(0,1-abs(t-{peak + 0.15:.2f})/0.10)':eval=frame[vf]"
            last = "vf"
    for i, (png, t0) in enumerate(stickers):
        k = 1 + i
        s = (f"max(0.05,if(lt(t-{t0:.2f},0.12),0.3+7*(t-{t0:.2f}),1.14-0.14*min(1,(t-{t0:.2f}-0.12)/0.15)))")
        xpos = 0.74 if i % 2 == 0 else 0.26
        ypos = 0.40 if o["layout"] != "split" else 0.43
        g += (f";[{k}:v]format=rgba,scale=w='trunc(270*{s}/2)*2':h='trunc(270*{s}/2)*2':eval=frame[st{i}];"
              f"[{last}][st{i}]overlay=x='W*{xpos}-w/2':y='H*{ypos}-h/2':eval=frame:"
              f"enable='between(t,{t0:.2f},{t0 + 1.4:.2f})'[vs{i}]")
        last = f"vs{i}"
    if ass:
        g += f";[{last}]subtitles={ass},format=yuv420p[o]"
    else:
        g += f";[{last}]format=yuv420p[o]"
    return g

def build_audio(info, dur, n_in_before_audio, music_idx, sfx_idx, o):
    """Original stream at full level; SFX quiet; music ducked under the streamer. -> filter string, label"""
    parts, labels = [], []
    if info["has_a"]:
        parts.append(f"[0:a]asetpts=PTS-STARTPTS,aresample={SR}[ao]")
        if music_idx is not None:
            parts.append("[ao]asplit=2[ao1][ak]")
            labels.append("[ao1]")
        else:
            labels.append("[ao]")
    if sfx_idx is not None:
        parts.append(f"[{sfx_idx}:a]aresample={SR},volume={o['sfx_gain']}[sx]")
        labels.append("[sx]")
    if music_idx is not None:
        parts.append(f"[{music_idx}:a]aresample={SR},atrim=0:{dur:.2f},asetpts=PTS-STARTPTS[mu]")
        if info["has_a"]:
            parts.append("[mu][ak]sidechaincompress=threshold=0.02:ratio=12:attack=10:release=400[md]")
            parts.append(f"[md]volume={o['music_gain']}[mv]")
        else:
            parts.append(f"[mu]volume={o['music_gain']}[mv]")
        labels.append("[mv]")
    if not labels:
        return "", None
    parts.append("".join(labels) + f"amix=inputs={len(labels)}:duration=first:normalize=0:dropout_transition=0,"
                 f"alimiter=limit=0.89:level=disabled,afade=t=out:st={max(0, dur - 0.4):.2f}:d=0.4[aout]")
    return ";".join(parts), "[aout]"

# ============================================================== render + verify
def render(extract, out, info, words, peak, o, tmpdir):
    dur = info["dur"]
    ass = None
    if words:
        ass = os.path.join(tmpdir, "caps.ass")
        style = o["cap_style"] if o["cap_style"] != "auto" else ("pop" if o["layout"] == "split" else "group")
        make_ass(words, ass, o["font"], style, o["layout"], peak if o["effects"] else None)
    stickers = []
    if o["stickers"] and words:
        for i, (emo, t0) in enumerate(pick_stickers(words, dur)):
            png = os.path.join(tmpdir, f"st{i}.png")
            if make_sticker_png(emo, png):
                stickers.append((png, t0))
        if not stickers:
            print("[edit] stickers requested but no emoji font/keywords matched; skipped", flush=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", extract]
    for png, _ in stickers:
        cmd += ["-loop", "1", "-framerate", f"{info['fps']:.3f}", "-t", f"{dur:.2f}", "-i", png]
    idx = 1 + len(stickers)
    sfx_idx = music_idx = None
    if o["sfx"]:
        sfx = os.path.join(tmpdir, "sfx.wav")
        write_wav(sfx, synth_sfx(dur, peak if o["effects"] else None, True, [t for _, t in stickers]))
        cmd += ["-i", sfx]; sfx_idx = idx; idx += 1
    if o["music"]:
        cmd += ["-stream_loop", "-1", "-i", o["music"]]; music_idx = idx; idx += 1
    vf = build_filter(o, dur, peak, ass, stickers)
    af, alabel = build_audio(info, dur, 1 + len(stickers), music_idx, sfx_idx, o)
    graph = vf + (";" + af if af else "")
    cmd += ["-filter_complex", graph, "-map", "[o]"]
    cmd += ["-map", alabel] if alabel else (["-map", "0:a?"] if info["has_a"] else [])
    cmd += ["-t", f"{dur:.2f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-fps_mode", "passthrough", "-c:a", "aac", "-b:a", "192k", "-ar", str(SR),
            "-movflags", "+faststart", out]
    run(cmd, timeout=900)
    return len(stickers)

def verify(out, src_info):
    """Check the FINISHED file. Returns list of problems (empty = ok)."""
    bad = []
    pi = probe(out)
    if not pi["has_v"] or pi["dur"] < 5:
        return ["no/short video"]
    if (pi["w"], pi["h"]) != (1080, 1920):
        bad.append(f"resolution {pi['w']}x{pi['h']}")
    if src_info["fps"] and abs(pi["fps"] - src_info["fps"]) > 0.6:
        bad.append(f"fps {pi['fps']:.1f} != source {src_info['fps']:.1f}")
    if abs(pi["dur"] - src_info["dur"]) > 1.5:
        bad.append(f"duration {pi['dur']:.1f}s vs {src_info['dur']:.1f}s")
    if src_info["has_a"]:
        if not pi["has_a"]:
            bad.append("audio missing")
        elif abs(pi["adur"] - pi["vdur"]) > 0.4:
            bad.append(f"A/V length mismatch {pi['adur']:.1f}/{pi['vdur']:.1f}")
    a = analyse(out)
    if a["black"] > 0.45:
        bad.append(f"black {a['black']:.0%}")
    if a["frozen"] > 0.6:
        bad.append(f"frozen {a['frozen']:.0%}")
    if src_info["has_a"] and (a["integrated"] is None or a["integrated"] < -60):
        bad.append("audio silent")
    if pi["has_a"]:
        v = subprocess.run(["ffmpeg", "-hide_banner", "-i", out, "-vn", "-af", "volumedetect", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=120).stderr
        m = re.search(r"max_volume: (-?[0-9.]+) dB", v)
        if m and float(m.group(1)) > -0.05:
            bad.append("audio clipping")
    return bad

# ============================================================== one candidate, start to finish
def process(rec, start, total, name, chat_score, chat_info, o, outdir):
    t_start = time.time()
    with tempfile.TemporaryDirectory() as td:
        ext = os.path.join(td, "extract.mp4")
        run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", rec, "-t", f"{total:.2f}",
             "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", ext], timeout=180)
        a = analyse(ext)
        info = a["info"]
        # ---- gate 1: is the recording valid at all?
        if not info["has_v"] or info["dur"] < min(20, total * 0.6):
            raise RuntimeError(f"invalid recording segment (dur={info['dur']:.1f}s)")
        if a["black"] > 0.5:
            raise RuntimeError(f"rejected before render: black screen {a['black']:.0%}")
        if a["frozen"] > 0.6:
            raise RuntimeError(f"rejected before render: frozen footage {a['frozen']:.0%}")
        boring = a["motion"] < o["min_motion"] and a["lift"] < o["min_lift"]
        if boring:
            raise RuntimeError(f"rejected before render: boring (motion {a['motion']:.2f}, loudness lift {a['lift']:.1f} LU)")
        # ---- gate 2 (optional): CLIP zero-shot frame scoring
        ai = clip_review(ext, info["dur"]) if o["ai_review"] else None
        if ai and ai["bad"] > 0.7:
            raise RuntimeError(f"rejected before render: CLIP says menu/loading/black ({ai['bad']:.0%})")
        # ---- editing timeline
        peak = a["peak_t"] if (a["peak_t"] is not None and a["lift"] >= 5) else None
        if peak is None:
            peak = min(info["dur"] - 5, max(3.0, o["pre_used"] - 3.0))
            peak_src = "chat-time"
        else:
            peak_src = "loudness"
        words = transcribe(ext) if o["captions"] else []
        n_st = render(ext, name, info, words, peak, o, td)
    problems = verify(name, info)
    if problems:
        os.replace(name, name + ".REJECTED")
        raise RuntimeError("failed final check: " + "; ".join(problems))
    exc = 100 * (0.4 * min(1, max(0, a["lift"]) / 12) + 0.3 * min(1, a["motion"] / 12) + 0.3 * min(1, chat_score / 40))
    note = (f"peak={peak:.1f}s({peak_src}) lift={a['lift']:.1f}LU motion={a['motion']:.1f} words={len(words)} "
            f"stickers={n_st} ai={ai['score'] if ai else 'n/a'} render={time.time() - t_start:.0f}s")
    row = [os.path.basename(name), f"{exc:.0f}", f"{chat_score:.1f}", chat_info, note]
    idx = os.path.join(outdir, "index.csv")
    new = not os.path.exists(idx)
    with open(idx, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "excitement(0-100, heuristic)", "chat_score", "chat", "details"])
        w.writerow(row)
    return note

# ============================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("channel", nargs="?", default="")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--mult", type=float, default=2.5)
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--min-users", type=int, default=5)
    ap.add_argument("--pre", type=int, default=28)
    ap.add_argument("--post", type=int, default=10)
    ap.add_argument("--cooldown", type=int, default=45)
    ap.add_argument("--out", default="clips")
    ap.add_argument("--font", default="Arial")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--layout", choices=["blur", "split", "full"], default="blur")
    ap.add_argument("--cam", default="")
    ap.add_argument("--game", default="")
    ap.add_argument("--focus", type=float, default=0.5)
    ap.add_argument("--cap-style", choices=["auto", "group", "pop"], default="auto")
    ap.add_argument("--no-effects", action="store_true", help="disable zoom/flash")
    ap.add_argument("--zoom", type=float, default=1.0, help="zoom strength multiplier (0.5 subtle .. 1.5 strong)")
    ap.add_argument("--sfx", action="store_true", help="synthetic whoosh + riser/impact timed to the peak")
    ap.add_argument("--sfx-gain", type=float, default=0.5)
    ap.add_argument("--music", default="", help="your own royalty-free audio file or a folder of them")
    ap.add_argument("--music-gain", type=float, default=0.22)
    ap.add_argument("--stickers", action="store_true", help="keyword-triggered emoji stickers")
    ap.add_argument("--ai-review", action="store_true", help="optional CLIP zero-shot frame scoring")
    ap.add_argument("--min-motion", type=float, default=0.8, help="below this AND low loudness lift = boring")
    ap.add_argument("--min-lift", type=float, default=2.0)
    ap.add_argument("--from-file", default="", help="TEST MODE: edit a video file instead of a live stream")
    ap.add_argument("--at", type=float, default=0.0, help="with --from-file: spike time in seconds")
    a = ap.parse_args()

    cam = parse_box(a.cam, "cam") if a.cam else None
    game = parse_box(a.game, "game") if a.game else None
    if a.layout == "split" and not cam:
        sys.exit("--layout split needs --cam x,y,w,h")
    music = ""
    if a.music:
        cands = sorted(glob.glob(os.path.join(a.music, "*"))) if os.path.isdir(a.music) else [a.music]
        cands = [c for c in cands if c.lower().endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac"))]
        if cands:
            music = random.choice(cands)
            print(f"[music] using {os.path.basename(music)}", flush=True)
        else:
            print("[music] no audio files found; continuing without music", flush=True)
    for tool in ("ffmpeg", "ffprobe") + (() if a.from_file else ("streamlink",)):
        if not shutil.which(tool):
            sys.exit(f"Missing '{tool}' on PATH.")
    os.makedirs(a.out, exist_ok=True)
    o = dict(layout=a.layout, cam=cam, game=game, focus=min(1, max(0, a.focus)), cap_style=a.cap_style,
             font=a.font, captions=not a.no_captions, effects=not a.no_effects, zoom=a.zoom, sfx=a.sfx,
             sfx_gain=a.sfx_gain, music=music, music_gain=a.music_gain, stickers=a.stickers,
             ai_review=a.ai_review, min_motion=a.min_motion, min_lift=a.min_lift, pre_used=a.pre)

    if a.from_file:                                            # ---- offline test mode
        name = os.path.join(a.out, "test_clip.mp4")
        start = max(0.0, a.at - a.pre)
        try:
            print(f"[clip] DONE {os.path.basename(name)} | {process(a.from_file, start, a.pre + a.post, name, 20.0, 'test', o, a.out)}")
        except Exception as e:
            print(f"[clip] skipped: {e}")
        return

    if not re.fullmatch(r"[A-Za-z0-9_]+", a.channel or ""):
        sys.exit("Invalid Twitch channel name.")
    rec = os.path.join(a.out, f"_recording_{a.channel}.ts")
    deadline = time.time() + a.max_minutes * 60 if a.max_minutes else None
    print(f"Recording twitch.tv/{a.channel} (layout={a.layout}) ...", flush=True)
    t_begin = time.time()
    sl = subprocess.Popen(["streamlink", "--retry-streams", "3", "--retry-max", "20",
                           f"twitch.tv/{a.channel}", "best", "-o", rec, "--force"], stdout=subprocess.DEVNULL)
    chat = ChatCounter(a.channel); chat.start()

    jobs = queue.Queue()
    def worker():
        n = 0
        while True:
            spike_ts, score, info = jobs.get()
            wait = spike_ts + a.post - time.time()
            if wait > 0:
                time.sleep(wait)
            try:
                dur = probe(rec)["dur"]
                start = max(0.0, dur - (time.time() - spike_ts) - a.pre)
                n += 1
                name = os.path.join(a.out, f"clip_{time.strftime('%H%M%S')}_{n}_chat{int(score)}.mp4")
                print(f"[clip] candidate #{n} chat={score:.0f} -> analysing/rendering", flush=True)
                print(f"[clip] DONE {os.path.basename(name)} | {process(rec, start, a.pre + a.post, name, score, info, o, a.out)}", flush=True)
            except Exception as e:
                print(f"[clip] skipped: {e}", flush=True)
            finally:
                jobs.task_done()
    threading.Thread(target=worker, daemon=True).start()

    history = collections.deque(maxlen=12)
    last_spike = 0
    try:
        while sl.poll() is None and not (deadline and time.time() > deadline):
            t0 = time.time()
            time.sleep(a.window)
            size = os.path.getsize(rec) if os.path.exists(rec) else 0
            if time.time() - t_begin > 90 and size < 300000:
                print("[ERROR] Twitch recorder produced no video after 90s (offline, or Twitch blocked the runner). Stopping.", flush=True)
                break
            c, raw, users, top = chat.score_since(t0)
            base = (sum(history) / len(history)) if history else 0
            spike = (c >= a.min_count and users >= a.min_users and len(history) >= 3
                     and c >= max(base * a.mult, a.min_count) and time.time() - last_spike > a.cooldown)
            print(f"[chat] score={c:5.1f} raw={raw:3d} users={users:3d} baseline={base:4.1f}" + ("  <-- SPIKE" if spike else ""), flush=True)
            if spike:
                last_spike = time.time()
                jobs.put((time.time(), c, f"{raw} msgs, {users} chatters, top: {top[:40]}"))
            else:
                history.append(c)
        print("Stopped: stream ended / time limit / recorder problem.", flush=True)
    except KeyboardInterrupt:
        print("Stopping...", flush=True)
    finally:
        sl.terminate()
        time.sleep(1)
        jobs.join()

if __name__ == "__main__":
    main()
