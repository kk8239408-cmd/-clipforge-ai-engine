#!/usr/bin/env python3
"""
clip_watcher.py - 100% free auto-clipper for Twitch streams.

streamlink  -> records the live stream
twitch chat -> read anonymously (no account, no token)
spike       -> counts chat messages every 10s, flags unusual bursts
ffmpeg      -> cuts around each spike, reframes to 9:16 (blurred background)
whisper     -> (optional, faster-whisper, local & free) burns word-highlight captions

Usage:
    python clip_watcher.py CHANNEL
    python clip_watcher.py CHANNEL --mult 2.0 --min-count 15 --pre 25 --post 8 --no-captions

Needs: streamlink, ffmpeg/ffprobe on PATH.  Optional: pip install faster-whisper
"""
import argparse, collections, os, queue, shutil, socket, subprocess, sys, threading, time

# ---------------------------------------------------------------- chat reader
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
                            with self.lock:
                                self.stamps.append(time.time())
            except Exception as e:
                print(f"[chat] reconnecting ({e})", flush=True)
                time.sleep(5)

    def count_since(self, t0):
        with self.lock:
            while self.stamps and self.stamps[0] < t0 - 600:
                self.stamps.popleft()
            return sum(1 for t in self.stamps if t >= t0)

# ---------------------------------------------------------------- helpers
def probe_duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out)
    except Exception:
        return 0.0

def ass_time(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def make_ass(words, path, font="Arial"):
    """words: [(start, end, text)]. Groups of 3 words, current word highlighted yellow."""
    head = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,"
        "Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        f"Style: Default,{font},84,&H00FFFFFF,&H0000FFFF,&H00000000,&H64000000,"
        "-1,0,0,0,100,100,0,0,1,6,2,2,60,60,520,1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    lines = []
    for i in range(0, len(words), 3):
        grp = words[i:i + 3]
        for j, (st, en, _) in enumerate(grp):
            end = grp[j + 1][0] if j + 1 < len(grp) else en
            txt = " ".join(
                (r"{\c&H00FFFF&}" + w.upper() + r"{\c&HFFFFFF&}") if k == j else w.upper()
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

VF = ("[0:v]split=2[bgsrc][fgsrc];"
      "[bgsrc]scale=1080:1920:force_original_aspect_ratio=increase,"
      "crop=1080:1920,boxblur=30:5[bg];"
      "[fgsrc]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
      "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]")

def render(rec, start, dur, out_path, captions, font):
    tmp = out_path + ".raw.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", rec,
                    "-t", f"{dur:.2f}", "-c", "copy", tmp], check=True)
    ass = None
    if captions:
        words = transcribe(tmp)
        if words:
            ass = out_path + ".ass"
            make_ass(words, ass, font)
    graph = VF + (f";[v]subtitles={ass}[o]" if ass else ";[v]null[o]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
                    "-filter_complex", graph, "-map", "[o]", "-map", "0:a?",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                    "-movflags", "+faststart", out_path], check=True)
    for f in (tmp, ass):
        if f and os.path.exists(f):
            os.remove(f)

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("channel")
    ap.add_argument("--window", type=int, default=10, help="chat bucket seconds")
    ap.add_argument("--mult", type=float, default=2.5, help="spike = baseline x this")
    ap.add_argument("--min-count", type=int, default=10, help="min msgs in a bucket to count")
    ap.add_argument("--pre", type=int, default=25, help="seconds before spike")
    ap.add_argument("--post", type=int, default=8, help="seconds after spike")
    ap.add_argument("--cooldown", type=int, default=45)
    ap.add_argument("--out", default="clips")
    ap.add_argument("--font", default="Arial")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop after N minutes (0 = no limit)")
    a = ap.parse_args()
    if not a.channel or not all(ch.isalnum() or ch == "_" for ch in a.channel):
        ap.error("Channel must contain only letters, numbers, or underscores.")
    if a.max_minutes < 0 or a.max_minutes > 330:
        ap.error("--max-minutes must be between 0 and 330.")
    if a.window < 1 or a.pre < 1 or a.post < 1:
        ap.error("--window, --pre, and --post must be positive.")
    deadline = time.time() + a.max_minutes * 60 if a.max_minutes else None

    for tool in ("streamlink", "ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"Missing '{tool}' on PATH.")
    os.makedirs(a.out, exist_ok=True)
    rec = os.path.join(a.out, f"_recording_{a.channel}.ts")

    print(f"Recording twitch.tv/{a.channel} ... (Ctrl+C to stop)", flush=True)
    sl = subprocess.Popen(["streamlink", f"twitch.tv/{a.channel}", "best", "-o", rec, "--force"],
                          stdout=subprocess.DEVNULL)
    chat = ChatCounter(a.channel)
    chat.start()

    jobs = queue.Queue()
    def worker():
        n = 0
        while True:
            spike_ts, count = jobs.get()
            wait = spike_ts + a.post - time.time()
            if wait > 0:
                time.sleep(wait)
            try:
                dur = probe_duration(rec)
                age = time.time() - spike_ts
                total = a.pre + a.post
                start = max(0.0, dur - age - a.pre)
                n += 1
                name = os.path.join(a.out, f"clip_{time.strftime('%H%M%S')}_{n}.mp4")
                print(f"[clip] spike={count} msgs -> rendering {name}", flush=True)
                render(rec, start, total, name, not a.no_captions, a.font)
                print(f"[clip] DONE {name}", flush=True)
            except Exception as e:
                print(f"[clip] failed: {e}", flush=True)
            finally:
                jobs.task_done()
    threading.Thread(target=worker, daemon=True).start()

    history = collections.deque(maxlen=12)  # last ~2 min of buckets
    last_spike = 0
    try:
        while sl.poll() is None and not (deadline and time.time() > deadline):
            t0 = time.time()
            time.sleep(a.window)
            c = chat.count_since(t0)
            base = (sum(history) / len(history)) if history else 0
            is_spike = (c >= a.min_count and len(history) >= 3
                        and c >= max(base * a.mult, a.min_count)
                        and time.time() - last_spike > a.cooldown)
            print(f"[chat] {c:3d} msgs/{a.window}s  baseline={base:4.1f}"
                  + ("  <-- SPIKE" if is_spike else ""), flush=True)
            if is_spike:
                last_spike = time.time()
                jobs.put((time.time(), c))
            else:
                history.append(c)
        print("Stream ended / not live.")
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if sl.poll() is None:
            sl.terminate()
            try:
                sl.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sl.kill()
        jobs.join()  # finish any clips already queued

if __name__ == "__main__":
    main()
      
