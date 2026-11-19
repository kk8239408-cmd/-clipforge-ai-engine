#!/usr/bin/env python3
"""YouTube long-form clip discovery: scan captions, AI-rank moments, download only finalists."""
import argparse, base64, json, os, re, subprocess, urllib.request
from pathlib import Path
KEY=os.environ.get("CLIPFORGE_VISION_API_KEY","").strip()
MODEL=os.environ.get("CLIPFORGE_VISION_MODEL","gemini-2.5-flash")
def log(s): print(s,flush=True)
def run(cmd,check=True):
 p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if check and p.returncode: raise RuntimeError("Command failed: "+" ".join(cmd[:5])+"\n"+p.stderr[-1000:])
 return p
def ai(parts):
 if not KEY: raise RuntimeError("Missing CLIPFORGE_VISION_API_KEY GitHub secret.")
 body={"contents":[{"parts":parts}],"generationConfig":{"temperature":0.15,"responseMimeType":"application/json"}}
 url="https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s"%(MODEL,KEY)
 req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
 with urllib.request.urlopen(req,timeout=100) as r: return json.loads(r.read().decode())["candidates"][0]["content"]["parts"][0]["text"]
def parse_vtt(p):
 raw=Path(p).read_text(encoding="utf-8",errors="ignore"); out=[]
 for block in re.split(r"\n\s*\n",raw):
  ls=[x.strip() for x in block.splitlines() if x.strip()]
  ti=next((i for i,x in enumerate(ls) if "-->" in x),None)
  if ti is None: continue
  m=re.match(r"(\d{2}):(\d{2}):(\d{2})[.,](\d+)\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d+)",ls[ti])
  if not m: continue
  a,b,c,d,e,f,g,h=m.groups(); start=int(a)*3600+int(b)*60+int(c)+float("0."+d); end=int(e)*3600+int(f)*60+int(g)+float("0."+h)
  txt=re.sub(r"<[^>]+>"," "," ".join(ls[ti+1:])); txt=re.sub(r"\s+"," ",txt).strip()
  if txt and end>start: out.append({"start":start,"end":end,"text":txt})
 return out
def windows(cues,duration):
 out=[]; start=0.; span=55.
 signals=("no way","what","wait","oh my god","let's go","insane","bro","why","how","actually","i can't","you can't","are you","that's crazy","holy","wow","never","finally","look at","watch this","i swear")
 while start<duration:
  end=min(duration,start+span); rows=[c for c in cues if c["end"]>start and c["start"]<end]; txt=" ".join(c["text"] for c in rows)
  if len(txt.split())>=18:
   low=txt.lower(); out.append({"start":start,"end":end,"text":txt[:1800],"prefilter":sum(low.count(s) for s in signals)})
  start+=20
 out.sort(key=lambda x:x["prefilter"],reverse=True)
 return out[:36]
def stamp(t): return "%02d:%02d:%02d.%03d"%(int(t//3600),int(t%3600//60),int(t%60),int((t%1)*1000))
def ass_time(t): return "%d:%02d:%05.2f"%(int(t//3600),int(t%3600//60),t%60)
def write_ass(cues,start,end,path):
 lines=["[Script Info]","ScriptType: v4.00+","PlayResX: 1080","PlayResY: 1920","","[V4+ Styles]","Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding","Style: Default,Montserrat,68,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,7,3,2,70,70,240,1","","[Events]","Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text"]
 for c in cues:
  a=max(0,c["start"]-start); b=min(end-start,c["end"]-start); txt=re.sub(r"[{}]","",c["text"]).strip()
  if b>a and txt: lines.append("Dialogue: 0,%s,%s,Default,,0,0,0,,%s"%(ass_time(a),ass_time(max(a+.15,b)),txt.replace(",","\\,")))
 Path(path).write_text("\n".join(lines)+"\n",encoding="utf-8")
def probe(p):
 j=json.loads(run(["ffprobe","-v","error","-show_entries","format=duration:stream=codec_type,width,height","-of","json",p]).stdout)
 v=next((s for s in j.get("streams",[]) if s.get("codec_type")=="video"),{})
 return {"duration":float(j.get("format",{}).get("duration",0) or 0),"width":int(v.get("width",0) or 0),"height":int(v.get("height",0) or 0),"audio":any(s.get("codec_type")=="audio" for s in j.get("streams",[]))}
def image_parts(path):
 out=[]
 for sec in (4,18,34,48):
  jpg=path+".%d.jpg"%sec; run(["ffmpeg","-y","-ss",str(sec),"-i",path,"-frames:v","1","-vf","scale=768:-2","-q:v","4",jpg],False)
  if os.path.exists(jpg):
   out.append({"inline_data":{"mime_type":"image/jpeg","data":base64.b64encode(Path(jpg).read_bytes()).decode()}}); os.remove(jpg)
 return out
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--url",required=True); ap.add_argument("--max-clips",type=int,default=3); ap.add_argument("--out",default="youtube-clips"); a=ap.parse_args()
 if not KEY: raise SystemExit("Add CLIPFORGE_VISION_API_KEY to GitHub Actions secrets.")
 if not re.match(r"^https?://(www\.)?(youtube\.com|youtu\.be)/",a.url): raise SystemExit("Invalid YouTube URL.")
 os.makedirs(a.out,exist_ok=True); log("[scan] Reading metadata and captions only; full video is NOT downloaded.")
 info=json.loads(run(["yt-dlp","--dump-single-json","--skip-download","--no-warnings",a.url]).stdout); duration=float(info.get("duration") or 0)
 if duration<60: raise SystemExit("Could not read video duration.")
 log("[source] %s | %.2f hours"%(info.get("title","YouTube video"),duration/3600))
 run(["yt-dlp","--skip-download","--write-auto-subs","--write-subs","--sub-langs","en.*,en","--sub-format","vtt","--no-warnings","-o",a.out+"/source",a.url],False)
 vtts=list(Path(a.out).glob("source*.vtt"))
 if not vtts: raise SystemExit("No English captions found. Stopping rather than pretending the full VOD was scanned.")
 cues=parse_vtt(vtts[0])
 if len(cues)<50: raise SystemExit("Caption track too sparse for reliable discovery.")
 log("[scan] caption_cues=%d; covered_until=%.2fh"%(len(cues),max(x["end"] for x in cues)/3600))
 cand=windows(cues,duration)
 payload=[{"id":i,"start":round(c["start"],1),"end":round(c["end"],1),"transcript":c["text"]} for i,c in enumerate(cand)]
 raw=ai([{"text":"Act as a ruthless short-form editor. Rank these moments from an entire long YouTube VOD. Reward real funny exchanges, surprise, reactions, conflict, impressive skill, strong story payoff or compelling reveals. Penalize rambling, repetition, dead air, generic chat and missing context. Do not reward keywords alone. Return JSON {candidates:[{id:integer,score:0-100,reason:string,hook:string,clip_start_offset:0-8,clip_end_offset:35-60}]}. Include all ids.\\n"+json.dumps(payload,ensure_ascii=False)}])
 ranking=json.loads(raw).get("candidates",[]); scores={int(x["id"]):x for x in ranking if isinstance(x,dict) and "id" in x}
 for i,c in enumerate(cand): c["ai"]=scores.get(i,{"score":0})
 cand.sort(key=lambda x:float(x["ai"].get("score",0)),reverse=True)
 shortlist=[c for c in cand if float(c["ai"].get("score",0))>=55][:8]
 log("[AI] transcript_windows_reviewed=%d finalists=%d"%(len(cand),len(shortlist)))
 if not shortlist: raise SystemExit("AI found no strong moments. No filler clips exported.")
 finalists=[]
 for n,c in enumerate(shortlist,1):
  s=c["start"]+max(0,float(c["ai"].get("clip_start_offset",0))); e=min(c["end"],s+max(35,min(60,float(c["ai"].get("clip_end_offset",50)))))
  rawfile=os.path.join(a.out,"candidate-%02d.mp4"%n); log("[download] #%d at %s AI score=%s"%(n,stamp(s),c["ai"].get("score")))
  run(["yt-dlp","--no-warnings","--no-progress","--force-overwrites","--download-sections","*%s-%s"%(stamp(s),stamp(e)),"--force-keyframes-at-cuts","-f","bv*[height<=720]+ba/b[height<=720]","--merge-output-format","mp4","-o",rawfile,a.url],False)
  if not os.path.exists(rawfile):
   alt=list(Path(a.out).glob("candidate-%02d.*"%n))
   if not alt: continue
   rawfile=str(alt[0])
  try:
   meta=probe(rawfile)
   if not meta["audio"] or meta["duration"]<30: continue
   vis=json.loads(ai([{"text":"Evaluate these frames in temporal order from a proposed short clip. Score visual clarity, visible action/reaction and coherent payoff, without inventing events. Return JSON {visual_score:0-100,visual_summary:string,edit_note:string}."}]+image_parts(rawfile)))
   score=round(.72*float(c["ai"].get("score",0))+.28*float(vis.get("visual_score",0)))
   finalists.append({"path":rawfile,"start":s,"end":e,"score":score,"transcript_score":c["ai"].get("score",0),"visual_score":vis.get("visual_score",0),"reason":c["ai"].get("reason",""),"hook":c["ai"].get("hook",""),"visual":vis})
  except Exception as ex: log("[reject] "+str(ex)[:160])
 finalists.sort(key=lambda x:x["score"],reverse=True); chosen=[]
 for c in finalists:
  if c["score"]<65 or any(abs(c["start"]-x["start"])<90 for x in chosen): continue
  chosen.append(c)
  if len(chosen)>=min(3,a.max_clips): break
 exports=[]
 for i,c in enumerate(chosen,1):
  out=os.path.join(a.out,"clip-%d_score%d.mp4"%(i,c["score"])); ass=out+".ass"; write_ass(cues,c["start"],c["end"],ass)
  sub="subtitles=filename='%s'"%os.path.abspath(ass).replace("\\","/").replace(":","\\:")
  run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",c["path"],"-vf","scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"+sub+",format=yuv420p","-map","0:v:0","-map","0:a:0?","-c:v","libx264","-preset","superfast","-crf","23","-r","60","-fps_mode","cfr","-c:a","aac","-b:a","160k","-movflags","+faststart",out])
  meta=probe(out)
  if (meta["width"],meta["height"])!=(1080,1920) or not meta["audio"]: raise RuntimeError("Render validation failed: "+str(meta))
  exports.append({"file":os.path.basename(out),"score":c["score"],"timestamp_seconds":round(c["start"],1),"transcript_score":c["transcript_score"],"visual_score":c["visual_score"],"reason":c["reason"],"hook":c["hook"],"visual":c["visual"],"ffprobe":meta})
  os.remove(ass); log("[export] %s score=%d"%(os.path.basename(out),c["score"]))
 summary={"source_title":info.get("title"),"duration_seconds":duration,"caption_cues_scanned":len(cues),"transcript_windows_reviewed":len(cand),"AI_model":MODEL,"policy":"Only combined score >=65; max 3; no low-score fallback.","exports":exports}
 Path(os.path.join(a.out,"summary.json")).write_text(json.dumps(summary,indent=2),encoding="utf-8")
 if not exports: raise SystemExit("No clips passed quality threshold. No filler exported.")
 log("[done] exported=%d; full source video was never downloaded."%len(exports))
if __name__=="__main__": main()
