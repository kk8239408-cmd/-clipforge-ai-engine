# CLIPFORGE AI Engine
Render deployment starter. Upload these files to the ROOT of your GitHub repository:
- Dockerfile
- main.py
- requirements.txt
- render.yaml

This first version transcribes speech with faster-whisper, ranks transcript windows using simple heuristics, and renders up to five vertical clips with FFmpeg. It is a starter, not a trained virality model or the full five-style editor. Free hosting may lack enough RAM/CPU and may sleep; verify costs and limits before deploying. Do not add paid services without approval.
