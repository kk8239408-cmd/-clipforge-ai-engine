FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg fontconfig unzip fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install curated fonts at build time. Personal-use-only fonts are deliberately excluded.
RUN mkdir -p /usr/local/share/fonts/clipforge \
    && unzip -oq fonts/CLIPFORGE_Curated_Font_Library.zip -d /tmp/clipforge-fonts \
    && find /tmp/clipforge-fonts/01_Core_Captions /tmp/clipforge-fonts/02_Display_and_Accent /tmp/clipforge-fonts/03_Fallback \
       -type f \( -iname '*.ttf' -o -iname '*.otf' \) -exec cp {} /usr/local/share/fonts/clipforge/ \; \
    && fc-cache -f \
    && rm -rf /tmp/clipforge-fonts

ENV PORT=8000 DATA_DIR=/tmp/clipforge MAX_UPLOAD_MB=150

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
