# ---- Base image ----
FROM python:3.11-slim

# ---- Env & pip ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# ---- Dependencies ----
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- App code ----
COPY src /app/src
COPY app /app/app
# If you want to ship your own fine-tuned model inside the image:
# COPY models/distilbert_best /app/models/distilbert_best
# ENV LOCAL_MODEL_PATH=/app/models/distilbert_best

# Use the PORT provided by the platform, default to 8080 locally
ENV PORT=8080
EXPOSE 8080

# ---- Start the server (Gunicorn) ----
# 2 workers, gthread for IO, binds to $PORT
CMD ["sh", "-c", "gunicorn -w 2 -k gthread -b 0.0.0.0:${PORT} app.app:app"]

