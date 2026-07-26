# VoiceGuard — Audio Trust & Investigation service
# Hugging Face Spaces (Docker SDK) image. Listens on 7860 (see README frontmatter).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Spaces runs containers as a non-root user; create one with uid 1000.
RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 7860

# Binds $PORT when the platform injects one (Railway/Render), else 7860 (HF Spaces).
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 2
