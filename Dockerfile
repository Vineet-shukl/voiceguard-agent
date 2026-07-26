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

# Normalise line endings (Windows checkouts) and make the launcher executable.
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

USER appuser

EXPOSE 7860

# start.sh binds the platform-injected $PORT and ALSO 7860, so a public domain
# pointed at either port always reaches the app (Railway / Render / HF Spaces).
CMD ["bash", "/app/start.sh"]
