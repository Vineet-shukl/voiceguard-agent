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

# Verify the app can serve requests before the platform routes traffic to it.
# --start-period gives uvicorn time to initialise; 3 retries before unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request, os; urllib.request.urlopen('http://localhost:' + os.getenv('PORT', '7860') + '/health', timeout=8)"

CMD ["bash", "/app/start.sh"]
