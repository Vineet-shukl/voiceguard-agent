#!/usr/bin/env bash
# VoiceGuard launcher.
#
# Binds to $PORT (default 7860 — HF Spaces, Dockerfile EXPOSE, and local dev
# all use the same port). Worker count comes from $WEB_CONCURRENCY; Railway
# and Render set this automatically based on plan size. Default is 1, which is
# the safe choice for free-tier containers with limited RAM.
#
# `exec` replaces the shell with the uvicorn process so platform signals
# (SIGTERM for graceful shutdown, SIGHUP for reload) reach uvicorn directly
# rather than being swallowed by bash.

set -euo pipefail

PORT="${PORT:-7860}"
WORKERS="${WEB_CONCURRENCY:-1}"

exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers "$WORKERS"
