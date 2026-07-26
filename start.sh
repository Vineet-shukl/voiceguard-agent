#!/usr/bin/env bash
# VoiceGuard launcher.
#
# Listens on the platform-injected $PORT and, when that differs from 7860,
# ALSO on 7860. Public domains are sometimes generated pointing at the
# Dockerfile's EXPOSEd port while the platform injects a different runtime
# port — binding both makes the container reachable in every combination
# (Railway, Render, HF Spaces, local).
set -m

PRIMARY_PORT="${PORT:-7860}"

uvicorn app.main:app --host 0.0.0.0 --port "$PRIMARY_PORT" --workers 2 &

if [ "$PRIMARY_PORT" != "7860" ]; then
  uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1 &
fi

# Exit when any listener dies so the platform restarts the container.
wait -n
