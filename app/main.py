"""
VoiceGuard — Audio Trust & Investigation service.

Composition root: mounts the investigation agent's routes, proxies the
deployed acoustic detector, serves the demo console at /, and (optionally)
enforces an API key for machine-to-machine callers.

Run locally:      uvicorn app.main:app --reload
Docker / Spaces:  uvicorn app.main:app --host 0.0.0.0 --port 7860

Endpoints
    GET  /                    demo console (single-page UI)
    GET  /docs                interactive OpenAPI docs
    GET  /health              liveness probe
    GET  /agent/health        live capability report (keys, sources, ASR)
    POST /detect              acoustic deepfake detection (proxied upstream)
    POST /investigate         full pipeline -> structured JSON report
    POST /investigate/report  full pipeline -> Markdown report
    POST /investigate/render  re-render a saved JSON report as Markdown
"""

from __future__ import annotations

import os
import secrets as _secrets
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from app.agent import render_markdown
from app.agent.schemas import InvestigationReport
from app.core import model as remote_detector
from app.routes_investigate import router as investigate_router

APP_VERSION = "1.0.0"

DESCRIPTION = """
VoiceGuard answers **"should I trust this audio, and why?"** — it transcribes
speech, extracts the factual claims inside it, researches each claim across
fact-checkers, global news and reference sources, fuses everything with
acoustic deepfake detection, and returns an auditable 0-100 trust score.

Integration notes:
- All POST endpoints accept/return JSON; audio travels as base64.
- If the deployment sets `AGENT_API_KEY`, send it as an `X-API-Key` header.
- CORS is open, so browser apps can call this API directly.
"""

app = FastAPI(
    title="VoiceGuard — Audio Trust & Investigation API",
    version=APP_VERSION,
    description=DESCRIPTION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_PROTECTED_PATHS = {"/investigate", "/investigate/report", "/investigate/render", "/detect"}


@app.middleware("http")
async def _api_key_guard(request: Request, call_next):
    """Optional auth: set AGENT_API_KEY to require X-API-Key on POST endpoints."""
    required = os.getenv("AGENT_API_KEY", "").strip()
    if required and request.method == "POST" and request.url.path in _PROTECTED_PATHS:
        supplied = request.headers.get("x-api-key", "")
        if not _secrets.compare_digest(supplied, required):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing X-API-Key header."},
            )
    return await call_next(request)


app.include_router(investigate_router)


class DetectRequest(BaseModel):
    audioBase64: str = Field(description="Base64-encoded audio payload")
    audioFormat: str = "wav"
    language: str = "English"


@app.post("/detect", tags=["detection"])
async def detect(req: DetectRequest) -> dict:
    """
    Acoustic-only deepfake detection.

    Proxies the deployed VoiceGuard detection engine (Wav2Vec2 neural model +
    DSP forensics) and returns its full response: classification, confidence,
    per-analyzer forensics and audio profile.
    """
    status, body, note = await remote_detector.detect_raw(
        req.audioBase64, req.audioFormat, req.language
    )
    if status == 200 and isinstance(body, dict):
        return body
    raise HTTPException(status_code=502, detail=f"Upstream detector unavailable — {note}")


@app.post("/investigate/render", response_class=PlainTextResponse, tags=["investigation"])
async def render_report(report: InvestigationReport) -> str:
    """Render a previously returned /investigate JSON report as Markdown (no re-run)."""
    try:
        return render_markdown(report)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"could not render report: {exc}") from exc


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok", "service": "voiceguard-agent", "version": APP_VERSION}


_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.get("/", include_in_schema=False)
async def home():
    index = _STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return JSONResponse(
        {
            "service": "VoiceGuard — Audio Trust & Investigation API",
            "docs": "/docs",
            "health": "/agent/health",
            "endpoints": ["/investigate", "/investigate/report", "/detect"],
        }
    )
