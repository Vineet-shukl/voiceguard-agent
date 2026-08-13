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
    GET  /health              liveness probe + capability report
    GET  /agent/health        live capability report (keys, sources, ASR)
    POST /detect              acoustic deepfake detection (proxied upstream)
    POST /investigate         full pipeline -> structured JSON report
    POST /investigate/report  full pipeline -> Markdown report
    POST /investigate/render  re-render a saved JSON report as Markdown

Environment variables
    AGENT_API_KEY       require this key on POST requests (X-API-Key header)
    ALLOW_PUBLIC_DEMO   explicitly permit POSTs without AGENT_API_KEY
    CORS_ORIGINS        comma-separated allowed origins; default "*" (open)
    LOG_LEVEL           DEBUG | INFO | WARNING | ERROR  (default INFO)
    MAX_BODY_BYTES      request body cap in bytes; default 37748736 (36 MB)
    RATE_LIMIT_REQUESTS process-local POST budget per rate-limit window
    POST_CONCURRENCY    maximum simultaneous protected POST requests
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets as _secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from app.agent import render_markdown
from app.agent.config import get_config
from app.agent.schemas import MAX_AUDIO_BASE64_CHARS, InvestigationReport
from app.core import model as remote_detector
from app.routes_investigate import router as investigate_router

# ── Logging ───────────────────────────────────────────────────────────────────
# Configure once here so every module's getLogger(__name__) inherits the format.
_log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
_log = logging.getLogger(__name__)

APP_VERSION = "1.0.0"

# ── CORS ──────────────────────────────────────────────────────────────────────
# Default "*" keeps the public demo open.
# Production: CORS_ORIGINS=https://yourapp.com,https://admin.yourapp.com
_raw_origins = os.getenv("CORS_ORIGINS", "*").strip()
_cors_origins: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()] or ["*"]

def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


# The JSON envelope adds a little overhead to the maximum base64 payload.
_MAX_BODY_BYTES = _env_int("MAX_BODY_BYTES", 36 * 1024 * 1024)
_RATE_LIMIT_REQUESTS = _env_int("RATE_LIMIT_REQUESTS", 20)
_RATE_LIMIT_WINDOW_SECONDS = _env_int("RATE_LIMIT_WINDOW_SECONDS", 60)
_POST_CONCURRENCY = _env_int("POST_CONCURRENCY", 2)
_rate_lock = asyncio.Lock()
_rate_window_started = time.monotonic()
_rate_window_count = 0
_post_slots = asyncio.Semaphore(_POST_CONCURRENCY)


class _RequestBodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Count actual ASGI body bytes, including chunked requests."""

    def __init__(self, app, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        declared = headers.get(b"content-length")
        if declared:
            try:
                if int(declared) < 0 or int(declared) > self.max_body_bytes:
                    await self._reject(scope, receive, send)
                    return
            except ValueError:
                await self._reject(scope, receive, send, status_code=400)
                return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await self._reject(scope, receive, send)

    async def _reject(self, scope, receive, send, status_code: int = 413) -> None:
        if status_code == 413:
            mb = self.max_body_bytes // (1024 * 1024)
            detail = f"Request body too large; maximum is {mb} MB."
        else:
            detail = "Malformed Content-Length header."
        response = JSONResponse(status_code=status_code, content={"detail": detail})
        await response(scope, receive, send)

DESCRIPTION = """
VoiceGuard answers **"should I trust this audio, and why?"** — it transcribes
speech, extracts the factual claims inside it, researches each claim across
fact-checkers, global news and reference sources, fuses everything with
acoustic deepfake detection, and returns an auditable 0-100 trust score.

Integration notes:
- All POST endpoints accept/return JSON; audio travels as base64.
- POST endpoints require `AGENT_API_KEY` unless `ALLOW_PUBLIC_DEMO=true`.
- CORS is configurable via `CORS_ORIGINS`; default is open for public demos.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Log capability status on startup so operators immediately see degradations."""
    cfg = get_config()
    providers = "/".join(cfg.provider_order()) or "none (heuristic fallback)"
    asr = "available" if cfg.has_asr else "unavailable (transcriptOverride required)"
    _log.info("VoiceGuard %s starting — LLM: %s | ASR: %s", APP_VERSION, providers, asr)
    if not cfg.has_llm:
        _log.warning(
            "No LLM API key — extraction and assessment fall back to heuristics. "
            "Set GEMINI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY for full accuracy."
        )
    if not cfg.has_asr:
        _log.warning(
            "No ASR key — transcription unavailable. "
            "Requests must include transcriptOverride, or set GROQ_API_KEY / GEMINI_API_KEY."
        )
    yield
    _log.info("VoiceGuard shutting down.")


app = FastAPI(
    title="VoiceGuard — Audio Trust & Investigation API",
    version=APP_VERSION,
    description=DESCRIPTION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include trailing-slash variants so FastAPI's 307-redirect doesn't silently
# bypass the API-key guard. On some ASGI stacks the redirect fires before
# middleware re-runs, making /investigate/ effectively unprotected otherwise.
_PROTECTED_PATHS = {
    "/investigate", "/investigate/",
    "/investigate/report", "/investigate/report/",
    "/investigate/render", "/investigate/render/",
    "/detect", "/detect/",
}


@app.middleware("http")
async def _api_key_guard(request: Request, call_next):
    """Fail closed by default and bound protected endpoint resource use."""
    if request.method != "POST" or request.url.path not in _PROTECTED_PATHS:
        return await call_next(request)

    required = os.getenv("AGENT_API_KEY", "").strip()
    public_demo = os.getenv("ALLOW_PUBLIC_DEMO", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not required and not public_demo:
        return JSONResponse(
            status_code=503,
            content={"detail": "API authentication is not configured."},
        )
    if required:
        supplied = request.headers.get("x-api-key", "")
        if not _secrets.compare_digest(supplied, required):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing X-API-Key header."},
            )

    global _rate_window_started, _rate_window_count
    now = time.monotonic()
    async with _rate_lock:
        if now - _rate_window_started >= _RATE_LIMIT_WINDOW_SECONDS:
            _rate_window_started = now
            _rate_window_count = 0
        if _rate_window_count >= _RATE_LIMIT_REQUESTS:
            retry_after = max(
                1, int(_RATE_LIMIT_WINDOW_SECONDS - (now - _rate_window_started))
            )
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={"detail": "Request rate limit exceeded."},
            )
        _rate_window_count += 1

    try:
        await asyncio.wait_for(_post_slots.acquire(), timeout=0.1)
    except TimeoutError:
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "1"},
            content={"detail": "Service is at its processing limit; retry shortly."},
        )
    try:
        return await call_next(request)
    finally:
        _post_slots.release()


# Register last so actual body-byte enforcement wraps every other middleware.
app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=_MAX_BODY_BYTES)
app.include_router(investigate_router)


class DetectRequest(BaseModel):
    audioBase64: str = Field(
        max_length=MAX_AUDIO_BASE64_CHARS,
        description="Base64-encoded audio payload (24 MB decoded maximum)",
    )
    audioFormat: str = Field(
        "wav", min_length=1, max_length=10, pattern=r"^[A-Za-z0-9]+$"
    )
    language: str = Field("English", min_length=1, max_length=64)


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
    if status in {400, 413, 422}:
        raise HTTPException(status_code=status, detail=note)
    _log.warning("upstream detector unavailable: status=%s note=%s", status, note)
    raise HTTPException(status_code=502, detail="Upstream detector unavailable.")


@app.post("/investigate/render", response_class=PlainTextResponse, tags=["investigation"])
async def render_report(report: InvestigationReport) -> str:
    """Render a previously returned /investigate JSON report as Markdown (no re-run)."""
    try:
        return render_markdown(report)
    except Exception as exc:
        _log.exception("could not render investigation report")
        raise HTTPException(status_code=422, detail="Could not render report.") from exc


@app.get("/health", tags=["meta"])
async def health() -> dict:
    """Liveness probe. Also reports which LLM providers and ASR are configured."""
    cfg = get_config()
    return {
        "status": "ok",
        "service": "voiceguard-agent",
        "version": APP_VERSION,
        "capabilities": {
            "llm": cfg.has_llm,
            "asr": cfg.has_asr,
            "providers": cfg.provider_order(),
        },
    }


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
            "health": "/health",
            "endpoints": ["/investigate", "/investigate/report", "/detect"],
        }
    )
