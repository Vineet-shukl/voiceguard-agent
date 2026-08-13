"""
Remote adapter to the deployed VoiceGuard acoustic detection API.

The original VoiceGuard detector (Wav2Vec2 + DSP forensics) already runs as its
own service on Hugging Face Spaces. This module makes that service look like a
local function, so the agent's probe (app/agent/detector.py) finds it at its
first candidate: ``app.core.model.detect``. That keeps the two deployments
loosely coupled — either can be replaced without touching the other.

Design rules, identical to the rest of the pipeline:

- never raises; returns ``None`` so the caller records a degradation
- validates the base64 payload *before* spending any network time
- routes HTTP through ``app.agent.http`` late-bound, so the offline test
  suite's network kill-switch applies here too
- auto-negotiates the upstream request shape (field names, key transport) and
  caches whatever worked for the life of the process

Environment:
    VOICE_API_URL           upstream endpoint (default: the deployed Space)
    VOICE_API_KEY           API key for the detector
    VOICE_API_AUTH_MODE     x-api-key (default), bearer, or body
    VOICE_API_TIMEOUT       seconds; generous default because a sleeping
                            Space can take ~1 min to wake (default 75)
    VOICE_API_AUDIO_FIELD   pin the audio field name, skipping negotiation
    ENABLE_REMOTE_DETECTOR  set false to disable outbound detection calls
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from app.agent import http as _http  # late-bound on purpose: tests patch attrs
from app.agent.transcribe import decode_audio

DEFAULT_VOICE_API_URL = "https://pandaisop-voice-detection-api.hf.space/test"

_MAX_CALLS_PER_NEGOTIATION = 6
_MIN_AUDIO_BYTES = 16
_MAX_AUDIO_BYTES = 24 * 1024 * 1024

# Serialize detector access so cold-start negotiation cannot fan out per request.
_detector_lock = asyncio.Lock()
# Remembered request shape: {"fields": {...}}
_negotiated: dict[str, Any] | None = None

# Field-name shapes tried in order. Semantic keys: audio / format / language.
_SHAPES: list[dict[str, str]] = [
    {"audio": "audioBase64", "format": "audioFormat", "language": "language"},
    {"audio": "audio_base64", "format": "audio_format", "language": "language"},
    {"audio": "audio", "format": "format", "language": "language"},
]


def _flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _settings() -> dict[str, Any]:
    try:
        timeout = float(os.getenv("VOICE_API_TIMEOUT", "75"))
    except ValueError:
        timeout = 75.0
    auth_mode = os.getenv("VOICE_API_AUTH_MODE", "x-api-key").strip().lower()
    if auth_mode not in {"x-api-key", "bearer", "body"}:
        auth_mode = "x-api-key"
    return {
        "url": os.getenv("VOICE_API_URL", DEFAULT_VOICE_API_URL).strip(),
        "key": os.getenv("VOICE_API_KEY", "").strip(),
        "auth_mode": auth_mode,
        "timeout": timeout,
        "pin_field": os.getenv("VOICE_API_AUDIO_FIELD", "").strip(),
        "enabled": _flag("ENABLE_REMOTE_DETECTOR", True),
    }


def _reset_negotiation() -> None:
    """Test hook: forget the cached request shape."""
    global _negotiated
    _negotiated = None


def _headers(key: str, auth_mode: str) -> dict[str, str]:
    if not key:
        return {}
    if auth_mode == "bearer":
        return {"Authorization": f"Bearer {key}"}
    if auth_mode == "x-api-key":
        return {"X-API-Key": key}
    return {}


def _payload(
    fields: dict[str, str],
    b64: str,
    fmt: str,
    lang: str,
    key: str,
    auth_mode: str,
) -> dict[str, Any]:
    p: dict[str, Any] = {fields["audio"]: b64}
    if fields.get("format"):
        p[fields["format"]] = fmt
    if fields.get("language"):
        p[fields["language"]] = lang
    if key and auth_mode == "body":
        p["apiKey"] = key
    return p


def _is_success(status: int, body: Any) -> bool:
    if status != 200 or not isinstance(body, dict):
        return False
    if "classification" in body:
        return True
    return str(body.get("status", "")).lower() == "success"


def _failure_note(status: int, _body: Any) -> str:
    if status == 0:
        return "upstream transport failed"
    if status in {401, 403}:
        return "upstream authentication failed"
    return f"upstream returned HTTP {status}"


def _fields_from_422(body: Any) -> dict[str, str] | None:
    """Learn the server's real field names from a FastAPI validation error."""
    if not isinstance(body, dict):
        return None
    detail = body.get("detail")
    if not isinstance(detail, list):
        return None
    names: list[str] = []
    for item in detail:
        loc = item.get("loc") if isinstance(item, dict) else None
        if isinstance(loc, (list, tuple)) and loc:
            names.append(str(loc[-1]))
    learned: dict[str, str] = {}
    for name in names:
        low = name.lower()
        if "audio" in low or "b64" in low or "base64" in low or low == "data":
            learned.setdefault("audio", name)
        elif "format" in low or low in {"ext", "extension"}:
            learned.setdefault("format", name)
        elif "lang" in low:
            learned.setdefault("language", name)
    return learned if "audio" in learned else None


async def _post(
    url: str,
    payload: dict[str, Any],
    key: str,
    auth_mode: str,
    timeout: float,
):
    try:
        return await _http.post_json(
            url, payload, headers=_headers(key, auth_mode), timeout=timeout
        )
    except Exception as exc:  # post_json shouldn't raise, but stay paranoid
        return 0, {"_transport_error": str(exc)}


async def detect_raw(
    audio_b64: str, audio_format: str = "wav", language: str = "English"
) -> tuple[int, Any, str]:
    """Validate audio, then call the upstream detector with bounded concurrency."""
    if not audio_b64:
        return 400, None, "audioBase64 is required"
    raw = decode_audio(audio_b64)
    if raw is None or len(raw) < _MIN_AUDIO_BYTES:
        return 422, None, "audioBase64 is not valid audio data"
    if len(raw) > _MAX_AUDIO_BYTES:
        return 413, None, "decoded audio exceeds the 24 MB limit"
    async with _detector_lock:
        return await _detect_raw_locked(audio_b64, audio_format, language)


async def _detect_raw_locked(
    audio_b64: str, audio_format: str = "wav", language: str = "English"
) -> tuple[int, Any, str]:
    global _negotiated
    s = _settings()
    if not s["enabled"] or not s["url"]:
        return 0, None, "remote detector disabled or VOICE_API_URL unset"

    # Fast path: a shape already worked in this process.
    if _negotiated:
        fields = _negotiated["fields"]
        status, body = await _post(
            s["url"],
            _payload(
                fields, audio_b64, audio_format, language, s["key"], s["auth_mode"]
            ),
            s["key"],
            s["auth_mode"],
            s["timeout"],
        )
        if _is_success(status, body):
            return 200, body, f"ok via '{fields['audio']}'"
        if status == 0:
            return 0, body, _failure_note(status, body)
        _negotiated = None  # upstream changed; renegotiate below

    shapes: list[dict[str, str]]
    if s["pin_field"]:
        shapes = [
            {"audio": s["pin_field"], "format": "audioFormat", "language": "language"}
        ]
    else:
        shapes = list(_SHAPES)

    calls = 0
    last_status, last_body = -1, None
    tried_learned = False

    for fields in shapes:
        if calls >= _MAX_CALLS_PER_NEGOTIATION:
            break
        calls += 1
        status, body = await _post(
            s["url"],
            _payload(
                fields, audio_b64, audio_format, language, s["key"], s["auth_mode"]
            ),
            s["key"],
            s["auth_mode"],
            s["timeout"],
        )
        last_status, last_body = status, body

        if _is_success(status, body):
            _negotiated = {"fields": fields}
            return 200, body, f"ok via '{fields['audio']}'"

        if status == 0:  # transport dead: retrying other shapes is pointless
            return 0, body, _failure_note(status, body)

        if status in (401, 403):
            return status, body, _failure_note(status, body)

        if status == 422 and not tried_learned and calls < _MAX_CALLS_PER_NEGOTIATION:
            learned = _fields_from_422(body)
            if learned:
                tried_learned = True
                calls += 1
                status3, body3 = await _post(
                    s["url"],
                    _payload(
                        learned,
                        audio_b64,
                        audio_format,
                        language,
                        s["key"],
                        s["auth_mode"],
                    ),
                    s["key"],
                    s["auth_mode"],
                    s["timeout"],
                )
                last_status, last_body = status3, body3
                if _is_success(status3, body3):
                    _negotiated = {"fields": learned}
                    return 200, body3, f"ok via learned '{learned['audio']}'"

    note = (
        "all request shapes rejected — "
        + _failure_note(last_status, last_body)
        + "; pin VOICE_API_AUDIO_FIELD or check the detection API's /docs"
    )
    return last_status, last_body, note


def _enrich(body: dict[str, Any]) -> dict[str, Any]:
    """Surface the neural score as synthetic_probability for the verdict math."""
    try:
        if "synthetic_probability" not in body and "syntheticProbability" not in body:
            neural = body.get("forensics", {}).get("neural_model", {})
            score = neural.get("score")
            if isinstance(score, (int, float)):
                val = float(score)
                if val > 1.0:
                    val /= 100.0
                body["synthetic_probability"] = min(max(val, 0.0), 1.0)
    except Exception:
        pass
    return body


async def detect(
    audio_base64: str | None = None,
    audio_format: str = "wav",
    language: str = "English",
) -> dict[str, Any] | None:
    """Probe target for app/agent/detector.py. Returns the upstream JSON or None."""
    if not audio_base64:
        return None
    status, body, _note = await detect_raw(audio_base64, audio_format, language)
    if status == 200 and isinstance(body, dict) and _is_success(status, body):
        return _enrich(dict(body))
    return None
