"""
Speech-to-text via Groq's hosted Whisper (free tier, OpenAI-compatible).

Requires a multipart upload, which we build by hand so the module stays
dependency-free.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import uuid

from .config import get_config
from .schemas import Transcript

GROQ_ASR_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

_MIME = {
    "mp3": "audio/mpeg",
    "mpeg": "audio/mpeg",
    "wav": "audio/wav",
    "m4a": "audio/mp4",
    "mp4": "audio/mp4",
    "ogg": "audio/ogg",
    "opus": "audio/opus",
    "webm": "audio/webm",
    "flac": "audio/flac",
}


def decode_audio(audio_b64: str) -> bytes | None:
    """Tolerant base64 decode: handles data URLs, whitespace, missing padding."""
    if not audio_b64:
        return None
    raw = audio_b64.strip()
    if raw.startswith("data:"):
        _, _, raw = raw.partition(",")
    raw = "".join(raw.split())
    raw = raw.replace("-", "+").replace("_", "/")
    padding = len(raw) % 4
    if padding:
        raw += "=" * (4 - padding)
    try:
        return base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError):
        return None


def _multipart(audio: bytes, filename: str, model: str) -> tuple[bytes, str]:
    boundary = f"----VoiceGuard{uuid.uuid4().hex}"
    extension = filename.rsplit(".", 1)[-1].lower()
    if extension not in _MIME:
        extension = "wav"
    filename = f"audio.{extension}"
    mime = _MIME[extension]
    parts: list[bytes] = []

    def field(name: str, value: str) -> None:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )

    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode()
    )
    parts.append(audio)
    parts.append(b"\r\n")
    field("model", model)
    field("response_format", "verbose_json")
    field("temperature", "0")
    parts.append(f"--{boundary}--\r\n".encode())

    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _blocking_transcribe(audio: bytes, fmt: str) -> Transcript:
    import json
    import urllib.error
    import urllib.request

    from .http import _ssl_context

    cfg = get_config()
    body, content_type = _multipart(audio, f"audio.{fmt}", cfg.groq_whisper_model)
    req = urllib.request.Request(
        GROQ_ASR_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {cfg.groq_api_key}",
            "Content-Type": content_type,
            "User-Agent": cfg.user_agent,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.llm_timeout, context=_ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        return Transcript(available=False, note=f"ASR http {exc.code}: {detail}")
    except Exception as exc:
        return Transcript(available=False, note=f"ASR transport error: {exc}")

    text = (data.get("text") or "").strip()
    return Transcript(
        text=text,
        language=data.get("language"),
        duration_seconds=data.get("duration"),
        engine=cfg.groq_whisper_model,
        available=bool(text),
        note="" if text else "ASR returned empty text (silent or music-only audio?)",
    )


async def _transcribe_gemini(audio: bytes, fmt: str) -> Transcript:
    """Gemini accepts audio inline — transcription with no separate ASR service.

    Inline base64 keeps request size ~1.37x the audio; stay under ~15 MB raw so
    the JSON body remains within Gemini's 20 MB request cap.
    """
    import base64

    from .http import post_json

    cfg = get_config()
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{cfg.gemini_model}:generateContent"
    )
    mime = _MIME.get(fmt, "audio/wav")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Transcribe this audio verbatim. Return ONLY the spoken "
                            "words, no commentary, no timestamps, no speaker labels. "
                            "If there is no speech, return an empty response."
                        )
                    },
                    {"inline_data": {"mime_type": mime, "data": base64.b64encode(audio).decode()}},
                ],
            }
        ],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4096},
    }
    status, data = await post_json(
        url, payload, headers={"x-goog-api-key": cfg.gemini_api_key}, timeout=cfg.llm_timeout
    )
    if status != 200 or not isinstance(data, dict):
        detail = ""
        if isinstance(data, dict):
            detail = str(data.get("error", {}) or data.get("_transport_error", ""))[:160]
        return Transcript(available=False, engine="gemini", note=f"Gemini ASR http {status} {detail}".strip())
    try:
        text = "".join(
            p.get("text", "") for p in data["candidates"][0]["content"]["parts"]
        ).strip()
    except Exception:
        return Transcript(available=False, engine="gemini", note="Gemini ASR: unparseable response")
    return Transcript(
        text=text,
        engine=f"gemini:{cfg.gemini_model}",
        available=bool(text),
        note="" if text else "Gemini heard no speech in the audio",
    )


async def transcribe(audio_b64: str | None, audio_format: str = "wav") -> Transcript:
    """Groq Whisper first (purpose-built ASR), Gemini native audio as fallback,
    so a single Google key is enough to run the whole pipeline."""
    cfg = get_config()

    if not audio_b64:
        return Transcript(available=False, note="no audio supplied")
    if not cfg.groq_api_key and not cfg.gemini_api_key:
        return Transcript(
            available=False,
            note="No GROQ_API_KEY or GEMINI_API_KEY — transcription unavailable. "
            "Pass transcriptOverride to investigate text directly.",
        )

    audio = decode_audio(audio_b64)
    if not audio:
        return Transcript(available=False, note="audioBase64 could not be decoded")
    if len(audio) < 512:
        return Transcript(available=False, note="audio payload too small to transcribe")
    # Groq caps free-tier uploads around 25 MB
    if len(audio) > 24 * 1024 * 1024:
        return Transcript(
            available=False, note=f"audio too large for ASR ({len(audio) // 1024 // 1024} MB)"
        )

    fmt = (audio_format or "wav").lower().lstrip(".")
    if fmt not in _MIME:
        return Transcript(available=False, note="unsupported audio format")

    if cfg.groq_api_key:
        result = await asyncio.to_thread(_blocking_transcribe, audio, fmt)
        if result.available:
            return result
        # Whisper failed — try Gemini before giving up
        if cfg.gemini_api_key and len(audio) <= 15 * 1024 * 1024:
            gem = await _transcribe_gemini(audio, fmt)
            if gem.available:
                gem.note = f"Groq Whisper failed ({result.note}); Gemini fallback used"
                return gem
        return result

    if len(audio) > 15 * 1024 * 1024:
        return Transcript(
            available=False,
            note="audio too large for Gemini inline ASR (15 MB max); set GROQ_API_KEY",
        )
    return await _transcribe_gemini(audio, fmt)
