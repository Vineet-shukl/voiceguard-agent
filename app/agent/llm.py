"""
Multi-provider LLM client with automatic failover and strict JSON extraction.

Order: Gemini (largest free tier) -> Groq (fastest) -> OpenRouter.
If every provider fails or no key exists, callers fall back to deterministic
heuristics so the pipeline still produces a report. That is deliberate: a demo
must never hard-fail because a free tier hit its rate limit.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .config import get_config
from .http import post_json

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMResult:
    def __init__(
        self,
        text: str = "",
        provider: str = "none",
        ok: bool = False,
        error: str = "",
    ) -> None:
        self.text = text
        self.provider = provider
        self.ok = ok
        self.error = error

    def __repr__(self) -> str:  # pragma: no cover
        return f"LLMResult(ok={self.ok}, provider={self.provider!r}, error={self.error!r})"


# --------------------------------------------------------------------------
# JSON salvage — LLMs wrap JSON in prose or fences more often than they should
# --------------------------------------------------------------------------


def extract_json(text: str) -> Any | None:
    if not text:
        return None
    text = text.strip()

    # strip ``` fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # brace/bracket matching for the first complete structure
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        break
    return None


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------


async def _call_gemini(system: str, user: str, want_json: bool) -> LLMResult:
    cfg = get_config()
    url = GEMINI_URL.format(model=cfg.gemini_model)
    payload: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
    }
    if want_json:
        payload["generationConfig"]["responseMimeType"] = "application/json"

    status, data = await post_json(
        url,
        payload,
        headers={"x-goog-api-key": cfg.gemini_api_key},
        timeout=cfg.llm_timeout,
    )
    if status != 200 or not isinstance(data, dict):
        detail = ""
        if isinstance(data, dict):
            detail = str(data.get("error", {}) or data.get("_transport_error", ""))[:200]
        return LLMResult(provider="gemini", error=f"http {status} {detail}".strip())

    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
    except Exception:
        finish = ""
        try:
            finish = data["candidates"][0].get("finishReason", "")
        except Exception:
            pass
        return LLMResult(provider="gemini", error=f"unparseable response {finish}")
    return LLMResult(text=text, provider="gemini", ok=bool(text.strip()))


async def _call_openai_compatible(
    provider: str, url: str, key: str, model: str, system: str, user: str, want_json: bool
) -> LLMResult:
    cfg = get_config()
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
    }
    if want_json:
        payload["response_format"] = {"type": "json_object"}

    headers = {"Authorization": f"Bearer {key}"}
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/voiceguard"
        headers["X-Title"] = "VoiceGuard Investigation Agent"

    status, data = await post_json(url, payload, headers=headers, timeout=cfg.llm_timeout)

    # Some free models reject response_format; retry once without it.
    if status == 400 and want_json:
        payload.pop("response_format", None)
        payload["messages"][0]["content"] = system + "\n\nRespond with raw JSON only."
        status, data = await post_json(url, payload, headers=headers, timeout=cfg.llm_timeout)

    if status != 200 or not isinstance(data, dict):
        detail = ""
        if isinstance(data, dict):
            detail = str(data.get("error", "") or data.get("_transport_error", ""))[:200]
        return LLMResult(provider=provider, error=f"http {status} {detail}".strip())

    try:
        text = data["choices"][0]["message"]["content"] or ""
    except Exception:
        return LLMResult(provider=provider, error="unparseable response")
    return LLMResult(text=text, provider=provider, ok=bool(text.strip()))


async def complete(system: str, user: str, *, want_json: bool = True) -> LLMResult:
    """Try each configured provider in order; return the first success."""
    cfg = get_config()
    errors: list[str] = []

    for provider in cfg.provider_order():
        if provider == "gemini":
            res = await _call_gemini(system, user, want_json)
        elif provider == "groq":
            res = await _call_openai_compatible(
                "groq", GROQ_URL, cfg.groq_api_key, cfg.groq_model, system, user, want_json
            )
        else:
            res = await _call_openai_compatible(
                "openrouter",
                OPENROUTER_URL,
                cfg.openrouter_api_key,
                cfg.openrouter_model,
                system,
                user,
                want_json,
            )
        if res.ok:
            return res
        errors.append(f"{provider}: {res.error}")

    return LLMResult(
        provider="none",
        error="; ".join(errors) if errors else "no LLM provider configured",
    )


async def complete_json(system: str, user: str) -> tuple[Any | None, LLMResult]:
    res = await complete(system, user, want_json=True)
    if not res.ok:
        return None, res
    parsed = extract_json(res.text)
    if parsed is None:
        res.ok = False
        res.error = "response was not valid JSON"
    return parsed, res
