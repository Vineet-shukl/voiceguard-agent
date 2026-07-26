"""
Adapter to the existing VoiceGuard acoustic detector.

Written defensively on purpose: I do not know the exact function signature in
your `app/core/model.py`, so this probes several plausible entry points and
normalises whatever shape comes back into `DetectionResult`. If nothing matches,
it returns a `degraded` result and the pipeline continues — the investigation is
still useful without the acoustic score, and the verdict engine knows to stop
weighting it.

If you know your entry point, delete the probing and call it directly:
    from app.core.model import detect_audio
"""

from __future__ import annotations

import inspect
from typing import Any

from .schemas import Classification, DetectionResult

# Candidate (module, attribute) pairs, most likely first.
_CANDIDATES = [
    ("app.core.model", "detect"),
    ("app.core.model", "detect_audio"),
    ("app.core.model", "predict"),
    ("app.core.model", "run_detection"),
    ("app.core.model", "analyze"),
    ("app.core.detector", "detect"),
    ("app.core.pipeline", "detect"),
    ("app.services.detection", "detect"),
]

_CLASS_CANDIDATES = [
    ("app.core.model", "VoiceGuardModel", "predict"),
    ("app.core.model", "Detector", "detect"),
    ("app.core.model", "VoiceDetector", "detect"),
]


def _coerce(raw: Any) -> DetectionResult | None:
    """Normalise dict / object / tuple outputs into DetectionResult."""
    if raw is None:
        return None

    data: dict[str, Any] = {}
    if isinstance(raw, dict):
        data = raw
    elif hasattr(raw, "model_dump"):
        try:
            data = raw.model_dump()
        except Exception:
            return None
    elif hasattr(raw, "__dict__"):
        data = {k: v for k, v in vars(raw).items() if not k.startswith("_")}
    else:
        return None

    lowered = {str(k).lower().replace("_", ""): v for k, v in data.items()}

    label = str(
        lowered.get("classification")
        or lowered.get("label")
        or lowered.get("prediction")
        or lowered.get("result")
        or ""
    ).upper()

    prob = lowered.get("syntheticprobability")
    if prob is None:
        prob = lowered.get("aiprobability") or lowered.get("fakeprobability")
    conf = lowered.get("confidencescore") or lowered.get("confidence") or 0.0

    try:
        conf_f = float(conf)
    except (TypeError, ValueError):
        conf_f = 0.0
    if conf_f > 1.0:  # someone returned a percentage
        conf_f /= 100.0

    if prob is None:
        # Derive from label + confidence
        if "AI" in label or "FAKE" in label or "SYNTH" in label or "SPOOF" in label:
            prob_f = conf_f if conf_f else 0.75
        elif "HUMAN" in label or "REAL" in label or "BONA" in label:
            prob_f = 1.0 - (conf_f if conf_f else 0.75)
        else:
            prob_f = 0.5
    else:
        try:
            prob_f = float(prob)
        except (TypeError, ValueError):
            prob_f = 0.5
        if prob_f > 1.0:
            prob_f /= 100.0

    if "AI" in label or "FAKE" in label or "SYNTH" in label or "SPOOF" in label:
        classification = Classification.AI_GENERATED
    elif "HUMAN" in label or "REAL" in label or "BONA" in label:
        classification = Classification.HUMAN
    elif prob_f >= 0.65:
        classification = Classification.AI_GENERATED
    elif prob_f <= 0.35:
        classification = Classification.HUMAN
    else:
        classification = Classification.INCONCLUSIVE

    agree = lowered.get("analyzersagree")
    inference = lowered.get("inferencetimems")
    try:
        inference_f = float(inference) if inference is not None else None
    except (TypeError, ValueError):
        inference_f = None

    return DetectionResult(
        classification=classification,
        synthetic_probability=min(max(prob_f, 0.0), 1.0),
        confidence=min(max(conf_f or abs(prob_f - 0.5) * 2, 0.0), 1.0),
        analyzers_agree=bool(agree) if agree is not None else None,
        explanation=str(lowered.get("explanation") or "")[:1000],
        inference_time_ms=inference_f,
        engine="voiceguard",
    )


async def _try_call(fn: Any, audio_b64: str, audio_format: str, language: str) -> Any:
    """Call fn with whichever argument names it actually accepts."""
    try:
        sig = inspect.signature(fn)
        params = set(sig.parameters)
    except (TypeError, ValueError):
        params = set()

    kwargs: dict[str, Any] = {}
    for name, value in [
        ("audio_base64", audio_b64), ("audioBase64", audio_b64), ("audio_b64", audio_b64),
        ("audio_format", audio_format), ("audioFormat", audio_format), ("format", audio_format),
        ("language", language),
    ]:
        if name in params:
            kwargs[name] = value

    if kwargs:
        out = fn(**kwargs)
    else:
        # positional: base64 first is the overwhelmingly common convention
        out = fn(audio_b64)

    if inspect.isawaitable(out):
        out = await out
    return out


async def detect(
    audio_b64: str | None, audio_format: str = "wav", language: str = "English"
) -> tuple[DetectionResult, str]:
    """Returns (result, note). Never raises."""
    if not audio_b64:
        return (
            DetectionResult(degraded=True, explanation="No audio supplied; text-only investigation."),
            "skipped (no audio)",
        )

    import importlib

    errors: list[str] = []

    for module_name, attr in _CANDIDATES:
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        fn = getattr(mod, attr, None)
        if not callable(fn):
            continue
        try:
            raw = await _try_call(fn, audio_b64, audio_format, language)
        except Exception as exc:
            errors.append(f"{module_name}.{attr}: {exc}")
            continue
        coerced = _coerce(raw)
        if coerced:
            return coerced, f"detected via {module_name}.{attr}"

    for module_name, cls_name, method in _CLASS_CANDIDATES:
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            instance = cls()
            fn = getattr(instance, method, None)
            if not callable(fn):
                continue
            raw = await _try_call(fn, audio_b64, audio_format, language)
        except Exception as exc:
            errors.append(f"{module_name}.{cls_name}: {exc}")
            continue
        coerced = _coerce(raw)
        if coerced:
            return coerced, f"detected via {module_name}.{cls_name}.{method}"

    note = "VoiceGuard detector not reachable"
    if errors:
        note += f" ({errors[0][:160]})"
    return (
        DetectionResult(
            degraded=True,
            explanation="Acoustic detector unavailable; investigation proceeded on transcript "
            "and web evidence only. Wire app/agent/detector.py to your detect() function.",
        ),
        note,
    )
