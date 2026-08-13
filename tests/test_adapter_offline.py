"""
Offline tests for the remote-detector adapter (app/core/model.py).

The network is faked at app.agent.http.post_json — the same seam
tests/test_offline.py uses — so this runs with zero keys and no network.

Run:  python3 tests/test_adapter_offline.py
"""
from __future__ import annotations

import asyncio
import base64
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
try:
    import pydantic  # noqa: F401
except ImportError:
    import _pydantic_shim  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _k in (
    "VOICE_API_KEY",
    "VOICE_API_AUTH_MODE",
    "VOICE_API_AUDIO_FIELD",
    "VOICE_API_URL",
    "VOICE_API_TIMEOUT",
    "ENABLE_REMOTE_DETECTOR",
):
    os.environ.pop(_k, None)

import app.agent.http as http_mod  # noqa: E402
from app.agent import detector  # noqa: E402
from app.agent.schemas import Classification  # noqa: E402
from app.core import model  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: object, info: object = ""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + str(info)) if info and not cond else ''}")


# Condensed real response from the deployed detection API.
SAMPLE = {
    "status": "success",
    "language": "English",
    "classification": "HUMAN",
    "confidenceScore": 0.88,
    "explanation": "Authentic human speech patterns and natural variations detected.",
    "inferenceTimeMs": 2874.9,
    "analyzersAgree": True,
    "forensics": {
        "neural_model": {
            "score": 0.1562,
            "verdict": "HUMAN",
            "segments_analyzed": 3,
            "per_segment_scores": [0.1576, 0.1554, 0.1555],
        },
        "spectral_analysis": {"score": 0, "verdict": "HUMAN", "artifacts_found": []},
    },
    "audioProfile": {"duration_sec": 30, "sample_rate": 16000},
    "artifactsSummary": ["excessive_inter_frame_correlation"],
}

GOOD_B64 = base64.b64encode(b"RIFF" + b"\x00" * 128).decode()


class FakeNet:
    """Scriptable stand-in for app.agent.http.post_json."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def post_json(self, url, payload, *, headers=None, timeout=None):
        self.calls.append(
            {"url": url, "payload": payload, "headers": headers or {}, "timeout": timeout}
        )
        if not self.script:
            return 0, {"_transport_error": "script exhausted"}
        item = self.script.pop(0)
        if callable(item):
            return item(url, payload, headers or {})
        return item


def wire(script):
    net = FakeNet(script)
    http_mod.post_json = net.post_json  # adapter late-binds through the module
    model._reset_negotiation()
    return net


def test_no_network_for_bad_input():
    print("\n[A1] Invalid input never touches the network")
    net = wire([])
    check("None input -> None", asyncio.run(model.detect(None)) is None)
    check("garbage b64 -> None", asyncio.run(model.detect(audio_base64="!!!not base64!!!")) is None)
    check("tiny payload -> None", asyncio.run(model.detect(audio_base64="QUJD")) is None)
    check("zero HTTP calls made", len(net.calls) == 0, net.calls)


def test_happy_path_and_probe():
    print("\n[A2] Happy path + probe integration")
    net = wire([(200, SAMPLE), (200, SAMPLE)])

    out = asyncio.run(model.detect(GOOD_B64, "mp3", "English"))
    check("dict returned", isinstance(out, dict))
    check("camelCase shape used first", "audioBase64" in net.calls[0]["payload"], net.calls[0]["payload"].keys())
    check("format field sent", net.calls[0]["payload"].get("audioFormat") == "mp3")
    check("neural score surfaced", abs(out.get("synthetic_probability", -1) - 0.1562) < 1e-9)

    res, note = asyncio.run(detector.detect(GOOD_B64, "mp3", "English"))
    check("probe found adapter", "app.core.model" in note, note)
    check("classification HUMAN", res.classification == Classification.HUMAN)
    check("synthetic prob from neural model", abs(res.synthetic_probability - 0.1562) < 1e-9)
    check("confidence coerced", abs(res.confidence - 0.88) < 1e-9)
    check("analyzers agree", res.analyzers_agree is True)
    check("inference time kept", res.inference_time_ms == 2874.9)
    check("not degraded", not res.degraded)
    check("negotiation cached (1 call each)", len(net.calls) == 2, len(net.calls))


def test_key_transport():
    print("\n[A3] API key uses one explicit transport")
    os.environ["VOICE_API_KEY"] = "sekret-123"
    try:
        net = wire([(200, SAMPLE)])
        asyncio.run(model.detect(GOOD_B64))
        c = net.calls[0]
        check("X-API-Key header", c["headers"].get("X-API-Key") == "sekret-123")
        check("no redundant Bearer header", "Authorization" not in c["headers"])
        check("no key in request body", "apiKey" not in c["payload"])
        check("no key in request URL", "sekret-123" not in c["url"])
    finally:
        os.environ.pop("VOICE_API_KEY", None)


def test_negotiation_via_422():
    print("\n[A4] Learns field names from a FastAPI 422")
    err_422 = (
        422,
        {
            "detail": [
                {"loc": ["body", "audio_data"], "msg": "Field required"},
                {"loc": ["body", "lang"], "msg": "Field required"},
            ]
        },
    )

    def learned_ok(url, payload, headers):
        if "audio_data" in payload and "lang" in payload:
            return 200, SAMPLE
        return 500, {"detail": "wrong shape"}

    net = wire([err_422, learned_ok])
    out = asyncio.run(model.detect(GOOD_B64))
    check("success after learning", isinstance(out, dict))
    check("exactly two calls", len(net.calls) == 2, len(net.calls))
    check("learned field used", "audio_data" in net.calls[1]["payload"])

    net2 = wire([])  # fresh net, but negotiation cache survives wire()? No — reset.
    net2.script = [(200, SAMPLE)]
    model._negotiated = {"fields": {"audio": "audio_data", "language": "lang"}}
    out2 = asyncio.run(model.detect(GOOD_B64))
    check("cached shape reused", isinstance(out2, dict) and "audio_data" in net2.calls[0]["payload"])


def test_explicit_auth_modes():
    print("\n[A5] Auth modes never expose keys in URLs")
    os.environ["VOICE_API_KEY"] = "mode-key-9"
    try:
        os.environ["VOICE_API_AUTH_MODE"] = "bearer"
        bearer = wire([(200, SAMPLE)])
        out = asyncio.run(model.detect(GOOD_B64))
        check("Bearer mode succeeds", isinstance(out, dict))
        check(
            "Bearer mode uses Authorization only",
            bearer.calls[0]["headers"] == {"Authorization": "Bearer mode-key-9"}
            and "apiKey" not in bearer.calls[0]["payload"],
            bearer.calls[0],
        )

        os.environ["VOICE_API_AUTH_MODE"] = "body"
        body = wire([(200, SAMPLE)])
        out = asyncio.run(model.detect(GOOD_B64))
        check("body mode succeeds", isinstance(out, dict))
        check(
            "body mode omits auth headers",
            body.calls[0]["headers"] == {}
            and body.calls[0]["payload"].get("apiKey") == "mode-key-9",
            body.calls[0],
        )
        check(
            "auth modes never put keys in URLs",
            all("mode-key-9" not in call["url"] for call in bearer.calls + body.calls),
        )
    finally:
        os.environ.pop("VOICE_API_AUTH_MODE", None)
        os.environ.pop("VOICE_API_KEY", None)


def test_transport_dead():
    print("\n[A6] Dead transport degrades after a single attempt")
    net = wire([(0, {"_transport_error": "network mocked out"})])
    out = asyncio.run(model.detect(GOOD_B64))
    check("returns None", out is None)
    check("single call only", len(net.calls) == 1, len(net.calls))

    res, note = asyncio.run(detector.detect(GOOD_B64))
    check("probe degrades gracefully", res.degraded, note)


def test_pinned_field():
    print("\n[A7] VOICE_API_AUDIO_FIELD pins the shape")
    os.environ["VOICE_API_AUDIO_FIELD"] = "clipData"
    try:
        net = wire([(200, SAMPLE)])
        out = asyncio.run(model.detect(GOOD_B64))
        check("pinned field used", "clipData" in net.calls[0]["payload"], net.calls[0]["payload"].keys())
        check("success", isinstance(out, dict))
    finally:
        os.environ.pop("VOICE_API_AUDIO_FIELD", None)


def test_soft_error_body():
    print("\n[A8] 200-with-error body is not success")
    soft = (200, {"status": "error", "message": "could not decode audio"})
    net = wire([soft, soft, soft])
    out = asyncio.run(model.detect(GOOD_B64))
    check("returns None", out is None)
    check("all shapes tried", len(net.calls) == 3, len(net.calls))


if __name__ == "__main__":
    print("=" * 74)
    print("VoiceGuard remote-detector adapter — offline verification")
    print("=" * 74)
    test_no_network_for_bad_input()
    test_happy_path_and_probe()
    test_key_transport()
    test_negotiation_via_422()
    test_explicit_auth_modes()
    test_transport_dead()
    test_pinned_field()
    test_soft_error_body()
    print("\n" + "=" * 74)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILURES:")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
