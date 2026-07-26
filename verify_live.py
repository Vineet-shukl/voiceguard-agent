"""
Live smoke test for a deployed VoiceGuard agent — works on any platform
(Railway, Render, HF Spaces, localhost). Zero dependencies (stdlib only).

    python verify_live.py https://your-service.up.railway.app
    python verify_live.py https://voiceguard-agent.onrender.com --api-key MYKEY
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
import wave

RBI_DEMO_CLAIM = (
    "This is the Reserve Bank of India. All UPI transactions above 5000 rupees "
    "are banned from Monday. Withdraw your cash immediately."
)


def tiny_wav_b64() -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        frames = bytearray()
        for i in range(8000):
            val = 40 if (i // 400) % 2 == 0 else -40
            frames += int(val).to_bytes(2, "little", signed=True)
        w.writeframes(bytes(frames))
    return base64.b64encode(buf.getvalue()).decode()


def call(method: str, url: str, body=None, headers=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    hdrs = {"Content-Type": "application/json", "User-Agent": "voiceguard-verify/1.0"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(raw)
            except Exception:
                return r.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw
    except Exception as e:
        return 0, str(e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="base URL, e.g. https://xyz.up.railway.app")
    ap.add_argument("--api-key", default="", help="X-API-Key if the service is locked")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    failures = 0

    print(f"Target: {base}\n")

    print("[1/4] GET /health (wakes the service if asleep)")
    status = 0
    for attempt in range(6):
        status, body = call("GET", f"{base}/health", timeout=90)
        if status == 200:
            print(f"      [ok] {body}")
            break
        print(f"      ... attempt {attempt + 1}: HTTP {status}, retrying in 10 s")
        time.sleep(10)
    if status != 200:
        print("      [!!] service not answering — check the platform's deploy logs")
        return 1

    print("[2/4] GET /agent/health")
    status, body = call("GET", f"{base}/agent/health", timeout=120)
    if status == 200 and isinstance(body, dict):
        caps = body.get("capabilities", {})
        print(f"      [ok] llm={caps.get('llm_providers')}")
        print(f"           asr={caps.get('asr')}")
        print(f"           grounding={caps.get('google_search_grounding')} "
              f"factcheck={caps.get('factcheck_api')} sources={caps.get('sources')}")
        if caps.get("llm_providers") == ["none (heuristic fallback)"]:
            print("      [!?] no LLM key visible — set GEMINI_API_KEY in the platform's variables")
    else:
        failures += 1
        print(f"      [!!] HTTP {status}: {str(body)[:200]}")

    print("[3/4] POST /investigate (text-only, ~10-40 s)")
    t0 = time.time()
    status, body = call(
        "POST", f"{base}/investigate",
        {"transcriptOverride": RBI_DEMO_CLAIM, "maxResearchRounds": 1},
        headers=headers,
    )
    if status == 200 and isinstance(body, dict):
        v = body.get("verdict", {})
        ev = len(body.get("harvest", {}).get("evidence", []))
        print(f"      [ok] trust={v.get('trust_score')}/100 band={v.get('risk_band')} "
              f"evidence={ev} in {time.time() - t0:.1f}s")
        if ev == 0:
            print("      [!?] zero evidence retrieved — check keys / try again (sources may be rate-limited)")
    else:
        failures += 1
        print(f"      [!!] HTTP {status}: {str(body)[:300]}")

    print("[4/4] POST /detect (round-trip to the detection Space; may wake it, ~1-2 min)")
    status, body = call(
        "POST", f"{base}/detect",
        {"audioBase64": tiny_wav_b64(), "audioFormat": "wav", "language": "English"},
        headers=headers,
    )
    if status == 200 and isinstance(body, dict):
        print(f"      [ok] upstream answered: classification={body.get('classification')} "
              f"confidence={body.get('confidenceScore')}")
    else:
        detail = body.get("detail", "") if isinstance(body, dict) else str(body)[:200]
        print(f"      [!?] upstream not confirmed (HTTP {status}): {detail}")
        print("           Investigations still work — the acoustic stage degrades gracefully.")
        print("           If this persists, send me the detection API's exact request JSON")
        print("           or set the VOICE_API_AUDIO_FIELD variable.")

    print("\n" + "=" * 60)
    print("ALL GOOD" if failures == 0 else f"{failures} check(s) failed — paste this output to Claude")
    print("=" * 60)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
