"""
One-command deployment of the VoiceGuard agent to Hugging Face Spaces.

Run this ON YOUR MACHINE (it needs internet):

    pip install -U huggingface_hub
    python deploy.py

What it does:
  1. logs into Hugging Face with your WRITE token
  2. creates (or reuses) a Docker Space:  <you>/voiceguard-agent
  3. sets your API keys as Space SECRETS (read from .env, or prompted)
  4. uploads this folder (minus .env and caches)
  5. waits for the build, then smoke-tests the live endpoints

Get a WRITE token at:  https://huggingface.co/settings/tokens
"""
from __future__ import annotations

import argparse
import base64
import getpass
import io
import os
import re
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RBI_DEMO_CLAIM = (
    "This is the Reserve Bank of India. All UPI transactions above 5000 rupees "
    "are banned from Monday. Withdraw your cash immediately."
)

SECRET_KEYS = [
    ("GEMINI_API_KEY", "Google AI Studio key (powers reasoning, grounded search, ASR)"),
    ("GROQ_API_KEY", "Groq key (Whisper ASR + LLM failover)"),
    ("OPENROUTER_API_KEY", "OpenRouter key (optional 3rd LLM fallback)"),
    ("FACTCHECK_API_KEY", "Fact Check Tools key (blank = reuse Gemini key)"),
    ("VOICE_API_KEY", "Key for the deployed voice-detection API"),
    ("AGENT_API_KEY", "Required unless --public-demo explicitly enables open POST endpoints"),
]


def read_env_file(path: Path) -> dict:
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        values[k.strip()] = v.strip()
    return values


def tiny_wav_b64() -> str:
    """0.5 s of near-silence, 16 kHz mono — just to prove the transport."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        frames = bytearray()
        for i in range(8000):
            val = 40 if (i // 400) % 2 == 0 else -40  # faint square wobble
            frames += int(val).to_bytes(2, "little", signed=True)
        w.writeframes(bytes(frames))
    return base64.b64encode(buf.getvalue()).decode()


def space_subdomain(owner: str, name: str) -> str:
    sub = f"{owner}-{name}".lower()
    return re.sub(r"[^a-z0-9-]", "-", sub)


def main() -> int:
    ap = argparse.ArgumentParser(description="Deploy VoiceGuard agent to HF Spaces")
    ap.add_argument("--name", default=os.getenv("SPACE_NAME", "voiceguard-agent"),
                    help="Space name (default: voiceguard-agent)")
    ap.add_argument("--private", action="store_true", help="make the Space private")
    ap.add_argument(
        "--public-demo",
        action="store_true",
        help="explicitly allow unauthenticated POST requests (rate limits still apply)",
    )
    ap.add_argument("--token", default=None, help="HF write token (else HF_TOKEN env / .env / prompt)")
    ap.add_argument("--skip-smoke", action="store_true", help="skip live endpoint tests")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[!!] huggingface_hub is not installed. Run:")
        print("     pip install -U huggingface_hub")
        return 1

    import requests  # dependency of huggingface_hub

    env = read_env_file(ROOT / ".env")

    token = args.token or os.getenv("HF_TOKEN") or env.get("HF_TOKEN") or ""
    if not token:
        print("A Hugging Face WRITE token is needed (https://huggingface.co/settings/tokens)")
        token = getpass.getpass("HF token (input hidden): ").strip()
    if not token:
        print("[!!] No token supplied; aborting.")
        return 1

    api = HfApi(token=token)
    try:
        who = api.whoami()
    except Exception as exc:
        print(f"[!!] Token rejected: {exc}")
        return 1
    owner = str(who.get("name") or "").strip()
    if not owner:
        print("[!!] Hugging Face account response did not include an owner name; aborting.")
        return 1
    print(f"[ok] Logged in as: {owner}")

    repo_id = f"{owner}/{args.name}"
    print(f"[..] Creating Space {repo_id} (docker sdk) ...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )
    print("[ok] Space exists.")

    # ---- secrets -----------------------------------------------------------
    print("\nSecrets (blank = skip). Values come from .env when present.")
    configured_secrets: dict[str, str] = {}
    for key, help_text in SECRET_KEYS:
        val = env.get(key, "")
        if key == "AGENT_API_KEY" and args.public_demo:
            print("  [--] skipped:    AGENT_API_KEY (--public-demo enabled)")
            continue
        if not val and key in ("GEMINI_API_KEY", "GROQ_API_KEY", "AGENT_API_KEY"):
            try:
                val = getpass.getpass(f"  {key} — {help_text}\n    value (hidden, Enter to skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                val = ""
        if key == "AGENT_API_KEY" and not val:
            print("[!!] AGENT_API_KEY is required unless --public-demo is supplied; aborting.")
            return 1
        if val:
            api.add_space_secret(repo_id=repo_id, key=key, value=val)
            configured_secrets[key] = val
            print(f"  [ok] secret set: {key}")
        else:
            print(f"  [--] skipped:    {key}")

    api.add_space_variable(
        repo_id=repo_id,
        key="ALLOW_PUBLIC_DEMO",
        value="true" if args.public_demo else "false",
    )
    voice_url = env.get("VOICE_API_URL", "https://pandaisop-voice-detection-api.hf.space/detect")
    api.add_space_variable(repo_id=repo_id, key="VOICE_API_URL", value=voice_url)
    print(f"  [ok] variable set: VOICE_API_URL = {voice_url}")

    # ---- upload ------------------------------------------------------------
    print("\n[..] Uploading project files ...")
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy VoiceGuard investigation agent",
        ignore_patterns=[
            ".env", ".env.*", "**/.env", "**/.env.*", "*.pem", "*.key",
            "*.p12", "*.pfx", ".git*", "**/.git/**", "**/__pycache__/**",
            "*.pyc", "**/.pytest_cache/**", "venv/**", ".venv/**",
            "b64.txt", "*.wav", "*.mp3", ".DS_Store", ".claude/**", ".zed/**",
        ],
    )
    print("[ok] Upload complete.")

    try:
        api.restart_space(repo_id=repo_id)
    except Exception:
        pass  # first build starts automatically

    # ---- wait for build ----------------------------------------------------
    print("\n[..] Waiting for the Space to build (first build ~1-3 min) ...")
    base = f"https://{space_subdomain(owner, args.name)}.hf.space"
    deadline = time.time() + 900
    stage = "UNKNOWN"
    while time.time() < deadline:
        try:
            stage = str(api.get_space_runtime(repo_id).stage)
        except Exception:
            stage = "UNKNOWN"
        print(f"     stage: {stage:<20}", end="\r")
        if "RUNNING" in stage:
            print(f"\n[ok] Space is RUNNING: {base}")
            break
        if "ERROR" in stage:
            print(f"\n[!!] Build failed with stage {stage}.")
            print(f"     Check logs: https://huggingface.co/spaces/{repo_id} -> Logs tab")
            return 1
        time.sleep(6)
    else:
        print("\n[!!] Timed out waiting for RUNNING. Check the Space page logs.")
        return 1

    if args.skip_smoke:
        print("\n[ok] Deployed (smoke tests skipped).")
        print(f"     UI:   {base}\n     Docs: {base}/docs")
        return 0

    # ---- live smoke tests ----------------------------------------------------
    agent_key = configured_secrets.get("AGENT_API_KEY", "")
    headers = {"X-API-Key": agent_key} if agent_key else {}
    ok = True

    print("\n[1/3] GET /agent/health")
    health = None
    for attempt in range(5):
        try:
            r = requests.get(f"{base}/agent/health", timeout=120)
            if r.status_code == 200:
                health = r.json()
                break
        except Exception:
            pass
        time.sleep(10)
    if health:
        caps = health.get("capabilities", {})
        print(f"      [ok] llm={caps.get('llm_providers')} asr={caps.get('asr')}")
        print(f"           grounding={caps.get('google_search_grounding')} "
              f"factcheck={caps.get('factcheck_api')}")
    else:
        ok = False
        print("      [!!] health endpoint not answering")

    print("[2/3] POST /investigate (text-only demo claim, ~10-40 s)")
    try:
        r = requests.post(
            f"{base}/investigate",
            json={"transcriptOverride": RBI_DEMO_CLAIM, "maxResearchRounds": 1},
            headers=headers, timeout=300,
        )
        if r.status_code == 200:
            rep = r.json()
            v = rep.get("verdict", {})
            print(f"      [ok] trust={v.get('trust_score')}/100 band={v.get('risk_band')} "
                  f"evidence={len(rep.get('harvest', {}).get('evidence', []))} "
                  f"in {rep.get('total_duration_ms', 0)/1000:.1f}s")
        else:
            ok = False
            print(f"      [!!] HTTP {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        ok = False
        print(f"      [!!] {exc}")

    print("[3/3] POST /detect (tiny wav -> upstream detector; may wake that Space)")
    try:
        r = requests.post(
            f"{base}/detect",
            json={"audioBase64": tiny_wav_b64(), "audioFormat": "wav", "language": "English"},
            headers=headers, timeout=300,
        )
        if r.status_code == 200:
            d = r.json()
            print(f"      [ok] upstream answered: classification={d.get('classification')} "
                  f"confidence={d.get('confidenceScore')}")
        else:
            detail = ""
            try:
                detail = r.json().get("detail", "")
            except Exception:
                detail = r.text[:200]
            print(f"      [!?] upstream not reachable yet (HTTP {r.status_code}).")
            print(f"           {detail}")
            print("           The pipeline still works (degrades to web-only verdicts).")
            print("           If this persists, the detection API may expect different")
            print("           field names — set the VOICE_API_AUDIO_FIELD variable on the")
            print("           Space, or send me its exact request JSON.")
    except Exception as exc:
        print(f"      [!?] {exc}")

    print("\n" + "=" * 62)
    print("DEPLOYED" if ok else "DEPLOYED (with warnings above)")
    print(f"  Demo UI     : {base}")
    print(f"  API docs    : {base}/docs")
    print(f"  Health      : {base}/agent/health")
    print(f"  Investigate : POST {base}/investigate")
    print(f"  Detect      : POST {base}/detect")
    if agent_key:
        print("  Auth        : X-API-Key header required on POSTs")
    print("  Integrate from other projects: see INTEGRATE.md")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
