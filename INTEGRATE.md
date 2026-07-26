# Integrating VoiceGuard into other projects

VoiceGuard is deployed as a plain REST API — any language, framework, app or
agent that can send JSON over HTTPS can use it. Base URL after deployment
(depending on platform):

```
https://<your-service>.up.railway.app      (Railway)
https://voiceguard-agent.onrender.com      (Render)
https://<you>-voiceguard-agent.hf.space    (HF Spaces, PRO)
```

## Architecture (two independent services)

```
your app ──► voiceguard-agent (this repo)          voice-detection-api
              │  transcribe → claims → research      │  Wav2Vec2 + DSP
              │  → auditable trust score             │  forensics
              └────────── /detect proxy ────────────►│
```

Call **one base URL** for everything — the agent proxies acoustic detection to
the detection Space internally, so future projects only integrate one API.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/investigate` | Full pipeline → structured JSON report |
| POST | `/investigate/report` | Same pipeline → Markdown report (text/markdown) |
| POST | `/investigate/render` | Re-render a saved report JSON as Markdown (no re-run) |
| POST | `/detect` | Acoustic-only detection (full forensics JSON) |
| GET | `/agent/health` | Live capabilities (which keys/sources are active) |
| GET | `/health` | Liveness probe |
| GET | `/docs` | Interactive OpenAPI reference |

### Auth

Open by default. If the deployment sets the `AGENT_API_KEY` secret, every POST
requires the header `X-API-Key: <value>` (401 otherwise).

### Request

```json
{
  "audioBase64": "<base64 audio>",     // OR use transcriptOverride
  "audioFormat": "mp3",                 // wav | mp3 | webm | ogg | m4a ...
  "language": "English",
  "transcriptOverride": null,           // text-only investigation (no audio/ASR)
  "maxResearchRounds": 2,               // 1 = fastest, 3 = deepest
  "includeTrace": true
}
```

### Response essentials (`/investigate`)

```json
{
  "investigation_id": "inv_…",
  "audio_status":  { "classification": "AI_GENERATED|HUMAN|INCONCLUSIVE",
                     "synthetic_probability": 0.97, "degraded": false },
  "transcript":    { "text": "…", "engine": "whisper-large-v3-turbo" },
  "extraction":    { "claims": [ { "id": "c1", "text": "…" } ] },
  "verdict":       { "trust_score": 8, "risk_band": "CRITICAL",
                     "recommendation": "…",
                     "factors": [ { "name": "…", "delta": -34.0 } ],
                     "claim_assessments": [ { "status": "refuted", "…": "…" } ],
                     "caveats": [ "…" ] },
  "harvest":       { "evidence": [ { "url": "…", "credibility": "fact_checker" } ] },
  "trace":         [ { "node": "harvest_web", "status": "ok", "duration_ms": 2140 } ]
}
```

Decision rule of thumb: `trust_score` < 30 or `risk_band` in
{CRITICAL, HIGH} → treat as untrustworthy; always surface `caveats`.

## Client snippets

### curl (text-only — fastest smoke test)

```bash
curl -X POST https://<you>-voiceguard-agent.hf.space/investigate \
  -H 'Content-Type: application/json' \
  -d '{"transcriptOverride":"RBI has banned UPI above 5000 rupees.","maxResearchRounds":1}'
```

### Python

```python
import base64, httpx

BASE = "https://<you>-voiceguard-agent.hf.space"
HEADERS = {}  # or {"X-API-Key": "..."} if the deployment is locked

def investigate_audio(path: str, fmt: str = "mp3") -> dict:
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    r = httpx.post(f"{BASE}/investigate",
                   json={"audioBase64": b64, "audioFormat": fmt},
                   headers=HEADERS, timeout=180)
    r.raise_for_status()
    return r.json()

report = investigate_audio("clip.mp3")
print(report["verdict"]["trust_score"], report["verdict"]["recommendation"])
```

Async: same call with `httpx.AsyncClient` — the API is fully async-friendly.

### JavaScript / TypeScript

```js
const BASE = "https://<you>-voiceguard-agent.hf.space";

async function investigateText(claim) {
  const r = await fetch(`${BASE}/investigate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" }, // + "X-API-Key" if locked
    body: JSON.stringify({ transcriptOverride: claim, maxResearchRounds: 1 }),
  });
  if (!r.ok) throw new Error(`VoiceGuard ${r.status}`);
  return r.json();
}
```

CORS is open (`*`), so browser front-ends can call the API directly.

## Operational notes

- **Latency**: text-only ≈ 5-20 s; with audio ≈ 10-45 s (parallel detection +
  ASR, then research rounds). Set client timeouts ≥ 120 s.
- **Cold starts**: free Spaces sleep after ~48 h idle; first request wakes them
  in ~30-60 s. Send a `GET /health` warm-up ping before demos.
- **Payloads**: base64 inflates audio ~33%; keep uploads under ~15 MB.
- **Errors**: `422` missing input · `401` bad/missing X-API-Key · `502`
  upstream detector unreachable (investigations still work — the acoustic
  stage degrades and the verdict rests on web evidence, disclosed in
  `audio_status.degraded` and `verdict.caveats`).
- **Idempotency**: every call re-runs the research; store the returned JSON if
  you need the exact report again (re-render via `/investigate/render`).
