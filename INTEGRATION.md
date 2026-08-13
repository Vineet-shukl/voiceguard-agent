# VoiceGuard Investigation Agent — Deep Dive & Integration Guide

Comprehensive technical reference covering architecture, integration steps, and system design decisions.

---

## Part 1 — Integration (15 minutes)

### Step 1. Copy files into your project

```
voice-detection-api/
├── app/
│   ├── main.py                  ← you edit this (2 lines)
│   ├── core/                    ← untouched
│   ├── routes_investigate.py    ← NEW (copy in)
│   └── agent/                   ← NEW (copy the whole folder)
│       ├── __init__.py
│       ├── config.py            env + graceful-degradation flags
│       ├── schemas.py           all data contracts
│       ├── http.py              httpx-or-stdlib HTTP layer
│       ├── llm.py               Gemini → Groq → OpenRouter failover
│       ├── transcribe.py        Groq Whisper ASR
│       ├── credibility.py       domain → trust tier table
│       ├── extract.py           transcript → structured claims
│       ├── harvest.py           multi-source parallel research
│       ├── fusion.py            deterministic trust scoring
│       ├── report.py            Markdown report renderer
│       ├── detector.py          adapter to YOUR VoiceGuard model
│       └── sources/
│           ├── googlegrounding.py  Grounded Google Search (primary)
│           ├── websearch.py     DuckDuckGo   (keyless fallback)
│           ├── news.py          GDELT        (no key)
│           ├── wikipedia.py     Wikipedia    (no key)
│           └── factcheck.py     Google Fact Check Tools
├── tests/test_offline.py        87 assertions, runs with no network
└── .env.example
```

**No new dependencies.** The agent uses only the standard library plus pydantic,
which FastAPI already installs. `httpx` is used if present, otherwise it falls
back to `urllib`. Nothing to `pip install`.

### Step 2. Register the router in `app/main.py`

```python
from app.routes_investigate import router as investigate_router
app.include_router(investigate_router)
```

### Step 3. Set keys

```bash
cp .env.example .env
# paste your Gemini key and set a strong AGENT_API_KEY
```

**One free Google AI Studio key is enough**: Gemini handles claim extraction and
stance analysis, runs real Google searches server-side (Search grounding, the
primary web source), and transcribes audio natively. Adding a Groq key upgrades
transcription to purpose-built Whisper and gives LLM failover. Get the Fact
Check Tools key too if you have five spare minutes — enable the API on the same
Google Cloud project and reuse the Gemini key.

### Step 4. Wire the detector (the one manual step)

`app/agent/detector.py` probes for your detection function automatically, trying
`app.core.model.detect`, `.detect_audio`, `.predict`, `.analyze` and a few class
forms. **Check whether the probe found yours:**

```bash
uvicorn app.main:app --reload
curl -s localhost:8000/agent/health | python -m json.tool
```

Then run a real investigation and look at the `detect_audio` trace row. If it
says *"VoiceGuard detector not reachable"*, open `detector.py` and replace the
probing block in `detect()` with a direct call:

```python
from app.core.model import your_actual_function
raw = your_actual_function(audio_b64)          # or whatever its signature is
coerced = _coerce(raw)                          # normalises dict/object output
if coerced:
    return coerced, "detected via app.core.model.your_actual_function"
```

`_coerce()` already handles `classification`/`label`/`prediction` keys,
confidences as either `0.97` or `97`, and dict, pydantic, or plain-object
returns. **The pipeline still works if you skip this** — it reports
`degraded: true` for the acoustic stage and scores on web evidence alone. Do not
let this block your demo.

### Step 5. Verify

```bash
python tests/test_offline.py     # expect: 87 passed, 0 failed
```

---

## Part 2 — Demo it

**Text-only** (fastest path, no audio or ASR needed — good for a first smoke test):

```bash
curl -X POST localhost:8000/investigate/report \
  -H 'Content-Type: application/json' \
  -d '{"transcriptOverride":"This is the Reserve Bank of India. All UPI transactions above 5000 rupees are banned from Monday. Withdraw your cash immediately.","maxResearchRounds":2}'
```

Returns a rendered Markdown forensic report.

**Full pipeline** with audio:

```bash
python -c "import base64;print(base64.b64encode(open('sample.mp3','rb').read()).decode())" > b64.txt
curl -X POST localhost:8000/investigate \
  -H 'Content-Type: application/json' \
  -d "{\"audioBase64\":\"$(cat b64.txt)\",\"audioFormat\":\"mp3\"}"
```

Endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /detect` | your original endpoint, unchanged |
| `POST /investigate` | full pipeline, structured JSON |
| `POST /investigate/report` | full pipeline, Markdown report |
| `GET /agent/health` | which capabilities are live |

---

## Part 3 — What to say about it

### The pitch, in one sentence

> VoiceGuard used to answer "is this audio fake?". It now answers the question
> that actually matters: **"should I trust this, and why?"** — by transcribing the
> audio, extracting the factual claims inside it, researching each claim across
> fact-checkers, global news and reference sources, and fusing everything into an
> auditable trust score.

### Problem-statement alignment

| Requirement | How it is met |
|---|---|
| Collect unstructured public web data | Grounded Google Search (official Gemini API) + DuckDuckGo fallback, GDELT (~100k outlets, 65 languages), Wikipedia, Google ClaimReview |
| Transform into structured, actionable context | Raw search results/JSON → typed `Claim`, `Evidence`, `ClaimAssessment` records with provenance, credibility tier and stance |
| AI / autonomous agents | 6-node graph; the agent reads its own round-1 results, judges coverage weak, reformulates queries and searches again |
| Real-world problem | Deepfake-driven misinformation — audio fraud and election disinformation |
| Research | Multi-source parallel harvest across 4 tool families |
| Decision support | 0–100 trust score, risk band, explicit recommendation |
| Recommendations | Concrete action per risk band, from "do not share" to "no concerns found" |
| Workflow automation | One API call replaces a manual OSINT workflow |
| Domain-specific assistance | Credibility table tuned for India (PIB Fact Check, Alt News, BOOM, PTI) |

### Architecture

```
                    POST /investigate
                           │
        ┌──────────────────┴──────────────────┐
        │            STAGE 1 (parallel)        │
        │  ┌────────────────┐ ┌──────────────┐ │
        │  │ VoiceGuard     │ │ Whisper ASR  │ │
        │  │ Wav2Vec2 +     │ │ (Groq)       │ │
        │  │ forensics      │ │              │ │
        │  └───────┬────────┘ └──────┬───────┘ │
        └──────────┼─────────────────┼─────────┘
                   │                 ▼
                   │      ┌────────────────────────┐
                   │      │ extract_claims (LLM)   │
                   │      │ → Claim[] + entities   │
                   │      │   + search queries     │
                   │      └──────────┬─────────────┘
                   │                 ▼
                   │      ┌────────────────────────────────┐
                   │      │ harvest_web  (round 1)         │
                   │      │  Google Search grounding ║     │
                   │      │  GDELT ║ FactCheck ║ Wikipedia │
                   │      │  (DDG fallback)   (parallel)   │
                   │      └──────────┬─────────────────────┘
                   │                 ▼
                   │        ╱ coverage weak? ╲──yes──┐
                   │        ╲               ╱        │
                   │              │ no        reformulate
                   │              │           queries,
                   │              │           round 2 ──┐
                   │              ▼                      │
                   │      ┌────────────────────────┐    │
                   │      │ assess_claims (LLM)    │◀───┘
                   │      │ stance per source      │
                   │      └──────────┬─────────────┘
                   ▼                 ▼
        ┌──────────────────────────────────────┐
        │ fuse_verdict — deterministic scoring │
        │ trust 0-100 + risk band + factors[]  │
        └──────────────────┬───────────────────┘
                           ▼
                  Investigation Report
                  (JSON or Markdown)
```

### Core design decisions & architectural rationale

**The trust score is arithmetic, not an LLM opinion.** Starts at 50 (no
information) and moves by fixed weights: up to −35 for synthetic audio, up to
−40 for refuted claims, −20 for fact-checker "false" ratings, −10 for
social-only circulation. Every point is itemised in `verdict.factors`, so an
evaluator can recompute the total by hand. LLMs are used only for what they are
good at — reading a snippet and judging stance. A test asserts
`score == 50 + sum(deltas)`.

**A fact-checker's explicit "false" overrides the LLM.** If Google ClaimReview
returns a `false` rating from an IFCN-signatory domain, the claim is marked
refuted regardless of what the stance model said. Human expert review outranks
model inference.

**Absence of evidence is never treated as truth.** If nothing is retrieved, the
score stays near neutral and a caveat says so explicitly. A misinformation tool
that scores unknown content as trustworthy is worse than no tool.

**Everything degrades instead of failing.** No LLM key → heuristic claim
extraction. No Groq key → Gemini transcribes the audio natively. Grounding quota
exhausted → DuckDuckGo fallback. Sources rate-limited → skipped and caveated.
Detector unreachable → web-only verdict. Verified by running the whole graph
with the network dead: 87/87 assertions pass and a full report still renders.

**The Google stack is load-bearing, not decorative.** One AI Studio key powers
four distinct capabilities: Gemini reasoning, server-side Google Search with
citation metadata (grounding), native audio transcription, and the ClaimReview
fact-check database.

### Known limitations & engineering trade-offs

1. **No reverse-audio search exists publicly.** There is no "Google Images for
   sound", so the agent cannot find where an audio *file* first appeared. It
   verifies the *claims inside* the audio instead. `earliest_reference` is the
   earliest publication date among retrieved articles, and the report says so.
2. **Speaker identity comes from transcript wording only** — "I am X", or a
   narrator naming them. There is no voice biometric matching, so this is a
   hypothesis, not an identification. Flagged as a caveat in every report.
3. **Grounded search quota.** Grounded Google Search has a free daily allowance.
   When exhausted, the agent auto-falls back to DuckDuckGo scraping. Empty
   results are treated as degradation, not truth: the primary web source is an
   official Google API, with a keyless scraper as the safety net.
4. **X/Twitter is not searched** — the API is paid. Reddit and YouTube surface
   indirectly through web search instead.
5. **English-centric.** Whisper transcribes ~99 languages, but the credibility
   table and fact-check queries are tuned for English and Indian sources.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `detect_audio` shows degraded | probe missed your function | wire `detector.py` directly (Step 4) |
| `transcribe` degraded | no Groq key and Gemini ASR failed | check `/agent/health`; or pass `transcriptOverride` |
| Sources all from `duckduckgo` | grounding failed or quota out | check Gemini key; grounding resumes next day |
| 0 sources retrieved | all web sources down/rate-limited | wait 60s; GDELT usually still responds |
| No fact-check evidence | key missing or API not enabled | enable Fact Check Tools API on the Gemini project |
| Claims look crude | LLM key missing → heuristics | check `/agent/health` shows a provider |
| `extract_claims` degraded | LLM refused or rate-limited | trace `detail` names the provider error |
