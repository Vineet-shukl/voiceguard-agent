---
title: VoiceGuard Investigation Agent
emoji: 🎙️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Deepfake detection + web research -> auditable trust score
---

# 🎙️ VoiceGuard — AI Deepfake Investigation Agent

**From "is this audio fake?" to "should I trust this, and why?"**

VoiceGuard combines a Wav2Vec2-based deepfake audio detector with an autonomous
research agent. It does not stop at classifying a waveform. It transcribes the
speech, extracts the factual claims being made, researches those claims across
fact-checking databases, global news archives and reference sources, and fuses
everything into an auditable trust score with a concrete recommendation.

Detecting a synthetic voice is a classification problem. Deciding whether to
believe what the voice said is an intelligence problem. This project does both.

---

## 🚀 What it does

```
POST /investigate  ──►  Investigation Report
```

| Stage | What happens |
|---|---|
| 1. Acoustic analysis | Wav2Vec2 neural classifier + spectral/temporal/formant/artifact forensics |
| 2. Transcription | Groq Whisper large-v3, or Gemini native audio (runs in parallel with stage 1) |
| 3. Claim extraction | LLM converts speech into typed, searchable claims with entities |
| 4. Web research | **Grounded Google Search (Gemini API)**, GDELT, Wikipedia and Google ClaimReview in parallel |
| 5. Adaptive re-search | Agent evaluates its own coverage and reformulates queries if it is thin |
| 6. Stance assessment | Each retrieved source scored as supporting, refuting or unrelated |
| 7. Verdict fusion | Deterministic 0–100 trust score, risk band, recommendation |

### Sample output

```
CRITICAL RISK — Trust score 8/100
████░░░░░░░░░░░░░░░░

Summary:  97% likely AI-generated; 1 claim refuted by public sources.
Recommendation:  Do not trust or share. Treat as fabricated until proven otherwise.

Claim c1 — REFUTED
  "The Reserve Bank of India has banned UPI transactions above 5000 rupees."
  Fact-checker ratings:  PIB Fact Check: false · Alt News: false
  Refuting sources:      factcheck.pib.gov.in/... · altnews.in/...

Trust score derivation (starts at neutral 50):
  Audio is synthetic                          −34
  1 claim refuted by sources                  −22
  Rated false by professional fact-checkers   −10
  Circulating only on user-generated platforms −10
  ────────────────────────────────────────────────
  Final                                    8 / 100
```

---

## 🎯 Problem-statement alignment

| Requirement | Implementation |
|---|---|
| Collect unstructured public web data | **Grounded Google Search (official Gemini API)** with DuckDuckGo fallback, GDELT (~100k outlets / 65 languages), Wikipedia, Google ClaimReview |
| Transform into structured, actionable context | Typed `Claim`, `Evidence`, `ClaimAssessment` records with provenance, credibility tier, stance |
| Use AI / autonomous agents | 6-node graph with self-evaluated, adaptive multi-round research |
| Solve a real-world problem | Deepfake-driven audio fraud and election disinformation |
| Research | Parallel multi-source harvest with dedupe and credibility ranking |
| Recommendations | Explicit action per risk band |
| Decision support | Auditable 0–100 trust score with itemised factors |
| Workflow automation | One API call replaces a manual OSINT investigation |
| Domain-specific assistance | Credibility table tuned for Indian fact-checkers and wires |

---

## 🛠️ Setup

### Prerequisites
- Python 3.9+, 4 GB RAM (8 GB recommended)
- **One free [Google AI Studio key](https://aistudio.google.com/apikey) runs everything**:
  Gemini reasoning, Google Search grounding, native audio transcription, and the
  Fact Check Tools API (enable it on the same Cloud project)
- Optional: [Groq](https://console.groq.com/keys) key for faster Whisper ASR and LLM failover

**No dependencies beyond your existing FastAPI stack.** The agent uses only the
standard library plus pydantic; `httpx` is used when available and falls back to
`urllib` otherwise.

```bash
cd voiceguard-agent

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # add provider keys and a strong AGENT_API_KEY

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open <http://localhost:8000> for the demo console, `/docs` for the API. POST
endpoints fail closed unless `AGENT_API_KEY` is set. For an intentionally open,
rate-limited demo, set `ALLOW_PUBLIC_DEMO=true` instead.

**Deploy to the cloud in one command** — `python deploy.py` (see
[DEPLOY.md](DEPLOY.md)). The acoustic Wav2Vec2 engine runs as its own deployed
service and is called over HTTP (`app/core/model.py`); everything degrades
gracefully if it is unreachable.

Verify the agent's live capabilities:

```bash
curl localhost:8000/agent/health
```

---

## 📖 API

### `POST /investigate` — full pipeline, structured JSON

```json
{
  "audioBase64": "<base64 audio>",
  "audioFormat": "mp3",
  "language": "English",
  "maxResearchRounds": 2,
  "includeTrace": true
}
```

Response (abridged):

```json
{
  "investigation_id": "inv_a3f8c21b9e04",
  "audio_status": {
    "classification": "AI_GENERATED",
    "synthetic_probability": 0.97,
    "analyzers_agree": true
  },
  "extraction": {
    "claims": [{
      "id": "c1",
      "text": "The RBI has banned UPI transactions above 5000 rupees.",
      "entities": ["Reserve Bank of India", "UPI"],
      "search_queries": ["RBI UPI transaction limit 5000", "RBI UPI ban fact check"]
    }]
  },
  "verdict": {
    "trust_score": 8,
    "risk_band": "CRITICAL",
    "recommendation": "Do not trust or share. Treat as fabricated until proven otherwise.",
    "factors": [
      {"name": "Audio is synthetic", "delta": -34.0},
      {"name": "1 claim refuted by sources", "delta": -22.0}
    ],
    "caveats": ["Speaker identity is inferred from transcript wording only..."]
  },
  "trace": [
    {"node": "harvest_web", "status": "ok", "duration_ms": 2140.5,
     "detail": "17 sources from 8 queries over 2 round(s)"}
  ]
}
```

### `POST /investigate/report` — same pipeline, Markdown report

### `POST /detect` — original acoustic-only endpoint, unchanged

Pass `transcriptOverride` instead of `audioBase64` to investigate a text claim
directly, with no audio and no ASR key.

---

## 🧠 Architecture

```
              ┌─────────────────┐   ┌─────────────────┐
              │ VoiceGuard      │   │  Whisper ASR    │   parallel
              │ Wav2Vec2 +      │   │  (Groq)         │
              │ forensics       │   │                 │
              └────────┬────────┘   └────────┬────────┘
                       │                     ▼
                       │          ┌──────────────────────┐
                       │          │  extract_claims      │
                       │          │  speech → Claim[]    │
                       │          └──────────┬───────────┘
                       │                     ▼
                       │   ┌─────────────────────────────────────┐
                       │   │ harvest_web        (parallel fan-out)│
                       │   │ DuckDuckGo ║ GDELT ║ ClaimReview ║   │
                       │   │ Wikipedia                            │
                       │   └─────────────────┬───────────────────┘
                       │                     ▼
                       │            ╱ coverage weak? ╲──yes──► reformulate,
                       │            ╲                ╱          round 2
                       │                    │ no                  │
                       │                    ▼                     │
                       │          ┌──────────────────────┐        │
                       │          │  assess_claims       │◀───────┘
                       │          │  stance per source   │
                       │          └──────────┬───────────┘
                       ▼                     ▼
              ┌────────────────────────────────────────┐
              │ fuse_verdict — deterministic scoring   │
              └────────────────────┬───────────────────┘
                                   ▼
                         Investigation Report
```

### 1. Acoustic engine
- **Backbone**: fine-tuned Wav2Vec 2.0 (XLSR-53) with attentive statistics pooling
- **Forensics**: spectral smoothness (vocoder artefacts), temporal micro-jitter,
  formant transitions, phase discontinuities
- **Fusion**: agreement between engines boosts confidence; disagreement lowers it
  and reduces the acoustic term's weight in the final score

### 2. Research agent
Claims are extracted with entities and pre-built search queries, then fanned out
across four source families concurrently. The primary web source is **Grounded
Google Search through the Gemini API**: Gemini runs real Google searches
server-side and returns an answer with citation metadata, and each cited source
becomes an Evidence record carrying the answer segments that cite it — far
richer input for stance assessment than a scraped snippet. DuckDuckGo scraping
remains as the keyless fallback and quota safety net. Evidence is deduplicated
on normalised URLs (tracking parameters stripped) and ranked by credibility
tier. The agent inspects its own results: if a claim has fewer than two credible
sources, it reformulates the queries and searches again.

### 3. Credibility model
A fixed, auditable domain table rather than an LLM judgement:

| Tier | Weight | Examples |
|---|---|---|
| Fact-checker | 1.00 | PIB Fact Check, Alt News, BOOM, Snopes, PolitiFact |
| Wire / major outlet | 0.85 | Reuters, AP, PTI, BBC, The Hindu |
| Reference | 0.70 | Wikipedia, `.gov`, `.edu`, RBI, ECI |
| General media | 0.50 | unrecognised news domains |
| User-generated | 0.25 | Reddit, X, YouTube, Telegram |

### 4. Verdict fusion
The score is arithmetic, not an LLM opinion. It starts at 50 (no information)
and moves by fixed, itemised weights — up to −35 for synthetic audio, up to −40
for refuted claims, −20 for fact-checker "false" ratings, −10 for social-only
circulation. Every point appears in `verdict.factors`, so the total can be
recomputed by hand. A test asserts `score == 50 + sum(deltas)`.

An explicit "false" from an IFCN-signatory fact-checker overrides the stance
model: human expert review outranks model inference.

---

## 🛡️ Graceful degradation

Every dependency is optional. Missing capability reduces confidence and is
disclosed in the report; it never crashes the pipeline.

| Missing | Behaviour |
|---|---|
| LLM key | Deterministic heuristic claim extraction |
| Groq key | Gemini native-audio transcription used instead |
| Both ASR paths | Use `transcriptOverride` for text-only investigation |
| Grounding quota exhausted | Automatic fallback to DuckDuckGo scraping |
| Fact Check key | Falls back to plain web search |
| Search rate-limited | Source skipped, caveat added, score stays neutral |
| Detector unreachable | Web-evidence-only verdict, acoustic stage marked degraded |
| No network at all | Full graph still executes and renders a report |

Verified: the complete pipeline runs with the network dead and all keys removed.

---

## 🧪 Tests

```bash
python tests/test_offline.py     # 87 assertions, no network required
pytest                           # your existing acoustic tests
```

Covers JSON salvage from malformed LLM output, credibility tiering, fact-check
rating normalisation, Google grounding response parsing (including redirect-URI
domain recovery), heuristic extraction and speaker-name edge cases, URL dedupe
and ranking, five fusion scenarios with arithmetic verification, full graph
execution with mocked network, and malformed-input safety.

---

## ⚠️ Limitations

1. **No reverse-audio search exists publicly.** There is no "Google Images for
   sound", so the agent cannot trace where an audio *file* first appeared. It
   verifies the *claims inside* the audio instead. `earliest_reference` is the
   earliest publication date among retrieved articles, not the file's origin.
2. **Speaker identity is inferred from transcript wording only**, never from
   voice biometrics. It is a hypothesis, and every report says so.
3. **Grounded search has a daily free quota.** Grounded Google Search queries
   have a free daily allowance (generous for a demo). When exhausted, the agent
   falls back to DuckDuckGo scraping, which is fragile and rate-limited. Empty
   results are treated as degradation, never as evidence of truth.
4. **X/Twitter is not queried** (paid API). Reddit and YouTube surface indirectly.
5. **English-centric** credibility table and query formulation.

This is decision-support output, not a legal or forensic determination.

---

## 🔌 Integration

This service is built to be called from other projects — one base URL covers
the full pipeline *and* proxies the acoustic detector. See
[INTEGRATE.md](INTEGRATE.md) for the REST reference, auth, and ready-made
Python / JavaScript / curl clients. Deployment guide: [DEPLOY.md](DEPLOY.md).

## 📄 License

MIT. See `LICENSE`.
