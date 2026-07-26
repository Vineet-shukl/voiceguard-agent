"""
Claim extraction: transcript -> structured, searchable claims.

This is the hinge of the whole project. Audio is unstructured; a claim with
entities and pre-built search queries is structured context. Everything
downstream operates on these objects.

An LLM does the extraction. A deterministic heuristic covers the no-key /
rate-limited case so the pipeline always produces something searchable.
"""

from __future__ import annotations

import re

from .config import get_config
from .llm import complete_json
from .schemas import Claim, ExtractionResult, SpeakerHypothesis

SYSTEM = """You are a forensic misinformation analyst. You extract verifiable factual \
claims from a transcript of speech that may be an AI-generated deepfake.

Rules:
- Extract only assertions that could be checked against public sources.
- Ignore opinions, greetings, filler, and vague sentiment.
- Restate each claim as a standalone sentence with no pronouns, so it can be \
searched on its own.
- For each claim, write 2 concise web search queries a journalist would actually type. \
No quotes, no boolean operators, under 12 words each.
- Identify the speaker ONLY if the transcript names them or they self-identify. \
Never guess from voice or style; you cannot hear the audio.
- Output valid JSON only.

Schema:
{
  "language": "English",
  "summary": "one sentence on what the audio asserts",
  "topics": ["short topic labels"],
  "speaker_hypotheses": [
    {"name": "Full Name", "basis": "self-identification|named by narrator|addressed by name",
     "confidence": 0.0-1.0}
  ],
  "claims": [
    {"text": "standalone claim", "checkability": "high|medium|low",
     "entities": ["people, orgs, places, dates"],
     "search_queries": ["query one", "query two"]}
  ]
}"""

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "this", "that", "these",
    "those", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "shall", "should", "can",
    "could", "may", "might", "must", "i", "you", "he", "she", "it", "we",
    "they", "my", "your", "his", "her", "its", "our", "their", "me", "him",
    "them", "us", "of", "in", "on", "at", "to", "for", "with", "from", "by",
    "about", "as", "so", "very", "just", "now", "today", "here", "there",
    "what", "when", "where", "who", "why", "how", "all", "not", "no", "yes",
}

_FACTUAL_HINTS = (
    "percent", "%", "million", "billion", "crore", "lakh", "thousand",
    "announced", "confirmed", "said", "stated", "reported", "declared",
    "will", "banned", "arrested", "resigned", "died", "killed", "won",
    "lost", "signed", "approved", "rejected", "launched", "increase",
    "decrease", "rose", "fell", "study", "研究", "according to", "government",
    "court", "police", "election", "vote", "rupees", "dollars", "deaths",
    "cases", "vaccine", "law", "bill", "act", "treaty", "deal",
)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 25]


def _keywords(sentence: str, limit: int = 8) -> list[str]:
    words = re.findall(r"[A-Za-z0-9%₹$][A-Za-z0-9'%₹$.-]*", sentence)
    out: list[str] = []
    for w in words:
        if w.lower() in _STOPWORDS or len(w) < 2:
            continue
        out.append(w)
        if len(out) >= limit:
            break
    return out


def _proper_nouns(text: str) -> list[str]:
    """Capitalised runs not at sentence start — crude but dependency-free NER."""
    found: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        tokens = sent.split()
        for i, tok in enumerate(tokens):
            clean = tok.strip(".,!?;:\"'()")
            if i == 0 or not clean or not clean[0].isupper():
                continue
            if clean.lower() in _STOPWORDS:
                continue
            # extend the run
            run = [clean]
            j = i + 1
            while j < len(tokens):
                nxt = tokens[j].strip(".,!?;:\"'()")
                if nxt and nxt[0].isupper() and nxt.lower() not in _STOPWORDS:
                    run.append(nxt)
                    j += 1
                else:
                    break
            phrase = " ".join(run)
            if 3 <= len(phrase) <= 60 and phrase not in found:
                found.append(phrase)
    return found[:10]


def heuristic_extract(text: str) -> ExtractionResult:
    """Deterministic fallback. Ranks sentences by factual-signal density."""
    if not text.strip():
        return ExtractionResult(degraded=True, summary="No transcript available.")

    cfg = get_config()
    scored: list[tuple[float, str]] = []
    for sent in _sentences(text):
        low = sent.lower()
        score = sum(1.5 for h in _FACTUAL_HINTS if h in low)
        score += 1.0 * len(re.findall(r"\b\d[\d,.]*\b", sent))
        score += 0.5 * len(_proper_nouns(sent))
        if score > 0:
            scored.append((score, sent))
    scored.sort(key=lambda x: -x[0])

    claims: list[Claim] = []
    for idx, (_, sent) in enumerate(scored[: cfg.max_claims], start=1):
        kws = _keywords(sent)
        base = " ".join(kws[:6])
        claims.append(
            Claim(
                id=f"c{idx}",
                text=sent[:400],
                checkability="medium",
                entities=_proper_nouns(sent),
                search_queries=[q for q in [base, f"{base} fact check"] if q.strip()],
            )
        )

    # Self-identification: "I am X" / "This is X speaking" / "My name is X".
    # The prefix is matched case-insensitively via a scoped inline flag; a global
    # re.IGNORECASE would also neuter the [A-Z] name test and match any word.
    speakers: list[SpeakerHypothesis] = []
    for m in re.finditer(
        r"\b(?i:i am|i'm|this is|my name is|you're listening to|speaking to you is)\s+"
        # A name token is either an initial ("J.") or a capitalised word ("Smith").
        # Crucially the word form excludes '.', so a run cannot cross a sentence
        # boundary: "my name is Sarah Chen. Police..." stops at "Chen".
        r"((?:[A-Z]\.|[A-Z][A-Za-z'-]+)(?:\s+(?:[A-Z]\.|[A-Z][A-Za-z'-]+)){0,3})",
        text,
    ):
        name = re.sub(r"\s+(?:speaking|here|again)$", "", m.group(1).strip(), flags=re.IGNORECASE)
        if len(name) < 3 or name.lower() in _STOPWORDS:
            continue
        if not any(s.name == name for s in speakers):
            speakers.append(
                SpeakerHypothesis(
                    name=name, basis="self-identification in transcript", confidence=0.45
                )
            )

    return ExtractionResult(
        claims=claims,
        speaker_hypotheses=speakers[:2],
        topics=_proper_nouns(text)[:5],
        summary=(text[:200] + "...") if len(text) > 200 else text,
        degraded=True,
    )


async def extract(transcript_text: str) -> tuple[ExtractionResult, str]:
    """Returns (result, note). Falls back to heuristics on any LLM failure."""
    cfg = get_config()
    text = (transcript_text or "").strip()
    if not text:
        return ExtractionResult(degraded=True, summary="No transcript available."), "empty transcript"

    if not cfg.has_llm:
        return heuristic_extract(text), "no LLM key — heuristic extraction"

    payload = text[:8000]
    parsed, res = await complete_json(SYSTEM, f"Transcript:\n\"\"\"\n{payload}\n\"\"\"")
    if not isinstance(parsed, dict):
        return heuristic_extract(text), f"LLM extraction failed ({res.error}) — heuristic fallback"

    claims: list[Claim] = []
    for idx, raw in enumerate((parsed.get("claims") or [])[: cfg.max_claims], start=1):
        if not isinstance(raw, dict):
            continue
        ctext = str(raw.get("text") or "").strip()
        if not ctext:
            continue
        queries = [
            str(q).strip()
            for q in (raw.get("search_queries") or [])
            if str(q).strip()
        ][:2]
        if not queries:
            queries = [" ".join(_keywords(ctext)[:6])]
        checkability = str(raw.get("checkability") or "medium").lower()
        if checkability not in {"high", "medium", "low"}:
            checkability = "medium"
        claims.append(
            Claim(
                id=f"c{idx}",
                text=ctext[:400],
                checkability=checkability,  # type: ignore[arg-type]
                entities=[str(e).strip() for e in (raw.get("entities") or []) if str(e).strip()][:8],
                search_queries=queries,
            )
        )

    speakers: list[SpeakerHypothesis] = []
    for raw in (parsed.get("speaker_hypotheses") or [])[:3]:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        try:
            conf = float(raw.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        speakers.append(
            SpeakerHypothesis(
                name=name[:120],
                basis=str(raw.get("basis") or "unspecified")[:200],
                confidence=min(max(conf, 0.0), 1.0),
            )
        )

    result = ExtractionResult(
        claims=claims,
        speaker_hypotheses=speakers,
        topics=[str(t).strip() for t in (parsed.get("topics") or []) if str(t).strip()][:6],
        language=str(parsed.get("language") or "").strip() or None,
        summary=str(parsed.get("summary") or "").strip()[:500],
        degraded=False,
    )
    if not result.claims:
        fallback = heuristic_extract(text)
        fallback.summary = result.summary or fallback.summary
        fallback.speaker_hypotheses = result.speaker_hypotheses or fallback.speaker_hypotheses
        return fallback, f"LLM found no checkable claims (via {res.provider}) — heuristic fallback"

    return result, f"extracted via {res.provider}"
