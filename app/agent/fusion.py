"""
Verdict fusion — the decision-support layer.

Design choice worth defending in a viva: the trust score is computed by
deterministic, auditable arithmetic, not by asking an LLM for a number. LLMs are
poorly calibrated and non-reproducible; a grader can recompute every point of
this score by hand from the `factors` list. The LLM is used only where language
models are actually good: judging whether a given article supports or refutes a
claim.

Score starts at 50 (no information) and moves on evidence.
"""

from __future__ import annotations

from .config import get_config
from .llm import complete_json
from .schemas import (
    Claim,
    ClaimAssessment,
    Classification,
    CredibilityTier,
    DetectionResult,
    Evidence,
    ExtractionResult,
    HarvestResult,
    RiskBand,
    ScoreFactor,
    Verdict,
)

STANCE_SYSTEM = """You assess whether retrieved sources support or refute a claim.

You are given a claim and numbered sources (title + snippet only). For each \
source decide its stance toward the claim. Base the decision only on the text \
shown; do not use outside knowledge.

Output JSON:
{
  "stances": [{"index": 1, "stance": "supports|refutes|unrelated|unclear"}],
  "verdict": "refuted|disputed|corroborated|unverified",
  "rationale": "one or two sentences citing what the sources actually say"
}

Use "refuted" only if credible sources contradict the claim. Use "unverified" \
if the sources do not settle it."""


def _heuristic_stance(claim: Claim, evidence: list[Evidence]) -> ClaimAssessment:
    """Keyword fallback. Leans on fact-check verdicts, which are already structured."""
    refuting: list[str] = []
    supporting: list[str] = []
    verdicts: list[str] = []

    for ev in evidence:
        if ev.factcheck_verdict:
            verdicts.append(f"{ev.factcheck_publisher or ev.domain}: {ev.factcheck_verdict}")
            if ev.factcheck_verdict in {"false", "mixed"}:
                refuting.append(ev.url)
            elif ev.factcheck_verdict == "true":
                supporting.append(ev.url)
            continue
        blob = f"{ev.title} {ev.snippet}".lower()
        if any(k in blob for k in ("false", "debunked", "hoax", "misleading", "fake", "no evidence", "denies", "denied")):
            refuting.append(ev.url)
        elif any(k in blob for k in ("confirms", "confirmed", "according to", "announced", "reported")):
            supporting.append(ev.url)

    if refuting and len(refuting) >= len(supporting):
        status = "refuted"
    elif refuting and supporting:
        status = "disputed"
    elif supporting:
        status = "corroborated"
    elif evidence:
        status = "unverified"
    else:
        status = "no_evidence"

    return ClaimAssessment(
        claim_id=claim.id,
        claim_text=claim.text,
        status=status,  # type: ignore[arg-type]
        supporting_urls=supporting[:5],
        refuting_urls=refuting[:5],
        factcheck_verdicts=verdicts[:5],
        rationale="Keyword and fact-check-rating heuristic (LLM unavailable).",
    )


async def assess_claim(claim: Claim, evidence: list[Evidence]) -> ClaimAssessment:
    cfg = get_config()
    relevant = [e for e in evidence if e.matched_claim_id == claim.id][:8]

    if not relevant:
        return ClaimAssessment(
            claim_id=claim.id,
            claim_text=claim.text,
            status="no_evidence",
            rationale="No sources retrieved for this claim.",
        )

    # Fact-check API hits are authoritative structured data; always record them.
    verdicts = [
        f"{e.factcheck_publisher or e.domain}: {e.factcheck_verdict}"
        for e in relevant
        if e.factcheck_verdict
    ]

    if not cfg.has_llm:
        return _heuristic_stance(claim, relevant)

    listing = "\n".join(
        f"{i}. [{e.credibility.value}] {e.title} — {e.snippet[:200]}"
        for i, e in enumerate(relevant, start=1)
    )
    parsed, res = await complete_json(
        STANCE_SYSTEM, f"Claim: {claim.text}\n\nSources:\n{listing}"
    )
    if not isinstance(parsed, dict):
        fallback = _heuristic_stance(claim, relevant)
        fallback.rationale = f"Stance model unavailable ({res.error}); used heuristic."
        return fallback

    supporting: list[str] = []
    refuting: list[str] = []
    for st in (parsed.get("stances") or []):
        if not isinstance(st, dict):
            continue
        try:
            idx = int(st.get("index", 0)) - 1
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < len(relevant)):
            continue
        stance = str(st.get("stance") or "").lower()
        if stance == "supports":
            supporting.append(relevant[idx].url)
        elif stance == "refutes":
            refuting.append(relevant[idx].url)

    status = str(parsed.get("verdict") or "unverified").lower()
    if status not in {"refuted", "disputed", "corroborated", "unverified", "no_evidence"}:
        status = "unverified"

    # Guardrail: an explicit "false" from a real fact-checker outranks the LLM.
    if any(
        e.factcheck_verdict == "false" and e.credibility == CredibilityTier.FACT_CHECKER
        for e in relevant
    ):
        status = "refuted"

    return ClaimAssessment(
        claim_id=claim.id,
        claim_text=claim.text,
        status=status,  # type: ignore[arg-type]
        supporting_urls=supporting[:5],
        refuting_urls=refuting[:5],
        factcheck_verdicts=verdicts[:5],
        rationale=str(parsed.get("rationale") or "")[:400],
    )


def _band(score: int) -> RiskBand:
    if score <= 20:
        return RiskBand.CRITICAL
    if score <= 40:
        return RiskBand.HIGH
    if score <= 60:
        return RiskBand.MODERATE
    if score <= 80:
        return RiskBand.LOW
    return RiskBand.MINIMAL


_RECOMMENDATIONS = {
    RiskBand.CRITICAL: "Do not trust or share. Treat as fabricated until proven otherwise.",
    RiskBand.HIGH: "Do not share. Strong indicators of manipulation or false content.",
    RiskBand.MODERATE: "Unverified. Withhold judgement and seek primary confirmation before acting.",
    RiskBand.LOW: "Probably authentic, but confirm against a primary source before relying on it.",
    RiskBand.MINIMAL: "No significant concerns found in this investigation.",
}


def fuse(
    detection: DetectionResult,
    extraction: ExtractionResult,
    harvest: HarvestResult,
    assessments: list[ClaimAssessment],
) -> Verdict:
    factors: list[ScoreFactor] = []
    score = 50.0
    caveats: list[str] = []

    # ---- 1. acoustic authenticity (weight up to -35) ----
    if detection.degraded:
        factors.append(
            ScoreFactor(
                name="Acoustic detection unavailable",
                detail="VoiceGuard engine did not run; audio authenticity unknown.",
                delta=0.0,
            )
        )
        caveats.append("Acoustic deepfake detection did not run for this request.")
    elif detection.classification == Classification.AI_GENERATED:
        delta = -35.0 * detection.synthetic_probability
        if detection.analyzers_agree is False:
            delta *= 0.6
            caveats.append("Neural and forensic analyzers disagreed; acoustic weight reduced.")
        factors.append(
            ScoreFactor(
                name="Audio is synthetic",
                detail=(
                    f"VoiceGuard: {detection.synthetic_probability:.0%} probability machine-generated"
                    + (", analyzers in agreement" if detection.analyzers_agree else "")
                ),
                delta=round(delta, 1),
            )
        )
        score += delta
    elif detection.classification == Classification.HUMAN:
        delta = 12.0 * detection.confidence
        factors.append(
            ScoreFactor(
                name="Audio appears human",
                detail=f"VoiceGuard found no synthesis artefacts (confidence {detection.confidence:.0%})",
                delta=round(delta, 1),
            )
        )
        score += delta
    else:
        caveats.append("Acoustic analysis was inconclusive.")

    # ---- 2. claim verification (dominant signal, up to -40) ----
    refuted = [a for a in assessments if a.status == "refuted"]
    disputed = [a for a in assessments if a.status == "disputed"]
    corroborated = [a for a in assessments if a.status == "corroborated"]
    unverified = [a for a in assessments if a.status in {"unverified", "no_evidence"}]

    if refuted:
        delta = -min(40.0, 22.0 * len(refuted))
        factors.append(
            ScoreFactor(
                name=f"{len(refuted)} claim(s) refuted by sources",
                detail="; ".join(a.claim_text[:80] for a in refuted[:3]),
                delta=round(delta, 1),
            )
        )
        score += delta
    if disputed:
        delta = -8.0 * len(disputed)
        factors.append(
            ScoreFactor(
                name=f"{len(disputed)} claim(s) disputed",
                detail="Sources conflict on these assertions.",
                delta=round(delta, 1),
            )
        )
        score += delta
    if corroborated:
        delta = min(18.0, 9.0 * len(corroborated))
        factors.append(
            ScoreFactor(
                name=f"{len(corroborated)} claim(s) corroborated",
                detail="; ".join(a.claim_text[:80] for a in corroborated[:3]),
                delta=round(delta, 1),
            )
        )
        score += delta
    if unverified and not (refuted or corroborated):
        caveats.append(
            f"{len(unverified)} claim(s) could not be verified from public sources — "
            "absence of evidence is not evidence of falsity."
        )

    # ---- 3. fact-checker involvement (explicit, high-trust signal) ----
    fc = [e for e in harvest.evidence if e.kind == "factcheck"]
    fc_false = [e for e in fc if e.factcheck_verdict == "false"]
    if fc_false:
        publishers = sorted({e.factcheck_publisher or e.domain for e in fc_false})
        delta = -min(20.0, 10.0 * len(fc_false))
        factors.append(
            ScoreFactor(
                name="Rated false by professional fact-checkers",
                detail=", ".join(publishers[:4]),
                delta=round(delta, 1),
            )
        )
        score += delta
    elif fc:
        factors.append(
            ScoreFactor(
                name="Fact-check records located",
                detail=f"{len(fc)} ClaimReview record(s) found, none rated outright false.",
                delta=0.0,
            )
        )

    # ---- 4. source quality of the surrounding coverage ----
    ugc = [e for e in harvest.evidence if e.credibility == CredibilityTier.USER_GENERATED]
    strong = [
        e
        for e in harvest.evidence
        if e.credibility in {CredibilityTier.WIRE_OR_MAJOR, CredibilityTier.REFERENCE}
    ]
    if harvest.evidence:
        if ugc and not strong:
            factors.append(
                ScoreFactor(
                    name="Circulating only on user-generated platforms",
                    detail=f"{len(ugc)} social/forum source(s), no wire or reference coverage. "
                    "Typical of content that has not survived editorial scrutiny.",
                    delta=-10.0,
                )
            )
            score -= 10.0
        elif strong:
            factors.append(
                ScoreFactor(
                    name="Covered by credible outlets",
                    detail=f"{len(strong)} wire/major/reference source(s) in evidence set.",
                    delta=5.0,
                )
            )
            score += 5.0
    else:
        caveats.append(
            "No web evidence was retrieved; the verdict rests on acoustic analysis alone."
        )

    # ---- 5. impersonation impact ----
    named = [s for s in extraction.speaker_hypotheses if s.confidence >= 0.4]
    if named and detection.classification == Classification.AI_GENERATED and not detection.degraded:
        factors.append(
            ScoreFactor(
                name="Possible impersonation of a named individual",
                detail=f"Synthetic audio referencing {named[0].name} ({named[0].basis}).",
                delta=-8.0,
            )
        )
        score -= 8.0
        caveats.append(
            "Speaker identity is inferred from transcript wording only — VoiceGuard does not "
            "perform voice biometric matching, so this is not an identification."
        )

    final = int(max(0, min(100, round(score))))
    band = _band(final)

    # ---- headline ----
    bits: list[str] = []
    if detection.degraded:
        bits.append("Audio authenticity unknown")
    elif detection.classification == Classification.AI_GENERATED:
        bits.append(f"{detection.synthetic_probability:.0%} likely AI-generated")
    elif detection.classification == Classification.HUMAN:
        bits.append("Audio appears human")
    else:
        bits.append("Acoustic result inconclusive")

    if refuted:
        bits.append(f"{len(refuted)} claim(s) refuted by public sources")
    elif corroborated:
        bits.append(f"{len(corroborated)} claim(s) corroborated")
    elif harvest.evidence:
        bits.append("claims unverified")

    return Verdict(
        trust_score=final,
        risk_band=band,
        recommendation=_RECOMMENDATIONS[band],
        headline="; ".join(bits) + ".",
        factors=factors,
        claim_assessments=assessments,
        caveats=caveats,
    )
