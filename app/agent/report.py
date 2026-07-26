"""Render an InvestigationReport as a human-readable Markdown forensic report."""

from __future__ import annotations

from .schemas import CredibilityTier, InvestigationReport, RiskBand

_BADGE = {
    RiskBand.CRITICAL: "CRITICAL RISK",
    RiskBand.HIGH: "HIGH RISK",
    RiskBand.MODERATE: "MODERATE RISK",
    RiskBand.LOW: "LOW RISK",
    RiskBand.MINIMAL: "MINIMAL RISK",
}

_STATUS_LABEL = {
    "refuted": "REFUTED",
    "disputed": "DISPUTED",
    "corroborated": "CORROBORATED",
    "unverified": "UNVERIFIED",
    "no_evidence": "NO EVIDENCE FOUND",
}

_TIER_LABEL = {
    CredibilityTier.FACT_CHECKER: "fact-checker",
    CredibilityTier.WIRE_OR_MAJOR: "wire/major outlet",
    CredibilityTier.REFERENCE: "reference",
    CredibilityTier.GENERAL_MEDIA: "general media",
    CredibilityTier.USER_GENERATED: "user-generated",
    CredibilityTier.UNKNOWN: "unclassified",
}


def _bar(score: int, width: int = 20) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def render_markdown(report: InvestigationReport) -> str:
    v = report.verdict
    d = report.audio_status
    lines: list[str] = []

    lines.append("# Audio Investigation Report")
    lines.append("")
    lines.append(f"**Investigation ID:** `{report.investigation_id}`  ")
    lines.append(f"**Generated:** {report.created_at}  ")
    lines.append(f"**Elapsed:** {report.total_duration_ms:.0f} ms")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- verdict ----
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"### {_BADGE[v.risk_band]} — Trust score {v.trust_score}/100")
    lines.append("")
    lines.append(f"`{_bar(v.trust_score)}`")
    lines.append("")
    lines.append(f"**Summary:** {v.headline}")
    lines.append("")
    lines.append(f"**Recommendation:** {v.recommendation}")
    lines.append("")

    # ---- acoustic ----
    lines.append("## 1. Acoustic analysis (VoiceGuard)")
    lines.append("")
    if d.degraded:
        lines.append("Detector did not run for this request. " + d.explanation)
    else:
        lines.append(f"- **Classification:** {d.classification.value}")
        lines.append(f"- **Synthetic probability:** {d.synthetic_probability:.1%}")
        lines.append(f"- **Confidence:** {d.confidence:.1%}")
        if d.analyzers_agree is not None:
            lines.append(
                f"- **Neural/forensic agreement:** {'yes' if d.analyzers_agree else 'no — confidence reduced'}"
            )
        if d.inference_time_ms:
            lines.append(f"- **Inference time:** {d.inference_time_ms:.0f} ms")
        if d.explanation:
            lines.append("")
            lines.append(f"> {d.explanation}")
    lines.append("")

    # ---- transcript ----
    lines.append("## 2. Transcript")
    lines.append("")
    if report.transcript.text:
        meta: list[str] = [f"engine: {report.transcript.engine}"]
        if report.transcript.language:
            meta.append(f"language: {report.transcript.language}")
        if report.transcript.duration_seconds:
            meta.append(f"duration: {report.transcript.duration_seconds:.1f}s")
        lines.append(f"*({', '.join(meta)})*")
        lines.append("")
        text = report.transcript.text
        lines.append("> " + (text[:1200] + ("..." if len(text) > 1200 else "")).replace("\n", "\n> "))
    else:
        lines.append(f"Not available — {report.transcript.note}")
    lines.append("")

    # ---- extracted context ----
    lines.append("## 3. Structured context extracted")
    lines.append("")
    ex = report.extraction
    if ex.summary:
        lines.append(f"**What the audio asserts:** {ex.summary}")
        lines.append("")
    if ex.speaker_hypotheses:
        lines.append("**Speaker hypotheses** (from transcript wording only — not voice biometrics):")
        lines.append("")
        for s in ex.speaker_hypotheses:
            lines.append(f"- {s.name} — {s.basis} (confidence {s.confidence:.0%})")
        lines.append("")
    if ex.topics:
        lines.append(f"**Topics:** {', '.join(ex.topics)}")
        lines.append("")
    if ex.claims:
        lines.append(f"**Checkable claims identified:** {len(ex.claims)}")
        lines.append("")
        for c in ex.claims:
            lines.append(f"- `{c.id}` ({c.checkability} checkability) {c.text}")
        lines.append("")

    # ---- claim by claim ----
    lines.append("## 4. Claim verification")
    lines.append("")
    if v.claim_assessments:
        for a in v.claim_assessments:
            lines.append(f"### `{a.claim_id}` — {_STATUS_LABEL.get(a.status, a.status.upper())}")
            lines.append("")
            lines.append(f"*Claim:* {a.claim_text}")
            lines.append("")
            if a.rationale:
                lines.append(f"*Assessment:* {a.rationale}")
                lines.append("")
            if a.factcheck_verdicts:
                lines.append("*Fact-checker ratings:*")
                lines.append("")
                for fv in a.factcheck_verdicts:
                    lines.append(f"- {fv}")
                lines.append("")
            if a.refuting_urls:
                lines.append("*Refuting sources:*")
                lines.append("")
                for u in a.refuting_urls:
                    lines.append(f"- {u}")
                lines.append("")
            if a.supporting_urls:
                lines.append("*Supporting sources:*")
                lines.append("")
                for u in a.supporting_urls:
                    lines.append(f"- {u}")
                lines.append("")
    else:
        lines.append("No claims were assessed.")
        lines.append("")

    # ---- provenance ----
    lines.append("## 5. Circulation and provenance")
    lines.append("")
    lines.append(f"- **Platforms observed:** {', '.join(report.platforms_seen) or 'none identified'}")
    lines.append(f"- **Earliest dated reference:** {report.earliest_reference or 'unknown'}")
    lines.append(f"- **Sources retrieved:** {report.evidence_count}")
    lines.append(f"- **Search rounds:** {report.harvest.rounds}")
    lines.append(f"- **Tools used:** {', '.join(report.harvest.tools_used) or 'none'}")
    lines.append("")
    lines.append(
        "> Note: earliest dated reference reflects publication dates of retrieved articles, "
        "not the origin of the audio file itself. No public reverse-audio-search index exists, "
        "so true first-upload provenance cannot be established by this method."
    )
    lines.append("")

    # ---- score breakdown ----
    lines.append("## 6. How the trust score was computed")
    lines.append("")
    lines.append("Starting from a neutral 50/100:")
    lines.append("")
    lines.append("| Factor | Detail | Δ |")
    lines.append("|---|---|---|")
    for f in v.factors:
        sign = f"+{f.delta:g}" if f.delta > 0 else f"{f.delta:g}"
        detail = f.detail.replace("|", "\\|")[:120]
        lines.append(f"| {f.name} | {detail} | {sign} |")
    lines.append(f"| **Final** | | **{v.trust_score}/100** |")
    lines.append("")

    if v.caveats:
        lines.append("## 7. Caveats and limitations")
        lines.append("")
        for c in v.caveats:
            lines.append(f"- {c}")
        lines.append("")

    # ---- evidence table ----
    if report.harvest.evidence:
        lines.append("## 8. Evidence set")
        lines.append("")
        lines.append("| # | Source | Tier | Kind | Title |")
        lines.append("|---|---|---|---|---|")
        for i, e in enumerate(report.harvest.evidence[:25], start=1):
            title = (e.title or "(untitled)").replace("|", "\\|")[:70]
            lines.append(
                f"| {i} | [{e.domain}]({e.url}) | {_TIER_LABEL.get(e.credibility, '?')} | {e.kind} | {title} |"
            )
        lines.append("")

    # ---- trace ----
    if report.trace:
        lines.append("## 9. Execution trace")
        lines.append("")
        lines.append("| Node | Status | Duration | Detail |")
        lines.append("|---|---|---|---|")
        for s in report.trace:
            detail = s.detail.replace("|", "\\|")[:90]
            lines.append(f"| `{s.node}` | {s.status} | {s.duration_ms:.0f} ms | {detail} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "*Generated by the VoiceGuard Investigation Agent. This is decision-support output, "
        "not a legal or forensic determination. Verify against primary sources before acting.*"
    )
    return "\n".join(lines)
