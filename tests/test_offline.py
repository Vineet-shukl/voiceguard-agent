"""
Offline end-to-end verification of the investigation pipeline.

Runs the full agent graph with network and LLM calls mocked, proving:
  1. every node executes and the graph wires together
  2. heuristic fallbacks work with zero API keys
  3. the fusion arithmetic is correct and auditable
  4. a report renders

Run:  python3 tests/test_offline.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
try:
    import pydantic  # noqa: F401
except ImportError:
    import _pydantic_shim  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent import credibility, extract, fusion, harvest, report  # noqa: E402
from app.agent.config import get_config  # noqa: E402
from app.agent.graph import investigate  # noqa: E402
from app.agent.llm import extract_json  # noqa: E402
from app.agent.schemas import (  # noqa: E402
    Claim,
    Classification,
    CredibilityTier,
    DetectionResult,
    Evidence,
    ExtractionResult,
    HarvestResult,
    InvestigateRequest,
    RiskBand,
)
from app.agent.sources import factcheck  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + str(info)) if info and not cond else ''}")


TRANSCRIPT = (
    "This is Rajesh Kumar speaking. I am announcing that the Reserve Bank of India "
    "has banned all UPI transactions above 5000 rupees starting next Monday. "
    "The government confirmed this decision yesterday and 40 percent of digital "
    "payments will be affected. Please withdraw your cash immediately."
)


def test_json_salvage():
    print("\n[1] LLM JSON extraction")
    check("plain object", extract_json('{"a": 1}') == {"a": 1})
    check("fenced json", extract_json('```json\n{"a": 2}\n```') == {"a": 2})
    check("prose-wrapped", extract_json('Sure! {"a": 3} hope that helps') == {"a": 3})
    check("brace in string", extract_json('{"a": "}"}') == {"a": "}"})
    check("array", extract_json("[1, 2]") == [1, 2])
    check("garbage -> None", extract_json("no json at all") is None)


def test_credibility():
    print("\n[2] Credibility tiering")
    cases = [
        ("https://www.altnews.in/story", CredibilityTier.FACT_CHECKER),
        ("https://factcheck.pib.gov.in/x", CredibilityTier.FACT_CHECKER),
        ("https://www.reuters.com/world/x", CredibilityTier.WIRE_OR_MAJOR),
        ("https://timesofindia.indiatimes.com/x", CredibilityTier.WIRE_OR_MAJOR),
        ("https://en.wikipedia.org/wiki/X", CredibilityTier.REFERENCE),
        ("https://rbi.org.in/notice", CredibilityTier.REFERENCE),
        ("https://www.reddit.com/r/x", CredibilityTier.USER_GENERATED),
        ("https://x.com/user/status/1", CredibilityTier.USER_GENERATED),
        ("https://randomblog.xyz/post", CredibilityTier.UNKNOWN),
    ]
    for url, expected in cases:
        got = credibility.classify(url)
        check(f"{url[:42]:<42} -> {expected.value}", got == expected, f"got {got}")
    check("platform label X", credibility.platform_label("https://x.com/a/status/1") == "X (Twitter)")
    check("subdomain match", credibility.classify("https://edition.cnn.com/x") == CredibilityTier.WIRE_OR_MAJOR)


def test_factcheck_ratings():
    print("\n[3] Fact-check rating normalisation")
    for raw, want in [
        ("False", "false"), ("Pants on Fire", "false"), ("Misleading", "false"),
        ("Fabricated", "false"), ("True", "true"), ("Mostly True", "true"),
        ("Partly false", "mixed"), ("Lacks context", "mixed"), ("", "unclear"),
    ]:
        got = factcheck.normalise_rating(raw)
        check(f"{raw or '(empty)':<16} -> {want}", got == want, f"got {got}")


def test_google_grounding_parser():
    print("\n[3b] Google Search grounding parser")
    from app.agent.sources.googlegrounding import parse_grounding

    # Shape mirrors the real generateContent response with groundingMetadata
    fake = {
        "candidates": [{
            "content": {"parts": [{"text": "The RBI has issued no such ban. PIB Fact Check rated the claim false."}]},
            "groundingMetadata": {
                "groundingChunks": [
                    {"web": {"uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC123",
                             "title": "pib.gov.in"}},
                    {"web": {"uri": "https://www.reuters.com/world/india/rbi-statement",
                             "title": "RBI dismisses viral UPI ban claim"}},
                ],
                "groundingSupports": [
                    {"segment": {"text": "The RBI has issued no such ban."},
                     "groundingChunkIndices": [0, 1]},
                    {"segment": {"text": "PIB Fact Check rated the claim false."},
                     "groundingChunkIndices": [0]},
                ],
            },
        }]
    }
    ev = parse_grounding(fake, "RBI UPI ban")
    check("two evidence records", len(ev) == 2, len(ev))
    check("redirect URI -> domain from title", ev[0].domain == "pib.gov.in", ev[0].domain)
    check("redirect classified as fact-checker",
          ev[0].credibility == CredibilityTier.FACT_CHECKER, ev[0].credibility)
    check("direct URI classified wire", ev[1].credibility == CredibilityTier.WIRE_OR_MAJOR,
          ev[1].credibility)
    check("segments become snippet", "no such ban" in ev[0].snippet, ev[0].snippet[:60])
    check("tool tagged", ev[0].source_tool == "google_grounding")
    check("empty response safe", parse_grounding({}, "q") == [])
    check("malformed response safe", parse_grounding({"candidates": [{}]}, "q") == [])


def test_heuristic_extraction():
    print("\n[4] Heuristic claim extraction (no LLM key)")
    res = extract.heuristic_extract(TRANSCRIPT)
    check("claims found", len(res.claims) > 0, f"{len(res.claims)}")
    check("queries generated", all(c.search_queries for c in res.claims))
    check("marked degraded", res.degraded)
    names = [s.name for s in res.speaker_hypotheses]
    check("self-ID speaker found", any("Rajesh" in n for n in names), names)
    check("entities extracted", any(c.entities for c in res.claims))
    print(f"       e.g. claim: {res.claims[0].text[:70]}...")
    print(f"       e.g. query: {res.claims[0].search_queries[0][:60]}")
    check("empty input safe", extract.heuristic_extract("").claims == [])

    # Regression: speaker runs must not cross sentence boundaries, must survive
    # lowercase prefixes, must handle initials, and must not fire on non-names.
    for text, want in [
        ("This is Rajesh Kumar speaking. The RBI banned UPI above 5000 rupees.", ["Rajesh Kumar"]),
        ("I am Narendra Modi and the government confirmed 40 percent growth.", ["Narendra Modi"]),
        ("my name is Sarah Chen. Police arrested 200 people according to reports.", ["Sarah Chen"]),
        ("This is J. K. Rowling here. The court approved the 5 million settlement.", ["J. K. Rowling"]),
        ("The RBI announced a policy affecting 40 percent of users.", []),
        ("I'm speaking about the election. Officials confirmed 300 deaths today.", []),
    ]:
        got = [s.name for s in extract.heuristic_extract(text).speaker_hypotheses]
        check(f"speaker {want or '(none)'} from {text[:34]!r}", got == want, f"got {got}")


def test_dedupe_and_rank():
    print("\n[5] Evidence dedupe and ranking")
    ev = [
        Evidence(url="https://a.com/x?utm_source=tw", domain="a.com", kind="web"),
        Evidence(url="https://a.com/x", domain="a.com", kind="web"),
        Evidence(url="https://altnews.in/y", domain="altnews.in", kind="factcheck",
                 credibility=CredibilityTier.FACT_CHECKER, credibility_weight=1.0),
        Evidence(url="https://reddit.com/z", domain="reddit.com", kind="web",
                 credibility=CredibilityTier.USER_GENERATED, credibility_weight=0.25),
    ]
    deduped = harvest._dedupe(ev)
    check("tracking params deduped", len(deduped) == 3, f"{len(deduped)}")
    ranked = harvest._rank(deduped)
    check("fact-check ranked first", ranked[0].kind == "factcheck", ranked[0].domain)
    check("weak coverage detected", harvest._coverage_is_weak(deduped[:1]))
    check("strong coverage ok", not harvest._coverage_is_weak([
        Evidence(url=f"https://reuters.com/{i}", credibility=CredibilityTier.WIRE_OR_MAJOR,
                 credibility_weight=.85) for i in range(4)]))


def test_fusion_scenarios():
    print("\n[6] Fusion arithmetic (deterministic, auditable)")
    claim = Claim(id="c1", text="RBI banned UPI above 5000 rupees", search_queries=["q"])
    ex = ExtractionResult(claims=[claim])

    # Worst case: synthetic audio + fact-checker says false
    det = DetectionResult(classification=Classification.AI_GENERATED,
                          synthetic_probability=0.97, confidence=0.97, analyzers_agree=True)
    hv = HarvestResult(evidence=[
        Evidence(url="https://altnews.in/x", kind="factcheck", domain="altnews.in",
                 credibility=CredibilityTier.FACT_CHECKER, credibility_weight=1.0,
                 factcheck_verdict="false", factcheck_publisher="Alt News",
                 matched_claim_id="c1"),
    ])
    asmt = fusion._heuristic_stance(claim, hv.evidence)
    check("fact-check false -> refuted", asmt.status == "refuted", asmt.status)
    v = fusion.fuse(det, ex, hv, [asmt])
    check(f"worst case critical (score {v.trust_score})",
          v.risk_band == RiskBand.CRITICAL, v.risk_band)
    check("factors explain score", len(v.factors) >= 3, len(v.factors))
    total = 50 + sum(f.delta for f in v.factors)
    check("score == 50 + sum(deltas)", abs(max(0, min(100, total)) - v.trust_score) <= 1,
          f"{total} vs {v.trust_score}")

    # Best case: human audio, corroborated
    det2 = DetectionResult(classification=Classification.HUMAN, synthetic_probability=0.04,
                           confidence=0.95, analyzers_agree=True)
    hv2 = HarvestResult(evidence=[
        Evidence(url="https://reuters.com/a", kind="news", domain="reuters.com",
                 title="RBI confirms new limit", snippet="according to the central bank",
                 credibility=CredibilityTier.WIRE_OR_MAJOR, credibility_weight=0.85,
                 matched_claim_id="c1"),
    ])
    a2 = fusion._heuristic_stance(claim, hv2.evidence)
    v2 = fusion.fuse(det2, ex, hv2, [a2])
    check(f"best case low risk (score {v2.trust_score})",
          v2.trust_score > v.trust_score and v2.risk_band in {RiskBand.LOW, RiskBand.MINIMAL, RiskBand.MODERATE},
          v2.risk_band)

    # Degraded: nothing available
    v3 = fusion.fuse(DetectionResult(degraded=True), ExtractionResult(), HarvestResult(), [])
    check(f"no data -> moderate/neutral (score {v3.trust_score})", v3.risk_band == RiskBand.MODERATE)
    check("degradation caveated", len(v3.caveats) >= 2, v3.caveats)

    # Disagreeing analyzers reduce acoustic weight
    det4 = DetectionResult(classification=Classification.AI_GENERATED, synthetic_probability=0.9,
                           confidence=0.6, analyzers_agree=False)
    v4 = fusion.fuse(det4, ex, HarvestResult(), [])
    v5 = fusion.fuse(DetectionResult(classification=Classification.AI_GENERATED,
                                     synthetic_probability=0.9, confidence=0.9,
                                     analyzers_agree=True), ex, HarvestResult(), [])
    check("disagreement softens penalty", v4.trust_score > v5.trust_score,
          f"{v4.trust_score} vs {v5.trust_score}")
    check("score always 0-100", all(0 <= x.trust_score <= 100 for x in (v, v2, v3, v4, v5)))


def test_full_graph_offline():
    print("\n[7] Full graph, network mocked, zero API keys")
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "FACTCHECK_API_KEY"):
        os.environ.pop(k, None)
    get_config(refresh=True)

    import app.agent.http as http_mod

    async def dead_get_json(*a, **k):
        return None

    async def dead_get_text(*a, **k):
        return None

    async def dead_post_json(*a, **k):
        return 0, {"_transport_error": "network mocked out"}

    http_mod.get_json = dead_get_json
    http_mod.get_text = dead_get_text
    http_mod.post_json = dead_post_json
    for mod in ("websearch", "news", "factcheck", "wikipedia", "googlegrounding"):
        m = sys.modules[f"app.agent.sources.{mod}"]
        if hasattr(m, "get_json"):
            m.get_json = dead_get_json
        if hasattr(m, "get_text"):
            m.get_text = dead_get_text
        if hasattr(m, "post_json"):
            m.post_json = dead_post_json

    req = InvestigateRequest(transcriptOverride=TRANSCRIPT, maxResearchRounds=2)
    rep = asyncio.run(investigate(req))

    check("report produced", rep is not None)
    check("investigation id set", rep.investigation_id.startswith("inv_"))
    nodes = [t.node for t in rep.trace]
    for expected in ["detect_audio", "transcribe", "extract_claims", "harvest_web",
                     "assess_claims", "fuse_verdict"]:
        check(f"node ran: {expected}", expected in nodes, nodes)
    check("claims extracted offline", len(rep.extraction.claims) > 0)
    check("queries were attempted", len(rep.harvest.queries_run) > 0, rep.harvest.queries_run)
    check("no evidence (network dead)", rep.evidence_count == 0)
    check("verdict still produced", 0 <= rep.verdict.trust_score <= 100, rep.verdict.trust_score)
    check("caveats explain missing evidence", any("evidence" in c.lower() for c in rep.verdict.caveats),
          rep.verdict.caveats)
    check("recommendation present", bool(rep.verdict.recommendation))
    check("detector degraded gracefully", rep.audio_status.degraded)
    check("json serialisable", isinstance(rep.model_dump(mode="json"), dict))

    md = report.render_markdown(rep)
    check("markdown renders", len(md) > 800, f"{len(md)} chars")
    for section in ["# Audio Investigation Report", "## Verdict", "## 4. Claim verification",
                    "## 6. How the trust score was computed", "## 9. Execution trace"]:
        check(f"report section: {section[:38]}", section in md)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_report.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"       wrote {out}")
    print(f"       trust score: {rep.verdict.trust_score}/100 ({rep.verdict.risk_band.value})")
    print(f"       elapsed: {rep.total_duration_ms:.0f} ms")


def test_bad_input():
    print("\n[8] Malformed input safety")
    from app.agent.transcribe import decode_audio
    check("data URL decoded", decode_audio("data:audio/wav;base64,QUJD") == b"ABC")
    check("unpadded decoded", decode_audio("QUJD") == b"ABC")
    check("whitespace tolerated", decode_audio("QU\nJD ") == b"ABC")
    check("empty -> None", decode_audio("") is None)
    rep = asyncio.run(investigate(InvestigateRequest(transcriptOverride="hi")))
    check("trivial transcript safe", rep.verdict.trust_score >= 0)
    rep2 = asyncio.run(investigate(InvestigateRequest(audioBase64="!!!not base64!!!")))
    check("garbage audio safe", rep2.audio_status.degraded)


if __name__ == "__main__":
    print("=" * 74)
    print("VoiceGuard Investigation Agent — offline verification")
    print("=" * 74)
    test_json_salvage()
    test_credibility()
    test_factcheck_ratings()
    test_google_grounding_parser()
    test_heuristic_extraction()
    test_dedupe_and_rank()
    test_fusion_scenarios()
    test_full_graph_offline()
    test_bad_input()
    print("\n" + "=" * 74)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILURES:")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
