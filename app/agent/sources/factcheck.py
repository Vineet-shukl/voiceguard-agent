"""
Google Fact Check Tools API — the highest-value source in this pipeline.

It returns structured ClaimReview markup: the claim text, who reviewed it, and
a textual rating ("False", "Misleading", "Pants on Fire"). That is already the
"structured, actionable context" the problem statement asks for, so we normalise
its verdicts into a small controlled vocabulary the fusion engine can score.

Free, and enabled on the same Google Cloud project as a Gemini key.
"""

from __future__ import annotations

from typing import Any

from ..config import get_config
from ..credibility import classify, domain_of, weight
from ..http import get_json
from ..schemas import Evidence

FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# Textual ratings are free-form per publisher; map them to a usable signal.
_FALSE_MARKERS = (
    "false", "incorrect", "untrue", "fake", "misleading", "misinformation",
    "pants on fire", "no evidence", "unfounded", "debunked", "hoax",
    "manipulated", "altered", "fabricated", "not true", "baseless",
    "distorts", "misrepresent", "deceptive", "scam", "doctored",
)
_TRUE_MARKERS = ("true", "correct", "accurate", "confirmed", "mostly true", "verified")
_MIXED_MARKERS = ("partly", "partially", "mixture", "half", "mixed", "exaggerated", "lacks context")


def normalise_rating(rating: str) -> str:
    """-> 'false' | 'true' | 'mixed' | 'unclear'"""
    r = (rating or "").strip().lower()
    if not r:
        return "unclear"
    for m in _MIXED_MARKERS:
        if m in r:
            return "mixed"
    for m in _FALSE_MARKERS:
        if m in r:
            return "false"
    for m in _TRUE_MARKERS:
        if m in r:
            return "true"
    return "unclear"


async def search(query: str, *, limit: int | None = None) -> list[Evidence]:
    cfg = get_config()
    if not cfg.factcheck_api_key:
        return []
    cap = limit or cfg.max_results_per_query

    data: Any = await get_json(
        FACTCHECK_URL,
        params={
            "query": query,
            "pageSize": max(cap, 5),
            "languageCode": "en",
        },
        headers={"X-Goog-Api-Key": cfg.factcheck_api_key},
    )
    if not isinstance(data, dict):
        return []

    out: list[Evidence] = []
    for claim in (data.get("claims") or []):
        claim_text = (claim.get("text") or "").strip()
        claimant = (claim.get("claimant") or "").strip()
        for review in (claim.get("claimReview") or [])[:2]:
            url = review.get("url") or ""
            if not url:
                continue
            publisher = ((review.get("publisher") or {}).get("name") or "").strip()
            rating = (review.get("textualRating") or "").strip()
            tier = classify(url, hint="factcheck")
            snippet = f"Claim: {claim_text[:220]}"
            if claimant:
                snippet += f" | Claimant: {claimant}"
            snippet += f" | Rating: {rating or 'unrated'}"

            out.append(
                Evidence(
                    url=url,
                    title=(review.get("title") or f"{publisher} fact check").strip()[:300],
                    snippet=snippet[:600],
                    domain=domain_of(url),
                    kind="factcheck",
                    source_tool="google_factcheck",
                    published=review.get("reviewDate") or claim.get("claimDate"),
                    credibility=tier,
                    credibility_weight=weight(tier),
                    query=query,
                    factcheck_verdict=normalise_rating(rating),
                    factcheck_publisher=publisher or None,
                )
            )
            if len(out) >= cap:
                return out
    return out
