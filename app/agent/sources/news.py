"""
News retrieval via GDELT DOC 2.0 — keyless, global, and it returns publication
timestamps, which is what lets the agent establish "first seen" for a claim.

GDELT indexes ~100k news outlets in 65 languages with a ~15 minute lag, which
makes it a better fit for this project than NewsAPI's key-gated 100 req/day.
"""

from __future__ import annotations

import re
from typing import Any

from ..config import get_config
from ..credibility import classify, domain_of, weight
from ..http import get_json
from ..schemas import Evidence

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def _iso_from_gdelt(stamp: str | None) -> str | None:
    """GDELT seendate looks like 20260712T143000Z."""
    if not stamp:
        return None
    m = re.match(r"(\d{4})(\d{2})(\d{2})T?(\d{2})?(\d{2})?(\d{2})?", stamp)
    if not m:
        return None
    y, mo, d, hh, mm, ss = (m.group(i) or "00" for i in range(1, 7))
    return f"{y}-{mo}-{d}T{hh}:{mm}:{ss}Z"


async def search(query: str, *, limit: int | None = None, timespan: str = "12m") -> list[Evidence]:
    cfg = get_config()
    cap = limit or cfg.max_results_per_query

    data: Any = await get_json(
        GDELT_URL,
        params={
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": max(cap, 10),
            "sort": "DateDesc",
            "timespan": timespan,
        },
    )
    if not isinstance(data, dict):
        return []

    out: list[Evidence] = []
    for art in (data.get("articles") or [])[:cap]:
        url = art.get("url") or ""
        if not url:
            continue
        tier = classify(url)
        out.append(
            Evidence(
                url=url,
                title=(art.get("title") or "").strip()[:300],
                snippet=(art.get("domain") or "") + " — " + (art.get("sourcecountry") or ""),
                domain=domain_of(url),
                kind="news",
                source_tool="gdelt",
                published=_iso_from_gdelt(art.get("seendate")),
                credibility=tier,
                credibility_weight=weight(tier),
                query=query,
            )
        )
    return out
