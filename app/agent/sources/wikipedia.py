"""
Wikipedia lookup — used for speaker/entity grounding rather than claim checking.

When the audio names a public figure, this establishes who they are, which in
turn tells the agent whether a synthetic clip of them is high-impact (a head of
state) or low-impact (an unknown private individual). That distinction drives
the risk band in the final verdict.
"""

from __future__ import annotations

from typing import Any

from ..credibility import classify, weight
from ..http import get_json
from ..schemas import Evidence

SEARCH_URL = "https://en.wikipedia.org/w/api.php"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

# Signals that a named entity is a high-profile public figure
_HIGH_PROFILE = (
    "president", "prime minister", "minister", "senator", "governor",
    "chief executive", "ceo", "chairman", "politician", "leader",
    "monarch", "king", "queen", "chancellor", "mayor", "judge", "justice",
    "billionaire", "actor", "actress", "singer", "celebrity", "cricketer",
    "footballer", "commissioner", "secretary", "diplomat", "spokesperson",
)


async def lookup(name: str) -> tuple[Evidence | None, dict[str, Any]]:
    """Return (evidence, profile) where profile carries prominence signals."""
    profile: dict[str, Any] = {"name": name, "found": False, "high_profile": False}
    if not name or len(name) < 3:
        return None, profile

    data: Any = await get_json(
        SEARCH_URL,
        params={
            "action": "query",
            "list": "search",
            "srsearch": name,
            "srlimit": 1,
            "format": "json",
            "origin": "*",
        },
    )
    hits = []
    if isinstance(data, dict):
        hits = ((data.get("query") or {}).get("search") or [])
    if not hits:
        return None, profile

    title = hits[0].get("title") or name
    import urllib.parse

    summary: Any = await get_json(SUMMARY_URL.format(title=urllib.parse.quote(title, safe="")))
    if not isinstance(summary, dict):
        return None, profile

    extract = (summary.get("extract") or "").strip()
    url = ((summary.get("content_urls") or {}).get("desktop") or {}).get("page") or (
        f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"
    )
    description = (summary.get("description") or "").strip()

    blob = f"{description} {extract}".lower()
    profile.update(
        {
            "found": True,
            "title": title,
            "description": description,
            "high_profile": any(k in blob for k in _HIGH_PROFILE),
            "is_person": summary.get("type") == "standard",
        }
    )

    tier = classify(url)
    ev = Evidence(
        url=url,
        title=f"{title} — Wikipedia",
        snippet=(description + ". " + extract)[:600] if description else extract[:600],
        domain="en.wikipedia.org",
        kind="reference",
        source_tool="wikipedia",
        credibility=tier,
        credibility_weight=weight(tier),
        query=name,
    )
    return ev, profile
