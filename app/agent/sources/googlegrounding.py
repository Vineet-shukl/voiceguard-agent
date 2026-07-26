"""
Grounded Google Search via the Gemini API — the primary web-data collector
when a Gemini key is present.

One call does search + read + synthesise: Gemini executes real Google Search
queries server-side, reads the results, and returns an answer with citations
(`groundingMetadata`). Each citation chunk becomes an Evidence record, and the
answer segments that cite it become its snippet — which gives the stance
assessor much richer text than a scraped result snippet.

Two parsing subtleties, learned from the API's actual shape:
  * Citation URIs are Google redirect links (vertexaisearch.cloud.google.com/
    grounding-api-redirect/...). The real source domain arrives in the chunk
    *title*, so credibility classification uses that when the URI is a redirect.
  * Gemini 2.0+ uses the `google_search` tool; older models used
    `google_search_retrieval`. On a 400 we retry once with the legacy name so a
    GEMINI_MODEL override to 1.5 still works.

Free-tier note: grounded queries have a daily free allowance (a few hundred per
day at the time of writing) — comfortably enough for a demo. On failure or an
exhausted quota the harvester falls back to DuckDuckGo automatically.
"""

from __future__ import annotations

from typing import Any

from ..config import get_config
from ..credibility import classify, domain_of, weight
from ..http import post_json
from ..schemas import Evidence

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def parse_grounding(data: Any, query: str) -> list[Evidence]:
    """Pure function so tests can verify parsing without network."""
    if not isinstance(data, dict):
        return []
    candidates = data.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        return []
    cand = candidates[0]
    meta = cand.get("groundingMetadata") or {}
    chunks = meta.get("groundingChunks") or []
    supports = meta.get("groundingSupports") or []

    try:
        answer = "".join(p.get("text", "") for p in cand["content"]["parts"])
    except Exception:
        answer = ""

    # chunk index -> the answer segments that cite it
    seg_by_chunk: dict[int, list[str]] = {}
    for sup in supports:
        if not isinstance(sup, dict):
            continue
        seg = ((sup.get("segment") or {}).get("text") or "").strip()
        if not seg:
            continue
        for idx in sup.get("groundingChunkIndices") or []:
            try:
                seg_by_chunk.setdefault(int(idx), []).append(seg)
            except (TypeError, ValueError):
                continue

    out: list[Evidence] = []
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        web = chunk.get("web") or {}
        uri = web.get("uri") or ""
        title = (web.get("title") or "").strip()
        if not uri:
            continue

        # Redirect URIs hide the real domain; Gemini puts the domain in title.
        if "grounding-api-redirect" in uri and title and "." in title and " " not in title:
            domain = title.lower()
            cred_url = f"https://{domain}/"
        else:
            domain = domain_of(uri)
            cred_url = uri

        tier = classify(cred_url)
        snippet = " … ".join(seg_by_chunk.get(i, [])[:3]) or answer[:300]
        out.append(
            Evidence(
                url=uri,
                title=title or domain,
                snippet=snippet[:600],
                domain=domain,
                kind="web",
                source_tool="google_grounding",
                credibility=tier,
                credibility_weight=weight(tier),
                query=query,
            )
        )
    return out


def _payload(query: str, legacy: bool) -> dict[str, Any]:
    tool: dict[str, Any] = (
        {"google_search_retrieval": {}} if legacy else {"google_search": {}}
    )
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Search the web and report factually what credible "
                            f"sources say about: {query}"
                        )
                    }
                ],
            }
        ],
        "tools": [tool],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1024},
    }


async def search(query: str, *, limit: int | None = None) -> list[Evidence]:
    cfg = get_config()
    if not cfg.gemini_api_key or not cfg.enable_grounding:
        return []
    cap = limit or cfg.max_results_per_query
    url = URL.format(model=cfg.gemini_model)
    headers = {"x-goog-api-key": cfg.gemini_api_key}

    status, data = await post_json(
        url, _payload(query, legacy=False), headers=headers, timeout=cfg.llm_timeout
    )
    if status == 400:  # older model — retry with the legacy tool name
        status, data = await post_json(
            url, _payload(query, legacy=True), headers=headers, timeout=cfg.llm_timeout
        )
    if status != 200:
        return []
    return parse_grounding(data, query)[:cap]
