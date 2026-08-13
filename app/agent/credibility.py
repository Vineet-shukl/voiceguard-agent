"""
Domain -> credibility tier mapping.

This is the component that turns a bag of URLs into weighted evidence. Tiers
are deliberately conservative and auditable: a fixed table beats asking an LLM
"is this source trustworthy", which is unstable and unciteable.

India-relevant fact-checkers and wires are included because deepfake audio
targeting Indian elections and public figures is a live problem domain.
"""

from __future__ import annotations

import urllib.parse

from .schemas import CREDIBILITY_WEIGHT, CredibilityTier

# IFCN signatories and major fact-check desks
FACT_CHECKERS = {
    "factcheck.org", "politifact.com", "snopes.com", "fullfact.org",
    "factcheck.afp.com", "reuters.com/fact-check", "apnews.com/hub/ap-fact-check",
    "altnews.in", "boomlive.in", "factchecker.in", "vishvasnews.com",
    "factly.in", "thequint.com/news/webqoof", "newschecker.in", "digiteye.in",
    "pib.gov.in", "factcheck.pib.gov.in", "leadstories.com", "checkyourfact.com",
    "truthorfiction.com", "washingtonpost.com/news/fact-checker",
    "fatabyyano.net", "maldita.es", "newtral.es", "correctiv.org",
    "mimikama.at", "faktisk.no", "tjekdet.dk", "logically.ai",
}

# Wire services and major outlets of record
WIRE_OR_MAJOR = {
    "reuters.com", "apnews.com", "afp.com", "bbc.com", "bbc.co.uk",
    "ptinews.com", "aninews.in", "thehindu.com", "indianexpress.com",
    "hindustantimes.com", "timesofindia.indiatimes.com", "ndtv.com",
    "livemint.com", "business-standard.com", "economictimes.indiatimes.com",
    "theprint.in", "scroll.in", "thewire.in", "deccanherald.com",
    "nytimes.com", "washingtonpost.com", "wsj.com", "theguardian.com",
    "ft.com", "economist.com", "bloomberg.com", "cnn.com", "nbcnews.com",
    "cbsnews.com", "abcnews.go.com", "npr.org", "aljazeera.com",
    "dw.com", "france24.com", "cnbc.com", "forbes.com", "time.com",
    "nature.com", "science.org", "scientificamerican.com",
}

REFERENCE = {
    "wikipedia.org", "wikidata.org", "britannica.com", "who.int", "un.org",
    "europa.eu", "nih.gov", "cdc.gov", "nasa.gov", "noaa.gov",
    "arxiv.org", "doi.org", "pubmed.ncbi.nlm.nih.gov", "eci.gov.in",
    "rbi.org.in", "sebi.gov.in", "supremecourt.gov.in",
}

USER_GENERATED = {
    "reddit.com", "x.com", "twitter.com", "youtube.com", "youtu.be",
    "facebook.com", "instagram.com", "tiktok.com", "threads.net",
    "quora.com", "medium.com", "substack.com", "t.me", "telegram.me",
    "whatsapp.com", "4chan.org", "rumble.com", "bitchute.com",
    "linkedin.com", "tumblr.com", "vk.com", "discord.com",
}

PLATFORM_LABELS = {
    "x.com": "X (Twitter)", "twitter.com": "X (Twitter)",
    "reddit.com": "Reddit", "youtube.com": "YouTube", "youtu.be": "YouTube",
    "facebook.com": "Facebook", "instagram.com": "Instagram",
    "tiktok.com": "TikTok", "t.me": "Telegram", "telegram.me": "Telegram",
    "threads.net": "Threads", "whatsapp.com": "WhatsApp",
    "rumble.com": "Rumble", "bitchute.com": "BitChute",
    "discord.com": "Discord", "linkedin.com": "LinkedIn",
    "quora.com": "Quora", "medium.com": "Medium",
    "substack.com": "Substack", "tumblr.com": "Tumblr",
    "vk.com": "VKontakte", "4chan.org": "4chan",
}


def domain_of(url: str) -> str:
    try:
        netloc = urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return ""
    netloc = netloc.split("@")[-1].split(":")[0]
    return netloc[4:] if netloc.startswith("www.") else netloc


def _registrable(domain: str) -> str:
    """Crude eTLD+1. Handles common two-part suffixes like co.uk / co.in."""
    parts = domain.split(".")
    if len(parts) <= 2:
        return domain
    two = {"co", "com", "org", "net", "gov", "ac", "edu", "nic", "gob"}
    if parts[-2] in two and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _in_set(domain: str, url: str, candidates: set[str]) -> bool:
    reg = _registrable(domain)
    lowered = url.lower()
    for cand in candidates:
        if "/" in cand:  # path-scoped entry, e.g. reuters.com/fact-check
            if cand in lowered:
                return True
            continue
        if domain == cand or reg == cand or domain.endswith("." + cand):
            return True
    return False


def classify(url: str, *, hint: str | None = None) -> CredibilityTier:
    """Assign a credibility tier. `hint` lets an API assert fact-checker status."""
    domain = domain_of(url)
    if not domain:
        return CredibilityTier.UNKNOWN

    if hint == "factcheck":
        return CredibilityTier.FACT_CHECKER
    if _in_set(domain, url, FACT_CHECKERS):
        return CredibilityTier.FACT_CHECKER
    if _in_set(domain, url, WIRE_OR_MAJOR):
        return CredibilityTier.WIRE_OR_MAJOR
    if _in_set(domain, url, REFERENCE) or domain.endswith((".gov", ".edu", ".gov.in", ".ac.in")):
        return CredibilityTier.REFERENCE
    if _in_set(domain, url, USER_GENERATED):
        return CredibilityTier.USER_GENERATED

    # A plausible news domain we simply don't have on file
    if any(tok in domain for tok in ("news", "times", "post", "herald", "tribune", "journal")):
        return CredibilityTier.GENERAL_MEDIA
    return CredibilityTier.UNKNOWN


def weight(tier: CredibilityTier) -> float:
    return CREDIBILITY_WEIGHT.get(tier, 0.2)


def platform_label(url: str) -> str | None:
    domain = _registrable(domain_of(url))
    return PLATFORM_LABELS.get(domain)
