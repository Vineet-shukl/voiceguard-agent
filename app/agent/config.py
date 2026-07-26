"""Runtime configuration. Everything degrades gracefully when keys are absent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.getenv(n)
        if v:
            return v.strip()
    return default


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class AgentConfig:
    # ---- LLM providers (either is sufficient; both = automatic failover) ----
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", default="gemini-2.0-flash"))

    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    groq_model: str = field(
        default_factory=lambda: _env("GROQ_MODEL", default="llama-3.3-70b-versatile")
    )
    groq_whisper_model: str = field(
        default_factory=lambda: _env("GROQ_WHISPER_MODEL", default="whisper-large-v3-turbo")
    )

    openrouter_api_key: str = field(default_factory=lambda: _env("OPENROUTER_API_KEY"))
    openrouter_model: str = field(
        default_factory=lambda: _env(
            "OPENROUTER_MODEL", default="meta-llama/llama-3.3-70b-instruct:free"
        )
    )

    # ---- Web sources ----
    # Google Fact Check Tools API. Free; enable it on the same Google Cloud
    # project as your Gemini key. Falls back to plain web search if unset.
    factcheck_api_key: str = field(
        default_factory=lambda: _env("FACTCHECK_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY")
    )

    enable_web_search: bool = field(default_factory=lambda: _flag("ENABLE_WEB_SEARCH", True))
    enable_news: bool = field(default_factory=lambda: _flag("ENABLE_NEWS", True))
    enable_wikipedia: bool = field(default_factory=lambda: _flag("ENABLE_WIKIPEDIA", True))
    enable_factcheck: bool = field(default_factory=lambda: _flag("ENABLE_FACTCHECK", True))
    # Grounded Google Search through the Gemini API — primary web source when a
    # Gemini key exists; DuckDuckGo remains the keyless fallback.
    enable_grounding: bool = field(default_factory=lambda: _flag("ENABLE_GOOGLE_GROUNDING", True))

    # ---- Budgets: keep a demo inside a few seconds ----
    http_timeout: float = field(default_factory=lambda: float(_env("HTTP_TIMEOUT", default="12")))
    llm_timeout: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT", default="45")))
    max_claims: int = field(default_factory=lambda: int(_env("MAX_CLAIMS", default="4")))
    max_queries_per_round: int = field(
        default_factory=lambda: int(_env("MAX_QUERIES_PER_ROUND", default="6"))
    )
    max_results_per_query: int = field(
        default_factory=lambda: int(_env("MAX_RESULTS_PER_QUERY", default="6"))
    )
    max_evidence: int = field(default_factory=lambda: int(_env("MAX_EVIDENCE", default="40")))

    user_agent: str = (
        "VoiceGuardInvestigationAgent/1.0 (academic project; +https://example.edu)"
    )

    # ---- Derived ----
    @property
    def has_llm(self) -> bool:
        return bool(self.gemini_api_key or self.groq_api_key or self.openrouter_api_key)

    @property
    def has_asr(self) -> bool:
        # Groq Whisper is preferred; Gemini native audio is the fallback path.
        return bool(self.groq_api_key or self.gemini_api_key)

    @property
    def grounding_active(self) -> bool:
        return bool(self.gemini_api_key) and self.enable_grounding

    def provider_order(self) -> list[str]:
        """Cheapest/most generous free tier first."""
        order: list[str] = []
        if self.gemini_api_key:
            order.append("gemini")
        if self.groq_api_key:
            order.append("groq")
        if self.openrouter_api_key:
            order.append("openrouter")
        return order

    def describe(self) -> dict[str, object]:
        asr: str
        if self.groq_api_key:
            asr = self.groq_whisper_model
        elif self.gemini_api_key:
            asr = f"gemini-native ({self.gemini_model})"
        else:
            asr = "unavailable"
        return {
            "llm_providers": self.provider_order() or ["none (heuristic fallback)"],
            "asr": asr,
            "factcheck_api": bool(self.factcheck_api_key) and self.enable_factcheck,
            "google_search_grounding": self.grounding_active,
            "sources": [
                name
                for name, on in [
                    ("google_grounding", self.grounding_active),
                    ("web_search_ddg", self.enable_web_search),
                    ("news_gdelt", self.enable_news),
                    ("wikipedia", self.enable_wikipedia),
                    ("factcheck", self.enable_factcheck),
                ]
                if on
            ],
        }


_config: AgentConfig | None = None


def get_config(refresh: bool = False) -> AgentConfig:
    global _config
    if _config is None or refresh:
        _config = AgentConfig()
    return _config
