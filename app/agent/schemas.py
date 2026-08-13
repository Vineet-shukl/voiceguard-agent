"""
Data contracts for the VoiceGuard Investigation Agent.

Every stage of the pipeline emits one of these models. This is the backbone of
the "unstructured web data -> structured, actionable context" claim: raw HTML,
news wire JSON and free-form transcripts all get normalised into `Evidence`
and `Claim` records with explicit provenance and credibility.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Stage 1 — acoustic detection (existing VoiceGuard engine)
# --------------------------------------------------------------------------


class Classification(str, Enum):
    AI_GENERATED = "AI_GENERATED"
    HUMAN = "HUMAN"
    INCONCLUSIVE = "INCONCLUSIVE"


class DetectionResult(BaseModel):
    """Normalised output of the VoiceGuard acoustic detector."""

    classification: Classification = Classification.INCONCLUSIVE
    synthetic_probability: float = Field(
        0.5, ge=0.0, le=1.0, description="P(audio is machine generated)"
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    analyzers_agree: bool | None = None
    explanation: str = ""
    inference_time_ms: float | None = None
    engine: str = "voiceguard"
    degraded: bool = Field(
        False, description="True when the real detector was unavailable"
    )


# --------------------------------------------------------------------------
# Stage 2/3 — transcription and claim extraction
# --------------------------------------------------------------------------


class Transcript(BaseModel):
    text: str = ""
    language: str | None = None
    duration_seconds: float | None = None
    engine: str = "groq-whisper-large-v3"
    available: bool = True
    note: str = ""


class Claim(BaseModel):
    """A single checkable assertion made in the audio."""

    id: str
    text: str = Field(description="Claim restated as a standalone sentence")
    checkability: Literal["high", "medium", "low"] = "medium"
    entities: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)


class SpeakerHypothesis(BaseModel):
    name: str
    basis: str = Field(description="Why the agent believes this, e.g. self-identification")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    speaker_hypotheses: list[SpeakerHypothesis] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    language: str | None = None
    summary: str = ""
    degraded: bool = False


# --------------------------------------------------------------------------
# Stage 4/5 — web harvest and evidence structuring
# --------------------------------------------------------------------------

SourceKind = Literal["web", "news", "factcheck", "reference"]


class CredibilityTier(str, Enum):
    FACT_CHECKER = "fact_checker"      # IFCN signatories, PIB Fact Check
    WIRE_OR_MAJOR = "wire_or_major"    # Reuters, AP, PTI, major broadsheets
    REFERENCE = "reference"            # Wikipedia, gov/edu domains
    GENERAL_MEDIA = "general_media"
    USER_GENERATED = "user_generated"  # Reddit, X, YouTube, forums
    UNKNOWN = "unknown"


CREDIBILITY_WEIGHT: dict[CredibilityTier, float] = {
    CredibilityTier.FACT_CHECKER: 1.00,
    CredibilityTier.WIRE_OR_MAJOR: 0.85,
    CredibilityTier.REFERENCE: 0.70,
    CredibilityTier.GENERAL_MEDIA: 0.50,
    CredibilityTier.USER_GENERATED: 0.25,
    CredibilityTier.UNKNOWN: 0.20,
}


class Evidence(BaseModel):
    """One normalised web artefact. This is the atom of structured context."""

    url: str
    title: str = ""
    snippet: str = ""
    domain: str = ""
    kind: SourceKind = "web"
    source_tool: str = ""
    published: str | None = None
    credibility: CredibilityTier = CredibilityTier.UNKNOWN
    credibility_weight: float = 0.2
    matched_claim_id: str | None = None
    query: str = ""
    # Populated only for fact-check API hits
    factcheck_verdict: str | None = None
    factcheck_publisher: str | None = None


class HarvestResult(BaseModel):
    evidence: list[Evidence] = Field(default_factory=list)
    queries_run: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    rounds: int = 0


# --------------------------------------------------------------------------
# Stage 6 — verdict fusion
# --------------------------------------------------------------------------


class RiskBand(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    MINIMAL = "MINIMAL"


class ClaimAssessment(BaseModel):
    claim_id: str
    claim_text: str
    status: Literal[
        "refuted", "disputed", "unverified", "corroborated", "no_evidence"
    ] = "no_evidence"
    supporting_urls: list[str] = Field(default_factory=list)
    refuting_urls: list[str] = Field(default_factory=list)
    factcheck_verdicts: list[str] = Field(default_factory=list)
    rationale: str = ""


class ScoreFactor(BaseModel):
    """Explainability: every point of the trust score is attributable."""

    name: str
    detail: str
    delta: float = Field(description="Signed contribution to the trust score")


class Verdict(BaseModel):
    trust_score: int = Field(50, ge=0, le=100)
    risk_band: RiskBand = RiskBand.MODERATE
    recommendation: str = ""
    headline: str = ""
    factors: list[ScoreFactor] = Field(default_factory=list)
    claim_assessments: list[ClaimAssessment] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Trace + top-level report
# --------------------------------------------------------------------------


class TraceStep(BaseModel):
    node: str
    status: Literal["ok", "degraded", "skipped", "error"] = "ok"
    duration_ms: float = 0.0
    detail: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class InvestigationReport(BaseModel):
    """The single artefact returned to the caller and rendered as Markdown."""

    investigation_id: str
    created_at: str = Field(default_factory=_utc_now)
    audio_status: DetectionResult
    transcript: Transcript
    extraction: ExtractionResult
    harvest: HarvestResult
    verdict: Verdict
    platforms_seen: list[str] = Field(default_factory=list)
    earliest_reference: str | None = None
    trace: list[TraceStep] = Field(default_factory=list)
    total_duration_ms: float = 0.0

    # ---- convenience views -------------------------------------------------
    @property
    def evidence_count(self) -> int:
        return len(self.harvest.evidence)


MAX_AUDIO_BASE64_CHARS = 34 * 1024 * 1024
MAX_TRANSCRIPT_CHARS = 100_000


class InvestigateRequest(BaseModel):
    audioBase64: str | None = Field(
        None,
        max_length=MAX_AUDIO_BASE64_CHARS,
        description="Base64 audio payload (24 MB decoded maximum)",
    )
    audioFormat: str | None = Field(
        "wav", min_length=1, max_length=10, pattern=r"^[A-Za-z0-9]+$"
    )
    language: str = Field("English", min_length=1, max_length=64)
    transcriptOverride: str | None = Field(
        None,
        max_length=MAX_TRANSCRIPT_CHARS,
        description="Skip ASR and investigate this text directly. Useful for demos "
        "and for text-only claim verification.",
    )
    maxResearchRounds: int = Field(2, ge=0, le=4)
    includeTrace: bool = True
