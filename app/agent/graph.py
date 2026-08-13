"""
The agent graph.

    detect_audio ──┐
                   ├─> extract_claims ─> harvest_web ─> assess_claims ─> fuse ─> report
    transcribe ────┘
      (parallel)                          (1-2 rounds,
                                        agent-decided)

Stage 1 runs the two independent audio operations concurrently. Every node
records a TraceStep, so the final report shows exactly what ran, how long it
took, and where it degraded — which is the difference between a demo you can
defend and a black box.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable
from typing import Any

from .detector import detect
from .extract import extract
from .fusion import assess_claim, fuse
from .harvest import earliest_date, harvest, platforms_in
from .schemas import (
    ClaimAssessment,
    DetectionResult,
    ExtractionResult,
    HarvestResult,
    InvestigateRequest,
    InvestigationReport,
    TraceStep,
    Transcript,
)
from .transcribe import transcribe


class _Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def ms(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 1)


async def _timed(awaitable: Awaitable[Any]) -> tuple[Any, float]:
    timer = _Timer()
    result = await awaitable
    return result, timer.ms()


async def investigate(req: InvestigateRequest) -> InvestigationReport:
    overall = _Timer()
    trace: list[TraceStep] = []
    inv_id = f"inv_{uuid.uuid4().hex[:12]}"

    # ---------------- Stage 1: acoustic detection + ASR, in parallel ----------
    if req.transcriptOverride:
        detection_task = detect(req.audioBase64, req.audioFormat, req.language)
        transcript_task = asyncio.sleep(
            0, result=Transcript(text=req.transcriptOverride.strip(), engine="client-supplied")
        )
    else:
        detection_task = detect(req.audioBase64, req.audioFormat, req.language)
        transcript_task = transcribe(req.audioBase64, req.audioFormat)

    (det_timed, transcript_timed) = await asyncio.gather(
        _timed(detection_task), _timed(transcript_task)
    )
    det_out, detect_duration_ms = det_timed
    transcript, transcript_duration_ms = transcript_timed
    detection, det_note = det_out if isinstance(det_out, tuple) else (det_out, "")

    if not isinstance(detection, DetectionResult):
        detection = DetectionResult(
            synthetic_probability=0.5,
            confidence=0.0,
            degraded=True,
            explanation="detector returned unexpected type",
        )
    if not isinstance(transcript, Transcript):
        transcript = Transcript(available=False, note="transcription returned unexpected type")

    trace.append(
        TraceStep(
            node="detect_audio",
            status="degraded" if detection.degraded else "ok",
            duration_ms=detect_duration_ms,
            detail=det_note or detection.explanation[:160],
            payload={
                "classification": detection.classification.value,
                "synthetic_probability": round(detection.synthetic_probability, 3),
            },
        )
    )
    trace.append(
        TraceStep(
            node="transcribe",
            status="ok" if transcript.available else "degraded",
            duration_ms=transcript_duration_ms,
            detail=transcript.note or f"{len(transcript.text)} chars via {transcript.engine}",
            payload={"language": transcript.language or "unknown"},
        )
    )

    # ---------------- Stage 2: claim extraction ------------------------------
    t = _Timer()
    if transcript.text.strip():
        extraction, ex_note = await extract(transcript.text)
    else:
        extraction, ex_note = (
            ExtractionResult(degraded=True, summary="No transcript to analyse."),
            "skipped (no transcript)",
        )
    trace.append(
        TraceStep(
            node="extract_claims",
            status="degraded" if extraction.degraded else "ok",
            duration_ms=t.ms(),
            detail=ex_note,
            payload={
                "claims": len(extraction.claims),
                "speakers": [s.name for s in extraction.speaker_hypotheses],
            },
        )
    )

    # ---------------- Stage 3: web harvest ----------------------------------
    t = _Timer()
    if extraction.claims or extraction.speaker_hypotheses or extraction.topics:
        harvest_result = await harvest(extraction, max_rounds=req.maxResearchRounds)
    else:
        harvest_result = HarvestResult(errors=["nothing to research"])
    trace.append(
        TraceStep(
            node="harvest_web",
            status="degraded" if harvest_result.errors else "ok",
            duration_ms=t.ms(),
            detail=(
                f"{len(harvest_result.evidence)} sources from "
                f"{len(harvest_result.queries_run)} queries over {harvest_result.rounds} round(s)"
            ),
            payload={
                "tools": harvest_result.tools_used,
                "queries": harvest_result.queries_run[:8],
                "errors": harvest_result.errors[:3],
            },
        )
    )

    # ---------------- Stage 4: per-claim stance assessment ------------------
    t = _Timer()
    assessments: list[ClaimAssessment] = []
    if extraction.claims:
        results = await asyncio.gather(
            *[assess_claim(c, harvest_result.evidence) for c in extraction.claims],
            return_exceptions=True,
        )
        for claim, res in zip(extraction.claims, results):
            if isinstance(res, BaseException):
                assessments.append(
                    ClaimAssessment(
                        claim_id=claim.id,
                        claim_text=claim.text,
                        status="no_evidence",
                        rationale=f"Assessment failed: {res}",
                    )
                )
            else:
                assessments.append(res)
    trace.append(
        TraceStep(
            node="assess_claims",
            status="ok" if assessments else "skipped",
            duration_ms=t.ms(),
            detail=f"{len(assessments)} claim(s) assessed",
            payload={a.claim_id: a.status for a in assessments},
        )
    )

    # ---------------- Stage 5: fusion ---------------------------------------
    t = _Timer()
    verdict = fuse(detection, extraction, harvest_result, assessments)
    trace.append(
        TraceStep(
            node="fuse_verdict",
            status="ok",
            duration_ms=t.ms(),
            detail=f"trust {verdict.trust_score}/100 → {verdict.risk_band.value}",
            payload={"factors": len(verdict.factors)},
        )
    )

    return InvestigationReport(
        investigation_id=inv_id,
        audio_status=detection,
        transcript=transcript,
        extraction=extraction,
        harvest=harvest_result,
        verdict=verdict,
        platforms_seen=platforms_in(harvest_result.evidence),
        earliest_reference=earliest_date(harvest_result.evidence),
        trace=trace if req.includeTrace else [],
        total_duration_ms=overall.ms(),
    )
