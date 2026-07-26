"""
FastAPI routes for the investigation agent.

Wire into your existing app/main.py with two lines:

    from app.routes_investigate import router as investigate_router
    app.include_router(investigate_router)

Your existing /detect endpoint is untouched.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from .agent import get_config, investigate, render_markdown
from .agent.schemas import InvestigateRequest

router = APIRouter(tags=["investigation"])


@router.get("/agent/health")
async def agent_health() -> dict[str, object]:
    """Shows which capabilities are live given current env configuration."""
    cfg = get_config(refresh=True)
    return {
        "status": "ok",
        "capabilities": cfg.describe(),
        "degradation_note": (
            "Missing keys reduce capability but never break the pipeline: "
            "no LLM falls back to heuristic extraction, no ASR key requires "
            "transcriptOverride, blocked sources are skipped."
        ),
    }


@router.post("/investigate")
async def investigate_endpoint(req: InvestigateRequest) -> dict[str, object]:
    """Full pipeline: detect -> transcribe -> extract -> research -> verdict."""
    if not req.audioBase64 and not req.transcriptOverride:
        raise HTTPException(
            status_code=422,
            detail="Provide audioBase64, or transcriptOverride for a text-only investigation.",
        )
    try:
        report = await investigate(req)
    except Exception as exc:  # never leak a stack trace to the client
        raise HTTPException(status_code=500, detail=f"investigation failed: {exc}") from exc
    return report.model_dump(mode="json")


@router.post("/investigate/report", response_class=PlainTextResponse)
async def investigate_markdown(req: InvestigateRequest) -> str:
    """Same pipeline, rendered as a Markdown forensic report."""
    if not req.audioBase64 and not req.transcriptOverride:
        raise HTTPException(
            status_code=422,
            detail="Provide audioBase64, or transcriptOverride for a text-only investigation.",
        )
    try:
        report = await investigate(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"investigation failed: {exc}") from exc
    return render_markdown(report)
