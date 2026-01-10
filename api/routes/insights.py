"""Insight routes for HelixForge API."""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status

from models.schemas import ErrorResponse, InsightRequest, InsightResult

router = APIRouter()


def get_state() -> Dict[str, Any]:
    """Get application state."""
    from api.server import get_app_state
    return get_app_state()


@router.post(
    "/generate",
    response_model=InsightResult,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}}
)
async def generate_insights(request: InsightRequest):
    """Generate insights for a fused dataset.

    Performs statistical analysis, correlation detection, outlier
    identification, clustering, and generates visualizations and
    narrative summaries.
    """
    state = get_state()
    insight_agent = state.get("insight")

    if not insight_agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Insight agent not available"
        )

    # Get fused dataset
    if request.fused_dataset_id not in state.get("fused", {}):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fused dataset not found: {request.fused_dataset_id}"
        )

    df = state["fused"][request.fused_dataset_id]["df"]

    try:
        # Generate insights
        result = insight_agent.process(
            fused_dataset_id=request.fused_dataset_id,
            df=df,
            analysis_types=request.analysis_types,
            generate_visualizations=request.generate_visualizations,
            generate_narrative=request.generate_narrative,
            export_formats=request.export_formats
        )

        # Store result
        state.setdefault("insights", {})[result.insight_id] = result

        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get(
    "/{insight_id}",
    response_model=InsightResult,
    responses={404: {"model": ErrorResponse}}
)
async def get_insight_result(insight_id: str):
    """Get insight generation result."""
    state = get_state()

    if insight_id not in state.get("insights", {}):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Insight not found: {insight_id}"
        )

    return state["insights"][insight_id]


@router.get("/")
async def list_insights():
    """List all generated insights."""
    state = get_state()
    insights = state.get("insights", {})

    return {
        "count": len(insights),
        "insights": [
            {
                "insight_id": iid,
                "fused_dataset_id": result.fused_dataset_id,
                "findings_count": len(result.key_findings),
                "generated_at": result.generated_at
            }
            for iid, result in insights.items()
        ]
    }


@router.get(
    "/{insight_id}/findings",
    responses={404: {"model": ErrorResponse}}
)
async def get_insight_findings(insight_id: str):
    """Get key findings from an insight."""
    state = get_state()

    if insight_id not in state.get("insights", {}):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Insight not found: {insight_id}"
        )

    result = state["insights"][insight_id]

    return {
        "insight_id": insight_id,
        "findings": [f.model_dump() for f in result.key_findings]
    }


@router.get(
    "/{insight_id}/narrative",
    responses={404: {"model": ErrorResponse}}
)
async def get_insight_narrative(insight_id: str):
    """Get narrative summary from an insight."""
    state = get_state()

    if insight_id not in state.get("insights", {}):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Insight not found: {insight_id}"
        )

    result = state["insights"][insight_id]

    return {
        "insight_id": insight_id,
        "narrative": result.narrative_summary
    }


@router.get(
    "/{insight_id}/statistics",
    responses={404: {"model": ErrorResponse}}
)
async def get_insight_statistics(insight_id: str):
    """Get statistics from an insight."""
    state = get_state()

    if insight_id not in state.get("insights", {}):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Insight not found: {insight_id}"
        )

    result = state["insights"][insight_id]

    return {
        "insight_id": insight_id,
        "statistics": result.statistics.model_dump()
    }


@router.get(
    "/{insight_id}/correlations",
    responses={404: {"model": ErrorResponse}}
)
async def get_insight_correlations(insight_id: str):
    """Get correlation analysis from an insight."""
    state = get_state()

    if insight_id not in state.get("insights", {}):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Insight not found: {insight_id}"
        )

    result = state["insights"][insight_id]

    if result.correlations is None:
        return {"insight_id": insight_id, "correlations": None}

    return {
        "insight_id": insight_id,
        "correlations": result.correlations.model_dump()
    }
