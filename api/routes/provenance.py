"""Provenance routes for HelixForge API."""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status

from models.schemas import ErrorResponse, ProvenanceReport, ProvenanceTrace

router = APIRouter()


def get_state() -> Dict[str, Any]:
    """Get application state."""
    from api.server import get_app_state
    return get_app_state()


@router.get(
    "/{dataset_id}/{field}",
    response_model=ProvenanceTrace,
    responses={404: {"model": ErrorResponse}}
)
async def get_field_provenance(dataset_id: str, field: str):
    """Get provenance trace for a specific field.

    Traces the field back to its original source(s) through
    all transformations.
    """
    state = get_state()
    provenance = state.get("provenance")

    if not provenance:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Provenance agent not available"
        )

    trace = provenance.query_lineage(dataset_id, field)

    if trace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No provenance found for {dataset_id}.{field}"
        )

    return trace


@router.get(
    "/{dataset_id}/report",
    response_model=ProvenanceReport,
    responses={404: {"model": ErrorResponse}}
)
async def get_provenance_report(dataset_id: str):
    """Get complete provenance report for a dataset.

    Includes all field lineages and transformation history.
    """
    state = get_state()
    provenance = state.get("provenance")

    if not provenance:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Provenance agent not available"
        )

    # Check if dataset exists (either raw or fused)
    exists = (
        dataset_id in state.get("datasets", {}) or
        dataset_id in state.get("fused", {})
    )

    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {dataset_id}"
        )

    try:
        report = provenance.process(dataset_id)
        return report
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get(
    "/{dataset_id}/graph",
    responses={404: {"model": ErrorResponse}}
)
async def get_lineage_graph(dataset_id: str):
    """Get lineage graph structure for visualization.

    Returns nodes and edges representing the data lineage.
    """
    state = get_state()
    provenance = state.get("provenance")

    if not provenance:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Provenance agent not available"
        )

    try:
        graph = provenance.build_lineage_graph(dataset_id)
        return {
            "dataset_id": dataset_id,
            "graph": graph
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/")
async def list_tracked_datasets():
    """List all datasets with provenance tracking."""
    state = get_state()
    provenance = state.get("provenance")

    if not provenance:
        return {"count": 0, "datasets": []}

    # Get unique dataset IDs from traces
    dataset_ids = set()
    for trace in provenance._traces.values():
        dataset_ids.add(trace.fused_dataset_id)

    return {
        "count": len(dataset_ids),
        "datasets": list(dataset_ids)
    }
