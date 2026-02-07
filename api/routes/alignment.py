"""Alignment routes for HelixForge API."""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status

from models.schemas import AlignmentRequest, AlignmentResult, ErrorResponse

router = APIRouter()


def get_state() -> Dict[str, Any]:
    """Get application state."""
    from api.server import get_app_state
    return get_app_state()


@router.post(
    "",
    response_model=AlignmentResult,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}}
)
async def align_datasets(request: AlignmentRequest):
    """Align fields across multiple datasets.

    Requires at least 2 dataset IDs. Datasets must have metadata generated.
    """
    state = get_state()
    aligner = state.get("aligner")

    if not aligner:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Alignment agent not available"
        )

    # Validate datasets exist and have metadata
    metadata_list = []
    for dataset_id in request.dataset_ids:
        if dataset_id not in state.get("metadata", {}):
            # Try to generate metadata if dataset exists
            if dataset_id in state.get("datasets", {}):
                interpreter = state.get("interpreter")
                if interpreter:
                    df = state["datasets"][dataset_id]["df"]
                    metadata = interpreter.process(dataset_id, df)
                    state["metadata"][dataset_id] = metadata
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Metadata not available for dataset: {dataset_id}"
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Dataset not found: {dataset_id}"
                )

        metadata_list.append(state["metadata"][dataset_id])

    try:
        # Perform alignment
        result = aligner.process(
            metadata_list,
            confidence_threshold=request.confidence_threshold,
            include_partial=request.include_partial_matches
        )

        # Store result
        state.setdefault("alignments", {})[result.alignment_job_id] = result

        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get(
    "/{job_id}",
    response_model=AlignmentResult,
    responses={404: {"model": ErrorResponse}}
)
async def get_alignment_result(job_id: str):
    """Get alignment job result."""
    state = get_state()

    if job_id not in state.get("alignments", {}):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alignment job not found: {job_id}"
        )

    return state["alignments"][job_id]


@router.get("/")
async def list_alignments():
    """List all alignment jobs."""
    state = get_state()
    alignments = state.get("alignments", {})

    return {
        "count": len(alignments),
        "alignments": [
            {
                "job_id": job_id,
                "datasets": result.datasets_aligned,
                "alignment_count": len(result.alignments),
                "completed_at": result.completed_at
            }
            for job_id, result in alignments.items()
        ]
    }


@router.post(
    "/{job_id}/validate/{alignment_id}",
    responses={404: {"model": ErrorResponse}}
)
async def validate_alignment(
    job_id: str,
    alignment_id: str,
    validated: bool = True
):
    """Mark an alignment as validated (human-reviewed)."""
    state = get_state()

    if job_id not in state.get("alignments", {}):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alignment job not found: {job_id}"
        )

    alignment_result = state["alignments"][job_id]

    for alignment in alignment_result.alignments:
        if alignment.alignment_id == alignment_id:
            alignment.validated = validated
            return {"status": "updated", "alignment_id": alignment_id, "validated": validated}

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Alignment not found: {alignment_id}"
    )
