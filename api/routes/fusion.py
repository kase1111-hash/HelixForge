"""Fusion routes for HelixForge API."""

import os
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from models.schemas import ErrorResponse, FusionRequest, FusionResult

router = APIRouter()


def get_state() -> Dict[str, Any]:
    """Get application state."""
    from api.server import get_app_state
    return get_app_state()


@router.post(
    "",
    response_model=FusionResult,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}}
)
async def fuse_datasets(request: FusionRequest):
    """Fuse datasets based on alignment results.

    Merges aligned datasets using the specified join strategy.
    """
    state = get_state()
    fusion_agent = state.get("fusion")

    if not fusion_agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fusion agent not available"
        )

    # Get alignment result
    if request.alignment_job_id not in state.get("alignments", {}):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alignment job not found: {request.alignment_job_id}"
        )

    alignment_result = state["alignments"][request.alignment_job_id]

    # Get DataFrames for aligned datasets
    dataframes = {}
    for dataset_id in alignment_result.datasets_aligned:
        if dataset_id not in state.get("datasets", {}):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset not found: {dataset_id}"
            )
        dataframes[dataset_id] = state["datasets"][dataset_id]["df"]

    try:
        # Perform fusion
        result = fusion_agent.process(
            dataframes=dataframes,
            alignment_result=alignment_result,
            join_strategy=request.join_strategy,
            imputation_method=request.imputation_method
        )

        # Store result
        state.setdefault("fused", {})[result.fused_dataset_id] = {
            "result": result,
            "df": fusion_agent.get_fused_dataframe(result.fused_dataset_id)
        }

        # Record provenance
        provenance = state.get("provenance")
        if provenance:
            field_mappings = {}
            for field in result.merged_fields:
                sources = []
                for alignment in alignment_result.alignments:
                    if alignment.target_field == field or alignment.source_field == field:
                        sources.append(f"{alignment.source_dataset}.{alignment.source_field}")
                        sources.append(f"{alignment.target_dataset}.{alignment.target_field}")
                field_mappings[field] = list(set(sources))

            provenance.record_fusion(
                source_datasets=result.source_datasets,
                fused_dataset_id=result.fused_dataset_id,
                join_strategy=result.join_strategy.value,
                field_mappings=field_mappings
            )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get(
    "/{fused_id}",
    response_model=FusionResult,
    responses={404: {"model": ErrorResponse}}
)
async def get_fused_dataset(fused_id: str):
    """Get fused dataset information."""
    state = get_state()

    if fused_id not in state.get("fused", {}):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fused dataset not found: {fused_id}"
        )

    return state["fused"][fused_id]["result"]


@router.get(
    "/{fused_id}/download",
    responses={404: {"model": ErrorResponse}}
)
async def download_fused_dataset(
    fused_id: str,
    format: str = "parquet"
):
    """Download the fused dataset as a file."""
    state = get_state()

    if fused_id not in state.get("fused", {}):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fused dataset not found: {fused_id}"
        )

    fused_data = state["fused"][fused_id]
    df = fused_data["df"]

    # Create output file
    output_dir = "./data/downloads"
    os.makedirs(output_dir, exist_ok=True)

    if format == "parquet":
        path = os.path.join(output_dir, f"{fused_id}.parquet")
        df.to_parquet(path, index=False)
        media_type = "application/octet-stream"
    else:
        path = os.path.join(output_dir, f"{fused_id}.csv")
        df.to_csv(path, index=False)
        media_type = "text/csv"

    return FileResponse(
        path=path,
        media_type=media_type,
        filename=os.path.basename(path)
    )


@router.get(
    "/{fused_id}/sample",
    responses={404: {"model": ErrorResponse}}
)
async def get_fused_sample(
    fused_id: str,
    rows: int = 10
):
    """Get sample rows from fused dataset."""
    state = get_state()

    if fused_id not in state.get("fused", {}):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fused dataset not found: {fused_id}"
        )

    df = state["fused"][fused_id]["df"]
    sample = df.head(rows).to_dict(orient="records")

    return {
        "fused_id": fused_id,
        "rows": len(sample),
        "data": sample
    }


@router.get("/")
async def list_fused_datasets():
    """List all fused datasets."""
    state = get_state()
    fused = state.get("fused", {})

    return {
        "count": len(fused),
        "fused_datasets": [
            {
                "fused_id": fid,
                "source_datasets": data["result"].source_datasets,
                "record_count": data["result"].record_count,
                "field_count": data["result"].field_count,
                "fused_at": data["result"].fused_at
            }
            for fid, data in fused.items()
        ]
    }
