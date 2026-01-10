"""Dataset routes for HelixForge API."""

import os
import tempfile
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from models.schemas import DatasetMetadata, ErrorResponse, IngestResult

router = APIRouter()


def get_state() -> Dict[str, Any]:
    """Get application state."""
    from api.server import get_app_state
    return get_app_state()


@router.post(
    "/upload",
    response_model=IngestResult,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}}
)
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_id: Optional[str] = None
):
    """Upload and ingest a new dataset.

    Supports CSV, Parquet, JSON, and Excel files.
    """
    state = get_state()
    ingestor = state.get("ingestor")

    if not ingestor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestor agent not available"
        )

    # Validate file extension
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    allowed = {".csv", ".parquet", ".json", ".jsonl", ".xlsx", ".xls"}

    if ext not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {ext}. Allowed: {allowed}"
        )

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Ingest the file
        result = ingestor.ingest_file(tmp_path, dataset_id=dataset_id)

        # Store in state
        state["datasets"][result.dataset_id] = {
            "result": result,
            "df": ingestor.get_dataframe(result.dataset_id)
        }

        # Record provenance
        if "provenance" in state:
            state["provenance"].record_ingestion(result)

        # Clean up temp file
        os.unlink(tmp_path)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get(
    "/{dataset_id}",
    response_model=DatasetMetadata,
    responses={404: {"model": ErrorResponse}}
)
async def get_dataset_metadata(dataset_id: str):
    """Get metadata for a dataset.

    If metadata hasn't been generated yet, it will be computed.
    """
    state = get_state()

    # Check if dataset exists
    if dataset_id not in state.get("datasets", {}):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {dataset_id}"
        )

    # Check for existing metadata
    if dataset_id in state.get("metadata", {}):
        return state["metadata"][dataset_id]

    # Generate metadata
    interpreter = state.get("interpreter")
    if not interpreter:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Interpreter agent not available"
        )

    df = state["datasets"][dataset_id]["df"]
    metadata = interpreter.process(dataset_id, df)

    state["metadata"][dataset_id] = metadata
    return metadata


@router.get(
    "/{dataset_id}/sample",
    responses={404: {"model": ErrorResponse}}
)
async def get_dataset_sample(
    dataset_id: str,
    rows: int = Query(default=10, ge=1, le=100)
):
    """Get sample rows from a dataset."""
    state = get_state()

    if dataset_id not in state.get("datasets", {}):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {dataset_id}"
        )

    df = state["datasets"][dataset_id]["df"]
    sample = df.head(rows).to_dict(orient="records")

    return {
        "dataset_id": dataset_id,
        "rows": len(sample),
        "data": sample
    }


@router.get("/")
async def list_datasets():
    """List all available datasets."""
    state = get_state()
    datasets = state.get("datasets", {})

    return {
        "count": len(datasets),
        "datasets": [
            {
                "dataset_id": did,
                "row_count": data["result"].row_count,
                "field_count": len(data["result"].schema_fields),
                "source_type": data["result"].source_type.value
            }
            for did, data in datasets.items()
        ]
    }


@router.delete(
    "/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}}
)
async def delete_dataset(dataset_id: str):
    """Delete a dataset."""
    state = get_state()

    if dataset_id not in state.get("datasets", {}):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {dataset_id}"
        )

    del state["datasets"][dataset_id]
    if dataset_id in state.get("metadata", {}):
        del state["metadata"][dataset_id]
