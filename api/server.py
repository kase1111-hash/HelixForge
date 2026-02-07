"""FastAPI server for HelixForge.

Main application entry point that configures the API server
with all routes, middleware, and dependencies.
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import alignment, datasets, fusion
from utils.config import load_config, reset_config
from utils.errors import ConfigurationError, HelixForgeError
from utils.logging import get_logger
from utils.validation import ValidationError

logger = get_logger(__name__)

# Global state
app_state: Dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Validates configuration on startup and fails fast with
    clear error messages if anything is wrong.
    """
    # Startup
    logger.info("Starting HelixForge API server")

    # Reset config cache so we pick up fresh values
    reset_config()

    try:
        config = load_config()
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e.message}. {e.detail}")
        raise SystemExit(f"FATAL: {e.message}. {e.detail}") from e

    app_state["config"] = config

    # Check OPENAI_API_KEY if using OpenAI provider
    llm_provider = config.get("llm", {}).get("provider", "openai")
    if llm_provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        logger.warning(
            "OPENAI_API_KEY not set. LLM features will fail. "
            "Set the environment variable or switch to a different provider."
        )

    # Initialize agents
    from agents.data_ingestor_agent import DataIngestorAgent
    from agents.metadata_interpreter_agent import MetadataInterpreterAgent
    from agents.ontology_alignment_agent import OntologyAlignmentAgent
    from agents.fusion_agent import FusionAgent

    app_state["ingestor"] = DataIngestorAgent(config)
    app_state["interpreter"] = MetadataInterpreterAgent(config)
    app_state["aligner"] = OntologyAlignmentAgent(config)
    app_state["fusion"] = FusionAgent(config)

    # Storage for datasets and results
    app_state["datasets"] = {}
    app_state["metadata"] = {}
    app_state["alignments"] = {}
    app_state["fused"] = {}

    logger.info("HelixForge API server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down HelixForge API server")


# OpenAPI tags metadata
tags_metadata = [
    {
        "name": "Datasets",
        "description": "Upload, manage, and query datasets. Supports CSV, JSON, Parquet, Excel, and REST API sources.",
    },
    {
        "name": "Alignment",
        "description": "Align schemas across multiple datasets using semantic similarity.",
    },
    {
        "name": "Fusion",
        "description": "Merge aligned datasets using various join strategies with transformation and imputation.",
    },
]

# Create FastAPI application
app = FastAPI(
    title="HelixForge API",
    description="""
# HelixForge - Cross-Dataset Insight Synthesizer

Transform heterogeneous datasets into unified insights through intelligent data fusion.

## Features

- **Multi-format Ingestion**: CSV, JSON, Parquet, Excel
- **Semantic Schema Alignment**: AI-powered field matching across datasets
- **Intelligent Fusion**: Multiple join strategies with conflict resolution

## Quick Start

1. Upload datasets via `/datasets/upload`
2. Align schemas via `/align/datasets`
3. Fuse data via `/fuse/execute`
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=tags_metadata,
    contact={
        "name": "HelixForge Support",
        "url": "https://github.com/helixforge/helixforge",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)


# ------------------------------------------------------------------ #
#  Structured exception handlers                                       #
# ------------------------------------------------------------------ #

@app.exception_handler(HelixForgeError)
async def helixforge_error_handler(request: Request, exc: HelixForgeError):
    """Return structured JSON for all HelixForge domain errors."""
    return JSONResponse(
        status_code=400,
        content={
            "error": type(exc).__name__,
            "detail": exc.message,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Return structured JSON for validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# ------------------------------------------------------------------ #
#  CORS                                                                #
# ------------------------------------------------------------------ #

config = load_config()
cors_origins = config.get("api", {}).get("cors_origins", ["*"])

if "*" in cors_origins:
    logger.warning(
        "CORS is configured with wildcard origin ('*'). "
        "This is not recommended for production deployments. "
        "Configure specific allowed origins in config.yaml under api.cors_origins."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
app.include_router(alignment.router, prefix="/align", tags=["Alignment"])
app.include_router(fusion.router, prefix="/fuse", tags=["Fusion"])


@app.get("/health")
async def health_check():
    """Health check endpoint.

    Verifies the application is running and reports on
    critical dependency availability.
    """
    checks = {"api": "ok"}

    # Check OpenAI API key availability
    if os.environ.get("OPENAI_API_KEY"):
        checks["openai_key"] = "configured"
    else:
        checks["openai_key"] = "missing"

    # Check agent initialization
    checks["agents"] = "ok" if app_state.get("ingestor") else "not_initialized"

    all_ok = checks["agents"] == "ok"
    return {
        "status": "healthy" if all_ok else "degraded",
        "checks": checks,
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "HelixForge API",
        "version": "1.0.0",
        "tagline": "From fragmented data to unified insight",
        "docs": "/docs"
    }


def get_app_state() -> Dict:
    """Get application state (for use in route handlers)."""
    return app_state
