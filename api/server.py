"""FastAPI server for HelixForge.

Main application entry point that configures the API server
with all routes, middleware, and dependencies.
"""

import os
from contextlib import asynccontextmanager
from typing import Dict

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import alignment, datasets, fusion
from utils.logging import get_logger

logger = get_logger(__name__)

# Global state
app_state: Dict = {}

# Cached configuration (singleton pattern to ensure consistent config)
_cached_config: dict = None


def load_config() -> dict:
    """Load configuration from config.yaml.

    Uses cached config to ensure consistent configuration across the application.
    """
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    config_path = os.environ.get("HELIXFORGE_CONFIG", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            _cached_config = yaml.safe_load(f) or {}
    else:
        _cached_config = {}
    return _cached_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting HelixForge API server")

    config = load_config()
    app_state["config"] = config

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

# Configure CORS
config = load_config()
cors_origins = config.get("api", {}).get("cors_origins", ["*"])

# Warn about wildcard CORS in production
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
    """Health check endpoint."""
    return {
        "status": "healthy",
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
