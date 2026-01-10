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

from api.routes import alignment, datasets, fusion, insights, provenance
from utils.logging import get_logger

logger = get_logger(__name__)

# Global state
app_state: Dict = {}


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = os.environ.get("HELIXFORGE_CONFIG", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


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
    from agents.insight_generator_agent import InsightGeneratorAgent
    from agents.provenance_tracker_agent import ProvenanceTrackerAgent

    app_state["ingestor"] = DataIngestorAgent(config)
    app_state["interpreter"] = MetadataInterpreterAgent(config)
    app_state["aligner"] = OntologyAlignmentAgent(config)
    app_state["fusion"] = FusionAgent(config)
    app_state["insight"] = InsightGeneratorAgent(config)
    app_state["provenance"] = ProvenanceTrackerAgent(config)

    # Storage for datasets and results
    app_state["datasets"] = {}
    app_state["metadata"] = {}
    app_state["alignments"] = {}
    app_state["fused"] = {}
    app_state["insights"] = {}

    logger.info("HelixForge API server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down HelixForge API server")
    if "provenance" in app_state:
        app_state["provenance"].close()


# Create FastAPI application
app = FastAPI(
    title="HelixForge API",
    description="Cross-Dataset Insight Synthesizer - Transform heterogeneous datasets into unified insights",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
config = load_config()
cors_origins = config.get("api", {}).get("cors_origins", ["*"])

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
app.include_router(insights.router, prefix="/insights", tags=["Insights"])
app.include_router(provenance.router, prefix="/trace", tags=["Provenance"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from models.schemas import HealthStatus

    return HealthStatus(
        status="healthy",
        database=True,  # Would check actual connections in production
        vector_store=True,
        graph_store=True
    )


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
