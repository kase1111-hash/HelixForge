# HelixForge - Claude Code Guide

## Project Overview

HelixForge is an AI-powered data integration platform that transforms heterogeneous datasets into harmonized, analysis-ready data products. It uses LLM-powered semantic analysis to automatically align schemas, fuse datasets, and generate insights.

**Tech Stack:** Python 3.10+, FastAPI, PostgreSQL, Neo4j, Weaviate, OpenAI API

## Quick Commands

```bash
# Development setup
make install-dev       # Install all dependencies
make run-dev           # Start API with hot reload (port 8000)
docker compose up -d   # Full stack with all services

# Testing
make test              # Run all tests
make test-unit         # Unit tests only
make test-int          # Integration tests only
pytest -xvs tests/test_<module>.py  # Run specific test file

# Code quality
make lint              # Run ruff linter
make format            # Format code with ruff
make typecheck         # Run mypy type checker
make check             # Run all checks (lint + typecheck)
```

## Project Structure

```
agents/           # 6-layer processing pipeline (core logic)
api/              # FastAPI routes and middleware
  routes/         # Endpoint handlers (datasets, alignment, fusion, insights, provenance)
  server.py       # Main application entry point
models/           # Pydantic schemas and data models
utils/            # Shared utilities (embeddings, similarity, logging, validation)
tests/            # Pytest test suite
config/           # Environment-specific YAML configs
docs/             # Architecture and style documentation
```

## Architecture

The system uses a 6-agent pipeline pattern:

1. **Data Ingestor** - Multi-format ingestion (CSV, Parquet, JSON, Excel, SQL, REST)
2. **Metadata Interpreter** - LLM semantic field labeling
3. **Ontology Alignment** - Cross-dataset schema matching
4. **Fusion Agent** - Dataset merging with multiple join strategies
5. **Insight Generator** - Statistical analysis and clustering
6. **Provenance Tracker** - Full lineage tracking in Neo4j

All agents inherit from `BaseAgent` in `agents/base_agent.py`.

## Code Conventions

- **Formatting:** ruff (line length 120)
- **Type hints:** Required on all public functions
- **Docstrings:** Google-style with Args, Returns, Raises sections
- **Naming:** snake_case for functions/modules, PascalCase for classes, UPPERCASE for constants

## Testing

- Use pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`
- Fixtures defined in `tests/conftest.py`
- Mock external services (OpenAI, databases) in tests
- Coverage targets: 80% overall, 90% for agents

## Configuration

- Main config: `config.yaml` (LLM settings, thresholds, database URLs)
- Environment configs: `config/development.yaml`, `config/staging.yaml`, `config/production.yaml`
- Secrets via environment variables (never commit credentials)

## Key Files

- `SPEC.md` - Complete technical specification (start here for deep understanding)
- `docs/ARCHITECTURE.md` - System design and data flows
- `docs/STYLE_GUIDE.md` - Detailed coding conventions
- `pyproject.toml` - Tool configurations (ruff, mypy, pytest)

## API

- OpenAPI docs available at `/docs` when server is running
- Health check: `GET /health`
- All endpoints are async FastAPI routes
