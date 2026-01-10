# Changelog

All notable changes to HelixForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-10

### Added

#### Core Agents
- **Data Ingestor Agent**: Multi-format data ingestion (CSV, JSON, Parquet, Excel, SQL, REST API)
  - Auto-detection of file encoding and delimiters
  - Content hashing for deduplication
  - Sample data extraction
- **Metadata Interpreter Agent**: LLM-powered semantic field labeling
  - Automatic data type inference
  - Semantic type classification (identifier, metric, timestamp, category, etc.)
  - Domain tag inference (healthcare, finance, HR, etc.)
  - OpenAI embedding generation for fields
- **Ontology Alignment Agent**: Cross-dataset schema alignment
  - String similarity matching (Jaccard, Levenshtein, Jaro-Winkler)
  - Semantic similarity using embeddings
  - Configurable similarity thresholds
  - Support for exact, similar, and derived alignments
- **Fusion Agent**: Dataset merging and transformation
  - Multiple join strategies (exact key, semantic similarity, probabilistic, temporal, concat)
  - Built-in transformations (uppercase, lowercase, trim, round, normalize, etc.)
  - Missing value imputation (mean, median, mode, KNN)
  - Conflict resolution strategies
- **Insight Generator Agent**: Statistical analysis and visualization
  - Descriptive statistics computation
  - Correlation analysis (Pearson/Spearman)
  - Outlier detection (IQR-based)
  - K-means clustering with silhouette scoring
  - Distribution type detection
  - Visualization generation (correlation heatmaps, box plots, histograms)
- **Provenance Tracker Agent**: Full data lineage tracking
  - Field-level origin tracking
  - Transformation chain recording
  - Neo4j graph storage integration
  - Confidence decay calculation
  - HTML/JSON report generation

#### API
- FastAPI REST API with async endpoints
- Dataset upload and management endpoints
- Schema alignment endpoints
- Dataset fusion endpoints
- Insight generation endpoints
- Provenance query endpoints
- Built-in OpenAPI/Swagger documentation

#### Infrastructure
- Docker and Docker Compose configuration
- PostgreSQL, Neo4j, and Weaviate service definitions
- Environment-based configuration
- Structured JSON logging with correlation IDs

#### Development
- Comprehensive unit test suite (200+ tests)
- Integration test suite
- Ruff linting configuration
- Mypy type checking configuration
- Makefile for common tasks
- GitHub Actions CI/CD pipeline

### Security
- Input validation and sanitization
- Secure configuration via environment variables
- API key authentication support

## [Unreleased]

### Planned
- Sentry error tracking integration
- ELK stack logging integration
- Performance benchmarking suite
- Additional data source connectors
- GraphQL API support
- Web dashboard UI
