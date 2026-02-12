# HelixForge

**Cross-Dataset Insight Synthesizer**

*From fragmented data to unified insight.*

---

## Overview

HelixForge transforms heterogeneous datasets with unique schemas, vocabularies, and formats into harmonized, analysis-ready data products. It functions as a Digital Language Processor (DLP) between data silos, using natural-language understanding to align meaning across sources.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HELIXFORGE CORE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│  │   CSV/JSON  │   │   Parquet   │   │  SQL/REST   │   │  Databases  │     │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘     │
│         └──────────────┬──┴──────────────────┴─────────────────┘           │
│                        ▼                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  LAYER 1: DATA INTAKE → LAYER 2: METADATA → LAYER 3: ALIGNMENT     │   │
│  │       ↓                       ↓                    ↓                │   │
│  │  LAYER 4: FUSION → LAYER 5: INSIGHTS → LAYER 6: PROVENANCE         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Vector Store │  │ Graph Store  │  │  PostgreSQL  │  │  Event Bus   │    │
│  │  (Weaviate)  │  │   (Neo4j)    │  │  (Metadata)  │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-format ingestion** - CSV, Parquet, JSON, SQL databases, REST APIs
- **Semantic field understanding** - LLM-powered inference of column meanings
- **Cross-dataset alignment** - Vector similarity matching for schema harmonization
- **Intelligent fusion** - Semantic joins, unit conversions, missing value imputation
- **Automated insights** - Correlations, outliers, clustering, narrative summaries
- **Full provenance** - Complete lineage tracking from source to output

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kase1111-hash/HelixForge.git
   cd HelixForge
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and set your OpenAI API key:
   # OPENAI_API_KEY=sk-your-api-key-here
   ```

   Or export directly:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

### Running with Docker

Start the API with Docker Compose:

```bash
docker compose up -d
```

This starts the HelixForge API on port 8000. To use external data stores (PostgreSQL, Neo4j, Weaviate), configure their connection details in `config.yaml` or environment variables and run them separately.

### Running Locally

Start just the API server:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

## Usage

### CLI

HelixForge includes a command-line interface for common operations:

```bash
python cli.py ingest <file>                    # Ingest a dataset
python cli.py describe <file>                  # Describe fields with semantic labels
python cli.py align <file1> <file2>            # Align schemas between datasets
python cli.py fuse <file1> <file2>             # Merge two datasets
python cli.py analyze <file>                   # Generate statistics and insights
```

All CLI commands support `--format json|table` and `--provider mock|openai` options.

### REST API

#### Upload a Dataset

```bash
curl -X POST "http://localhost:8000/datasets/upload" \
  -F "file=@data/samples/clinical.csv"
```

#### Get Dataset Metadata

```bash
curl "http://localhost:8000/datasets/{dataset_id}"
```

#### Align Datasets

```bash
curl -X POST "http://localhost:8000/align" \
  -H "Content-Type: application/json" \
  -d '{"dataset_ids": ["id1", "id2"]}'
```

#### Fuse Datasets

```bash
curl -X POST "http://localhost:8000/fuse" \
  -H "Content-Type: application/json" \
  -d '{"alignment_job_id": "job_123"}'
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/datasets/` | List all datasets |
| POST | `/datasets/upload` | Upload/register dataset |
| GET | `/datasets/{id}` | Get dataset metadata |
| GET | `/datasets/{id}/sample` | Get sample rows |
| DELETE | `/datasets/{id}` | Delete a dataset |
| GET | `/align/` | List all alignment jobs |
| POST | `/align` | Trigger alignment job |
| GET | `/align/{job_id}` | Get alignment result |
| POST | `/align/{job_id}/validate/{alignment_id}` | Validate an alignment |
| GET | `/fuse/` | List all fused datasets |
| POST | `/fuse` | Run fusion pipeline |
| GET | `/fuse/{id}` | Get fused dataset info |
| GET | `/fuse/{id}/download` | Download fused dataset |
| GET | `/fuse/{id}/sample` | Get sample rows from fused dataset |
| GET | `/health` | Health check |

Interactive API docs are available at `/docs` (Swagger UI) and `/redoc` (ReDoc).

## Project Structure

```
helixforge/
├── agents/                    # Core processing agents
│   ├── base_agent.py          # Abstract base class for all agents
│   ├── data_ingestor_agent.py # Data ingestion from multiple sources
│   ├── metadata_interpreter_agent.py  # Semantic field labeling
│   ├── ontology_alignment_agent.py    # Cross-dataset schema alignment
│   ├── fusion_agent.py        # Dataset merging and transformation
│   └── insight_agent.py       # Statistical analysis and insights
├── api/                       # FastAPI application
│   ├── server.py              # Main API server
│   └── routes/                # API route handlers
├── utils/                     # Utility modules
├── models/                    # Pydantic data models
├── tests/                     # Test suite
├── cli.py                     # Command-line interface
├── data/                      # Data storage
├── outputs/                   # Generated reports
├── docs/                      # Documentation
├── config.yaml               # Configuration
├── requirements.txt          # Python dependencies
├── Makefile                  # Build automation
├── Dockerfile
└── docker-compose.yaml
```

## Configuration

Configuration is managed through `config.yaml` and environment variables:

```yaml
llm:
  provider: "openai"
  model: "gpt-4o"
  embedding_model: "text-embedding-3-large"

processing:
  confidence_threshold: 0.80
  max_file_size_mb: 500

fusion:
  default_join_strategy: "semantic_similarity"
  similarity_threshold: 0.85
```

## Example Output

```json
{
  "fused_dataset_id": "fused_abc123",
  "merged_fields": ["gene_expression", "drug_response", "survival_months"],
  "record_count": 5000,
  "narrative_summary": "Analysis of 5000 patient records reveals a strong positive correlation (r=0.72) between gene expression levels and drug response rates. Patients with high gene_expression show 38% higher response to Drug X.",
  "key_findings": [
    {"type": "correlation", "description": "gene_expression and drug_response: r=0.72, p<0.001"},
    {"type": "outlier", "description": "15 records with survival_months > 120 (3 IQR above median)"}
  ],
  "provenance": {
    "sources": ["genomics.csv", "clinical_trial.parquet"],
    "transformations": 12
  }
}
```

## Development

### Running Tests

```bash
make test           # Run all tests
make test-unit      # Run unit tests only
make test-int       # Run integration tests only
make test-cov       # Run with coverage report
```

Or directly with pytest:
```bash
pytest tests/ -v --tb=short
```

### Code Quality

```bash
make lint           # Run ruff linter
make format         # Format code with ruff
make typecheck      # Run mypy type checker
make check          # Run lint + typecheck
```

## Documentation

- [SPEC.md](SPEC.md) - Technical specification
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture
- [docs/USER_STORIES.md](docs/USER_STORIES.md) - User stories and acceptance criteria
- [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md) - Coding conventions
- [docs/SECURITY.md](docs/SECURITY.md) - Security documentation
- [docs/FAQ.md](docs/FAQ.md) - Frequently asked questions
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [CHANGELOG.md](CHANGELOG.md) - Version history

## Roadmap

- [ ] Multi-modal support (text, image, omics)
- [ ] Ontology self-extension
- [ ] Interactive query agent
- [ ] Real-time streaming (Kafka/Kinesis)
- [ ] Federated mode for cross-organization alignment

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

---

## Connected Repositories

HelixForge is part of a broader ecosystem of natural language programming and AI agent tools:

### NatLangChain Ecosystem
- [NatLangChain](https://github.com/kase1111-hash/NatLangChain) - Prose-first, intent-native blockchain protocol
- [IntentLog](https://github.com/kase1111-hash/IntentLog) - Git for human reasoning
- [mediator-node](https://github.com/kase1111-hash/mediator-node) - LLM mediation layer

### Agent-OS Ecosystem
- [Agent-OS](https://github.com/kase1111-hash/Agent-OS) - Natural-language native operating system for AI agents
- [synth-mind](https://github.com/kase1111-hash/synth-mind) - NLOS-based agent with psychological modules
- [memory-vault](https://github.com/kase1111-hash/memory-vault) - Sovereign storage for cognitive artifacts

---

## License

See [LICENSE](LICENSE) for details.
