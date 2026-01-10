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
   export OPENAI_API_KEY="your-api-key"
   export NEO4J_PASSWORD="password"
   export DB_PASSWORD="password"
   ```

### Running with Docker

Start the full stack with Docker Compose:

```bash
docker-compose up -d
```

This starts:
- HelixForge API on port 8000
- PostgreSQL on port 5432
- Neo4j on ports 7474 (browser) and 7687 (bolt)
- Weaviate on port 8080

### Running Locally

Start just the API server:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

## Usage

### Upload a Dataset

```bash
curl -X POST "http://localhost:8000/datasets/upload" \
  -F "file=@data/samples/clinical.csv"
```

### Get Dataset Metadata

```bash
curl "http://localhost:8000/datasets/{dataset_id}"
```

### Align Datasets

```bash
curl -X POST "http://localhost:8000/align" \
  -H "Content-Type: application/json" \
  -d '{"dataset_ids": ["id1", "id2"]}'
```

### Fuse Datasets

```bash
curl -X POST "http://localhost:8000/fuse" \
  -H "Content-Type: application/json" \
  -d '{"alignment_job_id": "job_123"}'
```

### Generate Insights

```bash
curl -X POST "http://localhost:8000/insights/generate" \
  -H "Content-Type: application/json" \
  -d '{"fused_dataset_id": "fused_123", "generate_narrative": true}'
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/datasets/upload` | Upload/register dataset |
| GET | `/datasets/{id}` | Get dataset metadata |
| GET | `/datasets/{id}/sample` | Get sample rows |
| POST | `/align` | Trigger alignment job |
| GET | `/alignments/{job_id}` | Get alignment result |
| POST | `/fuse` | Run fusion pipeline |
| GET | `/fused/{id}` | Get fused dataset info |
| GET | `/fused/{id}/download` | Download fused dataset |
| POST | `/insights/generate` | Generate insights |
| GET | `/insights/{id}` | Get insight report |
| GET | `/trace/{dataset_id}/{field}` | Get field provenance |
| GET | `/health` | Health check |

## Project Structure

```
helixforge/
├── agents/                    # Core processing agents
│   ├── data_ingestor_agent.py
│   ├── metadata_interpreter_agent.py
│   ├── ontology_alignment_agent.py
│   ├── fusion_agent.py
│   ├── insight_generator_agent.py
│   └── provenance_tracker_agent.py
├── api/                       # FastAPI application
│   ├── routes/
│   └── middleware/
├── utils/                     # Utility modules
├── models/                    # Pydantic/SQLAlchemy models
├── tests/                     # Test suite
├── data/                      # Data storage
├── outputs/                   # Generated reports
├── docs/                      # Documentation
├── config.yaml               # Configuration
├── requirements.txt          # Python dependencies
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
pytest --cov=. --cov-report=html
```

### Code Quality

```bash
# Linting
ruff check .

# Formatting
ruff format .

# Type checking
mypy .
```

## Documentation

- [SPEC.md](SPEC.md) - Technical specification
- [docs/USER_STORIES.md](docs/USER_STORIES.md) - User stories and acceptance criteria
- [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md) - Coding conventions

## Roadmap

- [ ] Multi-modal support (text, image, omics)
- [ ] Ontology self-extension
- [ ] Interactive query agent
- [ ] Real-time streaming (Kafka/Kinesis)
- [ ] Federated mode for cross-organization alignment

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
