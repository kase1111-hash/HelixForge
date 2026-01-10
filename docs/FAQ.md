# HelixForge FAQ

## General Questions

### What is HelixForge?
HelixForge is a Cross-Dataset Insight Synthesizer that transforms heterogeneous data sources into unified insights. It uses AI-powered semantic understanding to automatically align schemas, merge datasets, and generate statistical insights while maintaining full data provenance.

### What data formats does HelixForge support?
- **Files**: CSV, JSON, JSON Lines, Parquet, Excel (.xlsx)
- **Databases**: PostgreSQL, MySQL, SQLite (via SQL connections)
- **APIs**: REST APIs returning JSON data

### Do I need an OpenAI API key?
Yes, HelixForge uses OpenAI's GPT-4o for semantic field labeling and text-embedding-3-large for generating field embeddings. You'll need to set the `OPENAI_API_KEY` environment variable.

### What are the system requirements?
- Python 3.10+
- 8GB RAM minimum (16GB recommended for large datasets)
- Docker (optional, for full stack deployment)

---

## Installation & Setup

### How do I install HelixForge?
```bash
# Clone the repository
git clone https://github.com/helixforge/helixforge.git
cd helixforge

# Install dependencies
pip install -r requirements.txt

# Set up environment
export OPENAI_API_KEY=your-key-here

# Run the server
make run
```

### How do I run with Docker?
```bash
docker compose up -d
```
This starts HelixForge along with PostgreSQL, Neo4j, and Weaviate.

### How do I configure HelixForge?
Edit `config.yaml` or use environment-specific configs in the `config/` directory:
- `config/development.yaml` - Local development settings
- `config/staging.yaml` - Pre-production settings
- `config/production.yaml` - Production settings

Set `HELIXFORGE_ENV=production` to use production config.

---

## Usage Questions

### How do I upload a dataset?
```bash
curl -X POST http://localhost:8000/datasets/upload \
  -F "file=@your-data.csv"
```

Or via the Swagger UI at `http://localhost:8000/docs`.

### How do I align two datasets?
```bash
curl -X POST http://localhost:8000/align/datasets \
  -H "Content-Type: application/json" \
  -d '{"dataset_ids": ["dataset-1", "dataset-2"]}'
```

### How do I merge aligned datasets?
```bash
curl -X POST http://localhost:8000/fuse/execute \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_ids": ["dataset-1", "dataset-2"],
    "alignment_id": "alignment-123",
    "join_strategy": "semantic_similarity"
  }'
```

### What join strategies are available?
| Strategy | Description |
|----------|-------------|
| `exact_key` | Join on exact field value matches |
| `semantic_similarity` | Join using embedding similarity |
| `probabilistic` | Fuzzy matching with confidence scores |
| `temporal` | Time-based joining for time series |
| `concat` | Simple concatenation (union) |

### How do I generate insights?
```bash
curl -X POST http://localhost:8000/insights/generate \
  -H "Content-Type: application/json" \
  -d '{"fused_dataset_id": "fused-123"}'
```

---

## Data & Privacy

### Is my data stored securely?
Yes. HelixForge:
- Stores data locally (or in configured databases)
- Never sends raw data to external services
- Only sends field names and sample values to OpenAI for semantic labeling
- Supports SSL/TLS for all database connections

### Can I use HelixForge with sensitive data?
For sensitive data:
1. Use the production config with `send_default_pii=False`
2. Enable API key authentication
3. Use on-premises deployment
4. Review the data being sent for embeddings

### How is data provenance tracked?
Every field in fused datasets has full lineage tracking:
- Original source file and column
- All transformations applied
- Confidence scores at each step
- Stored in Neo4j graph database

---

## Troubleshooting

### "Connection refused" errors
Ensure all services are running:
```bash
docker compose ps  # Check service status
docker compose logs postgres  # Check specific service logs
```

### "OpenAI API key not found"
Set the environment variable:
```bash
export OPENAI_API_KEY=sk-your-key-here
```

### Slow performance with large files
- Increase `processing.batch_size` in config
- Use Parquet format for large datasets
- Enable chunked processing for files >100MB

### Memory errors
- Reduce `processing.batch_size`
- Process datasets in smaller chunks
- Use streaming ingestion for very large files

### Type inference issues
If automatic type detection fails:
1. Check for mixed types in columns
2. Ensure consistent date formats
3. Use explicit dtype hints in the API call

---

## API & Integration

### How do I authenticate API requests?
When `security.api_key_required=true`:
```bash
curl -X GET http://localhost:8000/datasets \
  -H "X-API-Key: your-api-key"
```

### Can I use HelixForge programmatically?
Yes, import the agents directly:
```python
from agents.data_ingestor_agent import DataIngestorAgent
from agents.fusion_agent import FusionAgent

ingestor = DataIngestorAgent(config)
result = ingestor.ingest_file("data.csv")
```

### Is there a Python SDK?
The agents can be used as a Python library. Import from the `agents` package:
```python
from agents import (
    DataIngestorAgent,
    MetadataInterpreterAgent,
    OntologyAlignmentAgent,
    FusionAgent,
    InsightGeneratorAgent,
    ProvenanceTrackerAgent
)
```

### How do I extend HelixForge?
1. **Custom transformations**: Add to `BUILTIN_TRANSFORMS` in `fusion_agent.py`
2. **New data sources**: Implement `_ingest_*` method in `data_ingestor_agent.py`
3. **Custom insights**: Add analysis methods to `insight_generator_agent.py`

---

## Performance & Scaling

### How many datasets can HelixForge handle?
There's no hard limit. Performance depends on:
- Total data size
- Number of fields
- Complexity of alignments

### Can I run HelixForge in a cluster?
Yes, deploy multiple API instances behind a load balancer. All instances share the same databases.

### What's the maximum file size?
Default: 100MB (configurable via `storage.max_file_size_mb`)

For larger files:
1. Use chunked ingestion
2. Pre-convert to Parquet
3. Increase memory limits

---

## Contributing

### How do I report bugs?
Open an issue at: https://github.com/helixforge/helixforge/issues

### How do I contribute code?
1. Fork the repository
2. Create a feature branch
3. Run tests: `make test`
4. Run linting: `make check`
5. Submit a pull request

### What's the code style?
- Follow PEP 8
- Use type hints
- Run `ruff` for linting
- Run `mypy` for type checking
