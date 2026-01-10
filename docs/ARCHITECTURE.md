# HelixForge Architecture

## System Overview

HelixForge is a Cross-Dataset Insight Synthesizer that transforms heterogeneous data sources into unified insights through a multi-agent pipeline architecture.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HelixForge Architecture                            │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │   Client    │
                              │  (API/CLI)  │
                              └──────┬──────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    FastAPI Server     │
                         │   (REST API + Docs)   │
                         └───────────┬───────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
┌───────────────┐          ┌─────────────────┐          ┌─────────────────┐
│   Datasets    │          │    Alignment    │          │     Fusion      │
│   Endpoints   │          │    Endpoints    │          │    Endpoints    │
└───────┬───────┘          └────────┬────────┘          └────────┬────────┘
        │                           │                            │
        └───────────────────────────┼────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       Agent Pipeline          │
                    │   (Event-Driven Processing)   │
                    └───────────────┬───────────────┘
                                    │
     ┌──────────────────────────────┼──────────────────────────────┐
     │              │               │               │              │
     ▼              ▼               ▼               ▼              ▼
┌─────────┐  ┌───────────┐  ┌───────────┐  ┌─────────┐  ┌──────────┐
│Ingestor │─▶│Interpreter│─▶│  Aligner  │─▶│ Fusion  │─▶│ Insights │
│  Agent  │  │   Agent   │  │   Agent   │  │  Agent  │  │  Agent   │
└────┬────┘  └─────┬─────┘  └─────┬─────┘  └────┬────┘  └────┬─────┘
     │             │              │             │            │
     └─────────────┴──────────────┴─────────────┴────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────────┐
                    │    Provenance Tracker Agent   │
                    │   (Full Data Lineage Graph)   │
                    └───────────────┬───────────────┘
                                    │
     ┌──────────────────────────────┼──────────────────────────────┐
     │                              │                              │
     ▼                              ▼                              ▼
┌─────────────┐            ┌─────────────┐            ┌─────────────┐
│  PostgreSQL │            │    Neo4j    │            │  Weaviate   │
│  (Metadata) │            │   (Graph)   │            │  (Vectors)  │
└─────────────┘            └─────────────┘            └─────────────┘
```

## Agent Pipeline

The 6-agent pipeline processes data through distinct stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Agent Pipeline                                  │
└─────────────────────────────────────────────────────────────────────────────┘

   Data Sources                Agent Pipeline                    Outputs
   ────────────                ──────────────                    ───────

┌──────────┐
│   CSV    │──┐
└──────────┘  │
┌──────────┐  │    ┌─────────────────────────────────────────────────────┐
│   JSON   │──┤    │                                                     │
└──────────┘  │    │  ┌─────────┐   ┌────────────┐   ┌─────────────┐    │
┌──────────┐  ├───▶│  │ Data    │──▶│ Metadata   │──▶│  Ontology   │    │
│ Parquet  │──┤    │  │Ingestor │   │Interpreter │   │  Alignment  │    │
└──────────┘  │    │  └─────────┘   └────────────┘   └──────┬──────┘    │
┌──────────┐  │    │       │              │                 │           │
│   SQL    │──┤    │       │ data.ingested│ metadata.ready  │           │
└──────────┘  │    │       │              │                 │           │
┌──────────┐  │    │       ▼              ▼                 ▼           │
│   REST   │──┘    │                                                    │
└──────────┘       │  ┌─────────────┐   ┌────────────┐   ┌──────────┐  │
                   │  │  Provenance │◀──│   Fusion   │◀──│ alignment│  │
                   │  │   Tracker   │   │   Agent    │   │  .ready  │  │
                   │  └─────────────┘   └─────┬──────┘   └──────────┘  │
                   │        │                 │                        │
                   │        │ trace.updated   │ data.fused             │
                   │        │                 │                        │
                   │        ▼                 ▼                        │
                   └─────────────────────────────────────────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │    Insight      │
                                    │   Generator     │
                                    └────────┬────────┘
                                             │
                        ┌────────────────────┼────────────────────┐
                        │                    │                    │
                        ▼                    ▼                    ▼
                 ┌────────────┐      ┌────────────┐      ┌────────────┐
                 │ Statistics │      │Correlations│      │ Clustering │
                 │   Report   │      │   Matrix   │      │  Analysis  │
                 └────────────┘      └────────────┘      └────────────┘
```

## Data Flow

### 1. Ingestion Phase

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Data Ingestion Flow                                │
└─────────────────────────────────────────────────────────────────────────────┘

  Source File                    Processing                      Output
  ───────────                    ──────────                      ──────

┌────────────┐              ┌─────────────────┐            ┌───────────────┐
│            │              │                 │            │               │
│  Raw File  │──▶ detect ──▶│  Parse & Load   │──▶ hash ──▶│ IngestResult  │
│  (CSV/JSON)│    encoding  │  (pandas/pyarrow│    content │               │
│            │              │                 │            │ - dataset_id  │
└────────────┘              └────────┬────────┘            │ - schema      │
                                     │                     │ - dtypes      │
                                     ▼                     │ - row_count   │
                            ┌─────────────────┐            │ - content_hash│
                            │  Store Parquet  │            └───────────────┘
                            │  (Optimized)    │
                            └─────────────────┘
```

### 2. Metadata Interpretation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Metadata Interpretation Flow                            │
└─────────────────────────────────────────────────────────────────────────────┘

   DataFrame                   Analysis                       Output
   ─────────                   ────────                       ──────

┌────────────┐          ┌─────────────────┐          ┌─────────────────────┐
│            │          │                 │          │   DatasetMetadata   │
│  pandas    │───────▶  │  Infer Types    │          │                     │
│  DataFrame │          │  (int/float/str)│          │ For each field:     │
│            │          │                 │          │ ┌─────────────────┐ │
└────────────┘          └────────┬────────┘          │ │ FieldMetadata   │ │
                                 │                   │ │ - semantic_type │ │
                                 ▼                   │ │ - data_type     │ │
                        ┌─────────────────┐          │ │ - null_ratio    │ │
                        │   LLM Semantic  │          │ │ - unique_ratio  │ │
                        │    Labeling     │──────▶   │ │ - description   │ │
                        │   (GPT-4o)      │          │ │ - embedding     │ │
                        └────────┬────────┘          │ └─────────────────┘ │
                                 │                   └─────────────────────┘
                                 ▼
                        ┌─────────────────┐
                        │ Generate Field  │
                        │   Embeddings    │
                        │ (text-embed-3)  │
                        └─────────────────┘
```

### 3. Schema Alignment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Schema Alignment Flow                               │
└─────────────────────────────────────────────────────────────────────────────┘

  Dataset A                   Alignment                      Dataset B
  ─────────                   ─────────                      ─────────

┌────────────┐                                          ┌────────────┐
│ employee_id│◀─────────── EXACT ─────────────────────▶│  emp_id    │
│ first_name │◀─────────── SIMILAR (0.92) ────────────▶│  name      │
│ department │◀─────────── SIMILAR (0.87) ────────────▶│  dept      │
│ salary     │◀─────────── EXACT ─────────────────────▶│  salary    │
│ hire_date  │◀─────────── DERIVED ───────────────────▶│ start_date │
└────────────┘                                          └────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │     AlignmentResult       │
                    │                           │
                    │  - field_alignments[]     │
                    │  - similarity_scores      │
                    │  - canonical_names        │
                    │  - global_ontology        │
                    └───────────────────────────┘
```

### 4. Dataset Fusion

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Dataset Fusion Flow                                │
└─────────────────────────────────────────────────────────────────────────────┘

   Aligned Datasets              Fusion                    Fused Dataset
   ────────────────              ──────                    ─────────────

┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│   Dataset A     │        │  Join Strategy  │        │  Fused Dataset  │
│   (100 rows)    │───────▶│                 │───────▶│  (merged rows)  │
└─────────────────┘        │ - exact_key     │        └─────────────────┘
                           │ - semantic_sim  │               │
┌─────────────────┐        │ - probabilistic │               │
│   Dataset B     │───────▶│ - temporal      │               │
│   (80 rows)     │        │ - concat        │               │
└─────────────────┘        └────────┬────────┘               │
                                    │                        │
                                    ▼                        ▼
                           ┌─────────────────┐    ┌─────────────────────┐
                           │ Transformations │    │   FusionResult      │
                           │ - normalize     │    │                     │
                           │ - round         │    │ - fused_dataset_id  │
                           │ - uppercase     │    │ - total_records     │
                           │ - impute_mean   │    │ - transformations[] │
                           └─────────────────┘    │ - quality_scores    │
                                                  └─────────────────────┘
```

## Storage Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Storage Architecture                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │    PostgreSQL    │  │      Neo4j       │  │    Weaviate      │          │
│  │                  │  │                  │  │                  │          │
│  │  ┌────────────┐  │  │  ┌────────────┐  │  │  ┌────────────┐  │          │
│  │  │ datasets   │  │  │  │  :Source   │  │  │  │  Field     │  │          │
│  │  │ - id       │  │  │  │  - path    │  │  │  │ Embeddings │  │          │
│  │  │ - name     │  │  │  │  - hash    │  │  │  │            │  │          │
│  │  │ - schema   │  │  │  └────┬───────┘  │  │  │ 1536-dim   │  │          │
│  │  └────────────┘  │  │       │          │  │  │ vectors    │  │          │
│  │                  │  │       ▼          │  │  └────────────┘  │          │
│  │  ┌────────────┐  │  │  ┌────────────┐  │  │                  │          │
│  │  │ metadata   │  │  │  │  :Field    │  │  │  ┌────────────┐  │          │
│  │  │ - field    │  │  │  │  - name    │  │  │  │  Semantic  │  │          │
│  │  │ - type     │  │  │  │  - type    │──┼──┼──│  Search    │  │          │
│  │  │ - stats    │  │  │  └────┬───────┘  │  │  │  Index     │  │          │
│  │  └────────────┘  │  │       │          │  │  └────────────┘  │          │
│  │                  │  │       ▼          │  │                  │          │
│  │  ┌────────────┐  │  │  ┌────────────┐  │  └──────────────────┘          │
│  │  │ alignments │  │  │  │:Transform  │  │                                │
│  │  │ - pairs    │  │  │  │ - op       │  │                                │
│  │  │ - scores   │  │  │  │ - params   │  │                                │
│  │  └────────────┘  │  │  └────────────┘  │                                │
│  │                  │  │                  │                                │
│  └──────────────────┘  └──────────────────┘                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Event System

Agents communicate via a publish/subscribe event system:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Event System                                    │
└─────────────────────────────────────────────────────────────────────────────┘

  Event                 Publisher                    Subscribers
  ─────                 ─────────                    ───────────

  data.ingested    ◀── DataIngestorAgent      ──▶   MetadataInterpreter
                                                     ProvenanceTracker

  metadata.ready   ◀── MetadataInterpreter    ──▶   OntologyAligner
                                                     ProvenanceTracker

  alignment.ready  ◀── OntologyAligner        ──▶   FusionAgent
                                                     ProvenanceTracker

  data.fused       ◀── FusionAgent            ──▶   InsightGenerator
                                                     ProvenanceTracker

  insight.generated◀── InsightGenerator       ──▶   (External Systems)
                                                     ProvenanceTracker

  trace.updated    ◀── ProvenanceTracker      ──▶   (External Systems)
```

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Deployment Architecture                              │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌───────────────────┐
                         │   Load Balancer   │
                         │    (nginx/ALB)    │
                         └─────────┬─────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
     ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
     │  API Server 1   │  │  API Server 2   │  │  API Server N   │
     │   (uvicorn)     │  │   (uvicorn)     │  │   (uvicorn)     │
     └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
     ┌─────────────────────────────┼─────────────────────────────┐
     │                             │                             │
     ▼                             ▼                             ▼
┌─────────────┐           ┌─────────────┐           ┌─────────────┐
│ PostgreSQL  │           │   Neo4j     │           │  Weaviate   │
│  (Primary)  │           │  (Cluster)  │           │  (Cluster)  │
│      │      │           │             │           │             │
│      ▼      │           └─────────────┘           └─────────────┘
│  (Replica)  │
└─────────────┘

Docker Compose Services:
─────────────────────────
  - helixforge-api    (FastAPI application)
  - postgres          (PostgreSQL database)
  - neo4j             (Graph database)
  - weaviate          (Vector database)
```

## Security Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Security Architecture                               │
└─────────────────────────────────────────────────────────────────────────────┘

  Request Flow with Security Layers:

  Client Request
       │
       ▼
  ┌─────────────────┐
  │  TLS/HTTPS      │  ◀── Transport encryption
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Rate Limiting  │  ◀── DDoS protection
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  API Key Auth   │  ◀── Authentication
  │  (X-API-Key)    │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Input Valid.   │  ◀── Injection prevention
  │  (Pydantic)     │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  CORS Policy    │  ◀── Cross-origin protection
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Application    │
  │    Logic        │
  └─────────────────┘

  Secrets Management:
  ───────────────────
  - Environment variables (OPENAI_API_KEY, DB_PASSWORD, etc.)
  - .env files (development only)
  - Secrets manager integration (production)
```

## Performance Considerations

| Component | Optimization |
|-----------|--------------|
| Ingestion | Chunked processing, parallel file parsing |
| Embeddings | Batch API calls, caching |
| Alignment | Pre-computed similarity matrices |
| Fusion | Vectorized pandas operations |
| Insights | Sampling for large datasets |
| Storage | Parquet columnar format |

## Scalability

- **Horizontal**: Multiple API server instances behind load balancer
- **Vertical**: Configurable worker processes and batch sizes
- **Data**: Partitioning support for large datasets
- **Caching**: Redis integration for frequently accessed data
