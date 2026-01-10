# HelixForge — Cross-Dataset Insight Synthesizer
## Technical Specification v1.0

**Tagline:** From fragmented data to unified insight.

---

## 1. System Overview

### 1.1 Purpose

HelixForge transforms heterogeneous datasets with unique schemas, vocabularies, and formats into harmonized, analysis-ready data products. It functions as a Digital Language Processor (DLP) between data silos, using natural-language understanding to align meaning across sources.

### 1.2 Architecture Diagram

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
│  │                    LAYER 1: DATA INTAKE                             │   │
│  │                    data_ingestor_agent.py                           │   │
│  │         [Normalize → Hash → Store → Publish Metadata]               │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                ▼ Event: data.ingested                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    LAYER 2: METADATA UNDERSTANDING                  │   │
│  │                    metadata_interpreter_agent.py                    │   │
│  │         [Embed Fields → Infer Semantics → Label Confidence]         │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                ▼ Event: metadata.ready                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    LAYER 3: ONTOLOGY ALIGNMENT                      │   │
│  │                    ontology_alignment_agent.py                      │   │
│  │         [Similarity Match → Synonym Detection → Graph Build]        │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                ▼ Event: ontology.aligned                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    LAYER 4: FUSION                                  │   │
│  │                    fusion_agent.py                                  │   │
│  │         [Semantic Join → Transform → Impute → Merge]                │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                ▼ Event: dataset.fused                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    LAYER 5: INSIGHT GENERATION                      │   │
│  │                    insight_generator_agent.py                       │   │
│  │         [Analyze → Correlate → Visualize → Narrate]                 │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                ▼ Event: insight.generated                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    LAYER 6: PROVENANCE TRACKING                     │   │
│  │                    provenance_tracker_agent.py                      │   │
│  │         [Trace Lineage → Store Graph → Generate Reports]            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                           INFRASTRUCTURE                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Vector Store │  │ Graph Store  │  │  PostgreSQL  │  │ Agent-OS Bus │    │
│  │  (Weaviate)  │  │   (Neo4j)    │  │  (Metadata)  │  │ (NatLangChain│    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Layer Summary

| Layer | Agent | Input | Output | Event Published |
|-------|-------|-------|--------|-----------------|
| 1 | Data Ingestor | Raw files/APIs | Normalized DataFrame + metadata | `data.ingested` |
| 2 | Metadata Interpreter | Schema + sample data | Semantic field labels | `metadata.ready` |
| 3 | Ontology Alignment | Multiple metadata sets | Field mapping table + graph | `ontology.aligned` |
| 4 | Fusion | Aligned datasets | Merged dataset | `dataset.fused` |
| 5 | Insight Generator | Fused dataset | Narratives + visualizations | `insight.generated` |
| 6 | Provenance Tracker | All transformations | Lineage graph + reports | `trace.updated` |

---

## 2. Agent Specifications

### 2.1 Data Ingestor Agent

**File:** `agents/data_ingestor_agent.py`

#### 2.1.1 Responsibilities

| Function | Description |
|----------|-------------|
| `ingest_file()` | Load CSV, Parquet, JSON files |
| `ingest_sql()` | Query SQL databases via connection string |
| `ingest_rest()` | Fetch from REST API endpoints |
| `detect_encoding()` | Auto-detect file encoding (UTF-8, Latin-1, etc.) |
| `detect_delimiter()` | Auto-detect CSV delimiter |
| `infer_types()` | Infer column data types |
| `compute_hash()` | Generate content hash for deduplication |
| `publish_metadata()` | Emit `data.ingested` event |

#### 2.1.2 Input Sources

| Source Type | Handler | Config Required |
|-------------|---------|-----------------|
| CSV | `pandas.read_csv()` | delimiter, encoding |
| Parquet | `pyarrow.parquet.read_table()` | None |
| JSON | `pandas.read_json()` | orient, lines |
| SQL | `sqlalchemy.create_engine()` | connection_string, query |
| REST | `requests.get()` | url, headers, auth |

#### 2.1.3 Output Schema

```python
IngestResult = {
    "dataset_id": str,              # Unique identifier (UUID or user-provided)
    "source": str,                  # Origin path/URL
    "source_type": str,             # "csv" | "parquet" | "json" | "sql" | "rest"
    "schema": List[str],            # Column names in order
    "dtypes": Dict[str, str],       # Column name → pandas dtype
    "row_count": int,               # Total rows
    "sample_rows": int,             # Rows in preview (default: 10)
    "sample_data": List[Dict],      # First N rows as dicts
    "content_hash": str,            # SHA-256 of normalized content
    "encoding": str,                # Detected encoding
    "ingested_at": datetime,        # ISO timestamp
    "storage_path": str             # Internal DataFrame location
}
```

#### 2.1.4 Configuration

```python
IngestorConfig = {
    "max_file_size_mb": 500,
    "sample_size": 10,
    "supported_formats": ["csv", "parquet", "json", "xlsx"],
    "encoding_detection_sample_bytes": 10000,
    "sql_timeout_seconds": 300,
    "rest_timeout_seconds": 60,
    "temp_storage_path": "./data/raw/"
}
```

#### 2.1.5 Dependencies

```
pandas>=2.0
pyarrow>=14.0
sqlalchemy>=2.0
requests>=2.31
chardet>=5.0  # encoding detection
openpyxl>=3.1  # Excel support
```

---

### 2.2 Metadata Interpreter Agent

**File:** `agents/metadata_interpreter_agent.py`

#### 2.2.1 Responsibilities

| Function | Description |
|----------|-------------|
| `embed_field_names()` | Generate embeddings for column names |
| `embed_sample_values()` | Generate embeddings from sample data |
| `infer_semantics()` | LLM-based semantic label inference |
| `compute_confidence()` | Score semantic label certainty |
| `generate_description()` | Natural language field description |

#### 2.2.2 Semantic Inference Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Field Name  │───▶│  Embedding  │───▶│  LLM Infer  │───▶│  Semantic   │
│ + Samples   │    │  Generation │    │  + Context  │    │   Label     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │                  │
                          ▼                  ▼
                   ┌─────────────┐    ┌─────────────┐
                   │   Vector    │    │ Confidence  │
                   │   Store     │    │   Score     │
                   └─────────────┘    └─────────────┘
```

#### 2.2.3 Output Schema

```python
FieldMetadata = {
    "dataset_id": str,
    "field_name": str,              # Original column name
    "semantic_label": str,          # Inferred meaning
    "description": str,             # Natural language description
    "data_type": str,               # "string" | "integer" | "float" | "datetime" | "boolean"
    "semantic_type": str,           # "identifier" | "metric" | "category" | "timestamp" | "text"
    "embedding": List[float],       # Vector representation (1536-dim for text-embedding-3-large)
    "sample_values": List[Any],     # Representative values
    "null_ratio": float,            # Proportion of nulls
    "unique_ratio": float,          # Cardinality / row_count
    "confidence": float,            # 0.0 - 1.0
    "inferred_at": datetime
}

DatasetMetadata = {
    "dataset_id": str,
    "fields": List[FieldMetadata],
    "dataset_description": str,     # LLM-generated summary
    "domain_tags": List[str],       # Inferred domain (e.g., "healthcare", "genomics")
    "ready_at": datetime
}
```

#### 2.2.4 LLM Prompt Template

```
Given the following field information from a dataset:

Field Name: {field_name}
Data Type: {dtype}
Sample Values: {sample_values}
Null Ratio: {null_ratio}
Unique Ratio: {unique_ratio}

Infer the semantic meaning of this field. Respond with:
1. semantic_label: A concise label (2-4 words)
2. description: A one-sentence explanation
3. semantic_type: One of [identifier, metric, category, timestamp, text, unknown]
4. confidence: Your certainty (0.0-1.0)

Respond in JSON format only.
```

#### 2.2.5 Configuration

```python
InterpreterConfig = {
    "embedding_model": "text-embedding-3-large",
    "embedding_dimensions": 1536,
    "llm_model": "gpt-4o",
    "llm_temperature": 0.2,
    "max_sample_values": 20,
    "min_confidence_threshold": 0.5,
    "batch_size": 50  # Fields per LLM call
}
```

#### 2.2.6 Dependencies

```
openai>=1.0
numpy>=1.24
spacy>=3.7
tiktoken>=0.5
```

---

### 2.3 Ontology Alignment Agent

**File:** `agents/ontology_alignment_agent.py`

#### 2.3.1 Responsibilities

| Function | Description |
|----------|-------------|
| `compute_similarity()` | Cosine similarity between field embeddings |
| `detect_synonyms()` | Identify equivalent fields across datasets |
| `detect_relationships()` | Find hierarchical/associative relationships |
| `build_semantic_graph()` | Construct unified ontology graph |
| `generate_mapping_table()` | Output harmonized field mappings |

#### 2.3.2 Alignment Algorithm

```
FUNCTION align_datasets(dataset_a_metadata, dataset_b_metadata):
    alignments = []
    
    FOR field_a IN dataset_a_metadata.fields:
        FOR field_b IN dataset_b_metadata.fields:
            similarity = cosine_similarity(field_a.embedding, field_b.embedding)
            
            IF similarity >= confidence_threshold:
                alignment = {
                    "source_dataset": dataset_a_metadata.dataset_id,
                    "source_field": field_a.field_name,
                    "target_dataset": dataset_b_metadata.dataset_id,
                    "target_field": field_b.field_name,
                    "similarity": similarity,
                    "alignment_type": classify_alignment(field_a, field_b, similarity)
                }
                alignments.append(alignment)
    
    # Resolve conflicts (one field → multiple matches)
    alignments = resolve_conflicts(alignments)
    
    # Build graph
    graph = build_ontology_graph(alignments)
    
    RETURN alignments, graph
```

#### 2.3.3 Alignment Types

| Type | Condition | Description |
|------|-----------|-------------|
| `exact` | similarity ≥ 0.98 | Fields are semantically identical |
| `synonym` | 0.90 ≤ similarity < 0.98 | Different names, same meaning |
| `related` | 0.80 ≤ similarity < 0.90 | Conceptually related fields |
| `partial` | 0.70 ≤ similarity < 0.80 | Overlapping meaning |
| `none` | similarity < 0.70 | No meaningful relationship |

#### 2.3.4 Output Schema

```python
FieldAlignment = {
    "alignment_id": str,            # UUID
    "source_dataset": str,
    "source_field": str,
    "target_dataset": str,
    "target_field": str,
    "similarity": float,            # 0.0 - 1.0
    "alignment_type": str,          # exact | synonym | related | partial
    "transformation_hint": str,     # Optional: "unit_conversion", "type_cast", etc.
    "validated": bool,              # Human-reviewed flag
    "created_at": datetime
}

AlignmentResult = {
    "alignment_job_id": str,
    "datasets_aligned": List[str],
    "alignments": List[FieldAlignment],
    "unmatched_fields": List[str],
    "ontology_graph_uri": str,      # Neo4j/ArangoDB reference
    "completed_at": datetime
}
```

#### 2.3.5 Semantic Graph Schema (Neo4j)

```cypher
// Nodes
(:Field {
    id: string,
    dataset_id: string,
    name: string,
    semantic_label: string,
    embedding: list<float>
})

(:Dataset {
    id: string,
    name: string,
    domain: string
})

// Relationships
(:Field)-[:BELONGS_TO]->(:Dataset)
(:Field)-[:ALIGNS_WITH {similarity: float, type: string}]->(:Field)
(:Field)-[:DERIVED_FROM]->(:Field)
```

#### 2.3.6 Configuration

```python
AlignmentConfig = {
    "similarity_threshold": 0.80,
    "exact_match_threshold": 0.98,
    "synonym_threshold": 0.90,
    "max_alignments_per_field": 3,
    "vector_store": "weaviate",
    "graph_store": "neo4j",
    "graph_uri": "neo4j://localhost:7687",
    "conflict_resolution": "highest_similarity"  # or "manual"
}
```

#### 2.3.7 Dependencies

```
weaviate-client>=4.0
neo4j>=5.0
networkx>=3.0
scipy>=1.11
numpy>=1.24
```

---

### 2.4 Fusion Agent

**File:** `agents/fusion_agent.py`

#### 2.4.1 Responsibilities

| Function | Description |
|----------|-------------|
| `semantic_join()` | Join datasets on aligned fields (not just keys) |
| `transform_values()` | Apply unit conversions, type casts |
| `impute_missing()` | Handle nulls with statistical/ML methods |
| `probabilistic_match()` | Fuzzy record matching for non-exact joins |
| `merge_records()` | Combine aligned records into unified rows |
| `log_transformations()` | Record all operations for provenance |

#### 2.4.2 Fusion Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Dataset A  │    │   Dataset B  │    │   Dataset N  │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       └─────────┬─────────┴─────────┬─────────┘
                 ▼                   ▼
        ┌─────────────────────────────────────┐
        │         ALIGNMENT MAPPING           │
        │   (from Ontology Alignment Agent)   │
        └─────────────────┬───────────────────┘
                          ▼
        ┌─────────────────────────────────────┐
        │         TRANSFORMATION LAYER        │
        │  • Unit normalization               │
        │  • Type casting                     │
        │  • Value standardization            │
        └─────────────────┬───────────────────┘
                          ▼
        ┌─────────────────────────────────────┐
        │         JOIN STRATEGY SELECTOR      │
        │  • Exact key match                  │
        │  • Semantic similarity join         │
        │  • Probabilistic record matching    │
        └─────────────────┬───────────────────┘
                          ▼
        ┌─────────────────────────────────────┐
        │         IMPUTATION ENGINE           │
        │  • Mean/median/mode fill            │
        │  • KNN imputation                   │
        │  • Model-based prediction           │
        └─────────────────┬───────────────────┘
                          ▼
        ┌─────────────────────────────────────┐
        │         MERGED DATASET              │
        │    + Transformation Log             │
        └─────────────────────────────────────┘
```

#### 2.4.3 Join Strategies

| Strategy | Use Case | Method |
|----------|----------|--------|
| `exact_key` | Matching identifiers | SQL-style JOIN on key columns |
| `semantic_similarity` | No shared keys | Join on embedding similarity ≥ threshold |
| `probabilistic` | Fuzzy matching | Weighted combination of field similarities |
| `temporal` | Time-series data | Align by timestamp within tolerance window |

#### 2.4.4 Transformation Templates

```python
TransformationTemplate = {
    "template_id": str,
    "name": str,                    # e.g., "celsius_to_fahrenheit"
    "source_unit": str,
    "target_unit": str,
    "formula": str,                 # Python expression: "value * 9/5 + 32"
    "applicable_types": List[str]   # ["float", "integer"]
}

# Built-in templates
BUILTIN_TRANSFORMS = [
    {"template_id": "c_to_f", "name": "celsius_to_fahrenheit", "formula": "value * 9/5 + 32"},
    {"template_id": "kg_to_lb", "name": "kilograms_to_pounds", "formula": "value * 2.20462"},
    {"template_id": "m_to_ft", "name": "meters_to_feet", "formula": "value * 3.28084"},
    {"template_id": "days_to_months", "name": "days_to_months", "formula": "value / 30.44"},
    {"template_id": "normalize_0_1", "name": "min_max_normalize", "formula": "(value - min) / (max - min)"},
    {"template_id": "z_score", "name": "z_score_normalize", "formula": "(value - mean) / std"}
]
```

#### 2.4.5 Output Schema

```python
FusionResult = {
    "fused_dataset_id": str,
    "source_datasets": List[str],
    "record_count": int,
    "field_count": int,
    "merged_fields": List[str],
    "join_strategy": str,
    "transformations_applied": List[TransformationLog],
    "imputation_summary": ImputationSummary,
    "storage_path": str,
    "fused_at": datetime
}

TransformationLog = {
    "field": str,
    "operation": str,               # "unit_conversion" | "type_cast" | "rename" | "impute"
    "template_id": str,
    "source_value_sample": Any,
    "target_value_sample": Any,
    "records_affected": int
}

ImputationSummary = {
    "total_nulls_filled": int,
    "fields_imputed": Dict[str, int],  # field → count
    "method_used": str,             # "mean" | "knn" | "model"
    "imputation_quality_score": float
}
```

#### 2.4.6 Configuration

```python
FusionConfig = {
    "default_join_strategy": "semantic_similarity",
    "similarity_join_threshold": 0.85,
    "probabilistic_match_threshold": 0.75,
    "temporal_tolerance_seconds": 3600,
    "imputation_method": "knn",
    "knn_neighbors": 5,
    "max_null_ratio_for_inclusion": 0.5,
    "output_format": "parquet",
    "output_path": "./data/fused/"
}
```

#### 2.4.7 Dependencies

```
pandas>=2.0
pyarrow>=14.0
fuzzywuzzy>=0.18
python-Levenshtein>=0.21
scikit-learn>=1.3
numpy>=1.24
```

---

### 2.5 Insight Generator Agent

**File:** `agents/insight_generator_agent.py`

#### 2.5.1 Responsibilities

| Function | Description |
|----------|-------------|
| `compute_statistics()` | Descriptive stats for all fields |
| `detect_correlations()` | Pearson/Spearman correlation matrix |
| `detect_outliers()` | IQR/Z-score outlier detection |
| `cluster_records()` | K-means/DBSCAN clustering |
| `generate_visualizations()` | Charts and plots |
| `generate_narrative()` | LLM-powered natural language summary |
| `export_report()` | PDF/HTML/Notebook output |

#### 2.5.2 Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      FUSED DATASET                              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   STATISTICAL   │  │   CORRELATION   │  │    OUTLIER      │
│    ANALYSIS     │  │    ANALYSIS     │  │   DETECTION     │
│  • mean, std    │  │  • Pearson r    │  │  • IQR method   │
│  • distributions│  │  • Spearman ρ   │  │  • Z-score      │
│  • quartiles    │  │  • p-values     │  │  • Isolation    │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │    CLUSTERING   │
                    │  • K-means      │
                    │  • DBSCAN       │
                    │  • Hierarchical │
                    └────────┬────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ VISUALIZATIONS  │  │    NARRATIVE    │  │     EXPORT      │
│  • Correlation  │  │    GENERATION   │  │  • PDF report   │
│    matrix       │  │  • LLM summary  │  │  • HTML dash    │
│  • Scatter plots│  │  • Key findings │  │  • Jupyter NB   │
│  • Distributions│  │  • Anomalies    │  │  • JSON data    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

#### 2.5.3 Output Schema

```python
InsightResult = {
    "insight_id": str,
    "fused_dataset_id": str,
    "generated_at": datetime,
    "statistics": DatasetStatistics,
    "correlations": CorrelationMatrix,
    "outliers": OutlierReport,
    "clusters": ClusteringResult,
    "narrative_summary": str,
    "key_findings": List[Finding],
    "visualizations": List[Visualization],
    "export_paths": Dict[str, str]
}

DatasetStatistics = {
    "record_count": int,
    "field_count": int,
    "field_stats": Dict[str, FieldStatistics]
}

FieldStatistics = {
    "mean": float,
    "std": float,
    "min": float,
    "max": float,
    "median": float,
    "q1": float,
    "q3": float,
    "null_count": int,
    "unique_count": int,
    "distribution_type": str        # "normal" | "skewed" | "bimodal" | "uniform"
}

CorrelationMatrix = {
    "method": str,                  # "pearson" | "spearman"
    "correlations": List[CorrelationPair],
    "significant_pairs": List[CorrelationPair]  # p < 0.05
}

CorrelationPair = {
    "field_a": str,
    "field_b": str,
    "coefficient": float,           # -1.0 to 1.0
    "p_value": float,
    "significant": bool
}

Finding = {
    "finding_id": str,
    "type": str,                    # "correlation" | "outlier" | "cluster" | "trend"
    "severity": str,                # "high" | "medium" | "low"
    "description": str,
    "supporting_data": Dict,
    "visualization_ref": str
}

Visualization = {
    "viz_id": str,
    "type": str,                    # "correlation_matrix" | "scatter" | "histogram" | "boxplot"
    "title": str,
    "file_path": str,
    "format": str                   # "png" | "svg" | "html"
}
```

#### 2.5.4 Narrative Generation Prompt

```
You are a data analyst summarizing findings from a fused dataset.

Dataset: {dataset_description}
Record Count: {record_count}
Fields: {field_list}

Key Statistics:
{statistics_summary}

Significant Correlations:
{correlation_summary}

Detected Outliers:
{outlier_summary}

Clusters Found:
{cluster_summary}

Write a clear, professional narrative summary (3-5 paragraphs) highlighting:
1. The most important correlations and their implications
2. Notable outliers or anomalies
3. Any patterns revealed by clustering
4. Recommended next steps for analysis

Use specific numbers and field names. Avoid jargon.
```

#### 2.5.5 Configuration

```python
InsightConfig = {
    "llm_model": "gpt-4o",
    "correlation_method": "pearson",
    "correlation_significance_threshold": 0.05,
    "outlier_method": "iqr",
    "outlier_iqr_multiplier": 1.5,
    "clustering_algorithm": "kmeans",
    "clustering_k_range": [2, 10],
    "visualization_format": "plotly",
    "visualization_dpi": 150,
    "export_formats": ["html", "pdf"],
    "output_path": "./outputs/insights/"
}
```

#### 2.5.6 Dependencies

```
pandas>=2.0
numpy>=1.24
scipy>=1.11
scikit-learn>=1.3
matplotlib>=3.8
plotly>=5.18
seaborn>=0.13
openai>=1.0
jinja2>=3.1      # Report templating
weasyprint>=60   # PDF generation
```

---

### 2.6 Provenance Tracker Agent

**File:** `agents/provenance_tracker_agent.py`

#### 2.6.1 Responsibilities

| Function | Description |
|----------|-------------|
| `record_ingestion()` | Log dataset source and initial state |
| `record_transformation()` | Log every field transformation |
| `record_alignment()` | Log field mapping decisions |
| `record_fusion()` | Log join operations and record merges |
| `build_lineage_graph()` | Construct full provenance DAG |
| `query_lineage()` | Trace any field back to origins |
| `generate_provenance_report()` | Human-readable audit trail |

#### 2.6.2 Lineage Graph Structure

```
                    ┌───────────────────────┐
                    │   FUSED FIELD:        │
                    │   survival_months     │
                    └───────────┬───────────┘
                                │
            ┌───────────────────┴───────────────────┐
            │                                       │
   ┌────────▼────────┐                    ┌────────▼────────┐
   │ TRANSFORMATION: │                    │  ALIGNMENT:     │
   │ days_to_months  │                    │  synonym_match  │
   └────────┬────────┘                    └────────┬────────┘
            │                                       │
   ┌────────▼────────┐                    ┌────────▼────────┐
   │ SOURCE FIELD:   │                    │ SOURCE FIELD:   │
   │ survival_days   │                    │ time_to_death   │
   │ dataset: A      │                    │ dataset: B      │
   └────────┬────────┘                    └────────┬────────┘
            │                                       │
   ┌────────▼────────┐                    ┌────────▼────────┐
   │ RAW SOURCE:     │                    │ RAW SOURCE:     │
   │ clinical.csv    │                    │ registry.parquet│
   │ col: 12         │                    │ col: 8          │
   └─────────────────┘                    └─────────────────┘
```

#### 2.6.3 Output Schema

```python
ProvenanceTrace = {
    "trace_id": str,
    "field": str,                       # Current field name
    "fused_dataset_id": str,
    "lineage_depth": int,               # Levels of transformation
    "origins": List[FieldOrigin],
    "transformations": List[TransformationRecord],
    "confidence": float,                # Aggregate confidence through chain
    "traced_at": datetime
}

FieldOrigin = {
    "source_file": str,
    "source_column": str,
    "source_column_index": int,
    "dataset_id": str,
    "ingested_at": datetime,
    "content_hash": str
}

TransformationRecord = {
    "step_id": str,
    "operation": str,                   # "ingest" | "align" | "transform" | "impute" | "fuse"
    "input_fields": List[str],
    "output_field": str,
    "parameters": Dict,
    "agent": str,                       # Which agent performed it
    "timestamp": datetime,
    "confidence_delta": float           # Impact on confidence score
}

ProvenanceReport = {
    "report_id": str,
    "fused_dataset_id": str,
    "total_fields": int,
    "fields_with_complete_provenance": int,
    "coverage_percentage": float,
    "traces": List[ProvenanceTrace],
    "generated_at": datetime,
    "format": str                       # "json" | "json-ld" | "html"
}
```

#### 2.6.4 Graph Store Schema (Neo4j)

```cypher
// Provenance Nodes
(:RawSource {
    file_path: string,
    content_hash: string,
    ingested_at: datetime
})

(:SourceField {
    name: string,
    column_index: int,
    dataset_id: string
})

(:TransformedField {
    name: string,
    operation: string,
    confidence: float
})

(:FusedField {
    name: string,
    fused_dataset_id: string
})

// Provenance Relationships
(:SourceField)-[:EXTRACTED_FROM]->(:RawSource)
(:TransformedField)-[:DERIVED_FROM {operation: string, params: map}]->(:SourceField)
(:TransformedField)-[:DERIVED_FROM]->(:TransformedField)
(:FusedField)-[:MERGED_FROM {strategy: string}]->(:TransformedField)
(:FusedField)-[:MERGED_FROM]->(:SourceField)
```

#### 2.6.5 Configuration

```python
ProvenanceConfig = {
    "graph_store": "neo4j",
    "graph_uri": "neo4j://localhost:7687",
    "graph_user": "neo4j",
    "graph_password": "***",
    "json_ld_context": "https://www.w3.org/ns/prov",
    "report_format": "html",
    "report_output_path": "./outputs/provenance/",
    "confidence_decay_per_step": 0.02
}
```

#### 2.6.6 Dependencies

```
neo4j>=5.0
pyld>=2.0           # JSON-LD support
networkx>=3.0
jinja2>=3.1
```

---

## 3. Utility Modules

### 3.1 Embeddings Utility

**File:** `utils/embeddings.py`

```python
# Core functions
def get_embedding(text: str, model: str = "text-embedding-3-large") -> List[float]
def batch_embed(texts: List[str], model: str, batch_size: int = 100) -> List[List[float]]
def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float
def find_similar(query_vec: List[float], corpus: List[List[float]], top_k: int) -> List[Tuple[int, float]]
```

### 3.2 Similarity Utility

**File:** `utils/similarity.py`

```python
# Core functions
def string_similarity(a: str, b: str, method: str = "levenshtein") -> float
def semantic_similarity(a: str, b: str, embeddings: Dict) -> float
def record_similarity(row_a: Dict, row_b: Dict, field_weights: Dict) -> float
def find_best_match(record: Dict, candidates: List[Dict], threshold: float) -> Optional[Dict]
```

### 3.3 Logging Utility

**File:** `utils/logging.py`

```python
# Structured logging with correlation IDs
def get_logger(agent_name: str) -> Logger
def log_event(event_type: str, payload: Dict, correlation_id: str) -> None
def log_metric(metric_name: str, value: float, tags: Dict) -> None
```

---

## 4. API Specification

### 4.1 REST Endpoints

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| POST | `/datasets/upload` | Upload/register dataset | `multipart/form-data` or JSON | `IngestResult` |
| GET | `/datasets/{id}` | Get dataset metadata | - | `DatasetMetadata` |
| GET | `/datasets/{id}/sample` | Get sample rows | `?rows=N` | `List[Dict]` |
| POST | `/align` | Trigger alignment job | `AlignmentRequest` | `AlignmentResult` |
| GET | `/alignments/{job_id}` | Get alignment status/result | - | `AlignmentResult` |
| POST | `/fuse` | Run fusion pipeline | `FusionRequest` | `FusionResult` |
| GET | `/fused/{id}` | Get fused dataset info | - | `FusionResult` |
| GET | `/fused/{id}/download` | Download fused dataset | `?format=parquet|csv` | File |
| POST | `/insights/generate` | Generate insights | `InsightRequest` | `InsightResult` |
| GET | `/insights/{id}` | Get insight report | - | `InsightResult` |
| GET | `/trace/{dataset_id}/{field}` | Get field provenance | - | `ProvenanceTrace` |
| GET | `/trace/{dataset_id}/report` | Full provenance report | - | `ProvenanceReport` |
| GET | `/health` | System health check | - | `HealthStatus` |

### 4.2 Request Schemas

```python
AlignmentRequest = {
    "dataset_ids": List[str],           # 2+ datasets to align
    "confidence_threshold": float,       # Optional, default from config
    "include_partial_matches": bool      # Default: True
}

FusionRequest = {
    "alignment_job_id": str,            # Reference to completed alignment
    "join_strategy": str,               # Optional override
    "imputation_method": str,           # Optional override
    "output_format": str                # "parquet" | "csv"
}

InsightRequest = {
    "fused_dataset_id": str,
    "analysis_types": List[str],        # ["correlations", "outliers", "clusters"]
    "generate_visualizations": bool,
    "generate_narrative": bool,
    "export_formats": List[str]         # ["html", "pdf", "json"]
}
```

### 4.3 Event Bus Topics (Agent-OS / NatLangChain)

| Topic | Publisher | Payload | Subscribers |
|-------|-----------|---------|-------------|
| `data.ingested` | Data Ingestor | `IngestResult` | Metadata Interpreter |
| `metadata.ready` | Metadata Interpreter | `DatasetMetadata` | Ontology Alignment |
| `ontology.aligned` | Ontology Alignment | `AlignmentResult` | Fusion Agent |
| `dataset.fused` | Fusion Agent | `FusionResult` | Insight Generator, Provenance |
| `insight.generated` | Insight Generator | `InsightResult` | Provenance, API |
| `trace.updated` | Provenance Tracker | `ProvenanceTrace` | API, Dashboard |

---

## 5. Configuration

### 5.1 Master Configuration File

**File:** `config.yaml`

```yaml
# HelixForge Configuration

# LLM Settings
llm:
  provider: "openai"
  model: "gpt-4o"
  embedding_model: "text-embedding-3-large"
  temperature: 0.2
  max_tokens: 4096

# Vector Store
vector_store:
  provider: "weaviate"
  url: "http://localhost:8080"
  api_key: "${WEAVIATE_API_KEY}"

# Graph Store (Ontology + Provenance)
graph_store:
  provider: "neo4j"
  uri: "neo4j://localhost:7687"
  user: "neo4j"
  password: "${NEO4J_PASSWORD}"

# Relational Database (Metadata + Jobs)
database:
  url: "postgresql://helixforge:${DB_PASSWORD}@localhost:5432/helixforge"
  pool_size: 10

# Processing Settings
processing:
  confidence_threshold: 0.80
  max_file_size_mb: 500
  temp_storage_path: "./data/temp/"
  
# Fusion Settings
fusion:
  default_join_strategy: "semantic_similarity"
  similarity_threshold: 0.85
  imputation_method: "knn"
  
# Insight Settings
insights:
  correlation_significance: 0.05
  outlier_method: "iqr"
  visualization_format: "plotly"
  
# Output Settings
output:
  artifact_dir: "./outputs/"
  report_format: "html"
  
# API Settings
api:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["http://localhost:3000"]
  
# Logging
logging:
  level: "INFO"
  format: "json"
  output: "stdout"
```

---

## 6. Directory Structure

```
helixforge/
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py                 # Abstract base class
│   ├── data_ingestor_agent.py
│   ├── metadata_interpreter_agent.py
│   ├── ontology_alignment_agent.py
│   ├── fusion_agent.py
│   ├── insight_generator_agent.py
│   └── provenance_tracker_agent.py
│
├── api/
│   ├── __init__.py
│   ├── server.py                     # FastAPI application
│   ├── routes/
│   │   ├── datasets.py
│   │   ├── alignment.py
│   │   ├── fusion.py
│   │   ├── insights.py
│   │   └── provenance.py
│   └── middleware/
│       ├── auth.py
│       └── logging.py
│
├── utils/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── similarity.py
│   ├── logging.py
│   └── validation.py
│
├── models/
│   ├── __init__.py
│   ├── schemas.py                    # Pydantic models
│   └── database.py                   # SQLAlchemy models
│
├── templates/
│   ├── prompts/
│   │   ├── semantic_inference.txt
│   │   ├── narrative_generation.txt
│   │   └── alignment_validation.txt
│   └── visualizations/
│       ├── correlation_matrix.html
│       ├── scatter_template.html
│       └── report_template.html
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_ingestor.py
│   ├── test_interpreter.py
│   ├── test_alignment.py
│   ├── test_fusion.py
│   ├── test_insights.py
│   └── test_provenance.py
│
├── data/
│   ├── raw/                          # Ingested files (temp)
│   ├── fused/                        # Merged datasets
│   └── samples/                      # Test datasets
│
├── outputs/
│   ├── insights/                     # Generated reports
│   ├── provenance/                   # Lineage reports
│   └── visualizations/               # Charts and plots
│
├── config.yaml
├── requirements.txt
├── Dockerfile
├── docker-compose.yaml
└── README.md
```

---

## 7. Testing Requirements

### 7.1 Unit Tests

| Test Suite | Coverage Target | Key Assertions |
|------------|-----------------|----------------|
| `test_ingestor.py` | Parser accuracy ≥ 99% | Correct type inference, encoding detection |
| `test_interpreter.py` | Semantic labeling | Confidence scores valid, embeddings correct dims |
| `test_alignment.py` | Precision ≥ 0.90 | Synonyms detected, no false positives above threshold |
| `test_fusion.py` | Integrity ≥ 0.95 | No data loss, transformations reversible |
| `test_insights.py` | Statistical validity | Correlations match scipy, p-values correct |
| `test_provenance.py` | Coverage = 100% | Every field traceable to source |

### 7.2 Integration Tests

| Scenario | Datasets | Expected Outcome |
|----------|----------|------------------|
| Genomics + Clinical | 2 CSVs, ~5000 rows each | Fused dataset with survival correlations |
| Multi-format | CSV + Parquet + JSON | Unified schema, no format artifacts |
| High cardinality | 100+ columns, 1M rows | Completes in < 10 min |
| Missing data | 30% nulls | Imputation applied, documented |

### 7.3 Evaluation Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Alignment Precision | ≥ 0.90 | Correct matches / total matches |
| Alignment Recall | ≥ 0.85 | Correct matches / possible matches |
| Fusion Integrity | ≥ 0.95 | Records merged / total eligible |
| Insight Coherence | ≥ 0.85 | Human evaluation (1-5 scale) |
| Provenance Coverage | 100% | Fields with complete trace / total |
| End-to-end Latency | < 5 min | For 10K records, 50 fields |

---

## 8. Deployment

### 8.1 Container Configuration

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 8.2 Docker Compose (Full Stack)

```yaml
version: "3.9"

services:
  helixforge:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://helixforge:password@postgres:5432/helixforge
      - NEO4J_URI=neo4j://neo4j:7687
      - WEAVIATE_URL=http://weaviate:8080
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - postgres
      - neo4j
      - weaviate
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs

  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: helixforge
      POSTGRES_PASSWORD: password
      POSTGRES_DB: helixforge
    volumes:
      - postgres_data:/var/lib/postgresql/data

  neo4j:
    image: neo4j:5
    environment:
      NEO4J_AUTH: neo4j/password
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data

  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
      PERSISTENCE_DATA_PATH: "/var/lib/weaviate"
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  postgres_data:
  neo4j_data:
  weaviate_data:
```

### 8.3 Deployment Modes

| Mode | Configuration | Use Case |
|------|---------------|----------|
| **Microservice** | Docker Compose | Development, small-scale |
| **Serverless** | AWS Lambda + S3 + Aurora | Lightweight, burst workloads |
| **Kubernetes** | Helm chart + HPA | Production, high availability |

---

## 9. Dependencies Summary

### 9.1 Python Requirements

```
# Core
python>=3.10
pandas>=2.0
numpy>=1.24
pyarrow>=14.0

# API
fastapi>=0.109
uvicorn>=0.27
pydantic>=2.5

# Database
sqlalchemy>=2.0
psycopg2-binary>=2.9
neo4j>=5.0

# Vector Store
weaviate-client>=4.0

# LLM / Embeddings
openai>=1.0
tiktoken>=0.5

# NLP
spacy>=3.7

# ML / Stats
scikit-learn>=1.3
scipy>=1.11

# Visualization
matplotlib>=3.8
plotly>=5.18
seaborn>=0.13

# Utilities
requests>=2.31
chardet>=5.0
fuzzywuzzy>=0.18
python-Levenshtein>=0.21
networkx>=3.0
jinja2>=3.1
pyyaml>=6.0

# Testing
pytest>=7.4
pytest-asyncio>=0.23
httpx>=0.26
```

---

## 10. Future Extensions

| Extension | Description | Priority |
|-----------|-------------|----------|
| **Multimodal Alignment** | Integrate image + text (radiology + clinical notes) | High |
| **Interactive Query Agent** | Natural language questions on fused data | High |
| **Auto-Ontology Growth** | Dynamically extend ontology as new fields appear | Medium |
| **Data-to-LLM Fine-Tuning** | Export harmonized data for model training | Medium |
| **Real-time Streaming** | Kafka/Kinesis integration for live data fusion | Low |
| **Federated Mode** | Cross-organization alignment without data sharing | Low |

---

*Specification complete. Ready for implementation.*
