# HelixForge User Stories & Acceptance Criteria

## Overview

This document defines user stories and acceptance criteria for HelixForge, organized by the six core agent layers.

---

## 1. Data Ingestion

### US-1.1: Upload CSV Dataset
**As a** data analyst
**I want to** upload a CSV file to HelixForge
**So that** I can include it in cross-dataset analysis

**Acceptance Criteria:**
- [ ] System accepts CSV files up to 500MB
- [ ] System auto-detects file encoding (UTF-8, Latin-1, etc.)
- [ ] System auto-detects delimiter (comma, tab, semicolon, pipe)
- [ ] System infers column data types correctly
- [ ] System generates a unique dataset_id
- [ ] System computes SHA-256 content hash for deduplication
- [ ] System returns IngestResult with schema, dtypes, row_count, and sample_data
- [ ] System publishes `data.ingested` event

### US-1.2: Upload Parquet Dataset
**As a** data engineer
**I want to** upload Parquet files
**So that** I can leverage columnar storage format benefits

**Acceptance Criteria:**
- [ ] System reads Parquet files using PyArrow
- [ ] System preserves column metadata and types
- [ ] System handles nested/complex types appropriately

### US-1.3: Upload JSON Dataset
**As a** API developer
**I want to** upload JSON data
**So that** I can integrate API response data into analysis

**Acceptance Criteria:**
- [ ] System handles both JSON arrays and JSON Lines format
- [ ] System correctly flattens nested structures
- [ ] System infers types from JSON values

### US-1.4: Connect to SQL Database
**As a** database administrator
**I want to** connect HelixForge to a SQL database
**So that** I can analyze data directly from production systems

**Acceptance Criteria:**
- [ ] System accepts connection strings for PostgreSQL, MySQL, SQLite
- [ ] System executes provided SQL query with timeout (300s default)
- [ ] System handles connection errors gracefully
- [ ] Credentials are not logged or stored in plaintext

### US-1.5: Fetch from REST API
**As a** integration specialist
**I want to** ingest data from REST APIs
**So that** I can include external data sources in analysis

**Acceptance Criteria:**
- [ ] System accepts URL, headers, and authentication config
- [ ] System handles pagination if configured
- [ ] System respects timeout settings (60s default)
- [ ] System handles HTTP errors with appropriate messages

---

## 2. Metadata Interpretation

### US-2.1: Semantic Field Labeling
**As a** data analyst
**I want** HelixForge to automatically understand what each column means
**So that** I don't have to manually map field meanings

**Acceptance Criteria:**
- [ ] System generates embeddings for each field name
- [ ] System analyzes sample values to infer meaning
- [ ] System assigns semantic_label (2-4 word description)
- [ ] System assigns semantic_type (identifier, metric, category, timestamp, text)
- [ ] System provides confidence score (0.0-1.0) for each inference
- [ ] Fields with confidence < 0.5 are flagged for review

### US-2.2: Dataset Description Generation
**As a** data catalog curator
**I want** automatic dataset descriptions
**So that** teams can discover relevant datasets easily

**Acceptance Criteria:**
- [ ] System generates natural language dataset description
- [ ] System infers domain tags (e.g., "healthcare", "genomics", "finance")
- [ ] System publishes `metadata.ready` event when complete

### US-2.3: Field Statistics Computation
**As a** data quality analyst
**I want** to see null ratios and cardinality for each field
**So that** I can assess data quality before analysis

**Acceptance Criteria:**
- [ ] System computes null_ratio for each field
- [ ] System computes unique_ratio (cardinality / row_count)
- [ ] Statistics are included in FieldMetadata output

---

## 3. Ontology Alignment

### US-3.1: Cross-Dataset Field Matching
**As a** data integration specialist
**I want** HelixForge to find matching fields across datasets
**So that** I can merge data from different sources

**Acceptance Criteria:**
- [ ] System computes cosine similarity between field embeddings
- [ ] System identifies exact matches (similarity >= 0.98)
- [ ] System identifies synonyms (0.90 <= similarity < 0.98)
- [ ] System identifies related fields (0.80 <= similarity < 0.90)
- [ ] System resolves conflicts when one field matches multiple targets

### US-3.2: Semantic Graph Construction
**As a** ontology manager
**I want** field relationships stored in a graph database
**So that** I can visualize and query the data model

**Acceptance Criteria:**
- [ ] System creates Field nodes in Neo4j
- [ ] System creates ALIGNS_WITH relationships with similarity scores
- [ ] System creates BELONGS_TO relationships to Dataset nodes
- [ ] Graph is queryable via Cypher

### US-3.3: Alignment Validation
**As a** data steward
**I want** to review and validate automated alignments
**So that** I can ensure accuracy before fusion

**Acceptance Criteria:**
- [ ] System provides alignment review endpoint
- [ ] Alignments can be marked as validated=true/false
- [ ] System tracks who validated each alignment

---

## 4. Data Fusion

### US-4.1: Semantic Join
**As a** data scientist
**I want** to join datasets on semantically equivalent fields
**So that** I can combine data without exact key matches

**Acceptance Criteria:**
- [ ] System joins datasets using alignment mappings
- [ ] System supports exact_key, semantic_similarity, and probabilistic join strategies
- [ ] System logs all join operations for provenance

### US-4.2: Value Transformation
**As a** data engineer
**I want** automatic unit conversions during fusion
**So that** merged data is consistent

**Acceptance Criteria:**
- [ ] System applies built-in transformations (Celsius→Fahrenheit, kg→lb, etc.)
- [ ] System supports custom transformation formulas
- [ ] System logs source and target values for each transformation

### US-4.3: Missing Value Imputation
**As a** ML engineer
**I want** intelligent handling of missing values
**So that** merged datasets are analysis-ready

**Acceptance Criteria:**
- [ ] System supports mean/median/mode imputation
- [ ] System supports KNN imputation
- [ ] System excludes fields with null_ratio > 0.5 by default
- [ ] System reports imputation summary (fields imputed, counts, method)

### US-4.4: Fused Dataset Export
**As a** data consumer
**I want** to download the fused dataset
**So that** I can use it in external tools

**Acceptance Criteria:**
- [ ] System exports fused data as Parquet (default) or CSV
- [ ] System publishes `dataset.fused` event
- [ ] Download endpoint returns file with correct MIME type

---

## 5. Insight Generation

### US-5.1: Statistical Analysis
**As a** analyst
**I want** automatic descriptive statistics
**So that** I can quickly understand the fused data

**Acceptance Criteria:**
- [ ] System computes mean, std, min, max, median, Q1, Q3 for numeric fields
- [ ] System identifies distribution type (normal, skewed, bimodal, uniform)
- [ ] Statistics are included in InsightResult

### US-5.2: Correlation Detection
**As a** researcher
**I want** to find correlations between fields
**So that** I can identify potential relationships

**Acceptance Criteria:**
- [ ] System computes Pearson correlation matrix
- [ ] System computes p-values for each correlation
- [ ] System highlights significant correlations (p < 0.05)
- [ ] System generates correlation heatmap visualization

### US-5.3: Outlier Detection
**As a** quality analyst
**I want** automatic outlier identification
**So that** I can investigate anomalies

**Acceptance Criteria:**
- [ ] System detects outliers using IQR method (1.5x multiplier)
- [ ] System reports outlier counts per field
- [ ] System generates boxplot visualizations

### US-5.4: Clustering Analysis
**As a** data scientist
**I want** automatic clustering of records
**So that** I can discover natural groupings

**Acceptance Criteria:**
- [ ] System runs K-means clustering with automatic k selection (2-10)
- [ ] System reports cluster sizes and centroids
- [ ] System generates scatter plot with cluster coloring

### US-5.5: Narrative Summary
**As a** business stakeholder
**I want** a plain-English summary of findings
**So that** I can understand insights without technical expertise

**Acceptance Criteria:**
- [ ] System generates 3-5 paragraph narrative using LLM
- [ ] Narrative covers correlations, outliers, and clusters
- [ ] Narrative includes specific numbers and field names
- [ ] System publishes `insight.generated` event

### US-5.6: Report Export
**As a** report consumer
**I want** downloadable reports in multiple formats
**So that** I can share findings with stakeholders

**Acceptance Criteria:**
- [ ] System exports HTML report with interactive visualizations
- [ ] System exports PDF report for offline viewing
- [ ] System exports JSON for programmatic access

---

## 6. Provenance Tracking

### US-6.1: Field Lineage Tracing
**As a** auditor
**I want** to trace any field back to its source
**So that** I can verify data origins

**Acceptance Criteria:**
- [ ] System records every transformation step
- [ ] System builds lineage DAG in Neo4j
- [ ] API returns complete lineage for any field
- [ ] Lineage includes source file, column, and all transformations

### US-6.2: Provenance Report Generation
**As a** compliance officer
**I want** comprehensive provenance reports
**So that** I can demonstrate data governance

**Acceptance Criteria:**
- [ ] System generates report showing all field origins
- [ ] Report includes transformation parameters
- [ ] Report shows confidence scores through transformation chain
- [ ] Report available in JSON-LD, HTML formats

### US-6.3: Confidence Tracking
**As a** data quality manager
**I want** to see how confidence degrades through transformations
**So that** I can assess result reliability

**Acceptance Criteria:**
- [ ] System applies confidence_decay_per_step (default 0.02)
- [ ] Final confidence score reflects all transformations
- [ ] Low-confidence fields are flagged in reports

---

## API User Stories

### US-API.1: Health Check
**As a** operations engineer
**I want** a health check endpoint
**So that** I can monitor system availability

**Acceptance Criteria:**
- [ ] GET /health returns 200 when system is operational
- [ ] Response includes database and service connectivity status

### US-API.2: Dataset Management
**As a** API consumer
**I want** RESTful dataset management
**So that** I can integrate HelixForge into workflows

**Acceptance Criteria:**
- [ ] POST /datasets/upload accepts multipart/form-data
- [ ] GET /datasets/{id} returns dataset metadata
- [ ] GET /datasets/{id}/sample returns sample rows

---

## Non-Functional Requirements

### NFR-1: Performance
- End-to-end processing < 5 minutes for 10K records, 50 fields
- API response time < 2 seconds for metadata queries

### NFR-2: Scalability
- Support files up to 500MB
- Support datasets up to 1M rows

### NFR-3: Security
- No credentials in logs
- Environment variable configuration for secrets
- Input validation on all endpoints

### NFR-4: Reliability
- Graceful error handling with meaningful messages
- Transaction rollback on failures
