# HELIXFORGE REFOCUS PLAN

**Goal:** Strip HelixForge down to its differentiating core (LLM-powered semantic schema alignment), prove it works with real tests, and ship a minimal tool that solves one problem well.

**Guiding Principle:** Each phase must produce a usable, testable artifact. No phase should depend on infrastructure that hasn't been validated in a prior phase. Cut scope aggressively; earn complexity back through proven need.

---

## PHASE 0: Surgical Cuts (Infrastructure Debt Removal)

**Duration estimate: N/A - do this first**
**Risk: Low - removing unused code**
**Dependency: None**

### What to cut

| Target | Lines | Why |
|--------|-------|-----|
| `agents/provenance_tracker_agent.py` | ~525 | Separate product (Apache Atlas, OpenLineage). Neo4j integration is incomplete with silently-swallowed errors. Not core to alignment. |
| `agents/insight_generator_agent.py` | ~725 | Useful but not core. Correlations, clustering, and LLM narratives are downstream of alignment. Bring back in Phase 5. |
| `utils/metrics.py` | ~400 | Prometheus metrics for a system with no production deployment. |
| `api/routes/provenance.py` | ~140 | Routes for deleted agent. |
| `api/routes/insights.py` | ~200 | Routes for deferred agent. |
| `tests/fuzz/` | ~3 files | Premature - core unit tests don't verify correctness yet. |
| `tests/performance/` | ~2 files | Premature - no production baseline to benchmark against. |
| `tests/test_provenance.py` | ~1 file | Tests for deleted agent. |
| `tests/test_insights.py` | ~1 file | Tests for deferred agent. |
| `KEYWORDS.md` | ~1 file | Marketing fluff. |

### What to decouple

| Target | Action |
|--------|--------|
| Neo4j dependency | Remove from `docker-compose.yaml`, `requirements.txt`, `config.yaml`. OntologyAlignmentAgent's `_build_ontology_graph()` becomes a no-op or optional hook. |
| Weaviate dependency | Remove from `docker-compose.yaml`, `requirements.txt`, `config.yaml`. Never used by agent code anyway (only a health check in `server.py`). |
| PostgreSQL dependency | Keep only for SQL ingestion source type. Remove as a required backing service. App state stays in-memory for now. |
| Sentry integration | Remove from `utils/logging.py`. Keep structured JSON logging. |
| WeasyPrint / Jinja2 | Remove. No PDF report generation until Phase 5. |
| Locust | Remove from `requirements.txt`. |

### What to update after cuts

- `api/server.py`: Remove agent initialization for provenance and insight agents. Remove health checks for Neo4j and Weaviate. Remove `/metrics` endpoint.
- `api/routes/datasets.py`: Remove `provenance.record_ingestion()` call (line 72).
- `api/routes/alignment.py`: Remove `provenance.record_alignment()` call (line 77).
- `api/routes/fusion.py`: Remove `provenance.record_fusion()` call (line 86).
- `docker-compose.yaml`: Remove neo4j and weaviate services. Keep postgres as optional.
- `requirements.txt`: Remove neo4j, weaviate-client, prometheus-client, sentry-sdk, locust, weasyprint, jinja2, hypothesis.
- `config.yaml`: Remove `graph_store`, `vector_store`, `provenance`, `insights` sections.
- `Makefile`: Remove `test-fuzz`, `test-perf` targets if they exist.

### Exit criteria
- `make test-unit` passes with no import errors after deletions.
- `docker-compose up` starts with only helixforge + postgres (optional).
- No references to deleted modules remain in surviving code.

---

## PHASE 1: Harden the Ingestor

**Dependency: Phase 0 complete**
**Focus file:** `agents/data_ingestor_agent.py`
**Why first:** DataIngestorAgent is fully independent (no agent dependencies, no external services except optional SQL). It's the pipeline entry point. If ingestion is wrong, everything downstream is wrong.

### 1A. Fix the test suite

**Problem:** Current `test_ingestor.py` tests check that files are ingested without error, but don't validate data integrity rigorously.

**Tasks:**
- Create a `tests/fixtures/` directory with 10+ real-world-like CSV/JSON/Parquet files covering edge cases:
  - Mixed encodings (UTF-8, Latin-1, Shift-JIS)
  - Malformed CSV (unquoted commas, mixed delimiters, ragged rows)
  - Large files (100K+ rows) to test streaming behavior
  - Empty files, single-row files, header-only files
  - JSON with nested objects, arrays, nulls
  - Parquet with complex types (lists, maps, timestamps)
- Rewrite assertions to verify:
  - Exact row counts match expected
  - Column names are preserved correctly
  - Data types are inferred correctly (not just "something was returned")
  - Content hash is deterministic (same input -> same hash)
  - File size limits are enforced (expect rejection for oversized files)

### 1B. Add CLI entry point

**Tasks:**
- Create `cli.py` at project root with a minimal Click/Typer interface.
- First command: `helixforge ingest <file> [--format csv|json|parquet|excel]`
- Output: JSON summary of ingestion result (dataset_id, row_count, columns, dtypes).
- No API server required. Direct agent instantiation with in-memory config.

### 1C. Remove SQL and REST ingestion (defer)

**Rationale:** SQL and REST ingestion (`_ingest_sql`, `_ingest_rest`) add external dependencies and attack surface (SSRF risk in `_ingest_rest`). File-based ingestion is the core use case for schema alignment.

**Tasks:**
- Comment out or gate SQL/REST methods behind a `--experimental` flag.
- Remove `sqlalchemy`, `requests` from core requirements (move to optional extras).
- Simplify `validate_url()` usage - not needed without REST ingestion.

### Exit criteria
- `helixforge ingest sample.csv` works from the command line and prints a JSON result.
- All 10+ fixture files ingest correctly with verified assertions.
- No external service dependencies remain in the ingestor.

---

## PHASE 2: Harden the Metadata Interpreter

**Dependency: Phase 1 complete**
**Focus file:** `agents/metadata_interpreter_agent.py`
**Why second:** The interpreter generates embeddings and semantic labels that feed directly into alignment. If embeddings are bad, alignment is garbage.

### 2A. Make LLM calls testable

**Problem:** Every test mocks `batch_embed()` with `[0.1] * 1536`, bypassing the real embedding logic entirely.

**Tasks:**
- Record real OpenAI API responses using `vcrpy` or `pytest-recording` for 5-10 representative field sets.
- Store cassettes in `tests/cassettes/` (gitignored API keys, committed response shapes).
- Write tests that verify:
  - Embedding dimensions match expected (1536 for text-embedding-3-large).
  - Semantically similar fields produce similar embeddings (cosine > 0.8 for "employee_name" vs "worker_name").
  - Semantically different fields produce dissimilar embeddings (cosine < 0.4 for "employee_name" vs "hire_date").
  - Batch embedding handles edge cases (empty strings, very long field names, special characters).

### 2B. Add LLM provider abstraction

**Problem:** Hard-coded to OpenAI. If a user wants to use a local model or a different provider, they must rewrite agent code.

**Tasks:**
- Create `utils/llm.py` with a simple protocol/interface:
  ```python
  class LLMProvider(Protocol):
      def embed(self, texts: list[str]) -> list[list[float]]: ...
      def complete(self, messages: list[dict], **kwargs) -> str: ...
  ```
- Implement `OpenAIProvider` as the default.
- Implement `MockProvider` for testing (returns deterministic embeddings based on string hashing - better than uniform `[0.1]*1536`).
- Wire MetadataInterpreterAgent to accept a provider, not construct its own client.

### 2C. Validate semantic labeling quality

**Tasks:**
- Create a golden dataset: 50 field names with expected `SemanticType` and `DataType` labels.
  - Examples: `"patient_dob"` -> `SemanticType.DATE`, `"annual_salary"` -> `SemanticType.CURRENCY`
- Write a test that runs the interpreter against this golden set and asserts >= 80% label accuracy.
- Track accuracy as a regression metric - it should never decrease.

### 2D. Extend CLI

- Command: `helixforge describe <file>`
- Output: Table of field names, inferred types, semantic labels, and sample values.
- No API server required.

### Exit criteria
- `helixforge describe sample.csv` prints a readable field summary.
- Golden dataset test achieves >= 80% semantic labeling accuracy.
- Embedding similarity tests pass with recorded API responses.
- MockProvider produces better-than-uniform test embeddings.

---

## PHASE 3: Make Alignment Bulletproof

**Dependency: Phase 2 complete**
**Focus file:** `agents/ontology_alignment_agent.py`
**Why third:** This IS the product. If alignment is wrong, HelixForge has no reason to exist.

### 3A. Build a ground-truth alignment benchmark

**Tasks:**
- Curate 10 dataset pairs with manually-verified expected alignments:

  | Pair | Dataset A | Dataset B | Expected Alignments |
  |------|-----------|-----------|---------------------|
  | 1 | employees.csv (name, dept, salary) | workers.csv (worker_name, department, pay) | name<->worker_name, dept<->department, salary<->pay |
  | 2 | patients.csv (patient_id, dob, diagnosis) | medical.csv (id, birth_date, condition) | patient_id<->id, dob<->birth_date, diagnosis<->condition |
  | 3 | (intentionally non-overlapping schemas) | (no valid alignments) | empty |
  | ... | (mixed overlap, partial match, type conflicts, etc.) | | |

- Store in `tests/benchmarks/` with a manifest file describing expected results.
- Write a benchmark runner that:
  - Runs alignment on each pair.
  - Computes precision, recall, and F1 against expected alignments.
  - Fails if F1 drops below 0.75.

### 3B. Fix the alignment algorithm

**Problem:** Current alignment uses only cosine similarity on embeddings. This misses:
- Exact name matches that should always align (case-insensitive).
- Type incompatibilities that should prevent alignment (string vs float).
- Cardinality mismatches (unique ID vs categorical field).

**Tasks:**
- Implement a scoring pipeline (not just cosine similarity):
  1. **Name similarity** (weighted): Exact match, prefix match, edit distance via `string_similarity()`.
  2. **Embedding similarity** (weighted): Cosine similarity of field name + sample value embeddings.
  3. **Type compatibility** (gate): Reject alignments where types are fundamentally incompatible (e.g., datetime<->boolean).
  4. **Statistical profile similarity** (weighted): Similar null ratios, cardinality ranges, value distributions.
- Make weights configurable in `config.yaml` under `alignment.scoring_weights`.
- Replace the stub `validate_alignment()` method with real validation logic.

### 3C. Remove Neo4j dependency from alignment

**Tasks:**
- Delete `_build_ontology_graph()` (lines 324-376) from OntologyAlignmentAgent.
- Alignment results are returned as `AlignmentResult` Pydantic models - no graph database needed.
- If graph visualization is wanted later, it can be a separate export step, not baked into the alignment agent.

### 3D. Extend CLI

- Command: `helixforge align <file1> <file2> [--threshold 0.8] [--output json|table|csv]`
- Output: Alignment map showing matched fields, similarity scores, and alignment types.
- This is the **hero command** - it should work perfectly, be fast, and produce clear output.

### Exit criteria
- `helixforge align employees.csv workers.csv` produces correct alignments from the command line.
- Benchmark F1 >= 0.75 across 10 dataset pairs.
- No Neo4j dependency in alignment agent.
- Type-incompatible fields are never aligned.

---

## PHASE 4: Validate Fusion

**Dependency: Phase 3 complete**
**Focus file:** `agents/fusion_agent.py`
**Why fourth:** Fusion consumes alignment results. It only makes sense once alignment is reliable.

### 4A. Fix fusion tests

**Problem:** Current tests check that fusion returns a result, not that the fused data is correct.

**Tasks:**
- For each of the 10 benchmark dataset pairs from Phase 3, define expected fusion outputs:
  - Which rows should match.
  - How conflicts should be resolved (when both datasets have a value for the same entity+field).
  - Which rows should be left-joined (present in one dataset but not the other).
- Write tests that verify:
  - Fused DataFrame has expected row count.
  - Fused column names match expected (aligned field names, not duplicates).
  - Values are correct for specific rows (spot-check 5 rows per test).
  - Imputation fills nulls correctly (verify against known values, not just "not null").

### 4B. Simplify join strategies

**Problem:** Four join strategies (exact_key, semantic_similarity, probabilistic, temporal) is too many for a v1. Each strategy has edge cases that multiply testing burden.

**Tasks:**
- Keep `exact_key` (the simple, reliable case - join on a shared key column).
- Keep `semantic_similarity` (the differentiating case - join on embedding similarity).
- Defer `probabilistic` and `temporal` behind an `--experimental` flag.
- Ensure the default strategy is `exact_key` when a shared key is detected, falling back to `semantic_similarity`.

### 4C. Extend CLI

- Command: `helixforge fuse <file1> <file2> [--key id] [--strategy auto|exact|semantic] [--output fused.csv]`
- Output: Fused dataset written to file, with a summary printed to stdout (row count, columns, join stats).

### Exit criteria
- `helixforge fuse employees.csv workers.csv --key id --output merged.csv` produces a correct merged file.
- Spot-check assertions pass for all 10 benchmark pairs.
- Only `exact_key` and `semantic_similarity` strategies are enabled by default.

---

## PHASE 5: Bring Back Insights (Earned Complexity)

**Dependency: Phase 4 complete**
**Rationale:** Now that the core pipeline (ingest -> interpret -> align -> fuse) is proven, insights add genuine value on top of fused data. But only bring back what's validated.

### 5A. Reintroduce statistical analysis only

**Tasks:**
- Bring back `InsightGeneratorAgent` with ONLY these capabilities:
  - Descriptive statistics (mean, median, std, min, max, quartiles).
  - Correlation matrix (Pearson).
  - Outlier detection (IQR method).
- Remove (for now):
  - K-means clustering (nice-to-have, not core).
  - LLM narrative generation (expensive, unvalidated quality).
  - Visualization exports (Plotly, Matplotlib, Seaborn).
  - PDF report generation (WeasyPrint).

### 5B. Write correctness tests for statistics

**Problem:** Original tests assert `result.statistics is not None`. That's useless.

**Tasks:**
- Create test datasets with known statistical properties:
  - `[1, 2, 3, 4, 5]` -> mean=3.0, std=1.58, median=3.0
  - `[1, 1, 1, 100]` -> outlier=100 (IQR method)
  - `[x, 2x+noise]` -> correlation ~1.0
- Assert exact values (within floating-point tolerance).
- Assert outlier detection catches planted outliers.
- Assert correlation matrix is symmetric and diagonal is 1.0.

### 5C. Extend CLI

- Command: `helixforge analyze <file> [--correlations] [--outliers] [--stats]`
- Output: Statistical summary printed to stdout. Optional `--output report.json` for structured output.

### Exit criteria
- Statistical correctness tests pass with exact value assertions.
- No LLM dependency in the insights agent.
- No visualization library dependencies.
- `helixforge analyze fused.csv --stats --outliers` prints correct results.

---

## PHASE 6: Production Hardening

**Dependency: Phase 5 complete**
**Rationale:** Only harden what's proven to work.

### 6A. Add proper error handling

**Tasks:**
- Replace silent `except Exception: pass` patterns with explicit error types.
- Return structured error responses from API (not 500 with stack traces).
- Add input validation at API boundaries (file size limits, allowed formats, field count limits).
- Rate-limit OpenAI API calls to avoid quota exhaustion.

### 6B. Add configuration validation

**Tasks:**
- Validate `config.yaml` on startup - fail fast with clear error messages.
- Provide sensible defaults for all optional config keys.
- Support environment variable overrides for all config keys (12-factor app compliance).

### 6C. Restore Docker deployment

**Tasks:**
- Rebuild `Dockerfile` for the slimmed-down application.
- `docker-compose.yaml` with only helixforge + optional postgres.
- Health check endpoint that verifies OpenAI API connectivity.
- `.env.example` file documenting required environment variables.

### 6D. Add CI pipeline

**Tasks:**
- GitHub Actions workflow:
  - `lint` job: `ruff check .`
  - `typecheck` job: `mypy agents/ api/ models/ utils/`
  - `test` job: `pytest tests/ -m "not slow"` with recorded API cassettes.
  - `benchmark` job: Alignment F1 regression check.

### Exit criteria
- CI pipeline runs green on every push.
- `docker compose up` starts the full application.
- All config errors produce human-readable messages at startup.
- No silent exception swallowing in any agent.

---

## PHASE 7: Deferred Features (Backlog)

These features are intentionally deferred. Each should only be started when there is evidence of user demand.

| Feature | Prerequisite | Effort |
|---------|-------------|--------|
| Provenance/lineage tracking | Users request audit trails | Large - requires graph DB reintegration |
| LLM narrative generation | Users want natural-language summaries | Small - reintegrate `_generate_narrative()` |
| Visualization exports | Users want charts, not just data | Medium - reintegrate Plotly/Seaborn |
| PDF reports | Users want downloadable reports | Medium - reintegrate WeasyPrint/Jinja2 |
| K-means clustering | Users want segmentation | Small - already implemented, just re-enable |
| Probabilistic join strategy | Users have datasets without shared keys | Medium - needs validation |
| Temporal join strategy | Users have time-series data | Medium - needs validation |
| Event pub/sub system | Need async/streaming pipeline | Large - current system is ceremony |
| Weaviate vector store | In-memory similarity is too slow | Medium - need >100K fields to justify |
| Prometheus metrics | Production deployment exists | Small - reintegrate `utils/metrics.py` |
| Multi-language schema support | Users have non-English field names | Medium - embedding model handles this, but needs testing |
| REST API ingestion | Users want to pull from APIs | Small - already implemented, needs SSRF hardening |
| SQL database ingestion | Users want to pull from databases | Small - already implemented, needs connection pooling |

---

## DEPENDENCY GRAPH

```
Phase 0 ──→ Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4 ──→ Phase 5 ──→ Phase 6
  (Cut)      (Ingest)    (Metadata)   (Align)     (Fuse)      (Insights)  (Harden)
                                         ↑
                                    THIS IS THE
                                      PRODUCT
```

Phases 0-3 are **critical path**. If Phase 3 fails (alignment doesn't work reliably), the project has no value proposition and should be rethought.

Phases 4-5 are **value multipliers**. They make the alignment results useful.

Phase 6 is **operational hygiene**. Only invest here once Phases 0-5 are solid.

Phase 7 is **backlog**. Earn the right to build these through user feedback.

---

## SUCCESS METRICS

| Metric | Phase | Target |
|--------|-------|--------|
| Alignment F1 on benchmark | Phase 3 | >= 0.75 |
| Semantic labeling accuracy | Phase 2 | >= 80% |
| Test assertion quality | Phase 1-5 | Zero `assert x is not None` without value checks |
| External service dependencies | Phase 0 | Only OpenAI (for embeddings) |
| CLI commands working | Phase 4 | `ingest`, `describe`, `align`, `fuse` |
| Time from `git clone` to first alignment | Phase 3 | < 5 minutes (pip install + run) |

---

## WHAT THIS PLAN DOES NOT INCLUDE

- **Rewriting from scratch.** The existing code is clean Python. The architecture is sound. The problem is scope, not quality.
- **Changing the tech stack.** FastAPI, Pydantic, pandas are all appropriate. The problem is unused infrastructure, not wrong choices.
- **Adding features.** Every phase either removes or hardens. New features are only in Phase 7, gated by user demand.
- **A timeline.** Timelines create pressure to cut corners. Each phase has exit criteria instead. Move to the next phase only when the current one's exit criteria are met.
