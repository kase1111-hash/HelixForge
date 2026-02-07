"""Pydantic models for HelixForge.

This module defines all data schemas used across the system including
ingestion results, metadata, alignments, and fusion results.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Enums

class SourceType(str, Enum):
    """Supported data source types."""
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    SQL = "sql"
    REST = "rest"
    XLSX = "xlsx"


class DataType(str, Enum):
    """Data types for fields."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    OBJECT = "object"


class SemanticType(str, Enum):
    """Semantic types for fields."""
    IDENTIFIER = "identifier"
    METRIC = "metric"
    CATEGORY = "category"
    TIMESTAMP = "timestamp"
    TEXT = "text"
    UNKNOWN = "unknown"


class AlignmentType(str, Enum):
    """Types of field alignment."""
    EXACT = "exact"
    SYNONYM = "synonym"
    RELATED = "related"
    PARTIAL = "partial"
    NONE = "none"


class JoinStrategy(str, Enum):
    """Join strategies for fusion."""
    AUTO = "auto"
    EXACT_KEY = "exact_key"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    PROBABILISTIC = "probabilistic"
    TEMPORAL = "temporal"


class ImputationMethod(str, Enum):
    """Methods for missing value imputation."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    KNN = "knn"
    MODEL = "model"


# Layer 1: Data Ingestor

class IngestResult(BaseModel):
    """Result of data ingestion."""
    dataset_id: str = Field(..., description="Unique identifier (UUID)")
    source: str = Field(..., description="Origin path/URL")
    source_type: SourceType = Field(..., description="Type of data source")
    schema_fields: List[str] = Field(..., alias="schema", description="Column names in order")
    dtypes: Dict[str, str] = Field(..., description="Column name to pandas dtype mapping")
    row_count: int = Field(..., ge=0, description="Total rows")
    sample_rows: int = Field(default=10, description="Rows in preview")
    sample_data: List[Dict[str, Any]] = Field(default_factory=list, description="First N rows")
    content_hash: str = Field(..., description="SHA-256 of normalized content")
    encoding: Optional[str] = Field(default=None, description="Detected encoding")
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    storage_path: str = Field(..., description="Internal DataFrame location")

    class Config:
        populate_by_name = True


class IngestorConfig(BaseModel):
    """Configuration for Data Ingestor Agent."""
    max_file_size_mb: int = Field(default=500)
    sample_size: int = Field(default=10)
    supported_formats: List[str] = Field(
        default=["csv", "parquet", "json", "xlsx"]
    )
    encoding_detection_sample_bytes: int = Field(default=10000)
    sql_timeout_seconds: int = Field(default=300)
    rest_timeout_seconds: int = Field(default=60)
    temp_storage_path: str = Field(default="./data/raw/")
    experimental_sources: bool = Field(
        default=False,
        description="Enable experimental SQL and REST ingestion (requires sqlalchemy/requests)"
    )


# Layer 2: Metadata Interpreter

class FieldMetadata(BaseModel):
    """Metadata for a single field."""
    dataset_id: str
    field_name: str = Field(..., description="Original column name")
    semantic_label: str = Field(..., description="Inferred meaning (2-4 words)")
    description: str = Field(..., description="Natural language description")
    data_type: DataType
    semantic_type: SemanticType
    embedding: Optional[List[float]] = Field(default=None, description="Vector representation")
    sample_values: List[Any] = Field(default_factory=list)
    null_ratio: float = Field(..., ge=0, le=1)
    unique_ratio: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    inferred_at: datetime = Field(default_factory=datetime.utcnow)


class DatasetMetadata(BaseModel):
    """Complete metadata for a dataset."""
    dataset_id: str
    fields: List[FieldMetadata]
    dataset_description: str = Field(..., description="LLM-generated summary")
    domain_tags: List[str] = Field(default_factory=list, description="Inferred domains")
    ready_at: datetime = Field(default_factory=datetime.utcnow)


class InterpreterConfig(BaseModel):
    """Configuration for Metadata Interpreter Agent."""
    embedding_model: str = Field(default="text-embedding-3-large")
    embedding_dimensions: int = Field(default=1536)
    llm_model: str = Field(default="gpt-4o")
    llm_temperature: float = Field(default=0.2)
    max_sample_values: int = Field(default=20)
    min_confidence_threshold: float = Field(default=0.5)
    batch_size: int = Field(default=50)


# Layer 3: Ontology Alignment

class FieldAlignment(BaseModel):
    """Alignment between two fields."""
    alignment_id: str
    source_dataset: str
    source_field: str
    target_dataset: str
    target_field: str
    similarity: float = Field(..., ge=0, le=1)
    alignment_type: AlignmentType
    transformation_hint: Optional[str] = Field(
        default=None,
        description="e.g., 'unit_conversion', 'type_cast'"
    )
    validated: bool = Field(default=False, description="Human-reviewed flag")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AlignmentResult(BaseModel):
    """Result of alignment job."""
    alignment_job_id: str
    datasets_aligned: List[str]
    alignments: List[FieldAlignment]
    unmatched_fields: List[str] = Field(default_factory=list)
    ontology_graph_uri: Optional[str] = Field(default=None)
    completed_at: datetime = Field(default_factory=datetime.utcnow)


class ScoringWeights(BaseModel):
    """Weights for the multi-signal alignment scoring pipeline."""
    name: float = Field(default=0.30, ge=0, le=1, description="Field name similarity weight")
    embedding: float = Field(default=0.40, ge=0, le=1, description="Embedding cosine similarity weight")
    type_match: float = Field(default=0.15, ge=0, le=1, description="Type/semantic type match weight")
    stats: float = Field(default=0.15, ge=0, le=1, description="Statistical profile similarity weight")


class AlignmentConfig(BaseModel):
    """Configuration for Ontology Alignment Agent."""
    similarity_threshold: float = Field(default=0.50)
    exact_match_threshold: float = Field(default=0.95)
    synonym_threshold: float = Field(default=0.85)
    max_alignments_per_field: int = Field(default=3)
    conflict_resolution: str = Field(default="highest_similarity")
    scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights)
    enforce_type_compatibility: bool = Field(
        default=True,
        description="Reject alignments between fundamentally incompatible types"
    )


# Layer 4: Fusion

class TransformationTemplate(BaseModel):
    """Template for value transformation."""
    template_id: str
    name: str
    source_unit: Optional[str] = None
    target_unit: Optional[str] = None
    formula: str = Field(..., description="Python expression using 'value'")
    applicable_types: List[str] = Field(default_factory=list)


class TransformationLog(BaseModel):
    """Log of a transformation operation."""
    field: str
    operation: str = Field(..., description="unit_conversion | type_cast | rename | impute")
    template_id: Optional[str] = None
    source_value_sample: Optional[Any] = None
    target_value_sample: Optional[Any] = None
    records_affected: int


class ImputationSummary(BaseModel):
    """Summary of imputation operations."""
    total_nulls_filled: int
    fields_imputed: Dict[str, int] = Field(..., description="field -> count")
    method_used: ImputationMethod
    imputation_quality_score: Optional[float] = None


class FusionResult(BaseModel):
    """Result of fusion operation."""
    fused_dataset_id: str
    source_datasets: List[str]
    record_count: int
    field_count: int
    merged_fields: List[str]
    join_strategy: JoinStrategy
    transformations_applied: List[TransformationLog] = Field(default_factory=list)
    imputation_summary: Optional[ImputationSummary] = None
    storage_path: str
    fused_at: datetime = Field(default_factory=datetime.utcnow)


class FusionConfig(BaseModel):
    """Configuration for Fusion Agent."""
    default_join_strategy: JoinStrategy = Field(default=JoinStrategy.AUTO)
    similarity_join_threshold: float = Field(default=0.85)
    probabilistic_match_threshold: float = Field(default=0.75)
    temporal_tolerance_seconds: int = Field(default=3600)
    imputation_method: ImputationMethod = Field(default=ImputationMethod.MEAN)
    knn_neighbors: int = Field(default=5)
    max_null_ratio_for_inclusion: float = Field(default=0.5)
    output_format: str = Field(default="csv")
    output_path: str = Field(default="./data/fused/")
    experimental_strategies: bool = Field(
        default=False,
        description="Enable probabilistic and temporal join strategies"
    )


# API Request/Response Models

class AlignmentRequest(BaseModel):
    """Request to align datasets."""
    dataset_ids: List[str] = Field(..., min_length=2)
    confidence_threshold: Optional[float] = None
    include_partial_matches: bool = Field(default=True)


class FusionRequest(BaseModel):
    """Request to fuse datasets."""
    alignment_job_id: str
    join_strategy: Optional[JoinStrategy] = None
    imputation_method: Optional[ImputationMethod] = None
    output_format: str = Field(default="parquet")


# Layer 5: Insight Analysis

class FieldStatistics(BaseModel):
    """Descriptive statistics for a single numeric field."""
    field_name: str
    count: int
    mean: float
    std: float
    min: float
    q1: float
    median: float
    q3: float
    max: float
    null_count: int = 0
    unique_count: int = 0


class CorrelationPair(BaseModel):
    """A pair of correlated fields."""
    field_a: str
    field_b: str
    coefficient: float
    p_value: Optional[float] = None


class OutlierInfo(BaseModel):
    """Outlier detection results for a single field."""
    field_name: str
    outlier_count: int
    lower_bound: float
    upper_bound: float
    outlier_indices: List[int] = Field(default_factory=list)


class ClusterInfo(BaseModel):
    """K-means clustering results."""
    n_clusters: int
    labels: List[int] = Field(default_factory=list, description="Cluster assignment per row")
    centroids: List[List[float]] = Field(default_factory=list, description="Centroid coordinates")
    silhouette_score: Optional[float] = Field(default=None, description="Mean silhouette score (-1 to 1)")
    inertia: float = Field(default=0.0, description="Sum of squared distances to closest centroid")
    features_used: List[str] = Field(default_factory=list, description="Columns used for clustering")


class InsightResult(BaseModel):
    """Result of statistical analysis."""
    analysis_id: str
    source_description: str
    record_count: int
    field_count: int
    statistics: List[FieldStatistics] = Field(default_factory=list)
    correlations: List[CorrelationPair] = Field(default_factory=list)
    outliers: List[OutlierInfo] = Field(default_factory=list)
    clustering: Optional[ClusterInfo] = Field(default=None)
    narrative: Optional[str] = Field(default=None, description="LLM-generated insight summary")
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


class InsightConfig(BaseModel):
    """Configuration for Insight Agent."""
    correlation_method: str = Field(default="pearson")
    correlation_threshold: float = Field(default=0.5)
    outlier_iqr_multiplier: float = Field(default=1.5)
    include_stats: bool = Field(default=True)
    include_correlations: bool = Field(default=True)
    include_outliers: bool = Field(default=True)
    include_clustering: bool = Field(default=False)
    n_clusters: int = Field(default=3, ge=2, le=20)
    include_narrative: bool = Field(default=False)


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
