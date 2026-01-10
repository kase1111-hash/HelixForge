"""Pytest configuration and shared fixtures for HelixForge tests."""

import tempfile
from typing import Dict
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def sample_csv_data() -> str:
    """Sample CSV content for testing."""
    return """id,name,age,salary,department,hire_date
1,Alice,30,75000.50,Engineering,2020-01-15
2,Bob,25,65000.00,Marketing,2021-03-20
3,Charlie,35,85000.75,Engineering,2019-06-10
4,Diana,28,70000.25,Sales,2022-02-01
5,Eve,32,80000.00,Engineering,2020-08-15
"""


@pytest.fixture
def sample_csv_file(sample_csv_data, tmp_path) -> str:
    """Create a temporary CSV file for testing."""
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text(sample_csv_data)
    return str(csv_file)


@pytest.fixture
def sample_json_data() -> str:
    """Sample JSON content for testing."""
    return """[
    {"id": 1, "name": "Alice", "score": 95.5},
    {"id": 2, "name": "Bob", "score": 87.0},
    {"id": 3, "name": "Charlie", "score": 92.3}
]"""


@pytest.fixture
def sample_json_file(sample_json_data, tmp_path) -> str:
    """Create a temporary JSON file for testing."""
    json_file = tmp_path / "test_data.json"
    json_file.write_text(sample_json_data)
    return str(json_file)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [30, 25, 35, 28, 32],
        "salary": [75000.50, 65000.00, 85000.75, 70000.25, 80000.00],
        "department": ["Engineering", "Marketing", "Engineering", "Sales", "Engineering"],
        "hire_date": pd.to_datetime([
            "2020-01-15", "2021-03-20", "2019-06-10", "2022-02-01", "2020-08-15"
        ])
    })


@pytest.fixture
def sample_dataframe_with_nulls() -> pd.DataFrame:
    """Sample DataFrame with null values for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", None, "Charlie", "Diana", None],
        "age": [30, 25, None, 28, 32],
        "salary": [75000.50, None, 85000.75, 70000.25, None],
        "score": [95.0, 87.5, None, 91.0, 88.5]
    })


@pytest.fixture
def second_sample_dataframe() -> pd.DataFrame:
    """Second sample DataFrame for alignment/fusion testing."""
    return pd.DataFrame({
        "employee_id": [1, 2, 3, 6, 7],
        "full_name": ["Alice Smith", "Bob Jones", "Charlie Brown", "Frank", "Grace"],
        "years_old": [30, 25, 35, 40, 29],
        "annual_salary": [75000, 65000, 85000, 90000, 72000],
        "team": ["Eng", "Mkt", "Eng", "Eng", "Sales"]
    })


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch("openai.OpenAI") as mock:
        client = MagicMock()
        mock.return_value = client

        # Mock embeddings
        embedding_response = MagicMock()
        embedding_response.data = [MagicMock(embedding=[0.1] * 1536, index=0)]
        client.embeddings.create.return_value = embedding_response

        # Mock chat completions
        chat_response = MagicMock()
        chat_response.choices = [MagicMock(
            message=MagicMock(content='{"semantic_label": "Test Label", "description": "Test description", "semantic_type": "identifier", "confidence": 0.9}')
        )]
        client.chat.completions.create.return_value = chat_response

        yield client


@pytest.fixture
def mock_embedding():
    """Generate mock embedding vector."""
    return [0.1] * 1536


@pytest.fixture
def mock_embeddings_batch():
    """Generate batch of mock embedding vectors."""
    return [[0.1 + i * 0.01] * 1536 for i in range(5)]


@pytest.fixture
def ingestor_config() -> Dict:
    """Configuration for Data Ingestor Agent."""
    return {
        "ingestor": {
            "max_file_size_mb": 100,
            "sample_size": 5,
            "supported_formats": ["csv", "parquet", "json"],
            "temp_storage_path": tempfile.mkdtemp()
        }
    }


@pytest.fixture
def interpreter_config() -> Dict:
    """Configuration for Metadata Interpreter Agent."""
    return {
        "interpreter": {
            "embedding_model": "text-embedding-3-large",
            "embedding_dimensions": 1536,
            "llm_model": "gpt-4o",
            "llm_temperature": 0.2,
            "max_sample_values": 10,
            "min_confidence_threshold": 0.5,
            "batch_size": 10
        }
    }


@pytest.fixture
def alignment_config() -> Dict:
    """Configuration for Ontology Alignment Agent."""
    return {
        "alignment": {
            "similarity_threshold": 0.70,
            "exact_match_threshold": 0.95,
            "synonym_threshold": 0.85,
            "max_alignments_per_field": 3,
            "conflict_resolution": "highest_similarity"
        }
    }


@pytest.fixture
def fusion_config() -> Dict:
    """Configuration for Fusion Agent."""
    return {
        "fusion": {
            "default_join_strategy": "semantic_similarity",
            "similarity_join_threshold": 0.80,
            "imputation_method": "mean",
            "knn_neighbors": 3,
            "max_null_ratio_for_inclusion": 0.5,
            "output_format": "parquet",
            "output_path": tempfile.mkdtemp()
        }
    }


@pytest.fixture
def insight_config() -> Dict:
    """Configuration for Insight Generator Agent."""
    return {
        "insights": {
            "llm_model": "gpt-4o",
            "correlation_method": "pearson",
            "correlation_significance_threshold": 0.05,
            "outlier_method": "iqr",
            "outlier_iqr_multiplier": 1.5,
            "clustering_algorithm": "kmeans",
            "clustering_k_range": [2, 5],
            "output_path": tempfile.mkdtemp()
        }
    }


@pytest.fixture
def provenance_config() -> Dict:
    """Configuration for Provenance Tracker Agent."""
    return {
        "provenance": {
            "graph_store": "neo4j",
            "graph_uri": "neo4j://localhost:7687",
            "report_format": "json",
            "report_output_path": tempfile.mkdtemp(),
            "confidence_decay_per_step": 0.02
        }
    }


@pytest.fixture
def sample_ingest_result():
    """Sample IngestResult for testing."""
    from models.schemas import IngestResult, SourceType

    return IngestResult(
        dataset_id="test-dataset-001",
        source="/path/to/test.csv",
        source_type=SourceType.CSV,
        schema=["id", "name", "age", "salary"],
        dtypes={"id": "int64", "name": "object", "age": "int64", "salary": "float64"},
        row_count=100,
        sample_rows=5,
        sample_data=[{"id": 1, "name": "Alice", "age": 30, "salary": 75000.0}],
        content_hash="abc123def456",
        encoding="utf-8",
        storage_path="/tmp/test-dataset-001.parquet"
    )


@pytest.fixture
def sample_field_metadata():
    """Sample FieldMetadata for testing."""
    from models.schemas import DataType, FieldMetadata, SemanticType

    return FieldMetadata(
        dataset_id="test-dataset-001",
        field_name="salary",
        semantic_label="Annual Salary",
        description="Employee annual salary in USD",
        data_type=DataType.FLOAT,
        semantic_type=SemanticType.METRIC,
        embedding=[0.1] * 1536,
        sample_values=[75000.0, 65000.0, 85000.0],
        null_ratio=0.0,
        unique_ratio=0.95,
        confidence=0.85
    )


@pytest.fixture
def sample_dataset_metadata(sample_field_metadata):
    """Sample DatasetMetadata for testing."""
    from models.schemas import DatasetMetadata, DataType, FieldMetadata, SemanticType

    fields = [
        FieldMetadata(
            dataset_id="test-dataset-001",
            field_name="id",
            semantic_label="Employee ID",
            description="Unique employee identifier",
            data_type=DataType.INTEGER,
            semantic_type=SemanticType.IDENTIFIER,
            embedding=[0.2] * 1536,
            sample_values=[1, 2, 3],
            null_ratio=0.0,
            unique_ratio=1.0,
            confidence=0.95
        ),
        FieldMetadata(
            dataset_id="test-dataset-001",
            field_name="name",
            semantic_label="Employee Name",
            description="Full name of employee",
            data_type=DataType.STRING,
            semantic_type=SemanticType.TEXT,
            embedding=[0.3] * 1536,
            sample_values=["Alice", "Bob", "Charlie"],
            null_ratio=0.0,
            unique_ratio=0.9,
            confidence=0.90
        ),
        sample_field_metadata
    ]

    return DatasetMetadata(
        dataset_id="test-dataset-001",
        fields=fields,
        dataset_description="Employee records dataset with salary information",
        domain_tags=["hr", "finance"]
    )


@pytest.fixture(autouse=True)
def clean_temp_files(tmp_path):
    """Clean up temporary files after each test."""
    yield
    # Cleanup happens automatically with tmp_path fixture


@pytest.fixture
def disable_network_calls(monkeypatch):
    """Disable actual network calls during tests."""
    import requests

    def mock_get(*args, **kwargs):
        raise RuntimeError("Network calls disabled in tests")

    def mock_post(*args, **kwargs):
        raise RuntimeError("Network calls disabled in tests")

    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(requests, "post", mock_post)
