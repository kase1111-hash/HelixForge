"""Unit tests for Data Ingestor Agent.

Tests cover file-based ingestion (CSV, JSON, Parquet), encoding/delimiter
detection, content hashing, data integrity, and experimental source gating.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from agents.data_ingestor_agent import DataIngestorAgent, IngestionError
from models.schemas import SourceType

FIXTURES = Path(__file__).parent / "fixtures"


class TestCSVIngestion:
    """Tests for CSV file ingestion with various formats and encodings."""

    @pytest.fixture
    def agent(self, ingestor_config):
        return DataIngestorAgent(config=ingestor_config)

    def test_standard_csv(self, agent):
        """Standard UTF-8 CSV with 5 rows."""
        result = agent.ingest_file(str(FIXTURES / "employees.csv"))

        assert result.source_type == SourceType.CSV
        assert result.row_count == 5
        assert result.schema_fields == ["id", "name", "department", "salary", "hire_date"]
        assert result.encoding is not None
        assert result.content_hash is not None

        df = agent.get_dataframe(result.dataset_id)
        assert list(df.columns) == ["id", "name", "department", "salary", "hire_date"]
        assert df["id"].tolist() == [1, 2, 3, 4, 5]
        assert df["name"].tolist() == ["Alice", "Bob", "Charlie", "Diana", "Eve"]
        assert "float" in str(df["salary"].dtype)

    def test_semicolon_delimiter_detection(self, agent):
        """Semicolon-delimited CSV is auto-detected."""
        result = agent.ingest_file(str(FIXTURES / "semicolon_employees.csv"))

        assert result.row_count == 5
        assert "id" in result.schema_fields
        assert "name" in result.schema_fields
        assert "salary" in result.schema_fields

        df = agent.get_dataframe(result.dataset_id)
        assert df["id"].tolist() == [1, 2, 3, 4, 5]

    def test_pipe_delimiter_detection(self, agent):
        """Pipe-delimited CSV is auto-detected."""
        result = agent.ingest_file(str(FIXTURES / "pipe_data.csv"))

        assert result.row_count == 4
        assert result.schema_fields == ["id", "product", "price", "quantity"]

        df = agent.get_dataframe(result.dataset_id)
        assert df["product"].tolist() == ["Widget A", "Widget B", "Gadget C", "Gadget D"]
        assert "float" in str(df["price"].dtype)

    def test_tab_delimiter_detection(self, agent):
        """Tab-delimited CSV is auto-detected."""
        result = agent.ingest_file(str(FIXTURES / "tab_delimited.csv"))

        assert result.row_count == 4
        assert result.schema_fields == ["id", "name", "region", "revenue"]

        df = agent.get_dataframe(result.dataset_id)
        assert df["region"].tolist() == ["West", "East", "West", "North"]

    def test_header_only_csv(self, agent):
        """CSV with only headers (no data rows) returns row_count=0."""
        result = agent.ingest_file(str(FIXTURES / "header_only.csv"))

        assert result.row_count == 0
        assert result.schema_fields == ["id", "name", "value", "category"]

        df = agent.get_dataframe(result.dataset_id)
        assert len(df) == 0
        assert list(df.columns) == ["id", "name", "value", "category"]

    def test_single_row_csv(self, agent):
        """CSV with a single data row."""
        result = agent.ingest_file(str(FIXTURES / "single_row.csv"))

        assert result.row_count == 1
        assert result.schema_fields == ["id", "name", "score"]

        df = agent.get_dataframe(result.dataset_id)
        assert df.iloc[0]["name"] == "Alice"
        assert df.iloc[0]["score"] == pytest.approx(95.5)

    def test_mixed_types_csv(self, agent):
        """CSV with integer, float, boolean-like, and datetime-like columns."""
        result = agent.ingest_file(str(FIXTURES / "mixed_types.csv"))

        assert result.row_count == 5
        assert result.schema_fields == ["id", "label", "value", "flag", "timestamp"]

        df = agent.get_dataframe(result.dataset_id)
        assert "int" in str(df["id"].dtype)
        assert "float" in str(df["value"].dtype)

    def test_latin1_encoding_detection(self, agent):
        """Latin-1 encoded file with accented characters."""
        result = agent.ingest_file(str(FIXTURES / "latin1_names.csv"))

        assert result.row_count == 4
        assert result.schema_fields == ["id", "name", "city"]

        df = agent.get_dataframe(result.dataset_id)
        # Verify accented characters survived encoding detection
        names = df["name"].tolist()
        assert "Müller" in names or "M\xfcller" in names  # Latin-1 umlaut

    def test_shiftjis_encoding_fallback(self, agent):
        """Shift-JIS encoded file ingests via encoding fallback chain.

        chardet often misidentifies Shift-JIS for small files.
        The ingestor falls back through utf-8 -> latin-1 which always succeeds.
        """
        result = agent.ingest_file(str(FIXTURES / "shiftjis_data.csv"))

        assert result.row_count == 8
        assert "id" in result.schema_fields
        assert result.encoding is not None

    def test_large_csv_ingestion(self, agent, tmp_path):
        """Large CSV (100K rows) ingests correctly."""
        csv_file = tmp_path / "large.csv"
        # Generate 100K rows
        rows = ["id,value,category"]
        for i in range(100_000):
            rows.append(f"{i},{i * 1.5},{['A', 'B', 'C'][i % 3]}")
        csv_file.write_text("\n".join(rows) + "\n")

        result = agent.ingest_file(str(csv_file))

        assert result.row_count == 100_000
        assert result.schema_fields == ["id", "value", "category"]

        df = agent.get_dataframe(result.dataset_id)
        assert len(df) == 100_000
        assert df["id"].iloc[0] == 0
        assert df["id"].iloc[-1] == 99_999


class TestJSONIngestion:
    """Tests for JSON file ingestion."""

    @pytest.fixture
    def agent(self, ingestor_config):
        return DataIngestorAgent(config=ingestor_config)

    def test_flat_json_array(self, agent):
        """JSON array of flat objects."""
        result = agent.ingest_file(str(FIXTURES / "flat_records.json"))

        assert result.source_type == SourceType.JSON
        assert result.row_count == 3
        assert "id" in result.schema_fields
        assert "name" in result.schema_fields
        assert "score" in result.schema_fields
        assert "active" in result.schema_fields

        df = agent.get_dataframe(result.dataset_id)
        assert df["score"].tolist() == [95.5, 87.0, 92.3]
        assert df["active"].tolist() == [True, False, True]

    def test_json_with_nulls(self, agent):
        """JSON with null values preserves structure."""
        result = agent.ingest_file(str(FIXTURES / "with_nulls.json"))

        assert result.row_count == 4
        assert result.schema_fields == ["id", "name", "score", "grade"]

        df = agent.get_dataframe(result.dataset_id)
        assert pd.isna(df.iloc[1]["name"])
        assert pd.isna(df.iloc[2]["score"])
        assert pd.isna(df.iloc[2]["grade"])
        # Non-null values are correct
        assert df.iloc[0]["name"] == "Alice"
        assert df.iloc[0]["score"] == pytest.approx(95.5)

    def test_nested_json_flattens_to_columns(self, agent):
        """JSON with nested objects produces columns."""
        result = agent.ingest_file(str(FIXTURES / "nested_objects.json"))

        assert result.row_count == 3

        df = agent.get_dataframe(result.dataset_id)
        assert "id" in df.columns
        assert "name" in df.columns
        # Nested objects become columns (address, tags)
        assert "address" in df.columns or "address.city" in df.columns
        assert len(df) == 3

    def test_jsonl_format(self, agent, tmp_path):
        """JSON Lines format is auto-detected."""
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(
            '{"id": 1, "value": "alpha"}\n'
            '{"id": 2, "value": "beta"}\n'
            '{"id": 3, "value": "gamma"}\n'
        )

        result = agent.ingest_file(str(jsonl_file))

        assert result.row_count == 3
        df = agent.get_dataframe(result.dataset_id)
        assert df["value"].tolist() == ["alpha", "beta", "gamma"]


class TestParquetIngestion:
    """Tests for Parquet file ingestion."""

    @pytest.fixture
    def agent(self, ingestor_config):
        return DataIngestorAgent(config=ingestor_config)

    def test_typed_parquet(self, agent):
        """Parquet with int, float, datetime, and boolean columns."""
        result = agent.ingest_file(str(FIXTURES / "typed_data.parquet"))

        assert result.source_type == SourceType.PARQUET
        assert result.row_count == 5
        assert set(result.schema_fields) == {"id", "name", "score", "enrolled_date", "active"}

        df = agent.get_dataframe(result.dataset_id)
        assert "int" in str(df["id"].dtype)
        assert "float" in str(df["score"].dtype)
        assert pd.api.types.is_datetime64_any_dtype(df["enrolled_date"])
        assert df["active"].dtype == bool

    def test_nullable_parquet(self, agent):
        """Parquet with nullable integer columns."""
        result = agent.ingest_file(str(FIXTURES / "nullable_parquet.parquet"))

        assert result.row_count == 5
        assert "product_id" in result.schema_fields
        assert "stock" in result.schema_fields

        df = agent.get_dataframe(result.dataset_id)
        # stock has nulls at index 1 and 4
        assert pd.isna(df.iloc[1]["stock"])
        assert pd.isna(df.iloc[4]["stock"])
        assert df.iloc[0]["stock"] == 100

    def test_programmatic_parquet(self, agent, tmp_path):
        """Parquet generated at test time with known values."""
        df_source = pd.DataFrame({
            "x": [10, 20, 30],
            "y": [1.1, 2.2, 3.3],
        })
        parquet_file = tmp_path / "test.parquet"
        df_source.to_parquet(str(parquet_file), index=False)

        result = agent.ingest_file(str(parquet_file))

        assert result.row_count == 3
        df = agent.get_dataframe(result.dataset_id)
        assert df["x"].tolist() == [10, 20, 30]
        assert df["y"].tolist() == pytest.approx([1.1, 2.2, 3.3])


class TestContentHashing:
    """Tests for deterministic content hashing."""

    @pytest.fixture
    def agent(self, ingestor_config):
        return DataIngestorAgent(config=ingestor_config)

    def test_same_content_same_hash(self, agent):
        """Ingesting the same file twice produces the same content hash."""
        path = str(FIXTURES / "employees.csv")
        result1 = agent.ingest_file(path, dataset_id="run1")
        result2 = agent.ingest_file(path, dataset_id="run2")

        assert result1.content_hash == result2.content_hash

    def test_different_content_different_hash(self, agent, tmp_path):
        """Different data produces different hashes."""
        csv1 = tmp_path / "a.csv"
        csv1.write_text("id,name\n1,Alice\n")
        csv2 = tmp_path / "b.csv"
        csv2.write_text("id,name\n1,Bob\n")

        result1 = agent.ingest_file(str(csv1))
        result2 = agent.ingest_file(str(csv2))

        assert result1.content_hash != result2.content_hash

    def test_hash_is_sha256_hex(self, agent):
        """Content hash is a 64-character hex string (SHA-256)."""
        result = agent.ingest_file(str(FIXTURES / "single_row.csv"))

        assert len(result.content_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.content_hash)

    def test_hash_is_column_order_independent(self, agent, tmp_path):
        """Hash is stable regardless of original column order (sorted internally)."""
        csv1 = tmp_path / "ab.csv"
        csv1.write_text("a,b\n1,2\n")
        csv2 = tmp_path / "ba.csv"
        csv2.write_text("b,a\n2,1\n")

        r1 = agent.ingest_file(str(csv1))
        r2 = agent.ingest_file(str(csv2))

        # Both should produce the same hash since columns are sorted
        assert r1.content_hash == r2.content_hash


class TestStorageAndRetrieval:
    """Tests for DataFrame storage and retrieval."""

    @pytest.fixture
    def agent(self, ingestor_config):
        return DataIngestorAgent(config=ingestor_config)

    def test_get_dataframe_returns_correct_data(self, agent):
        """Retrieved DataFrame matches original ingested data."""
        result = agent.ingest_file(str(FIXTURES / "employees.csv"))
        df = agent.get_dataframe(result.dataset_id)

        assert df is not None
        assert len(df) == result.row_count
        assert list(df.columns) == result.schema_fields

    def test_get_dataframe_nonexistent_returns_none(self, agent):
        """Requesting a non-existent dataset ID returns None."""
        assert agent.get_dataframe("does-not-exist") is None

    def test_storage_path_exists_on_disk(self, agent):
        """Parquet file is written to disk at storage_path."""
        result = agent.ingest_file(str(FIXTURES / "employees.csv"))

        assert result.storage_path is not None
        assert os.path.exists(result.storage_path)
        assert result.storage_path.endswith(".parquet")

    def test_custom_dataset_id(self, agent):
        """Custom dataset_id is respected."""
        result = agent.ingest_file(str(FIXTURES / "employees.csv"), dataset_id="my-employees")

        assert result.dataset_id == "my-employees"
        df = agent.get_dataframe("my-employees")
        assert df is not None
        assert len(df) == 5

    def test_multiple_datasets_stored_independently(self, agent):
        """Multiple ingested datasets don't interfere."""
        r1 = agent.ingest_file(str(FIXTURES / "employees.csv"), dataset_id="ds1")
        r2 = agent.ingest_file(str(FIXTURES / "single_row.csv"), dataset_id="ds2")

        df1 = agent.get_dataframe("ds1")
        df2 = agent.get_dataframe("ds2")

        assert len(df1) == 5
        assert len(df2) == 1
        assert list(df1.columns) != list(df2.columns)


class TestDTypeInference:
    """Tests for data type inference in ingested DataFrames."""

    @pytest.fixture
    def agent(self, ingestor_config):
        return DataIngestorAgent(config=ingestor_config)

    def test_dtypes_mapping_populated(self, agent):
        """IngestResult.dtypes maps column names to dtype strings."""
        result = agent.ingest_file(str(FIXTURES / "employees.csv"))

        assert "id" in result.dtypes
        assert "int" in result.dtypes["id"].lower()
        assert "salary" in result.dtypes
        assert "float" in result.dtypes["salary"].lower()

    def test_numeric_columns_detected(self, agent):
        """Integer and float columns are correctly typed."""
        result = agent.ingest_file(str(FIXTURES / "pipe_data.csv"))
        df = agent.get_dataframe(result.dataset_id)

        assert "int" in str(df["id"].dtype)
        assert "float" in str(df["price"].dtype)
        assert "int" in str(df["quantity"].dtype)

    def test_sample_data_included(self, agent):
        """Sample data is included in the result."""
        result = agent.ingest_file(str(FIXTURES / "employees.csv"))

        assert result.sample_data is not None
        assert len(result.sample_data) <= result.sample_rows
        assert "id" in result.sample_data[0]
        assert result.sample_data[0]["name"] == "Alice"


class TestSourceTypeDetection:
    """Tests for auto-detecting source type from path."""

    @pytest.fixture
    def agent(self, ingestor_config):
        return DataIngestorAgent(config=ingestor_config)

    def test_csv_detected(self, agent):
        assert agent._detect_source_type("/path/to/file.csv") == "csv"

    def test_parquet_detected(self, agent):
        assert agent._detect_source_type("/path/to/file.parquet") == "parquet"

    def test_json_detected(self, agent):
        assert agent._detect_source_type("/path/to/file.json") == "json"

    def test_jsonl_detected(self, agent):
        assert agent._detect_source_type("/path/to/file.jsonl") == "json"

    def test_xlsx_detected(self, agent):
        assert agent._detect_source_type("/path/to/file.xlsx") == "xlsx"

    def test_sql_connection_string_detected(self, agent):
        """SQL connection strings are still detected (but gated at execution)."""
        assert agent._detect_source_type("postgresql://user:pass@host/db") == "sql"

    def test_rest_url_detected(self, agent):
        """REST URLs are still detected (but gated at execution)."""
        assert agent._detect_source_type("https://api.example.com/data") == "rest"

    def test_unknown_extension_raises(self, agent):
        with pytest.raises(IngestionError, match="Cannot auto-detect"):
            agent._detect_source_type("/path/to/file.xyz")


class TestExperimentalSourceGating:
    """Tests for SQL/REST experimental gating."""

    def test_sql_ingestion_gated_by_default(self, ingestor_config):
        """SQL ingestion raises IngestionError when experimental_sources=False."""
        agent = DataIngestorAgent(config=ingestor_config)

        with pytest.raises(IngestionError, match="experimental_sources=True"):
            agent.process("postgresql://user:pass@host/db", source_type="sql", query="SELECT 1")

    def test_rest_ingestion_gated_by_default(self, ingestor_config):
        """REST ingestion raises IngestionError when experimental_sources=False."""
        agent = DataIngestorAgent(config=ingestor_config)

        with pytest.raises(IngestionError, match="experimental_sources=True"):
            agent.ingest_rest("https://api.example.com/users")

    def test_sql_available_with_experimental_flag(self, ingestor_config):
        """SQL handler is registered when experimental_sources=True."""
        ingestor_config["ingestor"]["experimental_sources"] = True
        agent = DataIngestorAgent(config=ingestor_config)

        # We can't test actual SQL without a DB, but verify it tries
        # (will fail at sqlalchemy import or connection, not at gating)
        with pytest.raises(IngestionError) as exc_info:
            agent.process("postgresql://user:pass@host/db", source_type="sql", query="SELECT 1")
        # Should NOT be the gating error
        assert "experimental_sources" not in str(exc_info.value)


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    @pytest.fixture
    def agent(self, ingestor_config):
        return DataIngestorAgent(config=ingestor_config)

    def test_nonexistent_file_raises(self, agent):
        with pytest.raises(IngestionError, match="Failed to ingest"):
            agent.ingest_file("/nonexistent/file.csv")

    def test_unsupported_extension_raises(self, agent, tmp_path):
        bad_file = tmp_path / "data.xyz"
        bad_file.write_text("test")

        with pytest.raises(IngestionError, match="Cannot auto-detect"):
            agent.ingest_file(str(bad_file))


class TestEventPublishing:
    """Tests for event bus integration."""

    @pytest.fixture
    def agent(self, ingestor_config):
        return DataIngestorAgent(config=ingestor_config)

    def test_event_published_on_ingest(self, agent):
        """data.ingested event fires after successful ingestion."""
        events = []
        agent.subscribe("data.ingested", lambda e: events.append(e))

        agent.ingest_file(str(FIXTURES / "employees.csv"))

        assert len(events) == 1
        assert events[0]["event_type"] == "data.ingested"

    def test_event_type_property(self, agent):
        assert agent.event_type == "data.ingested"


class TestCorrelationID:
    """Tests for correlation ID propagation."""

    def test_custom_correlation_id(self, ingestor_config):
        agent = DataIngestorAgent(
            config=ingestor_config, correlation_id="test-123"
        )
        assert agent.correlation_id == "test-123"

    def test_new_correlation_id(self, ingestor_config):
        agent = DataIngestorAgent(config=ingestor_config)
        old_id = agent.correlation_id
        new_id = agent.new_correlation_id()
        assert new_id != old_id
        assert agent.correlation_id == new_id
