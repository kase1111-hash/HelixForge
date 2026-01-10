"""Unit tests for Data Ingestor Agent."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agents.data_ingestor_agent import DataIngestorAgent, IngestionError
from models.schemas import SourceType


class TestDataIngestorAgent:
    """Tests for DataIngestorAgent."""

    @pytest.fixture
    def agent(self, ingestor_config):
        """Create agent instance for testing."""
        return DataIngestorAgent(config=ingestor_config)

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent.agent_name == "DataIngestorAgent"
        assert agent.event_type == "data.ingested"

    def test_ingest_csv_success(self, agent, sample_csv_file):
        """Test successful CSV ingestion."""
        result = agent.ingest_file(sample_csv_file)

        assert result is not None
        assert result.source_type == SourceType.CSV
        assert result.row_count == 5
        assert "id" in result.schema_fields
        assert "name" in result.schema_fields
        assert result.content_hash is not None
        assert result.encoding is not None

    def test_ingest_csv_encoding_detection(self, agent, tmp_path):
        """Test CSV encoding detection."""
        # Create CSV with specific content
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob\n", encoding="utf-8")

        result = agent.ingest_file(str(csv_file))
        assert result.encoding == "utf-8" or result.encoding == "ascii"

    def test_ingest_csv_delimiter_detection(self, agent, tmp_path):
        """Test CSV delimiter detection."""
        # Create tab-delimited file
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id\tname\n1\tAlice\n2\tBob\n")

        result = agent.ingest_file(str(csv_file))
        assert result.row_count == 2

    def test_ingest_json_array(self, agent, sample_json_file):
        """Test JSON array ingestion."""
        result = agent.ingest_file(sample_json_file)

        assert result is not None
        assert result.source_type == SourceType.JSON
        assert result.row_count == 3
        assert "id" in result.schema_fields
        assert "score" in result.schema_fields

    def test_ingest_json_lines(self, agent, tmp_path):
        """Test JSON Lines format ingestion."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"id": 1, "name": "Alice"}\n{"id": 2, "name": "Bob"}\n')

        result = agent.ingest_file(str(jsonl_file))
        assert result.row_count == 2

    def test_ingest_parquet(self, agent, sample_dataframe, tmp_path):
        """Test Parquet file ingestion."""
        parquet_file = tmp_path / "test.parquet"
        sample_dataframe.to_parquet(str(parquet_file), index=False)

        result = agent.ingest_file(str(parquet_file))

        assert result.source_type == SourceType.PARQUET
        assert result.row_count == 5

    def test_ingest_with_custom_dataset_id(self, agent, sample_csv_file):
        """Test ingestion with custom dataset ID."""
        custom_id = "my-custom-dataset"
        result = agent.ingest_file(sample_csv_file, dataset_id=custom_id)

        assert result.dataset_id == custom_id

    def test_ingest_sample_data(self, agent, sample_csv_file):
        """Test that sample data is included in result."""
        result = agent.ingest_file(sample_csv_file)

        assert result.sample_data is not None
        assert len(result.sample_data) <= result.sample_rows
        assert "id" in result.sample_data[0]

    def test_ingest_dtypes_mapping(self, agent, sample_csv_file):
        """Test data type inference."""
        result = agent.ingest_file(sample_csv_file)

        assert "id" in result.dtypes
        assert "int" in result.dtypes["id"].lower()
        assert "salary" in result.dtypes
        assert "float" in result.dtypes["salary"].lower()

    def test_ingest_file_not_found(self, agent):
        """Test ingestion of non-existent file raises error."""
        with pytest.raises(IngestionError, match="Failed to ingest"):
            agent.ingest_file("/nonexistent/file.csv")

    def test_ingest_unsupported_format(self, agent, tmp_path):
        """Test ingestion of unsupported format raises error."""
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("test content")

        with pytest.raises(IngestionError, match="Cannot auto-detect"):
            agent.ingest_file(str(unsupported_file))

    def test_get_dataframe(self, agent, sample_csv_file):
        """Test retrieving stored DataFrame."""
        result = agent.ingest_file(sample_csv_file)
        df = agent.get_dataframe(result.dataset_id)

        assert df is not None
        assert len(df) == result.row_count

    def test_get_dataframe_not_found(self, agent):
        """Test retrieving non-existent DataFrame returns None."""
        df = agent.get_dataframe("nonexistent-id")
        assert df is None

    def test_content_hash_consistency(self, agent, sample_csv_file):
        """Test that same content produces same hash."""
        result1 = agent.ingest_file(sample_csv_file, dataset_id="test1")
        result2 = agent.ingest_file(sample_csv_file, dataset_id="test2")

        assert result1.content_hash == result2.content_hash

    def test_content_hash_different_content(self, agent, tmp_path):
        """Test that different content produces different hash."""
        csv1 = tmp_path / "test1.csv"
        csv1.write_text("id,name\n1,Alice\n")

        csv2 = tmp_path / "test2.csv"
        csv2.write_text("id,name\n1,Bob\n")

        result1 = agent.ingest_file(str(csv1))
        result2 = agent.ingest_file(str(csv2))

        assert result1.content_hash != result2.content_hash

    def test_storage_path_created(self, agent, sample_csv_file):
        """Test that storage path is created and valid."""
        result = agent.ingest_file(sample_csv_file)

        assert result.storage_path is not None
        assert os.path.exists(result.storage_path)

    def test_event_published(self, agent, sample_csv_file):
        """Test that data.ingested event is published."""
        events_received = []

        def event_handler(event):
            events_received.append(event)

        agent.subscribe("data.ingested", event_handler)
        agent.ingest_file(sample_csv_file)

        assert len(events_received) == 1
        assert events_received[0]["event_type"] == "data.ingested"

    def test_detect_source_type_csv(self, agent):
        """Test source type detection for CSV."""
        source_type = agent._detect_source_type("/path/to/file.csv")
        assert source_type == "csv"

    def test_detect_source_type_parquet(self, agent):
        """Test source type detection for Parquet."""
        source_type = agent._detect_source_type("/path/to/file.parquet")
        assert source_type == "parquet"

    def test_detect_source_type_json(self, agent):
        """Test source type detection for JSON."""
        source_type = agent._detect_source_type("/path/to/file.json")
        assert source_type == "json"

    def test_detect_source_type_sql(self, agent):
        """Test source type detection for SQL connection string."""
        source_type = agent._detect_source_type("postgresql://user:pass@host/db")
        assert source_type == "sql"

    def test_detect_source_type_rest(self, agent):
        """Test source type detection for REST URL."""
        source_type = agent._detect_source_type("https://api.example.com/data")
        assert source_type == "rest"

    @patch("agents.data_ingestor_agent.requests.get")
    def test_ingest_rest_api(self, mock_get, agent):
        """Test REST API ingestion."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"}
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = agent.ingest_rest("https://api.example.com/users")

        assert result.source_type == SourceType.REST
        assert result.row_count == 2

    def test_correlation_id_propagation(self, ingestor_config):
        """Test that correlation ID is set correctly."""
        agent = DataIngestorAgent(
            config=ingestor_config,
            correlation_id="test-correlation-123"
        )

        assert agent.correlation_id == "test-correlation-123"

    def test_new_correlation_id(self, agent):
        """Test generating new correlation ID."""
        old_id = agent.correlation_id
        new_id = agent.new_correlation_id()

        assert new_id != old_id
        assert agent.correlation_id == new_id
