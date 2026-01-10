"""Unit tests for Metadata Interpreter Agent."""

from unittest.mock import patch

import pandas as pd
import pytest

from agents.metadata_interpreter_agent import MetadataInterpreterAgent
from models.schemas import DataType, SemanticType


class TestMetadataInterpreterAgent:
    """Tests for MetadataInterpreterAgent."""

    @pytest.fixture
    def agent(self, interpreter_config):
        """Create agent instance for testing."""
        return MetadataInterpreterAgent(config=interpreter_config)

    @pytest.fixture
    def mock_agent(self, interpreter_config, mock_openai_client):
        """Create agent with mocked OpenAI client."""
        agent = MetadataInterpreterAgent(config=interpreter_config)
        agent._openai_client = mock_openai_client
        return agent

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent.agent_name == "MetadataInterpreterAgent"
        assert agent.event_type == "metadata.ready"

    def test_infer_data_type_integer(self, agent):
        """Test integer type inference."""
        series = pd.Series([1, 2, 3, 4, 5])
        data_type = agent._infer_data_type(series)
        assert data_type == DataType.INTEGER

    def test_infer_data_type_float(self, agent):
        """Test float type inference."""
        series = pd.Series([1.5, 2.5, 3.5])
        data_type = agent._infer_data_type(series)
        assert data_type == DataType.FLOAT

    def test_infer_data_type_string(self, agent):
        """Test string type inference."""
        series = pd.Series(["Alice", "Bob", "Charlie"])
        data_type = agent._infer_data_type(series)
        assert data_type == DataType.STRING

    def test_infer_data_type_boolean(self, agent):
        """Test boolean type inference."""
        series = pd.Series([True, False, True])
        data_type = agent._infer_data_type(series)
        assert data_type == DataType.BOOLEAN

    def test_infer_data_type_datetime(self, agent):
        """Test datetime type inference."""
        series = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
        data_type = agent._infer_data_type(series)
        assert data_type == DataType.DATETIME

    def test_heuristic_inference_identifier(self, agent):
        """Test heuristic inference for identifier fields."""
        result = agent._heuristic_inference("user_id", DataType.INTEGER, 1.0)

        assert result["semantic_type"] == "identifier"
        assert result["confidence"] > 0

    def test_heuristic_inference_timestamp(self, agent):
        """Test heuristic inference for timestamp fields."""
        result = agent._heuristic_inference("created_at", DataType.DATETIME, 0.9)

        assert result["semantic_type"] == "timestamp"

    def test_heuristic_inference_metric(self, agent):
        """Test heuristic inference for metric fields."""
        result = agent._heuristic_inference("total_amount", DataType.FLOAT, 0.8)

        assert result["semantic_type"] == "metric"

    def test_heuristic_inference_category(self, agent):
        """Test heuristic inference for category fields."""
        result = agent._heuristic_inference("status", DataType.STRING, 0.05)

        assert result["semantic_type"] == "category"

    @patch("agents.metadata_interpreter_agent.batch_embed")
    def test_generate_embeddings(self, mock_embed, agent, sample_dataframe):
        """Test embedding generation."""
        mock_embed.return_value = [[0.1] * 1536 for _ in range(len(sample_dataframe.columns))]

        embeddings = agent._generate_embeddings(
            list(sample_dataframe.columns),
            sample_dataframe
        )

        assert len(embeddings) == len(sample_dataframe.columns)
        assert len(embeddings[0]) == 1536

    @patch("agents.metadata_interpreter_agent.batch_embed")
    def test_generate_embeddings_fallback(self, mock_embed, agent, sample_dataframe):
        """Test embedding generation fallback on error."""
        mock_embed.side_effect = Exception("API Error")

        embeddings = agent._generate_embeddings(
            list(sample_dataframe.columns),
            sample_dataframe
        )

        # Should return zero vectors on failure
        assert len(embeddings) == len(sample_dataframe.columns)
        assert all(v == 0 for v in embeddings[0])

    def test_infer_domain_tags_healthcare(self, agent):
        """Test domain tag inference for healthcare data."""
        from models.schemas import FieldMetadata, DataType

        fields = [
            FieldMetadata(
                dataset_id="test",
                field_name="patient_id",
                semantic_label="Patient ID",
                description="Patient identifier",
                data_type=DataType.STRING,
                semantic_type=SemanticType.IDENTIFIER,
                null_ratio=0,
                unique_ratio=1,
                confidence=0.9
            ),
            FieldMetadata(
                dataset_id="test",
                field_name="diagnosis",
                semantic_label="Diagnosis",
                description="Medical diagnosis",
                data_type=DataType.STRING,
                semantic_type=SemanticType.TEXT,
                null_ratio=0,
                unique_ratio=0.5,
                confidence=0.8
            )
        ]

        tags = agent._infer_domain_tags(fields, "Medical patient records")
        assert "healthcare" in tags

    def test_infer_domain_tags_finance(self, agent):
        """Test domain tag inference for finance data."""
        from models.schemas import FieldMetadata, DataType

        fields = [
            FieldMetadata(
                dataset_id="test",
                field_name="transaction_id",
                semantic_label="Transaction ID",
                description="Transaction identifier",
                data_type=DataType.STRING,
                semantic_type=SemanticType.IDENTIFIER,
                null_ratio=0,
                unique_ratio=1,
                confidence=0.9
            ),
            FieldMetadata(
                dataset_id="test",
                field_name="revenue",
                semantic_label="Revenue",
                description="Revenue amount",
                data_type=DataType.FLOAT,
                semantic_type=SemanticType.METRIC,
                null_ratio=0,
                unique_ratio=0.8,
                confidence=0.85
            )
        ]

        tags = agent._infer_domain_tags(fields, "Financial transaction records")
        assert "finance" in tags

    @patch("agents.metadata_interpreter_agent.batch_embed")
    def test_process_full_pipeline(self, mock_embed, mock_agent, sample_dataframe):
        """Test full metadata interpretation pipeline."""
        mock_embed.return_value = [[0.1] * 1536 for _ in range(len(sample_dataframe.columns))]

        result = mock_agent.process("test-dataset", sample_dataframe)

        assert result is not None
        assert result.dataset_id == "test-dataset"
        assert len(result.fields) == len(sample_dataframe.columns)
        assert result.dataset_description is not None

    @patch("agents.metadata_interpreter_agent.batch_embed")
    def test_interpret_field(self, mock_embed, mock_agent, sample_dataframe):
        """Test single field interpretation."""
        mock_embed.return_value = [[0.1] * 1536]

        field_meta = mock_agent._interpret_field(
            "test-dataset",
            "salary",
            sample_dataframe["salary"]
        )

        assert field_meta.field_name == "salary"
        assert field_meta.data_type == DataType.FLOAT
        assert 0 <= field_meta.null_ratio <= 1
        assert 0 <= field_meta.confidence <= 1

    def test_null_ratio_calculation(self, agent):
        """Test null ratio calculation."""
        series_with_nulls = pd.Series([1, None, 3, None, 5])

        field_meta = agent._interpret_field(
            "test", "col", series_with_nulls
        )

        assert field_meta.null_ratio == 0.4  # 2 nulls out of 5

    def test_unique_ratio_calculation(self, agent):
        """Test unique ratio calculation."""
        series = pd.Series([1, 1, 2, 2, 3])

        field_meta = agent._interpret_field(
            "test", "col", series
        )

        assert field_meta.unique_ratio == 0.6  # 3 unique out of 5

    def test_sample_values_extraction(self, agent):
        """Test sample values extraction."""
        series = pd.Series(range(100))

        field_meta = agent._interpret_field(
            "test", "col", series
        )

        assert len(field_meta.sample_values) <= 5  # Limited to 5 in output

    def test_event_published(self, mock_agent, sample_dataframe):
        """Test that metadata.ready event is published."""
        events_received = []

        def event_handler(event):
            events_received.append(event)

        mock_agent.subscribe("metadata.ready", event_handler)

        with patch("agents.metadata_interpreter_agent.batch_embed") as mock_embed:
            mock_embed.return_value = [[0.1] * 1536 for _ in range(len(sample_dataframe.columns))]
            mock_agent.process("test-dataset", sample_dataframe)

        assert len(events_received) == 1
        assert events_received[0]["event_type"] == "metadata.ready"
