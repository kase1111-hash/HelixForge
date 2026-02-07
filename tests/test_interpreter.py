"""Unit tests for Metadata Interpreter Agent."""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from agents.metadata_interpreter_agent import MetadataInterpreterAgent
from models.schemas import DataType, SemanticType
from utils.embeddings import cosine_similarity
from utils.llm import MockProvider


class TestMetadataInterpreterAgent:
    """Tests for MetadataInterpreterAgent."""

    @pytest.fixture
    def agent(self, interpreter_config):
        """Create agent instance for testing."""
        return MetadataInterpreterAgent(config=interpreter_config)

    @pytest.fixture
    def mock_agent(self, interpreter_config, mock_openai_client):
        """Create agent with mocked OpenAI client (legacy path)."""
        agent = MetadataInterpreterAgent(config=interpreter_config)
        agent._openai_client = mock_openai_client
        return agent

    @pytest.fixture
    def provider_agent(self, interpreter_config, mock_provider):
        """Create agent with MockProvider (preferred path)."""
        return MetadataInterpreterAgent(
            config=interpreter_config, provider=mock_provider
        )

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent.agent_name == "MetadataInterpreterAgent"
        assert agent.event_type == "metadata.ready"

    def test_agent_accepts_provider(self, interpreter_config):
        """Test agent accepts an LLM provider via constructor."""
        provider = MockProvider()
        agent = MetadataInterpreterAgent(
            config=interpreter_config, provider=provider
        )
        assert agent._get_provider() is provider

    def test_agent_legacy_openai_client(self, interpreter_config, mock_openai_client):
        """Test backward compat: setting _openai_client wraps in OpenAIProvider."""
        agent = MetadataInterpreterAgent(config=interpreter_config)
        agent._openai_client = mock_openai_client
        provider = agent._get_provider()
        # Should be an OpenAIProvider wrapping the mock client
        assert provider._client is mock_openai_client

    # --- Data type inference ---

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
        """Test string type inference (object and StringDtype)."""
        series = pd.Series(["Alice", "Bob", "Charlie"])
        data_type = agent._infer_data_type(series)
        assert data_type == DataType.STRING

    def test_infer_data_type_string_explicit_dtype(self, agent):
        """Test string type inference with explicit StringDtype."""
        series = pd.Series(["Alice", "Bob", "Charlie"], dtype="string")
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

    def test_infer_data_type_datetime_from_strings(self, agent):
        """Test datetime inference from date-like strings."""
        series = pd.Series(["2020-01-01", "2020-02-01", "2020-03-01"])
        data_type = agent._infer_data_type(series)
        assert data_type == DataType.DATETIME

    # --- Heuristic inference ---

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

    # --- Embedding generation with MockProvider ---

    def test_generate_embeddings_with_provider(self, provider_agent, sample_dataframe):
        """Test embedding generation via MockProvider."""
        embeddings = provider_agent._generate_embeddings(
            list(sample_dataframe.columns),
            sample_dataframe
        )
        assert len(embeddings) == len(sample_dataframe.columns)
        assert len(embeddings[0]) == 1536
        # MockProvider should produce non-zero vectors
        assert any(v != 0.0 for v in embeddings[0])

    def test_embeddings_deterministic(self, interpreter_config):
        """Test that MockProvider produces identical embeddings for identical inputs."""
        p1 = MockProvider(dimensions=1536)
        p2 = MockProvider(dimensions=1536)
        emb1 = p1.embed(["employee_name", "salary"])
        emb2 = p2.embed(["employee_name", "salary"])
        assert emb1[0] == emb2[0]
        assert emb1[1] == emb2[1]

    def test_embeddings_similar_fields_have_higher_similarity(self):
        """Test that semantically related field names produce similar embeddings."""
        provider = MockProvider(dimensions=1536)

        # Fields that share words should be more similar
        embeddings = provider.embed([
            "employee_name",
            "worker_name",     # shares "name"
            "hire_date",       # completely different
        ])

        sim_name_worker = cosine_similarity(embeddings[0], embeddings[1])
        sim_name_date = cosine_similarity(embeddings[0], embeddings[2])

        # "employee_name" should be more similar to "worker_name" than to "hire_date"
        assert sim_name_worker > sim_name_date

    def test_embeddings_identical_inputs_have_perfect_similarity(self):
        """Test that identical inputs produce similarity = 1.0."""
        provider = MockProvider(dimensions=1536)
        embeddings = provider.embed(["salary", "salary"])
        sim = cosine_similarity(embeddings[0], embeddings[1])
        assert sim == pytest.approx(1.0)

    def test_embeddings_empty_string_returns_zero_vector(self):
        """Test that empty strings produce zero vectors."""
        provider = MockProvider(dimensions=1536)
        embeddings = provider.embed(["", "   "])
        assert all(v == 0.0 for v in embeddings[0])
        assert all(v == 0.0 for v in embeddings[1])

    def test_generate_embeddings_fallback_on_error(self, agent, sample_dataframe):
        """Test embedding generation fallback when provider fails."""
        # Agent with no provider configured will try to create OpenAI (which fails)
        # but the fallback should return zero vectors
        with patch("utils.llm.OpenAIProvider.embed", side_effect=Exception("API Error")):
            agent._provider = None
            agent._openai_client = None
            embeddings = agent._generate_embeddings(
                list(sample_dataframe.columns),
                sample_dataframe
            )
        assert len(embeddings) == len(sample_dataframe.columns)
        assert all(v == 0 for v in embeddings[0])

    # --- Domain tag inference ---

    def test_infer_domain_tags_healthcare(self, agent):
        """Test domain tag inference for healthcare data."""
        from models.schemas import FieldMetadata

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
        from models.schemas import FieldMetadata

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

    # --- Full pipeline with MockProvider ---

    def test_process_full_pipeline_with_provider(self, provider_agent, sample_dataframe):
        """Test full pipeline using MockProvider (no mocks needed)."""
        result = provider_agent.process("test-dataset", sample_dataframe)

        assert result is not None
        assert result.dataset_id == "test-dataset"
        assert len(result.fields) == len(sample_dataframe.columns)
        assert result.dataset_description is not None
        assert len(result.dataset_description) > 0
        assert len(result.domain_tags) > 0

        # Verify every field has an embedding
        for field in result.fields:
            assert field.embedding is not None
            assert len(field.embedding) == 1536

    def test_process_full_pipeline_legacy(self, mock_agent, sample_dataframe):
        """Test full pipeline via legacy _openai_client path."""
        with patch("agents.metadata_interpreter_agent.OpenAIProvider.embed") as mock_embed:
            mock_embed.return_value = [[0.1] * 1536 for _ in range(len(sample_dataframe.columns))]
            result = mock_agent.process("test-dataset", sample_dataframe)

        assert result is not None
        assert result.dataset_id == "test-dataset"
        assert len(result.fields) == len(sample_dataframe.columns)

    def test_interpret_field_with_provider(self, provider_agent, sample_dataframe):
        """Test single field interpretation with MockProvider."""
        field_meta = provider_agent._interpret_field(
            "test-dataset",
            "salary",
            sample_dataframe["salary"]
        )

        assert field_meta.field_name == "salary"
        assert field_meta.data_type == DataType.FLOAT
        assert 0 <= field_meta.null_ratio <= 1
        assert 0 <= field_meta.confidence <= 1
        # MockProvider should label salary as a metric
        assert field_meta.semantic_type == SemanticType.METRIC

    # --- Statistics ---

    def test_null_ratio_calculation(self, provider_agent):
        """Test null ratio calculation."""
        series_with_nulls = pd.Series([1, None, 3, None, 5])

        field_meta = provider_agent._interpret_field(
            "test", "col", series_with_nulls
        )
        assert field_meta.null_ratio == 0.4  # 2 nulls out of 5

    def test_unique_ratio_calculation(self, provider_agent):
        """Test unique ratio calculation."""
        series = pd.Series([1, 1, 2, 2, 3])

        field_meta = provider_agent._interpret_field(
            "test", "col", series
        )
        assert field_meta.unique_ratio == 0.6  # 3 unique out of 5

    def test_sample_values_extraction(self, provider_agent):
        """Test sample values extraction."""
        series = pd.Series(range(100))

        field_meta = provider_agent._interpret_field(
            "test", "col", series
        )
        assert len(field_meta.sample_values) <= 5  # Limited to 5 in output

    # --- Events ---

    def test_event_published(self, provider_agent, sample_dataframe):
        """Test that metadata.ready event is published."""
        events_received = []

        def event_handler(event):
            events_received.append(event)

        provider_agent.subscribe("metadata.ready", event_handler)
        provider_agent.process("test-dataset", sample_dataframe)

        assert len(events_received) == 1
        assert events_received[0]["event_type"] == "metadata.ready"


class TestMockProviderSemantics:
    """Tests validating MockProvider produces semantically meaningful results."""

    @pytest.fixture
    def provider(self):
        return MockProvider(dimensions=1536)

    def test_complete_returns_valid_json(self, provider):
        """Test that complete() returns parseable JSON for field prompts."""
        result_text = provider.complete(
            messages=[
                {"role": "system", "content": "You are a data analyst."},
                {"role": "user", "content": "Field Name: employee_id\nData Type: integer\nSample Values: [1, 2, 3]"}
            ]
        )
        result = json.loads(result_text)
        assert "semantic_label" in result
        assert "semantic_type" in result
        assert "confidence" in result
        assert result["semantic_type"] == "identifier"

    def test_complete_metric_field(self, provider):
        """Test metric field detection."""
        result = json.loads(provider.complete(
            messages=[{"role": "user", "content": "Field Name: total_salary\nData Type: float"}]
        ))
        assert result["semantic_type"] == "metric"

    def test_complete_timestamp_field(self, provider):
        """Test timestamp field detection."""
        result = json.loads(provider.complete(
            messages=[{"role": "user", "content": "Field Name: hire_date\nData Type: datetime"}]
        ))
        assert result["semantic_type"] == "timestamp"

    def test_complete_category_field(self, provider):
        """Test category field detection."""
        result = json.loads(provider.complete(
            messages=[{"role": "user", "content": "Field Name: department\nData Type: string"}]
        ))
        assert result["semantic_type"] == "category"

    def test_complete_text_field(self, provider):
        """Test text field detection."""
        result = json.loads(provider.complete(
            messages=[{"role": "user", "content": "Field Name: employee_name\nData Type: string"}]
        ))
        assert result["semantic_type"] == "text"

    def test_complete_dataset_description_fallback(self, provider):
        """Test that prompts without field info return a description string."""
        result = provider.complete(
            messages=[
                {"role": "system", "content": "You are a data doc expert."},
                {"role": "user", "content": "Describe this dataset."}
            ]
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_embedding_similarity_matrix(self, provider):
        """Test that embedding similarities reflect shared word content.

        MockProvider uses word-level hashing, so fields sharing words
        produce more similar embeddings than fields with disjoint words.
        """
        fields = [
            "employee_id",        # 0: shares "employee" with 2, "id" with 1
            "worker_id",          # 1: shares "id" with 0
            "employee_name",      # 2: shares "employee" with 0, "name" with 3
            "worker_name",        # 3: shares "name" with 2, "worker" with 1
            "hire_date",          # 4: disjoint from 0-3
        ]
        embeddings = provider.embed(fields)

        sim_emp_id_worker_id = cosine_similarity(embeddings[0], embeddings[1])  # share "id"
        sim_emp_id_emp_name = cosine_similarity(embeddings[0], embeddings[2])   # share "employee"
        sim_emp_id_hire_date = cosine_similarity(embeddings[0], embeddings[4])  # no shared words

        # Fields sharing a word should be more similar than disjoint fields
        assert sim_emp_id_worker_id > sim_emp_id_hire_date, \
            "employee_id should be more similar to worker_id (shared 'id') than hire_date"
        assert sim_emp_id_emp_name > sim_emp_id_hire_date, \
            "employee_id should be more similar to employee_name (shared 'employee') than hire_date"
