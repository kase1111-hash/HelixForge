"""Unit tests for Ontology Alignment Agent."""

from unittest.mock import MagicMock, patch

import pytest

from agents.ontology_alignment_agent import OntologyAlignmentAgent
from models.schemas import (
    AlignmentType,
    DatasetMetadata,
    DataType,
    FieldMetadata,
    SemanticType,
)


class TestOntologyAlignmentAgent:
    """Tests for OntologyAlignmentAgent."""

    @pytest.fixture
    def agent(self, alignment_config):
        """Create agent instance for testing."""
        return OntologyAlignmentAgent(config=alignment_config)

    @pytest.fixture
    def metadata_a(self):
        """Create first dataset metadata for testing."""
        fields = [
            FieldMetadata(
                dataset_id="dataset-a",
                field_name="employee_id",
                semantic_label="Employee ID",
                description="Unique employee identifier",
                data_type=DataType.INTEGER,
                semantic_type=SemanticType.IDENTIFIER,
                embedding=[0.1] * 1536,
                sample_values=[1, 2, 3],
                null_ratio=0,
                unique_ratio=1.0,
                confidence=0.95
            ),
            FieldMetadata(
                dataset_id="dataset-a",
                field_name="salary",
                semantic_label="Annual Salary",
                description="Employee annual salary",
                data_type=DataType.FLOAT,
                semantic_type=SemanticType.METRIC,
                embedding=[0.2] * 1536,
                sample_values=[75000, 80000, 85000],
                null_ratio=0,
                unique_ratio=0.9,
                confidence=0.9
            ),
            FieldMetadata(
                dataset_id="dataset-a",
                field_name="department",
                semantic_label="Department",
                description="Employee department",
                data_type=DataType.STRING,
                semantic_type=SemanticType.CATEGORY,
                embedding=[0.3] * 1536,
                sample_values=["Engineering", "Sales", "Marketing"],
                null_ratio=0,
                unique_ratio=0.1,
                confidence=0.85
            )
        ]

        return DatasetMetadata(
            dataset_id="dataset-a",
            fields=fields,
            dataset_description="Employee records",
            domain_tags=["hr"]
        )

    @pytest.fixture
    def metadata_b(self):
        """Create second dataset metadata for testing."""
        fields = [
            FieldMetadata(
                dataset_id="dataset-b",
                field_name="emp_id",
                semantic_label="Employee ID",
                description="Employee identifier",
                data_type=DataType.INTEGER,
                semantic_type=SemanticType.IDENTIFIER,
                embedding=[0.1] * 1536,  # Same as dataset-a employee_id
                sample_values=[1, 2, 3],
                null_ratio=0,
                unique_ratio=1.0,
                confidence=0.95
            ),
            FieldMetadata(
                dataset_id="dataset-b",
                field_name="annual_compensation",
                semantic_label="Annual Compensation",
                description="Yearly pay",
                data_type=DataType.FLOAT,
                semantic_type=SemanticType.METRIC,
                embedding=[0.2] * 1536,  # Same as dataset-a salary
                sample_values=[75000, 80000, 85000],
                null_ratio=0,
                unique_ratio=0.9,
                confidence=0.9
            ),
            FieldMetadata(
                dataset_id="dataset-b",
                field_name="team",
                semantic_label="Team Name",
                description="Employee team",
                data_type=DataType.STRING,
                semantic_type=SemanticType.CATEGORY,
                embedding=[0.35] * 1536,  # Slightly different
                sample_values=["Eng", "Sales", "Mkt"],
                null_ratio=0,
                unique_ratio=0.1,
                confidence=0.8
            )
        ]

        return DatasetMetadata(
            dataset_id="dataset-b",
            fields=fields,
            dataset_description="Compensation data",
            domain_tags=["hr", "finance"]
        )

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent.agent_name == "OntologyAlignmentAgent"
        assert agent.event_type == "ontology.aligned"

    def test_compute_similarity_identical(self, agent, metadata_a):
        """Test similarity computation for identical embeddings."""
        field = metadata_a.fields[0]
        similarity = agent._compute_similarity(field, field)
        assert similarity == 1.0

    def test_compute_similarity_different(self, agent, metadata_a):
        """Test similarity computation for different embeddings."""
        field_a = metadata_a.fields[0]
        field_b = metadata_a.fields[2]

        similarity = agent._compute_similarity(field_a, field_b)
        assert similarity < 1.0

    def test_classify_alignment_exact(self, agent):
        """Test alignment classification for exact match."""
        alignment_type = agent._classify_alignment(0.99)
        assert alignment_type == AlignmentType.EXACT

    def test_classify_alignment_synonym(self, agent):
        """Test alignment classification for synonym."""
        alignment_type = agent._classify_alignment(0.92)
        assert alignment_type == AlignmentType.SYNONYM

    def test_classify_alignment_related(self, agent):
        """Test alignment classification for related."""
        alignment_type = agent._classify_alignment(0.87)
        assert alignment_type == AlignmentType.RELATED

    def test_classify_alignment_partial(self, agent):
        """Test alignment classification for partial."""
        alignment_type = agent._classify_alignment(0.75)
        assert alignment_type == AlignmentType.PARTIAL

    def test_suggest_transformation_type_cast(self, agent, metadata_a, metadata_b):
        """Test transformation suggestion for type cast."""
        # Modify field to have different type
        field_a = metadata_a.fields[0]
        field_b = FieldMetadata(
            dataset_id="dataset-b",
            field_name="emp_id",
            semantic_label="Employee ID",
            description="Employee identifier",
            data_type=DataType.STRING,  # Different type
            semantic_type=SemanticType.IDENTIFIER,
            embedding=[0.1] * 1536,
            sample_values=["1", "2", "3"],
            null_ratio=0,
            unique_ratio=1.0,
            confidence=0.9
        )

        hint = agent._suggest_transformation(field_a, field_b)
        assert hint is not None
        assert "type_cast" in hint

    def test_align_datasets(self, agent, metadata_a, metadata_b):
        """Test dataset alignment."""
        alignments = agent._align_datasets(metadata_a, metadata_b)

        assert len(alignments) > 0

        # Check that employee_id and emp_id are aligned
        id_alignments = [
            a for a in alignments
            if a.source_field == "employee_id" or a.target_field == "emp_id"
        ]
        assert len(id_alignments) > 0

    def test_resolve_conflicts_highest_similarity(self, agent):
        """Test conflict resolution with highest similarity strategy."""
        from models.schemas import FieldAlignment

        alignments = [
            FieldAlignment(
                alignment_id="a1",
                source_dataset="ds-a",
                source_field="field1",
                target_dataset="ds-b",
                target_field="target",
                similarity=0.8,
                alignment_type=AlignmentType.RELATED
            ),
            FieldAlignment(
                alignment_id="a2",
                source_dataset="ds-a",
                source_field="field2",
                target_dataset="ds-b",
                target_field="target",
                similarity=0.9,
                alignment_type=AlignmentType.SYNONYM
            )
        ]

        resolved = agent._resolve_conflicts(alignments)

        # Should keep only the higher similarity alignment
        assert len(resolved) == 1
        assert resolved[0].alignment_id == "a2"

    @patch.object(OntologyAlignmentAgent, "_build_ontology_graph")
    def test_process_full_pipeline(self, mock_graph, agent, metadata_a, metadata_b):
        """Test full alignment process."""
        mock_graph.return_value = None

        result = agent.process([metadata_a, metadata_b])

        assert result is not None
        assert result.alignment_job_id is not None
        assert len(result.datasets_aligned) == 2
        assert len(result.alignments) > 0

    def test_unmatched_fields_detection(self, agent, metadata_a):
        """Test detection of unmatched fields."""
        # Create metadata with unique field
        metadata_c = DatasetMetadata(
            dataset_id="dataset-c",
            fields=[
                FieldMetadata(
                    dataset_id="dataset-c",
                    field_name="unique_field",
                    semantic_label="Unique Field",
                    description="A unique field",
                    data_type=DataType.STRING,
                    semantic_type=SemanticType.TEXT,
                    embedding=[0.9] * 1536,  # Very different embedding
                    sample_values=["x", "y", "z"],
                    null_ratio=0,
                    unique_ratio=1.0,
                    confidence=0.7
                )
            ],
            dataset_description="Dataset with unique field",
            domain_tags=["other"]
        )

        with patch.object(agent, "_build_ontology_graph", return_value=None):
            result = agent.process([metadata_a, metadata_c])

        # The unique field should be unmatched
        assert len(result.unmatched_fields) > 0

    def test_event_published(self, agent, metadata_a, metadata_b):
        """Test that ontology.aligned event is published."""
        events_received = []

        def event_handler(event):
            events_received.append(event)

        agent.subscribe("ontology.aligned", event_handler)

        with patch.object(agent, "_build_ontology_graph", return_value=None):
            agent.process([metadata_a, metadata_b])

        assert len(events_received) == 1
        assert events_received[0]["event_type"] == "ontology.aligned"

    def test_validate_alignment(self, agent):
        """Test alignment validation."""
        result = agent.validate_alignment("align-123", validated=True)
        assert result is True

    def test_similarity_fallback_no_embeddings(self, agent):
        """Test similarity computation falls back when no embeddings."""
        field_a = FieldMetadata(
            dataset_id="ds-a",
            field_name="salary",
            semantic_label="Annual Salary",
            description="Employee salary",
            data_type=DataType.FLOAT,
            semantic_type=SemanticType.METRIC,
            embedding=None,  # No embedding
            sample_values=[75000],
            null_ratio=0,
            unique_ratio=0.9,
            confidence=0.9
        )
        field_b = FieldMetadata(
            dataset_id="ds-b",
            field_name="compensation",
            semantic_label="Annual Compensation",
            description="Employee compensation",
            data_type=DataType.FLOAT,
            semantic_type=SemanticType.METRIC,
            embedding=None,  # No embedding
            sample_values=[75000],
            null_ratio=0,
            unique_ratio=0.9,
            confidence=0.9
        )

        similarity = agent._compute_similarity(field_a, field_b)
        assert 0 <= similarity <= 1
