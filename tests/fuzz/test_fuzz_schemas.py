"""
Fuzz Testing for Data Schemas and Models

Uses Hypothesis for property-based testing to validate
schema robustness with random inputs.

Run with:
    pytest tests/fuzz/test_fuzz_schemas.py -v
"""

import os
import sys
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.schemas import (
    FieldMetadata,
    DatasetMetadata,
    IngestResult,
    SemanticLabel,
    OntologyMapping,
    FusedRecord,
    Insight,
    ProvenanceRecord,
)


class TestFuzzFieldMetadata:
    """Fuzz tests for FieldMetadata schema."""

    @given(
        name=st.text(max_size=100),
        data_type=st.text(max_size=50),
        nullable=st.booleans(),
    )
    @settings(max_examples=100)
    def test_fuzz_field_metadata_creation(self, name, data_type, nullable):
        """Fuzz test: FieldMetadata with random values."""
        # Assume name is not empty (required field)
        assume(name.strip())

        try:
            field = FieldMetadata(
                name=name,
                data_type=data_type if data_type.strip() else "string",
                nullable=nullable,
                sample_values=[],
            )
            assert field.name == name
        except (ValueError, TypeError):
            pass

    @given(
        sample_values=st.lists(st.text(max_size=100), max_size=20),
    )
    @settings(max_examples=50)
    def test_fuzz_field_sample_values(self, sample_values):
        """Fuzz test: Sample values with random strings."""
        try:
            field = FieldMetadata(
                name="test_field",
                data_type="string",
                sample_values=sample_values,
            )
            assert field.sample_values == sample_values
        except (ValueError, TypeError):
            pass


class TestFuzzDatasetMetadata:
    """Fuzz tests for DatasetMetadata schema."""

    @given(
        name=st.text(min_size=1, max_size=100),
        source=st.text(max_size=200),
        row_count=st.integers(),
        column_count=st.integers(),
    )
    @settings(max_examples=100)
    def test_fuzz_dataset_metadata(self, name, source, row_count, column_count):
        """Fuzz test: DatasetMetadata with random values."""
        assume(name.strip())

        try:
            metadata = DatasetMetadata(
                id=str(uuid4()),
                name=name,
                source=source,
                row_count=max(0, row_count),  # Non-negative
                column_count=max(0, column_count),
                fields=[],
                created_at=datetime.now(),
            )
            assert metadata.name == name
        except (ValueError, TypeError):
            pass

    @given(
        num_fields=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=30)
    def test_fuzz_dataset_with_fields(self, num_fields):
        """Fuzz test: Dataset with varying number of fields."""
        fields = [
            FieldMetadata(
                name=f"field_{i}",
                data_type="string",
                sample_values=[],
            )
            for i in range(num_fields)
        ]

        try:
            metadata = DatasetMetadata(
                id=str(uuid4()),
                name="test_dataset",
                source="test",
                row_count=100,
                column_count=num_fields,
                fields=fields,
                created_at=datetime.now(),
            )
            assert len(metadata.fields) == num_fields
        except (ValueError, TypeError):
            pass


class TestFuzzSemanticLabel:
    """Fuzz tests for SemanticLabel schema."""

    @given(
        field_name=st.text(min_size=1, max_size=50),
        label=st.text(max_size=100),
        confidence=st.floats(min_value=0, max_value=1),
    )
    @settings(max_examples=100)
    def test_fuzz_semantic_label(self, field_name, label, confidence):
        """Fuzz test: SemanticLabel with random values."""
        assume(field_name.strip())

        try:
            semantic_label = SemanticLabel(
                field_name=field_name,
                label=label if label.strip() else "unknown",
                confidence=confidence,
            )
            assert 0 <= semantic_label.confidence <= 1
        except (ValueError, TypeError):
            pass

    @given(
        confidence=st.floats(),
    )
    @settings(max_examples=50)
    def test_fuzz_semantic_label_confidence_bounds(self, confidence):
        """Fuzz test: Confidence bounds validation."""
        try:
            semantic_label = SemanticLabel(
                field_name="test",
                label="test_label",
                confidence=confidence,
            )
            # If it passes validation, confidence should be valid
            assert 0 <= semantic_label.confidence <= 1
        except (ValueError, TypeError):
            # Expected for out-of-bounds values
            pass


class TestFuzzOntologyMapping:
    """Fuzz tests for OntologyMapping schema."""

    @given(
        source_field=st.text(min_size=1, max_size=50),
        target_concept=st.text(min_size=1, max_size=100),
        similarity_score=st.floats(min_value=0, max_value=1),
    )
    @settings(max_examples=100)
    def test_fuzz_ontology_mapping(self, source_field, target_concept, similarity_score):
        """Fuzz test: OntologyMapping with random values."""
        assume(source_field.strip() and target_concept.strip())

        try:
            mapping = OntologyMapping(
                source_field=source_field,
                target_concept=target_concept,
                similarity_score=similarity_score,
            )
            assert 0 <= mapping.similarity_score <= 1
        except (ValueError, TypeError):
            pass


class TestFuzzFusedRecord:
    """Fuzz tests for FusedRecord schema."""

    @given(
        source_datasets=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10),
        fields=st.dictionaries(
            st.text(min_size=1, max_size=30),
            st.one_of(st.text(max_size=100), st.integers(), st.floats(allow_nan=False)),
            max_size=20,
        ),
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    def test_fuzz_fused_record(self, source_datasets, fields):
        """Fuzz test: FusedRecord with random sources and fields."""
        assume(all(s.strip() for s in source_datasets))
        assume(all(k.strip() for k in fields.keys()))

        try:
            record = FusedRecord(
                id=str(uuid4()),
                source_datasets=source_datasets,
                fields=fields,
                confidence=0.9,
            )
            assert len(record.source_datasets) == len(source_datasets)
        except (ValueError, TypeError):
            pass


class TestFuzzInsight:
    """Fuzz tests for Insight schema."""

    @given(
        title=st.text(min_size=1, max_size=200),
        description=st.text(max_size=1000),
        insight_type=st.sampled_from(["correlation", "anomaly", "trend", "cluster", "pattern"]),
        confidence=st.floats(min_value=0, max_value=1),
    )
    @settings(max_examples=100)
    def test_fuzz_insight(self, title, description, insight_type, confidence):
        """Fuzz test: Insight with random values."""
        assume(title.strip())

        try:
            insight = Insight(
                id=str(uuid4()),
                title=title,
                description=description,
                insight_type=insight_type,
                confidence=confidence,
                supporting_data={},
                created_at=datetime.now(),
            )
            assert insight.title == title
        except (ValueError, TypeError):
            pass

    @given(
        supporting_data=st.dictionaries(
            st.text(min_size=1, max_size=30),
            st.one_of(
                st.text(max_size=100),
                st.integers(),
                st.floats(allow_nan=False),
                st.lists(st.integers(), max_size=10),
            ),
            max_size=10,
        ),
    )
    @settings(max_examples=50)
    def test_fuzz_insight_supporting_data(self, supporting_data):
        """Fuzz test: Insight with random supporting data."""
        assume(all(k.strip() for k in supporting_data.keys()))

        try:
            insight = Insight(
                id=str(uuid4()),
                title="Test Insight",
                description="Test description",
                insight_type="pattern",
                confidence=0.9,
                supporting_data=supporting_data,
                created_at=datetime.now(),
            )
            assert insight.supporting_data == supporting_data
        except (ValueError, TypeError):
            pass


class TestFuzzProvenanceRecord:
    """Fuzz tests for ProvenanceRecord schema."""

    @given(
        entity_id=st.text(min_size=1, max_size=50),
        entity_type=st.text(min_size=1, max_size=30),
        operation=st.text(min_size=1, max_size=50),
        agent=st.text(max_size=100),
    )
    @settings(max_examples=100)
    def test_fuzz_provenance_record(self, entity_id, entity_type, operation, agent):
        """Fuzz test: ProvenanceRecord with random values."""
        assume(entity_id.strip() and entity_type.strip() and operation.strip())

        try:
            provenance = ProvenanceRecord(
                id=str(uuid4()),
                entity_id=entity_id,
                entity_type=entity_type,
                operation=operation,
                agent=agent if agent.strip() else "unknown",
                timestamp=datetime.now(),
                inputs=[],
                outputs=[],
            )
            assert provenance.entity_id == entity_id
        except (ValueError, TypeError):
            pass

    @given(
        inputs=st.lists(st.text(min_size=1, max_size=50), max_size=20),
        outputs=st.lists(st.text(min_size=1, max_size=50), max_size=20),
    )
    @settings(max_examples=50)
    def test_fuzz_provenance_io(self, inputs, outputs):
        """Fuzz test: Provenance inputs and outputs."""
        inputs = [i for i in inputs if i.strip()]
        outputs = [o for o in outputs if o.strip()]

        try:
            provenance = ProvenanceRecord(
                id=str(uuid4()),
                entity_id="test_entity",
                entity_type="dataset",
                operation="transform",
                agent="test_agent",
                timestamp=datetime.now(),
                inputs=inputs,
                outputs=outputs,
            )
            assert len(provenance.inputs) == len(inputs)
            assert len(provenance.outputs) == len(outputs)
        except (ValueError, TypeError):
            pass


class TestFuzzDateTimeHandling:
    """Fuzz tests for datetime handling in schemas."""

    @given(
        days_offset=st.integers(min_value=-10000, max_value=10000),
    )
    @settings(max_examples=50)
    def test_fuzz_datetime_range(self, days_offset):
        """Fuzz test: Datetime with various offsets."""
        try:
            timestamp = datetime.now() + timedelta(days=days_offset)
            provenance = ProvenanceRecord(
                id=str(uuid4()),
                entity_id="test",
                entity_type="dataset",
                operation="test",
                agent="test",
                timestamp=timestamp,
                inputs=[],
                outputs=[],
            )
            assert provenance.timestamp == timestamp
        except (ValueError, OverflowError):
            # Expected for extreme values
            pass
