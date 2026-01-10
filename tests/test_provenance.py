"""Unit tests for Provenance Tracker Agent."""

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from agents.provenance_tracker_agent import ProvenanceTrackerAgent
from models.schemas import IngestResult, ProvenanceOperation, SourceType


class TestProvenanceTrackerAgent:
    """Tests for ProvenanceTrackerAgent."""

    @pytest.fixture
    def agent(self, provenance_config):
        """Create agent instance for testing."""
        return ProvenanceTrackerAgent(config=provenance_config)

    @pytest.fixture
    def sample_ingest_result(self):
        """Create sample ingest result."""
        return IngestResult(
            dataset_id="test-dataset-001",
            source="/data/test.csv",
            source_type=SourceType.CSV,
            schema=["id", "name", "salary", "department"],
            dtypes={"id": "int64", "name": "object", "salary": "float64", "department": "object"},
            row_count=100,
            sample_rows=5,
            sample_data=[],
            content_hash="abc123",
            encoding="utf-8",
            storage_path="/tmp/test.parquet"
        )

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent.agent_name == "ProvenanceTrackerAgent"
        assert agent.event_type == "trace.updated"

    def test_record_ingestion(self, agent, sample_ingest_result):
        """Test recording data ingestion."""
        agent.record_ingestion(sample_ingest_result)

        # Check traces were created
        assert len(agent._traces) == 4  # 4 fields

        # Check each field has a trace
        for field in sample_ingest_result.schema_fields:
            trace_key = f"{sample_ingest_result.dataset_id}.{field}"
            assert trace_key in agent._traces

            trace = agent._traces[trace_key]
            assert trace.field == field
            assert len(trace.origins) == 1
            assert trace.lineage_depth == 0
            assert trace.confidence == 1.0

    def test_record_ingestion_origin_details(self, agent, sample_ingest_result):
        """Test origin details are recorded correctly."""
        agent.record_ingestion(sample_ingest_result)

        trace_key = f"{sample_ingest_result.dataset_id}.id"
        trace = agent._traces[trace_key]

        origin = trace.origins[0]
        assert origin.source_file == sample_ingest_result.source
        assert origin.source_column == "id"
        assert origin.source_column_index == 0
        assert origin.dataset_id == sample_ingest_result.dataset_id
        assert origin.content_hash == sample_ingest_result.content_hash

    def test_record_transformation(self, agent, sample_ingest_result):
        """Test recording transformation."""
        agent.record_ingestion(sample_ingest_result)

        # Record a transformation
        agent.record_transformation(
            source_fields=[f"{sample_ingest_result.dataset_id}.salary"],
            target_field="salary_usd",
            operation="transform",
            parameters={"conversion": "eur_to_usd"},
            fused_dataset_id="fused-001",
            agent="FusionAgent"
        )

        # Check new trace was created
        trace_key = "fused-001.salary_usd"
        assert trace_key in agent._traces

        trace = agent._traces[trace_key]
        assert trace.lineage_depth == 1
        assert len(trace.transformations) == 1
        assert trace.confidence < 1.0  # Should be reduced

    def test_record_alignment(self, agent):
        """Test recording field alignment."""
        agent.record_alignment(
            source_dataset="dataset-a",
            source_field="employee_id",
            target_dataset="dataset-b",
            target_field="emp_id",
            similarity=0.95
        )

        # Check transformation was recorded
        assert len(agent._transformations) == 1

        transform = agent._transformations[0]
        assert transform.operation == ProvenanceOperation.ALIGN

    def test_record_fusion(self, agent, sample_ingest_result):
        """Test recording dataset fusion."""
        agent.record_ingestion(sample_ingest_result)

        field_mappings = {
            "merged_salary": [f"{sample_ingest_result.dataset_id}.salary"],
            "merged_dept": [f"{sample_ingest_result.dataset_id}.department"]
        }

        agent.record_fusion(
            source_datasets=[sample_ingest_result.dataset_id],
            fused_dataset_id="fused-001",
            join_strategy="semantic_similarity",
            field_mappings=field_mappings
        )

        # Check traces were created for merged fields
        assert "fused-001.merged_salary" in agent._traces
        assert "fused-001.merged_dept" in agent._traces

    def test_query_lineage(self, agent, sample_ingest_result):
        """Test querying lineage for a field."""
        agent.record_ingestion(sample_ingest_result)

        trace = agent.query_lineage(
            sample_ingest_result.dataset_id,
            "salary"
        )

        assert trace is not None
        assert trace.field == "salary"
        assert len(trace.origins) == 1

    def test_query_lineage_not_found(self, agent):
        """Test querying lineage for non-existent field."""
        trace = agent.query_lineage("nonexistent", "field")
        assert trace is None

    def test_build_lineage_graph(self, agent, sample_ingest_result):
        """Test building lineage graph."""
        agent.record_ingestion(sample_ingest_result)

        graph = agent.build_lineage_graph(sample_ingest_result.dataset_id)

        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) > 0

    def test_confidence_decay(self, agent, sample_ingest_result):
        """Test that confidence decays with transformations."""
        agent.record_ingestion(sample_ingest_result)

        # Multiple transformations
        for i in range(5):
            agent.record_transformation(
                source_fields=[f"{sample_ingest_result.dataset_id}.salary"] if i == 0 else [f"fused-{i-1}.transformed"],
                target_field="transformed",
                operation="transform",
                parameters={"step": i},
                fused_dataset_id=f"fused-{i}",
                agent="FusionAgent"
            )

        # Check final confidence is lower
        final_trace = agent._traces["fused-4.transformed"]
        assert final_trace.confidence < 1.0

    def test_process_generates_report(self, agent, sample_ingest_result):
        """Test that process generates a report."""
        agent.record_ingestion(sample_ingest_result)

        report = agent.process(sample_ingest_result.dataset_id)

        assert report is not None
        assert report.fused_dataset_id == sample_ingest_result.dataset_id
        assert report.total_fields > 0
        assert report.coverage_percentage > 0

    def test_report_coverage_calculation(self, agent, sample_ingest_result):
        """Test report coverage calculation."""
        agent.record_ingestion(sample_ingest_result)

        report = agent.process(sample_ingest_result.dataset_id)

        # All fields should have complete provenance (direct from source)
        assert report.coverage_percentage == 1.0

    def test_export_report_json(self, agent, sample_ingest_result):
        """Test JSON report export."""
        agent.record_ingestion(sample_ingest_result)
        report = agent.process(sample_ingest_result.dataset_id)

        path = agent._export_report(report)

        assert os.path.exists(path)
        assert path.endswith(".json")

    def test_generate_html_report(self, agent, sample_ingest_result):
        """Test HTML report generation."""
        agent.record_ingestion(sample_ingest_result)
        report = agent.process(sample_ingest_result.dataset_id)

        html = agent._generate_html_report(report)

        assert "<html>" in html
        assert sample_ingest_result.dataset_id in html
        assert "salary" in html  # Field name should appear

    def test_event_published(self, agent, sample_ingest_result):
        """Test that trace.updated event is published."""
        events_received = []

        def event_handler(event):
            events_received.append(event)

        agent.subscribe("trace.updated", event_handler)
        agent.record_ingestion(sample_ingest_result)
        agent.process(sample_ingest_result.dataset_id)

        assert len(events_received) == 1
        assert events_received[0]["event_type"] == "trace.updated"

    def test_transformation_chain_recorded(self, agent, sample_ingest_result):
        """Test that transformation chain is recorded."""
        agent.record_ingestion(sample_ingest_result)

        # Chain of transformations
        agent.record_transformation(
            source_fields=[f"{sample_ingest_result.dataset_id}.salary"],
            target_field="salary_normalized",
            operation="transform",
            parameters={"type": "normalization"},
            fused_dataset_id="step-1",
            agent="FusionAgent"
        )

        agent.record_transformation(
            source_fields=["step-1.salary_normalized"],
            target_field="salary_final",
            operation="transform",
            parameters={"type": "rounding"},
            fused_dataset_id="step-2",
            agent="FusionAgent"
        )

        # Check final trace has all transformations
        trace = agent._traces["step-2.salary_final"]
        assert trace.lineage_depth == 2
        assert len(trace.transformations) == 2

    @patch("agents.provenance_tracker_agent.GraphDatabase")
    def test_record_to_graph(self, mock_neo4j, agent, sample_ingest_result):
        """Test recording to Neo4j graph."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_neo4j.driver.return_value = mock_driver

        agent.record_ingestion(sample_ingest_result)

        # Graph should have been written to
        assert mock_session.run.called

    def test_close(self, agent):
        """Test closing graph connection."""
        agent._graph_driver = MagicMock()
        agent.close()

        agent._graph_driver.close.assert_called_once()
        assert agent._graph_driver is None

    def test_multiple_origins_for_merged_field(self, agent):
        """Test field with multiple origins."""
        # Ingest two datasets
        result1 = IngestResult(
            dataset_id="dataset-1",
            source="/data/file1.csv",
            source_type=SourceType.CSV,
            schema=["value"],
            dtypes={"value": "float64"},
            row_count=10,
            sample_rows=5,
            sample_data=[],
            content_hash="hash1",
            encoding="utf-8",
            storage_path="/tmp/1.parquet"
        )

        result2 = IngestResult(
            dataset_id="dataset-2",
            source="/data/file2.csv",
            source_type=SourceType.CSV,
            schema=["value"],
            dtypes={"value": "float64"},
            row_count=10,
            sample_rows=5,
            sample_data=[],
            content_hash="hash2",
            encoding="utf-8",
            storage_path="/tmp/2.parquet"
        )

        agent.record_ingestion(result1)
        agent.record_ingestion(result2)

        # Merge field from both sources
        agent.record_transformation(
            source_fields=["dataset-1.value", "dataset-2.value"],
            target_field="merged_value",
            operation="fuse",
            parameters={"strategy": "average"},
            fused_dataset_id="fused",
            agent="FusionAgent"
        )

        trace = agent._traces["fused.merged_value"]
        assert len(trace.origins) == 2
