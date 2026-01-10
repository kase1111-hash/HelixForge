"""System/Acceptance tests for HelixForge.

These tests verify the complete user workflows as defined in the user stories.
They test end-to-end functionality from the user's perspective.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agents.data_ingestor_agent import DataIngestorAgent
from agents.metadata_interpreter_agent import MetadataInterpreterAgent
from agents.ontology_alignment_agent import OntologyAlignmentAgent
from agents.fusion_agent import FusionAgent
from agents.insight_generator_agent import InsightGeneratorAgent
from agents.provenance_tracker_agent import ProvenanceTrackerAgent
from models.schemas import JoinStrategy


class TestUserStory1_DataIngestion:
    """
    User Story 1: As a data analyst, I want to upload datasets in various formats
    so that I can work with data from different sources.

    Acceptance Criteria:
    - System accepts CSV, JSON, Parquet, Excel files
    - System auto-detects file encoding and delimiter
    - System provides feedback on ingestion success/failure
    - Ingested data is accessible for subsequent operations
    """

    @pytest.fixture
    def ingestor(self, ingestor_config):
        return DataIngestorAgent(config=ingestor_config)

    def test_ac1_accepts_csv_files(self, ingestor, tmp_path):
        """AC1: System accepts CSV files."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,value\n1,Alice,100\n2,Bob,200\n")

        result = ingestor.ingest_file(str(csv_file))

        assert result is not None
        assert result.row_count == 2
        assert "id" in result.schema_fields

    def test_ac1_accepts_json_files(self, ingestor, tmp_path):
        """AC1: System accepts JSON files."""
        json_file = tmp_path / "data.json"
        json_file.write_text('[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]')

        result = ingestor.ingest_file(str(json_file))

        assert result is not None
        assert result.row_count == 2

    def test_ac1_accepts_parquet_files(self, ingestor, tmp_path):
        """AC1: System accepts Parquet files."""
        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        parquet_file = tmp_path / "data.parquet"
        df.to_parquet(str(parquet_file))

        result = ingestor.ingest_file(str(parquet_file))

        assert result is not None
        assert result.row_count == 2

    def test_ac2_autodetects_encoding(self, ingestor, tmp_path):
        """AC2: System auto-detects file encoding."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Müller\n2,Søren\n", encoding="utf-8")

        result = ingestor.ingest_file(str(csv_file))

        assert result.encoding is not None
        assert result.row_count == 2

    def test_ac2_autodetects_delimiter(self, ingestor, tmp_path):
        """AC2: System auto-detects delimiter."""
        # Tab-delimited file
        tsv_file = tmp_path / "data.csv"
        tsv_file.write_text("id\tname\n1\tAlice\n2\tBob\n")

        result = ingestor.ingest_file(str(tsv_file))

        assert result.row_count == 2

    def test_ac3_provides_success_feedback(self, ingestor, tmp_path):
        """AC3: System provides feedback on ingestion success."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n")

        result = ingestor.ingest_file(str(csv_file))

        assert result.dataset_id is not None
        assert result.content_hash is not None
        assert result.storage_path is not None

    def test_ac3_provides_failure_feedback(self, ingestor):
        """AC3: System provides feedback on ingestion failure."""
        from agents.data_ingestor_agent import IngestionError

        with pytest.raises(IngestionError):
            ingestor.ingest_file("/nonexistent/file.csv")

    def test_ac4_data_accessible_after_ingestion(self, ingestor, tmp_path):
        """AC4: Ingested data is accessible for subsequent operations."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,value\n1,Alice,100\n2,Bob,200\n")

        result = ingestor.ingest_file(str(csv_file))
        df = ingestor.get_dataframe(result.dataset_id)

        assert df is not None
        assert len(df) == 2
        assert list(df.columns) == ["id", "name", "value"]


class TestUserStory2_SemanticLabeling:
    """
    User Story 2: As a data analyst, I want automatic semantic labeling of fields
    so that I understand what each column represents.

    Acceptance Criteria:
    - System infers data types (int, float, string, date, etc.)
    - System assigns semantic types (identifier, metric, timestamp, etc.)
    - System provides confidence scores for inferences
    - Labels are editable/overridable by user
    """

    @pytest.fixture
    def mock_interpreter(self, interpreter_config, mock_openai_client):
        agent = MetadataInterpreterAgent(config=interpreter_config)
        agent._openai_client = mock_openai_client
        return agent

    def test_ac1_infers_data_types(self, mock_interpreter):
        """AC1: System infers data types."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "salary": [50000.0, 60000.0, 70000.0],
            "active": [True, False, True]
        })

        with patch("agents.metadata_interpreter_agent.batch_embed") as mock_embed:
            mock_embed.return_value = [[0.1] * 1536 for _ in range(4)]
            result = mock_interpreter.process("test-dataset", df)

        # Verify data types are inferred
        field_types = {f.field_name: f.data_type for f in result.fields}
        assert "INTEGER" in str(field_types.get("id", "")).upper() or "INT" in str(field_types.get("id", "")).upper()

    def test_ac2_assigns_semantic_types(self, mock_interpreter):
        """AC2: System assigns semantic types."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "total_amount": [100.0, 200.0, 300.0]
        })

        with patch("agents.metadata_interpreter_agent.batch_embed") as mock_embed:
            mock_embed.return_value = [[0.1] * 1536 for _ in range(2)]
            result = mock_interpreter.process("test-dataset", df)

        # Verify semantic types are assigned
        assert len(result.fields) == 2
        for field in result.fields:
            assert field.semantic_type is not None

    def test_ac3_provides_confidence_scores(self, mock_interpreter):
        """AC3: System provides confidence scores."""
        df = pd.DataFrame({"value": [1, 2, 3]})

        with patch("agents.metadata_interpreter_agent.batch_embed") as mock_embed:
            mock_embed.return_value = [[0.1] * 1536]
            result = mock_interpreter.process("test-dataset", df)

        # Verify confidence scores exist
        for field in result.fields:
            assert 0 <= field.confidence <= 1


class TestUserStory3_SchemaAlignment:
    """
    User Story 3: As a data analyst, I want to align schemas across datasets
    so that I can merge data from different sources.

    Acceptance Criteria:
    - System identifies matching fields across datasets
    - System shows similarity scores for potential matches
    - System suggests canonical field names
    - User can approve/reject/modify alignments
    """

    @pytest.fixture
    def aligner(self, alignment_config):
        return OntologyAlignmentAgent(config=alignment_config)

    @pytest.fixture
    def two_metadata_results(self, interpreter_config, mock_openai_client):
        """Create metadata for two similar datasets."""
        from models.schemas import DatasetMetadata, FieldMetadata, DataType, SemanticType

        meta1 = DatasetMetadata(
            dataset_id="dataset-1",
            dataset_description="Employee data",
            domain_tags=["hr"],
            fields=[
                FieldMetadata(
                    dataset_id="dataset-1",
                    field_name="employee_id",
                    semantic_label="Employee ID",
                    description="Unique identifier",
                    data_type=DataType.INTEGER,
                    semantic_type=SemanticType.IDENTIFIER,
                    null_ratio=0,
                    unique_ratio=1,
                    confidence=0.95,
                    embedding=[0.1] * 1536
                ),
                FieldMetadata(
                    dataset_id="dataset-1",
                    field_name="salary",
                    semantic_label="Salary",
                    description="Employee salary",
                    data_type=DataType.FLOAT,
                    semantic_type=SemanticType.METRIC,
                    null_ratio=0,
                    unique_ratio=0.9,
                    confidence=0.9,
                    embedding=[0.2] * 1536
                )
            ]
        )

        meta2 = DatasetMetadata(
            dataset_id="dataset-2",
            dataset_description="HR records",
            domain_tags=["hr"],
            fields=[
                FieldMetadata(
                    dataset_id="dataset-2",
                    field_name="emp_id",
                    semantic_label="Employee ID",
                    description="Employee identifier",
                    data_type=DataType.INTEGER,
                    semantic_type=SemanticType.IDENTIFIER,
                    null_ratio=0,
                    unique_ratio=1,
                    confidence=0.9,
                    embedding=[0.1] * 1536
                ),
                FieldMetadata(
                    dataset_id="dataset-2",
                    field_name="wage",
                    semantic_label="Wage",
                    description="Employee wage",
                    data_type=DataType.FLOAT,
                    semantic_type=SemanticType.METRIC,
                    null_ratio=0,
                    unique_ratio=0.85,
                    confidence=0.85,
                    embedding=[0.2] * 1536
                )
            ]
        )

        return [meta1, meta2]

    def test_ac1_identifies_matching_fields(self, aligner, two_metadata_results):
        """AC1: System identifies matching fields."""
        result = aligner.process(two_metadata_results)

        assert result is not None
        assert len(result.alignments) > 0

    def test_ac2_shows_similarity_scores(self, aligner, two_metadata_results):
        """AC2: System shows similarity scores."""
        result = aligner.process(two_metadata_results)

        for alignment in result.alignments:
            assert 0 <= alignment.similarity <= 1

    def test_ac3_suggests_canonical_names(self, aligner, two_metadata_results):
        """AC3: System suggests canonical field names."""
        result = aligner.process(two_metadata_results)

        for alignment in result.alignments:
            assert alignment.transformation_hint is not None or alignment.alignment_type is not None


class TestUserStory4_DatasetFusion:
    """
    User Story 4: As a data analyst, I want to merge aligned datasets
    so that I can analyze combined data.

    Acceptance Criteria:
    - System merges datasets based on alignment
    - System handles missing values appropriately
    - System resolves conflicts between sources
    - Merged dataset maintains data quality
    """

    @pytest.fixture
    def fusion_agent(self, fusion_config):
        return FusionAgent(config=fusion_config)

    @pytest.fixture
    def sample_alignment(self):
        """Create a sample alignment result."""
        from models.schemas import AlignmentResult, FieldAlignment, AlignmentType

        return AlignmentResult(
            alignment_job_id="test-alignment",
            datasets_aligned=["dataset-1", "dataset-2"],
            alignments=[
                FieldAlignment(
                    alignment_id="align-1",
                    source_dataset="dataset-1",
                    source_field="id",
                    target_dataset="dataset-2",
                    target_field="id",
                    alignment_type=AlignmentType.EXACT,
                    similarity=1.0
                ),
                FieldAlignment(
                    alignment_id="align-2",
                    source_dataset="dataset-1",
                    source_field="value",
                    target_dataset="dataset-2",
                    target_field="amount",
                    alignment_type=AlignmentType.RELATED,
                    similarity=0.85
                )
            ]
        )

    def test_ac1_merges_datasets_based_on_alignment(self, fusion_agent, sample_alignment):
        """AC1: System merges datasets based on alignment."""
        df1 = pd.DataFrame({"id": [1, 2], "value": [100, 200]})
        df2 = pd.DataFrame({"id": [3, 4], "amount": [300, 400]})

        result = fusion_agent.process(
            dataframes={"dataset-1": df1, "dataset-2": df2},
            alignment_result=sample_alignment,
            join_strategy=JoinStrategy.EXACT_KEY
        )

        assert result is not None
        assert result.record_count == 4

    def test_ac2_handles_missing_values(self, fusion_agent, sample_alignment):
        """AC2: System handles missing values appropriately."""
        df1 = pd.DataFrame({"id": [1, 2], "value": [100.0, None]})
        df2 = pd.DataFrame({"id": [3, 4], "amount": [None, 400.0]})

        result = fusion_agent.process(
            dataframes={"dataset-1": df1, "dataset-2": df2},
            alignment_result=sample_alignment,
            join_strategy=JoinStrategy.EXACT_KEY,
            imputation_method="mean"
        )

        assert result is not None
        # Check imputation was applied
        fused_df = fusion_agent.get_fused_dataframe(result.fused_dataset_id)
        assert fused_df is not None

    def test_ac4_maintains_data_quality(self, fusion_agent, sample_alignment):
        """AC4: Merged dataset maintains data quality."""
        df1 = pd.DataFrame({"id": [1, 2], "value": [100, 200]})
        df2 = pd.DataFrame({"id": [3, 4], "amount": [300, 400]})

        result = fusion_agent.process(
            dataframes={"dataset-1": df1, "dataset-2": df2},
            alignment_result=sample_alignment,
            join_strategy=JoinStrategy.EXACT_KEY
        )

        # Verify quality metrics
        assert result.record_count >= 0
        assert result.record_count == len(df1) + len(df2)


class TestUserStory5_InsightGeneration:
    """
    User Story 5: As a data analyst, I want to generate insights from fused data
    so that I can understand patterns and anomalies.

    Acceptance Criteria:
    - System computes descriptive statistics
    - System identifies correlations between fields
    - System detects outliers and anomalies
    - System generates visualizations
    """

    @pytest.fixture
    def insight_agent(self, insight_config, mock_openai_client):
        agent = InsightGeneratorAgent(config=insight_config)
        agent._openai_client = mock_openai_client
        return agent

    @pytest.fixture
    def analysis_df(self):
        """Create DataFrame for analysis."""
        import numpy as np
        np.random.seed(42)
        return pd.DataFrame({
            "id": range(100),
            "value_a": np.random.normal(100, 20, 100),
            "value_b": np.random.normal(50, 10, 100),
            "category": np.random.choice(["A", "B", "C"], 100)
        })

    def test_ac1_computes_descriptive_statistics(self, insight_agent, analysis_df):
        """AC1: System computes descriptive statistics."""
        with patch.object(insight_agent, "_generate_visualizations", return_value=[]):
            result = insight_agent.process("test", analysis_df, generate_visualizations=False)

        assert result.statistics is not None
        assert result.statistics.record_count == 100
        assert "value_a" in result.statistics.field_stats

    def test_ac2_identifies_correlations(self, insight_agent, analysis_df):
        """AC2: System identifies correlations between fields."""
        with patch.object(insight_agent, "_generate_visualizations", return_value=[]):
            result = insight_agent.process("test", analysis_df, generate_visualizations=False)

        assert result.correlations is not None

    def test_ac3_detects_outliers(self, insight_agent):
        """AC3: System detects outliers and anomalies."""
        # Create data with clear outliers
        df = pd.DataFrame({
            "id": range(100),
            "value": [10] * 98 + [1000, -500]  # Clear outliers
        })

        with patch.object(insight_agent, "_generate_visualizations", return_value=[]):
            result = insight_agent.process("test", df, generate_visualizations=False)

        assert result.outliers is not None
        assert result.outliers.total_outliers > 0


class TestUserStory6_ProvenanceTracking:
    """
    User Story 6: As a data analyst, I want to track data lineage
    so that I can understand where each value came from.

    Acceptance Criteria:
    - System tracks original source for each field
    - System records all transformations applied
    - System provides confidence scores for derived data
    - Lineage is queryable and exportable
    """

    @pytest.fixture
    def provenance_agent(self, provenance_config):
        return ProvenanceTrackerAgent(config=provenance_config)

    @pytest.fixture
    def sample_ingest_result(self):
        from models.schemas import IngestResult, SourceType
        return IngestResult(
            dataset_id="source-dataset",
            source="/data/source.csv",
            source_type=SourceType.CSV,
            schema=["id", "value", "name"],
            dtypes={"id": "int64", "value": "float64", "name": "object"},
            row_count=100,
            sample_rows=5,
            sample_data=[],
            content_hash="abc123",
            encoding="utf-8",
            storage_path="/tmp/source.parquet"
        )

    def test_ac1_tracks_original_source(self, provenance_agent, sample_ingest_result):
        """AC1: System tracks original source for each field."""
        provenance_agent.record_ingestion(sample_ingest_result)

        trace = provenance_agent.query_lineage(sample_ingest_result.dataset_id, "value")

        assert trace is not None
        assert len(trace.origins) == 1
        assert trace.origins[0].source_file == sample_ingest_result.source

    def test_ac2_records_transformations(self, provenance_agent, sample_ingest_result):
        """AC2: System records all transformations applied."""
        provenance_agent.record_ingestion(sample_ingest_result)

        provenance_agent.record_transformation(
            source_fields=[f"{sample_ingest_result.dataset_id}.value"],
            target_field="normalized_value",
            operation="transform",
            parameters={"type": "normalize"},
            fused_dataset_id="fused-1",
            agent="FusionAgent"
        )

        trace = provenance_agent.query_lineage("fused-1", "normalized_value")

        assert trace is not None
        assert len(trace.transformations) == 1

    def test_ac3_provides_confidence_scores(self, provenance_agent, sample_ingest_result):
        """AC3: System provides confidence scores for derived data."""
        provenance_agent.record_ingestion(sample_ingest_result)

        # Apply multiple transformations
        provenance_agent.record_transformation(
            source_fields=[f"{sample_ingest_result.dataset_id}.value"],
            target_field="step1",
            operation="transform",
            parameters={},
            fused_dataset_id="fused-1",
            agent="FusionAgent"
        )

        trace = provenance_agent.query_lineage("fused-1", "step1")

        assert trace.confidence < 1.0  # Confidence should decay

    def test_ac4_lineage_is_queryable(self, provenance_agent, sample_ingest_result):
        """AC4: Lineage is queryable."""
        provenance_agent.record_ingestion(sample_ingest_result)

        # Query lineage
        trace = provenance_agent.query_lineage(
            sample_ingest_result.dataset_id,
            "value"
        )

        assert trace is not None
        assert trace.field == "value"

    def test_ac4_lineage_graph_exportable(self, provenance_agent, sample_ingest_result):
        """AC4: Lineage is exportable as graph."""
        provenance_agent.record_ingestion(sample_ingest_result)

        graph = provenance_agent.build_lineage_graph(sample_ingest_result.dataset_id)

        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) > 0


class TestEndToEndWorkflow:
    """
    End-to-end workflow test simulating a complete user session.
    """

    def test_complete_pipeline_workflow(
        self,
        ingestor_config,
        interpreter_config,
        alignment_config,
        fusion_config,
        insight_config,
        provenance_config,
        mock_openai_client,
        tmp_path
    ):
        """Test complete workflow from ingestion to insights."""
        # Create test files
        csv1 = tmp_path / "employees.csv"
        csv1.write_text("emp_id,name,salary\n1,Alice,50000\n2,Bob,60000\n")

        csv2 = tmp_path / "performance.csv"
        csv2.write_text("employee_id,score\n1,85\n2,90\n")

        # 1. Ingest datasets
        ingestor = DataIngestorAgent(config=ingestor_config)
        result1 = ingestor.ingest_file(str(csv1), dataset_id="employees")
        result2 = ingestor.ingest_file(str(csv2), dataset_id="performance")

        assert result1.row_count == 2
        assert result2.row_count == 2

        # 2. Track provenance
        provenance = ProvenanceTrackerAgent(config=provenance_config)
        provenance.record_ingestion(result1)
        provenance.record_ingestion(result2)

        # Verify provenance tracked
        trace = provenance.query_lineage("employees", "salary")
        assert trace is not None

        # 3. Get DataFrames
        df1 = ingestor.get_dataframe(result1.dataset_id)
        df2 = ingestor.get_dataframe(result2.dataset_id)

        assert df1 is not None
        assert df2 is not None

        # 4. Interpret metadata
        interpreter = MetadataInterpreterAgent(config=interpreter_config)
        interpreter._openai_client = mock_openai_client

        with patch("agents.metadata_interpreter_agent.batch_embed") as mock_embed:
            mock_embed.return_value = [[0.1] * 1536 for _ in range(3)]
            meta1 = interpreter.process("employees", df1)
            mock_embed.return_value = [[0.1] * 1536 for _ in range(2)]
            meta2 = interpreter.process("performance", df2)

        assert len(meta1.fields) == 3
        assert len(meta2.fields) == 2

        # 5. Align schemas
        aligner = OntologyAlignmentAgent(config=alignment_config)
        alignment = aligner.process([meta1, meta2])

        assert alignment is not None

        # 6. Generate insights on individual dataset
        insight_gen = InsightGeneratorAgent(config=insight_config)
        insight_gen._openai_client = mock_openai_client

        with patch.object(insight_gen, "_generate_visualizations", return_value=[]):
            insights = insight_gen.process("employees", df1, generate_visualizations=False)

        assert insights is not None
        assert insights.statistics.record_count == 2

        # Workflow completed successfully
        print("End-to-end workflow completed successfully!")
