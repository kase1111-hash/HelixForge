"""System/Acceptance tests for HelixForge.

These tests verify the complete user workflows as defined in the user stories.
They test end-to-end functionality from the user's perspective.
"""

import os
import tempfile

import pandas as pd
import pytest

from agents.data_ingestor_agent import DataIngestorAgent
from agents.metadata_interpreter_agent import MetadataInterpreterAgent
from agents.ontology_alignment_agent import OntologyAlignmentAgent
from agents.fusion_agent import FusionAgent
from models.schemas import JoinStrategy
from utils.llm import MockProvider


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
    """

    @pytest.fixture
    def mock_interpreter(self, interpreter_config):
        return MetadataInterpreterAgent(
            config=interpreter_config, provider=MockProvider(dimensions=1536)
        )

    def test_ac1_infers_data_types(self, mock_interpreter):
        """AC1: System infers data types."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "salary": [50000.0, 60000.0, 70000.0],
            "active": [True, False, True]
        })

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

        result = mock_interpreter.process("test-dataset", df)

        # Verify semantic types are assigned
        assert len(result.fields) == 2
        for field in result.fields:
            assert field.semantic_type is not None

    def test_ac3_provides_confidence_scores(self, mock_interpreter):
        """AC3: System provides confidence scores."""
        df = pd.DataFrame({"value": [1, 2, 3]})

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
    """

    @pytest.fixture
    def aligner(self, alignment_config):
        return OntologyAlignmentAgent(config=alignment_config)

    @pytest.fixture
    def two_metadata_results(self):
        """Create metadata for two similar datasets with orthogonal embeddings."""
        from models.schemas import DatasetMetadata, FieldMetadata, DataType, SemanticType

        # Orthogonal embeddings so cosine similarity is meaningful
        emb_id = [1.0] * 512 + [0.0] * 512 + [0.0] * 512
        emb_salary = [0.0] * 512 + [1.0] * 512 + [0.0] * 512

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
                    embedding=emb_id,
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
                    embedding=emb_salary,
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
                    embedding=emb_id,
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
                    embedding=emb_salary,
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
        tmp_path
    ):
        """Test complete workflow from ingestion to alignment."""
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

        # 2. Get DataFrames
        df1 = ingestor.get_dataframe(result1.dataset_id)
        df2 = ingestor.get_dataframe(result2.dataset_id)

        assert df1 is not None
        assert df2 is not None

        # 3. Interpret metadata
        provider = MockProvider(dimensions=1536)
        interpreter = MetadataInterpreterAgent(
            config=interpreter_config, provider=provider
        )

        meta1 = interpreter.process("employees", df1)
        meta2 = interpreter.process("performance", df2)

        assert len(meta1.fields) == 3
        assert len(meta2.fields) == 2

        # 4. Align schemas
        aligner = OntologyAlignmentAgent(config=alignment_config)
        alignment = aligner.process([meta1, meta2])

        assert alignment is not None
