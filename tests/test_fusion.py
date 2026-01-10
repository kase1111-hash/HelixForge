"""Unit tests for Fusion Agent."""

import os

import pandas as pd
import pytest

from agents.fusion_agent import BUILTIN_TRANSFORMS, FusionAgent
from models.schemas import (
    AlignmentResult,
    AlignmentType,
    FieldAlignment,
    ImputationMethod,
    JoinStrategy,
)


class TestFusionAgent:
    """Tests for FusionAgent."""

    @pytest.fixture
    def agent(self, fusion_config):
        """Create agent instance for testing."""
        return FusionAgent(config=fusion_config)

    @pytest.fixture
    def sample_alignment_result(self):
        """Create sample alignment result."""
        alignments = [
            FieldAlignment(
                alignment_id="align-1",
                source_dataset="dataset-a",
                source_field="employee_id",
                target_dataset="dataset-b",
                target_field="emp_id",
                similarity=0.95,
                alignment_type=AlignmentType.EXACT
            ),
            FieldAlignment(
                alignment_id="align-2",
                source_dataset="dataset-a",
                source_field="salary",
                target_dataset="dataset-b",
                target_field="annual_salary",
                similarity=0.90,
                alignment_type=AlignmentType.SYNONYM
            )
        ]

        return AlignmentResult(
            alignment_job_id="job-123",
            datasets_aligned=["dataset-a", "dataset-b"],
            alignments=alignments,
            unmatched_fields=[]
        )

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent.agent_name == "FusionAgent"
        assert agent.event_type == "dataset.fused"

    def test_builtin_transforms_celsius_to_fahrenheit(self):
        """Test Celsius to Fahrenheit transformation."""
        transform = BUILTIN_TRANSFORMS["celsius_to_fahrenheit"]
        assert transform(0) == 32
        assert transform(100) == 212

    def test_builtin_transforms_kg_to_lb(self):
        """Test kg to lb transformation."""
        transform = BUILTIN_TRANSFORMS["kg_to_lb"]
        result = transform(1)
        assert abs(result - 2.20462) < 0.001

    def test_builtin_transforms_m_to_ft(self):
        """Test meters to feet transformation."""
        transform = BUILTIN_TRANSFORMS["m_to_ft"]
        result = transform(1)
        assert abs(result - 3.28084) < 0.001

    def test_impute_missing_mean(self, agent):
        """Test mean imputation."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, None, 4.0, 5.0],
            "b": [10.0, None, 30.0, 40.0, 50.0]
        })

        result_df, summary = agent._impute_missing(df, ImputationMethod.MEAN)

        assert not result_df["a"].isna().any()
        assert not result_df["b"].isna().any()
        assert summary.total_nulls_filled == 2

    def test_impute_missing_median(self, agent):
        """Test median imputation."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, None, 100.0, 5.0]  # Outlier to show median difference
        })

        result_df, summary = agent._impute_missing(df, ImputationMethod.MEDIAN)

        # Median of [1, 2, 5, 100] = 3.5
        assert abs(result_df["a"].iloc[2] - 3.5) < 0.1

    def test_impute_missing_mode(self, agent):
        """Test mode imputation."""
        df = pd.DataFrame({
            "a": [1.0, 1.0, None, 2.0, 1.0]
        })

        result_df, summary = agent._impute_missing(df, ImputationMethod.MODE)

        assert result_df["a"].iloc[2] == 1.0  # Mode is 1

    def test_impute_missing_knn(self, agent):
        """Test KNN imputation."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, None, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0]
        })

        result_df, summary = agent._impute_missing(df, ImputationMethod.KNN)

        assert not result_df["a"].isna().any()

    def test_impute_skips_high_null_ratio(self, agent):
        """Test that columns with high null ratio are skipped."""
        df = pd.DataFrame({
            "a": [1.0, None, None, None, None],  # 80% null
            "b": [10.0, 20.0, None, 40.0, 50.0]   # 20% null
        })

        result_df, summary = agent._impute_missing(df, ImputationMethod.MEAN)

        # Column 'a' should still have nulls (too many)
        assert result_df["a"].isna().sum() > 0 or "a" not in summary.fields_imputed

    def test_apply_transformation_unit_conversion(self, agent):
        """Test unit conversion transformation."""
        df = pd.DataFrame({"temp_c": [0.0, 100.0, 25.0]})

        alignment = FieldAlignment(
            alignment_id="a1",
            source_dataset="ds-a",
            source_field="temp_f",
            target_dataset="ds-b",
            target_field="temp_c",
            similarity=0.9,
            alignment_type=AlignmentType.SYNONYM,
            transformation_hint="unit_conversion:celsius_to_fahrenheit"
        )

        result_df, log = agent._apply_transformation(df, alignment)

        assert result_df["temp_c"].iloc[0] == 32.0  # 0°C = 32°F
        assert log is not None
        assert log.operation == "unit_conversion"

    def test_merge_datasets_exact_key(self, agent, sample_dataframe, second_sample_dataframe, sample_alignment_result):
        """Test exact key merge strategy."""
        # Rename columns to match alignment
        df_a = sample_dataframe.rename(columns={"id": "employee_id"})
        df_b = second_sample_dataframe.rename(columns={"employee_id": "emp_id", "annual_salary": "annual_salary"})

        # Adjust alignment for this test
        alignment = FieldAlignment(
            alignment_id="a1",
            source_dataset="dataset-a",
            source_field="employee_id",
            target_dataset="dataset-b",
            target_field="emp_id",
            similarity=0.95,
            alignment_type=AlignmentType.EXACT
        )

        merged, transforms = agent._merge_datasets(
            df_a, df_b, [alignment], JoinStrategy.EXACT_KEY
        )

        assert merged is not None
        assert len(merged) > 0

    def test_semantic_join(self, agent):
        """Test semantic similarity join."""
        df_a = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "value": [1, 2, 3]
        })
        df_b = pd.DataFrame({
            "full_name": ["Alice Smith", "Bob Jones", "David"],
            "score": [10, 20, 30]
        })

        alignment = FieldAlignment(
            alignment_id="a1",
            source_dataset="ds-a",
            source_field="name",
            target_dataset="ds-b",
            target_field="full_name",
            similarity=0.8,
            alignment_type=AlignmentType.RELATED
        )

        merged = agent._semantic_join(df_a, df_b, [alignment])

        assert len(merged) > 0

    def test_save_fused_dataset_parquet(self, agent, sample_dataframe):
        """Test saving fused dataset as parquet."""
        path = agent._save_fused_dataset(sample_dataframe, "test-fused")

        assert path.endswith(".parquet")
        assert os.path.exists(path)

    def test_get_fused_dataframe(self, agent, sample_dataframe, second_sample_dataframe, sample_alignment_result):
        """Test retrieving fused dataframe."""
        dataframes = {
            "dataset-a": sample_dataframe.rename(columns={"id": "employee_id"}),
            "dataset-b": second_sample_dataframe.rename(columns={"employee_id": "emp_id"})
        }

        result = agent.process(dataframes, sample_alignment_result)
        df = agent.get_fused_dataframe(result.fused_dataset_id)

        assert df is not None
        assert len(df) > 0

    def test_process_full_pipeline(self, agent, sample_dataframe, second_sample_dataframe, sample_alignment_result):
        """Test full fusion pipeline."""
        dataframes = {
            "dataset-a": sample_dataframe.rename(columns={"id": "employee_id"}),
            "dataset-b": second_sample_dataframe.rename(columns={"employee_id": "emp_id"})
        }

        result = agent.process(dataframes, sample_alignment_result)

        assert result is not None
        assert result.fused_dataset_id is not None
        assert result.record_count > 0
        assert result.field_count > 0
        assert len(result.source_datasets) == 2

    def test_event_published(self, agent, sample_dataframe, second_sample_dataframe, sample_alignment_result):
        """Test that dataset.fused event is published."""
        events_received = []

        def event_handler(event):
            events_received.append(event)

        agent.subscribe("dataset.fused", event_handler)

        dataframes = {
            "dataset-a": sample_dataframe.rename(columns={"id": "employee_id"}),
            "dataset-b": second_sample_dataframe.rename(columns={"employee_id": "emp_id"})
        }

        agent.process(dataframes, sample_alignment_result)

        assert len(events_received) == 1
        assert events_received[0]["event_type"] == "dataset.fused"

    def test_transformation_log_recorded(self, agent, sample_dataframe, second_sample_dataframe):
        """Test that transformations are logged."""
        alignment = FieldAlignment(
            alignment_id="a1",
            source_dataset="ds-a",
            source_field="temp_f",
            target_dataset="ds-b",
            target_field="temp_c",
            similarity=0.9,
            alignment_type=AlignmentType.SYNONYM,
            transformation_hint="unit_conversion:celsius_to_fahrenheit"
        )

        alignment_result = AlignmentResult(
            alignment_job_id="job-123",
            datasets_aligned=["ds-a", "ds-b"],
            alignments=[alignment],
            unmatched_fields=[]
        )

        df_a = pd.DataFrame({"temp_f": [32, 212]})
        df_b = pd.DataFrame({"temp_c": [0, 100]})

        result = agent.process(
            {"ds-a": df_a, "ds-b": df_b},
            alignment_result
        )

        # Check transformation was applied and logged
        assert result.transformations_applied is not None

    def test_imputation_summary_included(self, agent, sample_alignment_result):
        """Test that imputation summary is included in result."""
        df_a = pd.DataFrame({
            "employee_id": [1, 2, 3, 4, 5],
            "value": [1.0, None, 3.0, None, 5.0]
        })
        df_b = pd.DataFrame({
            "emp_id": [1, 2, 6],
            "score": [10.0, 20.0, 60.0]
        })

        result = agent.process(
            {"dataset-a": df_a, "dataset-b": df_b},
            sample_alignment_result,
            imputation_method=ImputationMethod.MEAN
        )

        assert result.imputation_summary is not None
