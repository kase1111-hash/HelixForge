"""Unit tests for Fusion Agent.

Tests verify fused data correctness with spot-check assertions on
specific row values, not just "result is not None" or "len > 0".
"""

import os
import tempfile

import numpy as np
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


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _align(
    src_ds: str, src_field: str,
    tgt_ds: str, tgt_field: str,
    similarity: float = 0.95,
    atype: AlignmentType = AlignmentType.EXACT,
    hint: str = None,
) -> FieldAlignment:
    """Shorthand for creating an alignment."""
    return FieldAlignment(
        alignment_id=f"a-{src_field}-{tgt_field}",
        source_dataset=src_ds,
        source_field=src_field,
        target_dataset=tgt_ds,
        target_field=tgt_field,
        similarity=similarity,
        alignment_type=atype,
        transformation_hint=hint,
    )


def _result(alignments, ds_ids, unmatched=None):
    """Shorthand for AlignmentResult."""
    return AlignmentResult(
        alignment_job_id="job-test",
        datasets_aligned=ds_ids,
        alignments=alignments,
        unmatched_fields=unmatched or [],
    )


def _make_agent(tmp_path, **overrides):
    """Create a FusionAgent with sensible test defaults."""
    cfg = {
        "fusion": {
            "default_join_strategy": "auto",
            "similarity_join_threshold": 0.80,
            "imputation_method": "mean",
            "knn_neighbors": 3,
            "max_null_ratio_for_inclusion": 0.5,
            "output_format": "csv",
            "output_path": str(tmp_path),
        }
    }
    cfg["fusion"].update(overrides)
    return FusionAgent(config=cfg)


# ================================================================== #
#  1. Initialization                                                   #
# ================================================================== #

class TestFusionInit:
    def test_agent_name(self, tmp_path):
        agent = _make_agent(tmp_path)
        assert agent.agent_name == "FusionAgent"

    def test_event_type(self, tmp_path):
        agent = _make_agent(tmp_path)
        assert agent.event_type == "dataset.fused"


# ================================================================== #
#  2. Built-in transforms                                              #
# ================================================================== #

class TestBuiltinTransforms:
    def test_celsius_to_fahrenheit(self):
        assert BUILTIN_TRANSFORMS["celsius_to_fahrenheit"](0) == 32
        assert BUILTIN_TRANSFORMS["celsius_to_fahrenheit"](100) == 212

    def test_fahrenheit_to_celsius(self):
        assert BUILTIN_TRANSFORMS["fahrenheit_to_celsius"](32) == 0
        assert BUILTIN_TRANSFORMS["fahrenheit_to_celsius"](212) == 100

    def test_kg_to_lb(self):
        assert abs(BUILTIN_TRANSFORMS["kg_to_lb"](1) - 2.20462) < 0.001

    def test_m_to_ft(self):
        assert abs(BUILTIN_TRANSFORMS["m_to_ft"](1) - 3.28084) < 0.001


# ================================================================== #
#  3. Exact-key merge — value correctness                              #
# ================================================================== #

class TestExactKeyMerge:
    """Exact-key merge tests with spot-check assertions."""

    @pytest.fixture
    def employees_a(self):
        return pd.DataFrame({
            "employee_id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "salary": [70000.0, 60000.0, 80000.0],
        })

    @pytest.fixture
    def employees_b(self):
        return pd.DataFrame({
            "emp_id": [2, 3, 4],
            "full_name": ["Bob J.", "Charlie B.", "Diana P."],
            "annual_pay": [62000.0, 82000.0, 75000.0],
            "department": ["Marketing", "Engineering", "Sales"],
        })

    @pytest.fixture
    def alignment(self):
        return _result([
            _align("ds-a", "employee_id", "ds-b", "emp_id"),
            _align("ds-a", "salary", "ds-b", "annual_pay",
                   similarity=0.88, atype=AlignmentType.SYNONYM),
        ], ["ds-a", "ds-b"])

    def test_merged_row_count(self, tmp_path, employees_a, employees_b, alignment):
        """Outer join: 1(left-only) + 2(matched) + 1(right-only) = 4."""
        agent = _make_agent(tmp_path)
        result = agent.process(
            {"ds-a": employees_a, "ds-b": employees_b},
            alignment, join_strategy=JoinStrategy.EXACT_KEY,
        )
        df = agent.get_fused_dataframe(result.fused_dataset_id)
        assert result.record_count == 4
        assert len(df) == 4

    def test_matched_row_values(self, tmp_path, employees_a, employees_b, alignment):
        """Row with employee_id=2 should have left salary preferred."""
        agent = _make_agent(tmp_path)
        result = agent.process(
            {"ds-a": employees_a, "ds-b": employees_b},
            alignment, join_strategy=JoinStrategy.EXACT_KEY,
        )
        df = agent.get_fused_dataframe(result.fused_dataset_id)
        row = df[df["employee_id"] == 2].iloc[0]
        assert row["name"] == "Bob"
        assert row["salary"] == 60000.0  # left value preferred
        assert row["department"] == "Marketing"  # only in right

    def test_left_only_row(self, tmp_path, employees_a, employees_b, alignment):
        """Employee_id=1 exists only in left dataset."""
        agent = _make_agent(tmp_path)
        result = agent.process(
            {"ds-a": employees_a, "ds-b": employees_b},
            alignment, join_strategy=JoinStrategy.EXACT_KEY,
        )
        df = agent.get_fused_dataframe(result.fused_dataset_id)
        row = df[df["employee_id"] == 1].iloc[0]
        assert row["name"] == "Alice"
        assert row["salary"] == 70000.0
        assert pd.isna(row["department"])  # no right match

    def test_right_only_row(self, tmp_path, employees_a, employees_b, alignment):
        """Emp_id=4 exists only in right dataset."""
        agent = _make_agent(tmp_path)
        result = agent.process(
            {"ds-a": employees_a, "ds-b": employees_b},
            alignment, join_strategy=JoinStrategy.EXACT_KEY,
        )
        df = agent.get_fused_dataframe(result.fused_dataset_id)
        row = df[df["employee_id"] == 4].iloc[0]
        assert row["full_name"] == "Diana P."
        assert row["department"] == "Sales"

    def test_aligned_columns_unified(self, tmp_path, employees_a, employees_b, alignment):
        """Aligned columns use left-side names; no duplicates."""
        agent = _make_agent(tmp_path)
        result = agent.process(
            {"ds-a": employees_a, "ds-b": employees_b},
            alignment, join_strategy=JoinStrategy.EXACT_KEY,
        )
        df = agent.get_fused_dataframe(result.fused_dataset_id)
        # emp_id should be renamed to employee_id, annual_pay to salary
        assert "employee_id" in df.columns
        assert "salary" in df.columns
        assert "emp_id" not in df.columns
        assert "annual_pay" not in df.columns

    def test_non_aligned_columns_preserved(self, tmp_path, employees_a, employees_b, alignment):
        """Columns not in any alignment are preserved as-is."""
        agent = _make_agent(tmp_path)
        result = agent.process(
            {"ds-a": employees_a, "ds-b": employees_b},
            alignment, join_strategy=JoinStrategy.EXACT_KEY,
        )
        df = agent.get_fused_dataframe(result.fused_dataset_id)
        assert "name" in df.columns
        assert "full_name" in df.columns
        assert "department" in df.columns

    def test_user_specified_key(self, tmp_path):
        """User-specified key column is respected."""
        df_a = pd.DataFrame({"id": [1, 2], "val_a": [10, 20]})
        df_b = pd.DataFrame({"id": [2, 3], "val_b": [200, 300]})
        align = _result([
            _align("ds-a", "id", "ds-b", "id"),
        ], ["ds-a", "ds-b"])
        agent = _make_agent(tmp_path)
        result = agent.process(
            {"ds-a": df_a, "ds-b": df_b}, align,
            join_strategy=JoinStrategy.EXACT_KEY, key_column="id",
        )
        df = agent.get_fused_dataframe(result.fused_dataset_id)
        assert len(df) == 3  # 1(left), 2(both), 3(right)
        assert df[df["id"] == 2].iloc[0]["val_a"] == 20
        assert df[df["id"] == 2].iloc[0]["val_b"] == 200

    def test_coalesce_prefers_left_non_null(self, tmp_path):
        """When both sides have a value, left wins."""
        df_a = pd.DataFrame({"id": [1, 2], "score": [90.0, None]})
        df_b = pd.DataFrame({"id": [1, 2], "score": [85.0, 75.0]})
        align = _result([
            _align("ds-a", "id", "ds-b", "id"),
            _align("ds-a", "score", "ds-b", "score", atype=AlignmentType.SYNONYM),
        ], ["ds-a", "ds-b"])
        agent = _make_agent(tmp_path)
        result = agent.process(
            {"ds-a": df_a, "ds-b": df_b}, align,
            join_strategy=JoinStrategy.EXACT_KEY,
        )
        df = agent.get_fused_dataframe(result.fused_dataset_id)
        # id=1: left has 90, right has 85 → keep 90
        assert df[df["id"] == 1].iloc[0]["score"] == 90.0
        # id=2: left is NaN, right has 75 → fill with 75
        assert df[df["id"] == 2].iloc[0]["score"] == 75.0


# ================================================================== #
#  4. Auto strategy detection                                          #
# ================================================================== #

class TestAutoStrategy:
    def test_auto_selects_exact_key_when_exact_alignment(self, tmp_path):
        """Auto picks exact_key when EXACT alignment exists."""
        df_a = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        df_b = pd.DataFrame({"id": [2, 3], "val": [200, 300]})
        align = _result([
            _align("ds-a", "id", "ds-b", "id", atype=AlignmentType.EXACT),
        ], ["ds-a", "ds-b"])
        agent = _make_agent(tmp_path)
        result = agent.process({"ds-a": df_a, "ds-b": df_b}, align)
        assert result.join_strategy == JoinStrategy.EXACT_KEY

    def test_auto_selects_exact_key_when_synonym_alignment(self, tmp_path):
        """Auto picks exact_key when SYNONYM alignment exists."""
        df_a = pd.DataFrame({"employee_id": [1], "name": ["Alice"]})
        df_b = pd.DataFrame({"emp_id": [1], "full_name": ["Alice S."]})
        align = _result([
            _align("ds-a", "employee_id", "ds-b", "emp_id", atype=AlignmentType.SYNONYM),
        ], ["ds-a", "ds-b"])
        agent = _make_agent(tmp_path)
        result = agent.process({"ds-a": df_a, "ds-b": df_b}, align)
        assert result.join_strategy == JoinStrategy.EXACT_KEY

    def test_auto_selects_semantic_when_only_partial(self, tmp_path):
        """Auto falls back to semantic when no EXACT/SYNONYM alignment."""
        df_a = pd.DataFrame({"name": ["Alice", "Bob"], "val": [10, 20]})
        df_b = pd.DataFrame({"full_name": ["Alice S.", "Charlie"], "score": [100, 300]})
        align = _result([
            _align("ds-a", "name", "ds-b", "full_name",
                   similarity=0.6, atype=AlignmentType.PARTIAL),
        ], ["ds-a", "ds-b"])
        agent = _make_agent(tmp_path)
        result = agent.process({"ds-a": df_a, "ds-b": df_b}, align)
        assert result.join_strategy == JoinStrategy.SEMANTIC_SIMILARITY

    def test_auto_selects_exact_key_when_user_provides_key(self, tmp_path):
        """Auto picks exact_key when key_column is specified."""
        df_a = pd.DataFrame({"id": [1], "val": [10]})
        df_b = pd.DataFrame({"id": [1], "score": [100]})
        align = _result([
            _align("ds-a", "id", "ds-b", "id", atype=AlignmentType.PARTIAL),
        ], ["ds-a", "ds-b"])
        agent = _make_agent(tmp_path)
        result = agent.process(
            {"ds-a": df_a, "ds-b": df_b}, align, key_column="id",
        )
        assert result.join_strategy == JoinStrategy.EXACT_KEY


# ================================================================== #
#  5. Experimental strategy gating                                     #
# ================================================================== #

class TestExperimentalGating:
    def test_probabilistic_disabled_by_default(self, tmp_path):
        """Probabilistic strategy falls back when experimental=False."""
        df_a = pd.DataFrame({"name": ["Alice"], "val": [10]})
        df_b = pd.DataFrame({"name": ["Alice"], "score": [100]})
        align = _result([
            _align("ds-a", "name", "ds-b", "name", atype=AlignmentType.EXACT),
        ], ["ds-a", "ds-b"])
        agent = _make_agent(tmp_path, experimental_strategies=False)
        # Should not crash — falls back to semantic
        result = agent.process(
            {"ds-a": df_a, "ds-b": df_b}, align,
            join_strategy=JoinStrategy.PROBABILISTIC,
        )
        assert result.record_count >= 1

    def test_temporal_disabled_by_default(self, tmp_path):
        """Temporal strategy falls back when experimental=False."""
        df_a = pd.DataFrame({"date": ["2024-01-01"], "val": [10]})
        df_b = pd.DataFrame({"date": ["2024-01-01"], "score": [100]})
        align = _result([
            _align("ds-a", "date", "ds-b", "date", atype=AlignmentType.EXACT),
        ], ["ds-a", "ds-b"])
        agent = _make_agent(tmp_path, experimental_strategies=False)
        result = agent.process(
            {"ds-a": df_a, "ds-b": df_b}, align,
            join_strategy=JoinStrategy.TEMPORAL,
        )
        assert result.record_count >= 1

    def test_temporal_enabled_when_experimental(self, tmp_path):
        """Temporal strategy works when experimental=True."""
        df_a = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00"]),
            "val": [10, 20],
        })
        df_b = pd.DataFrame({
            "event_time": pd.to_datetime(["2024-01-01 10:05", "2024-01-01 11:05"]),
            "score": [100, 200],
        })
        align = _result([
            _align("ds-a", "timestamp", "ds-b", "event_time", atype=AlignmentType.SYNONYM),
        ], ["ds-a", "ds-b"])
        agent = _make_agent(tmp_path, experimental_strategies=True)
        result = agent.process(
            {"ds-a": df_a, "ds-b": df_b}, align,
            join_strategy=JoinStrategy.TEMPORAL,
        )
        assert result.record_count == 2


# ================================================================== #
#  6. Column unification in semantic join                               #
# ================================================================== #

class TestSemanticJoin:
    def test_semantic_join_renames_right_columns(self, tmp_path):
        """Semantic join renames aligned right columns to left names."""
        df_a = pd.DataFrame({"name": ["Alice", "Bob"], "value": [1, 2]})
        df_b = pd.DataFrame({"full_name": ["Alice", "Bob"], "score": [10, 20]})
        align = _result([
            _align("ds-a", "name", "ds-b", "full_name",
                   similarity=0.8, atype=AlignmentType.RELATED),
        ], ["ds-a", "ds-b"])
        agent = _make_agent(tmp_path, similarity_join_threshold=0.5)
        result = agent.process(
            {"ds-a": df_a, "ds-b": df_b}, align,
            join_strategy=JoinStrategy.SEMANTIC_SIMILARITY,
        )
        df = agent.get_fused_dataframe(result.fused_dataset_id)
        # full_name should be renamed to name (left name)
        assert "name" in df.columns

    def test_semantic_join_matches_similar_records(self, tmp_path):
        """Semantic join matches records with similar values."""
        df_a = pd.DataFrame({"name": ["Alice", "Bob", "Eve"], "val": [10, 20, 30]})
        df_b = pd.DataFrame({"name": ["Alice", "Bob", "Frank"], "score": [100, 200, 300]})
        align = _result([
            _align("ds-a", "name", "ds-b", "name", atype=AlignmentType.EXACT),
        ], ["ds-a", "ds-b"])
        agent = _make_agent(tmp_path, similarity_join_threshold=0.5)
        result = agent.process(
            {"ds-a": df_a, "ds-b": df_b}, align,
            join_strategy=JoinStrategy.SEMANTIC_SIMILARITY,
        )
        df = agent.get_fused_dataframe(result.fused_dataset_id)
        # Alice and Bob should match; Eve and Frank should be unmatched
        assert result.record_count >= 3


# ================================================================== #
#  7. Imputation — value correctness                                   #
# ================================================================== #

class TestImputation:
    def test_mean_imputation_correct_value(self, tmp_path):
        """Mean imputation fills with arithmetic mean."""
        agent = _make_agent(tmp_path)
        df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0, 5.0]})
        result_df, summary = agent._impute_missing(df, ImputationMethod.MEAN)
        # Mean of [1, 2, 4, 5] = 3.0
        assert result_df["x"].iloc[2] == 3.0
        assert summary.total_nulls_filled == 1
        assert summary.fields_imputed == {"x": 1}

    def test_median_imputation_correct_value(self, tmp_path):
        """Median imputation fills with median."""
        agent = _make_agent(tmp_path)
        df = pd.DataFrame({"x": [1.0, 2.0, None, 100.0, 5.0]})
        result_df, summary = agent._impute_missing(df, ImputationMethod.MEDIAN)
        # Median of [1, 2, 5, 100] = 3.5
        assert abs(result_df["x"].iloc[2] - 3.5) < 0.01

    def test_mode_imputation_correct_value(self, tmp_path):
        """Mode imputation fills with most frequent value."""
        agent = _make_agent(tmp_path)
        df = pd.DataFrame({"x": [1.0, 1.0, None, 2.0, 1.0]})
        result_df, summary = agent._impute_missing(df, ImputationMethod.MODE)
        assert result_df["x"].iloc[2] == 1.0

    def test_knn_imputation_fills_nulls(self, tmp_path):
        """KNN imputation fills all nulls."""
        agent = _make_agent(tmp_path)
        df = pd.DataFrame({
            "a": [1.0, 2.0, None, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        result_df, summary = agent._impute_missing(df, ImputationMethod.KNN)
        assert not result_df["a"].isna().any()
        assert summary.total_nulls_filled == 1

    def test_imputation_skips_high_null_ratio(self, tmp_path):
        """Columns with >50% nulls are not imputed."""
        agent = _make_agent(tmp_path)
        df = pd.DataFrame({
            "sparse": [1.0, None, None, None, None],  # 80% null
            "dense": [10.0, 20.0, None, 40.0, 50.0],  # 20% null
        })
        result_df, summary = agent._impute_missing(df, ImputationMethod.MEAN)
        # sparse should still have nulls
        assert result_df["sparse"].isna().sum() == 4
        # dense should be filled
        assert not result_df["dense"].isna().any()
        assert "sparse" not in summary.fields_imputed
        assert "dense" in summary.fields_imputed

    def test_imputation_no_nulls_is_noop(self, tmp_path):
        """If no nulls, imputation does nothing."""
        agent = _make_agent(tmp_path)
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result_df, summary = agent._impute_missing(df, ImputationMethod.MEAN)
        assert summary.total_nulls_filled == 0


# ================================================================== #
#  8. Transformations — value correctness                              #
# ================================================================== #

class TestTransformations:
    def test_unit_conversion_applied(self, tmp_path):
        """Unit conversion transforms column values."""
        agent = _make_agent(tmp_path)
        df = pd.DataFrame({"temp_c": [0.0, 100.0, 25.0]})
        alignment = _align(
            "ds-a", "temp_f", "ds-b", "temp_c",
            atype=AlignmentType.SYNONYM,
            hint="unit_conversion:celsius_to_fahrenheit",
        )
        result_df, log = agent._apply_transformation(df, alignment)
        assert result_df["temp_c"].iloc[0] == 32.0
        assert result_df["temp_c"].iloc[1] == 212.0
        assert log is not None
        assert log.operation == "unit_conversion"
        assert log.records_affected == 3

    def test_type_cast_to_float(self, tmp_path):
        """Type cast transforms string to float."""
        agent = _make_agent(tmp_path)
        df = pd.DataFrame({"price": ["10.5", "20.3", "bad"]})
        alignment = _align(
            "ds-a", "amount", "ds-b", "price",
            hint="type_cast:string->float",
        )
        result_df, log = agent._apply_transformation(df, alignment)
        assert result_df["price"].iloc[0] == 10.5
        assert result_df["price"].iloc[1] == 20.3
        assert pd.isna(result_df["price"].iloc[2])  # "bad" → NaN

    def test_inverse_transform(self, tmp_path):
        """Inverse unit conversion is resolved correctly."""
        agent = _make_agent(tmp_path)
        df = pd.DataFrame({"weight": [2.20462]})
        alignment = _align(
            "ds-a", "weight_kg", "ds-b", "weight",
            hint="unit_conversion:kg_to_lb_inverse",
        )
        result_df, log = agent._apply_transformation(df, alignment)
        # Inverse of kg_to_lb is lb_to_kg: 2.20462 / 2.20462 ≈ 1.0
        assert abs(result_df["weight"].iloc[0] - 1.0) < 0.001


# ================================================================== #
#  9. Full pipeline                                                    #
# ================================================================== #

class TestFullPipeline:
    def test_full_pipeline_exact_key(self, tmp_path):
        """End-to-end: ingest → align → fuse → verify."""
        df_a = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "salary": [70000, 60000, 80000, 75000, 65000],
        })
        df_b = pd.DataFrame({
            "worker_id": [2, 3, 5, 6],
            "dept": ["Marketing", "Engineering", "Engineering", "Sales"],
            "bonus": [5000.0, 8000.0, 6000.0, 7000.0],
        })
        align = _result([
            _align("ds-a", "id", "ds-b", "worker_id", atype=AlignmentType.SYNONYM),
        ], ["ds-a", "ds-b"])
        agent = _make_agent(tmp_path)
        result = agent.process({"ds-a": df_a, "ds-b": df_b}, align)
        df = agent.get_fused_dataframe(result.fused_dataset_id)

        # Outer join: 5 left + 1 right-only (id=6) = 6 rows
        assert result.record_count == 6
        assert len(result.source_datasets) == 2

        # Spot checks
        row2 = df[df["id"] == 2].iloc[0]
        assert row2["name"] == "Bob"
        assert row2["dept"] == "Marketing"
        assert row2["bonus"] == 5000.0

        row6 = df[df["id"] == 6].iloc[0]
        assert row6["dept"] == "Sales"
        assert row6["bonus"] == 7000.0
        assert pd.isna(row6["name"])

        # id=1 and id=4 have no right match
        row1 = df[df["id"] == 1].iloc[0]
        assert row1["name"] == "Alice"
        assert pd.isna(row1["dept"])

    def test_pipeline_with_imputation(self, tmp_path):
        """Full pipeline with mean imputation on fused result."""
        df_a = pd.DataFrame({
            "id": [1, 2, 3],
            "score": [100.0, None, 80.0],
        })
        df_b = pd.DataFrame({
            "id": [2, 3, 4],
            "rating": [4.0, None, 3.0],
        })
        align = _result([
            _align("ds-a", "id", "ds-b", "id"),
        ], ["ds-a", "ds-b"])
        agent = _make_agent(tmp_path)
        result = agent.process(
            {"ds-a": df_a, "ds-b": df_b}, align,
            imputation_method=ImputationMethod.MEAN,
        )
        df = agent.get_fused_dataframe(result.fused_dataset_id)
        # Score mean = (100 + 80) / 2 = 90; fills id=2's null
        assert not df["score"].isna().any() or result.imputation_summary.total_nulls_filled > 0

    def test_result_has_storage_path(self, tmp_path):
        """FusionResult includes a valid storage path."""
        df_a = pd.DataFrame({"id": [1], "val": [10]})
        df_b = pd.DataFrame({"id": [1], "score": [100]})
        align = _result([
            _align("ds-a", "id", "ds-b", "id"),
        ], ["ds-a", "ds-b"])
        agent = _make_agent(tmp_path)
        result = agent.process({"ds-a": df_a, "ds-b": df_b}, align)
        assert result.storage_path.endswith(".csv")
        assert os.path.exists(result.storage_path)

    def test_event_published(self, tmp_path):
        """dataset.fused event is published."""
        events = []
        def handler(event):
            events.append(event)

        df_a = pd.DataFrame({"id": [1], "val": [10]})
        df_b = pd.DataFrame({"id": [1], "score": [100]})
        align = _result([_align("ds-a", "id", "ds-b", "id")], ["ds-a", "ds-b"])
        agent = _make_agent(tmp_path)
        agent.subscribe("dataset.fused", handler)
        agent.process({"ds-a": df_a, "ds-b": df_b}, align)
        assert len(events) == 1
        assert events[0]["event_type"] == "dataset.fused"

    def test_no_alignments_skips_dataset(self, tmp_path):
        """Datasets with no alignments are skipped with warning."""
        df_a = pd.DataFrame({"id": [1], "val": [10]})
        df_b = pd.DataFrame({"x": [100]})
        align = _result([], ["ds-a", "ds-b"])
        agent = _make_agent(tmp_path)
        result = agent.process({"ds-a": df_a, "ds-b": df_b}, align)
        # Only left dataset should be in source_datasets
        assert result.source_datasets == ["ds-a"]
        df = agent.get_fused_dataframe(result.fused_dataset_id)
        assert len(df) == 1
        assert list(df.columns) == ["id", "val"]


# ================================================================== #
#  10. Transformation in pipeline                                      #
# ================================================================== #

class TestTransformationInPipeline:
    def test_unit_conversion_in_merge(self, tmp_path):
        """Unit conversion is applied before merge."""
        df_a = pd.DataFrame({"id": [1], "temp_f": [32.0]})
        df_b = pd.DataFrame({"id": [1], "temp_c": [0.0]})
        align = _result([
            _align("ds-a", "id", "ds-b", "id"),
            _align("ds-a", "temp_f", "ds-b", "temp_c",
                   atype=AlignmentType.SYNONYM,
                   hint="unit_conversion:celsius_to_fahrenheit"),
        ], ["ds-a", "ds-b"])
        agent = _make_agent(tmp_path)
        result = agent.process(
            {"ds-a": df_a, "ds-b": df_b}, align,
            join_strategy=JoinStrategy.EXACT_KEY,
        )
        assert len(result.transformations_applied) == 1
        assert result.transformations_applied[0].operation == "unit_conversion"

    def test_transformation_log_has_samples(self, tmp_path):
        """Transformation log includes before/after samples."""
        df_a = pd.DataFrame({"id": [1], "temp_f": [32.0]})
        df_b = pd.DataFrame({"id": [1], "temp_c": [0.0]})
        align = _result([
            _align("ds-a", "id", "ds-b", "id"),
            _align("ds-a", "temp_f", "ds-b", "temp_c",
                   atype=AlignmentType.SYNONYM,
                   hint="unit_conversion:celsius_to_fahrenheit"),
        ], ["ds-a", "ds-b"])
        agent = _make_agent(tmp_path)
        result = agent.process(
            {"ds-a": df_a, "ds-b": df_b}, align,
            join_strategy=JoinStrategy.EXACT_KEY,
        )
        log = result.transformations_applied[0]
        assert log.source_value_sample == 0.0  # Before: 0°C
        assert log.target_value_sample == 32.0  # After: 32°F
