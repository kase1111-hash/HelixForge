"""Phase 7: Backlog feature tests.

Tests for:
  - K-means clustering (InsightAgent)
  - LLM narrative generation (InsightAgent + MockProvider)
  - Probabilistic join strategy (FusionAgent)
"""

import tempfile

import numpy as np
import pandas as pd
import pytest


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _make_agent(**insight_overrides):
    from agents.insight_agent import InsightAgent
    config = {"insight": insight_overrides}
    return InsightAgent(config=config)


def _make_agent_with_provider(**insight_overrides):
    from agents.insight_agent import InsightAgent
    from utils.llm import MockProvider
    provider = MockProvider(dimensions=1536)
    config = {"insight": insight_overrides}
    return InsightAgent(config=config, provider=provider)


def _cluster_df():
    """Three well-separated clusters for testing."""
    np.random.seed(42)
    c1 = np.random.normal(loc=[0, 0], scale=0.5, size=(20, 2))
    c2 = np.random.normal(loc=[10, 10], scale=0.5, size=(20, 2))
    c3 = np.random.normal(loc=[20, 0], scale=0.5, size=(20, 2))
    data = np.vstack([c1, c2, c3])
    return pd.DataFrame(data, columns=["x", "y"])


# ------------------------------------------------------------------ #
#  K-means clustering tests                                            #
# ------------------------------------------------------------------ #

class TestKMeansClustering:
    """Clustering produces correct labels and scores."""

    def test_basic_clustering(self):
        df = _cluster_df()
        agent = _make_agent()
        result = agent.process(df, include_stats=False, include_correlations=False,
                               include_outliers=False, include_clustering=True, n_clusters=3)
        assert result.clustering is not None
        assert result.clustering.n_clusters == 3
        assert len(result.clustering.labels) == 60
        assert set(result.clustering.labels) == {0, 1, 2}

    def test_silhouette_score_high_for_separated_clusters(self):
        df = _cluster_df()
        agent = _make_agent()
        result = agent.process(df, include_stats=False, include_correlations=False,
                               include_outliers=False, include_clustering=True, n_clusters=3)
        # Well-separated clusters should have high silhouette
        assert result.clustering.silhouette_score > 0.5

    def test_centroids_in_original_scale(self):
        df = _cluster_df()
        agent = _make_agent()
        result = agent.process(df, include_stats=False, include_correlations=False,
                               include_outliers=False, include_clustering=True, n_clusters=3)
        centroids = result.clustering.centroids
        assert len(centroids) == 3
        # Each centroid is [x, y]
        assert len(centroids[0]) == 2
        # Centroids should be near the cluster centers (0,0), (10,10), (20,0)
        flat = sorted(centroids, key=lambda c: c[0])
        assert abs(flat[0][0] - 0) < 2
        assert abs(flat[1][0] - 10) < 2
        assert abs(flat[2][0] - 20) < 2

    def test_features_used(self):
        df = _cluster_df()
        agent = _make_agent()
        result = agent.process(df, include_stats=False, include_correlations=False,
                               include_outliers=False, include_clustering=True, n_clusters=3)
        assert result.clustering.features_used == ["x", "y"]

    def test_inertia_positive(self):
        df = _cluster_df()
        agent = _make_agent()
        result = agent.process(df, include_stats=False, include_correlations=False,
                               include_outliers=False, include_clustering=True, n_clusters=3)
        assert result.clustering.inertia > 0

    def test_clustering_disabled_by_default(self):
        df = _cluster_df()
        agent = _make_agent()
        result = agent.process(df)
        assert result.clustering is None

    def test_too_few_rows_returns_none(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        agent = _make_agent()
        result = agent.process(df, include_stats=False, include_correlations=False,
                               include_outliers=False, include_clustering=True, n_clusters=3)
        assert result.clustering is None

    def test_no_numeric_columns_returns_none(self):
        df = pd.DataFrame({"a": ["x", "y", "z", "w", "v"]})
        agent = _make_agent()
        result = agent.process(df, include_stats=False, include_correlations=False,
                               include_outliers=False, include_clustering=True, n_clusters=2)
        assert result.clustering is None

    def test_custom_n_clusters(self):
        df = _cluster_df()
        agent = _make_agent()
        result = agent.process(df, include_stats=False, include_correlations=False,
                               include_outliers=False, include_clustering=True, n_clusters=2)
        assert result.clustering.n_clusters == 2
        assert set(result.clustering.labels) == {0, 1}

    def test_handles_nan_rows(self):
        df = _cluster_df()
        df.loc[5, "x"] = np.nan
        df.loc[15, "y"] = np.nan
        agent = _make_agent()
        result = agent.process(df, include_stats=False, include_correlations=False,
                               include_outliers=False, include_clustering=True, n_clusters=3)
        assert result.clustering is not None
        # Labels should be fewer than 60 since NaN rows are dropped
        assert len(result.clustering.labels) == 58


# ------------------------------------------------------------------ #
#  LLM narrative generation tests                                      #
# ------------------------------------------------------------------ #

class TestNarrativeGeneration:
    """Narrative uses LLMProvider.complete() and returns text."""

    def test_narrative_with_mock_provider(self):
        df = pd.DataFrame({"x": range(10), "y": range(10, 20)})
        agent = _make_agent_with_provider()
        result = agent.process(df, include_narrative=True)
        assert result.narrative is not None
        assert len(result.narrative) > 0

    def test_narrative_disabled_by_default(self):
        df = pd.DataFrame({"x": range(10), "y": range(10, 20)})
        agent = _make_agent_with_provider()
        result = agent.process(df)
        assert result.narrative is None

    def test_narrative_without_provider_returns_none(self):
        df = pd.DataFrame({"x": range(10), "y": range(10, 20)})
        agent = _make_agent()  # No provider
        result = agent.process(df, include_narrative=True)
        assert result.narrative is None

    def test_narrative_includes_stats_context(self):
        """MockProvider returns a dataset description string."""
        df = pd.DataFrame({"x": range(10), "y": range(10, 20)})
        agent = _make_agent_with_provider()
        result = agent.process(df, include_stats=True, include_narrative=True)
        # MockProvider.complete() returns a generic description
        assert isinstance(result.narrative, str)

    def test_narrative_with_clustering(self):
        df = _cluster_df()
        agent = _make_agent_with_provider()
        result = agent.process(df, include_clustering=True,
                               include_narrative=True, n_clusters=3)
        assert result.clustering is not None
        assert result.narrative is not None


# ------------------------------------------------------------------ #
#  Probabilistic join strategy tests                                   #
# ------------------------------------------------------------------ #

class TestProbabilisticJoin:
    """Probabilistic strategy matches records via weighted fuzzy similarity."""

    def _make_fusion_agent(self, experimental=True):
        from agents.fusion_agent import FusionAgent
        config = {
            "fusion": {
                "default_join_strategy": "probabilistic",
                "experimental_strategies": experimental,
                "probabilistic_match_threshold": 0.5,
                "output_format": "csv",
                "output_path": tempfile.mkdtemp(),
            }
        }
        return FusionAgent(config=config)

    def _align(self, src_ds, src_field, tgt_ds, tgt_field, atype="exact", sim=0.95):
        from models.schemas import AlignmentType, FieldAlignment
        return FieldAlignment(
            alignment_id=f"{src_field}-{tgt_field}",
            source_dataset=src_ds,
            source_field=src_field,
            target_dataset=tgt_ds,
            target_field=tgt_field,
            similarity=sim,
            alignment_type=AlignmentType(atype),
        )

    def _result(self, alignments, datasets):
        from models.schemas import AlignmentResult
        return AlignmentResult(
            alignment_job_id="test-job",
            datasets_aligned=datasets,
            alignments=alignments,
        )

    def test_exact_numeric_match(self):
        """Identical numeric records should match perfectly."""
        agent = self._make_fusion_agent()
        df_left = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        df_right = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        align = [self._align("A", "id", "B", "id"), self._align("A", "val", "B", "val")]
        ar = self._result(align, ["A", "B"])
        result = agent.process({"A": df_left, "B": df_right}, alignment_result=ar)
        assert result.record_count == 3

    def test_fuzzy_string_match(self):
        """Similar strings should be matched via fuzzy ratio."""
        agent = self._make_fusion_agent()
        df_left = pd.DataFrame({"name": ["Alice Smith", "Bob Jones", "Charlie"]})
        df_right = pd.DataFrame({"full_name": ["Alice S.", "Robert Jones", "Charlie B."]})
        align = [self._align("A", "name", "B", "full_name")]
        ar = self._result(align, ["A", "B"])
        result = agent.process({"A": df_left, "B": df_right}, alignment_result=ar)
        # All should match (fuzzy similarity above threshold)
        assert result.record_count == 3

    def test_unmatched_rows_preserved(self):
        """Rows below threshold appear as unmatched."""
        agent = self._make_fusion_agent()
        df_left = pd.DataFrame({"name": ["Alice"]})
        df_right = pd.DataFrame({"full_name": ["Zzzzz", "Yyyyy"]})
        align = [self._align("A", "name", "B", "full_name")]
        ar = self._result(align, ["A", "B"])
        result = agent.process({"A": df_left, "B": df_right}, alignment_result=ar)
        # 1 unmatched left + 2 unmatched right = 3
        assert result.record_count == 3

    def test_disabled_falls_back(self):
        """When experimental=False, falls back to semantic_similarity."""
        agent = self._make_fusion_agent(experimental=False)
        df_left = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        df_right = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        align = [self._align("A", "id", "B", "id"), self._align("A", "val", "B", "val")]
        ar = self._result(align, ["A", "B"])
        result = agent.process({"A": df_left, "B": df_right}, alignment_result=ar,
                               join_strategy="probabilistic")
        # Should still produce a result (falls back to semantic)
        assert result.record_count >= 2

    def test_multi_column_scoring(self):
        """Multiple aligned columns improve match precision."""
        agent = self._make_fusion_agent()
        df_left = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "age": [30, 25],
            "city": ["NYC", "LA"],
        })
        df_right = pd.DataFrame({
            "person": ["Alice", "Bob"],
            "years": [30, 25],
            "location": ["New York", "Los Angeles"],
        })
        align = [
            self._align("A", "name", "B", "person"),
            self._align("A", "age", "B", "years"),
            self._align("A", "city", "B", "location"),
        ]
        ar = self._result(align, ["A", "B"])
        result = agent.process({"A": df_left, "B": df_right}, alignment_result=ar)
        assert result.record_count == 2


# ------------------------------------------------------------------ #
#  Schema tests                                                        #
# ------------------------------------------------------------------ #

class TestBacklogSchemas:
    """New schema models work correctly."""

    def test_cluster_info_defaults(self):
        from models.schemas import ClusterInfo
        ci = ClusterInfo(n_clusters=3)
        assert ci.labels == []
        assert ci.centroids == []
        assert ci.silhouette_score is None
        assert ci.inertia == 0.0
        assert ci.features_used == []

    def test_insight_result_has_clustering_field(self):
        from models.schemas import InsightResult
        r = InsightResult(
            analysis_id="test",
            source_description="test",
            record_count=10,
            field_count=2,
        )
        assert r.clustering is None
        assert r.narrative is None

    def test_insight_config_new_defaults(self):
        from models.schemas import InsightConfig
        cfg = InsightConfig()
        assert cfg.include_clustering is False
        assert cfg.n_clusters == 3
        assert cfg.include_narrative is False

    def test_insight_config_n_clusters_validation(self):
        from models.schemas import InsightConfig
        with pytest.raises(Exception):
            InsightConfig(n_clusters=1)  # min is 2
        with pytest.raises(Exception):
            InsightConfig(n_clusters=21)  # max is 20
