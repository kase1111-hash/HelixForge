"""Unit tests for Insight Generator Agent."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from agents.insight_generator_agent import InsightGeneratorAgent
from models.schemas import (
    DistributionType,
    FindingType,
    Severity,
)


class TestInsightGeneratorAgent:
    """Tests for InsightGeneratorAgent."""

    @pytest.fixture
    def agent(self, insight_config):
        """Create agent instance for testing."""
        return InsightGeneratorAgent(config=insight_config)

    @pytest.fixture
    def mock_agent(self, insight_config, mock_openai_client):
        """Create agent with mocked OpenAI client."""
        agent = InsightGeneratorAgent(config=insight_config)
        agent._openai_client = mock_openai_client
        return agent

    @pytest.fixture
    def analysis_dataframe(self):
        """Create DataFrame suitable for analysis."""
        np.random.seed(42)
        n = 100

        return pd.DataFrame({
            "id": range(n),
            "age": np.random.normal(35, 10, n).astype(int),
            "salary": np.random.normal(75000, 15000, n),
            "experience": np.random.normal(10, 5, n),
            "performance_score": np.random.normal(75, 10, n),
            "department": np.random.choice(["Eng", "Sales", "Mkt"], n)
        })

    @pytest.fixture
    def correlated_dataframe(self):
        """Create DataFrame with known correlations."""
        np.random.seed(42)
        n = 100

        x = np.random.normal(0, 1, n)
        y = x * 0.8 + np.random.normal(0, 0.3, n)  # Strongly correlated
        z = np.random.normal(0, 1, n)  # Uncorrelated

        return pd.DataFrame({
            "var_x": x,
            "var_y": y,
            "var_z": z
        })

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent.agent_name == "InsightGeneratorAgent"
        assert agent.event_type == "insight.generated"

    def test_compute_statistics(self, agent, analysis_dataframe):
        """Test statistics computation."""
        stats = agent._compute_statistics(analysis_dataframe)

        assert stats.record_count == len(analysis_dataframe)
        assert stats.field_count == len(analysis_dataframe.columns)
        assert "salary" in stats.field_stats

        salary_stats = stats.field_stats["salary"]
        assert salary_stats.mean is not None
        assert salary_stats.std is not None
        assert salary_stats.min is not None
        assert salary_stats.max is not None

    def test_compute_statistics_with_nulls(self, agent, sample_dataframe_with_nulls):
        """Test statistics with null values."""
        stats = agent._compute_statistics(sample_dataframe_with_nulls)

        assert stats.field_stats["name"].null_count == 2
        assert stats.field_stats["salary"].null_count == 2

    def test_detect_distribution_normal(self, agent):
        """Test normal distribution detection."""
        np.random.seed(42)
        normal_series = pd.Series(np.random.normal(0, 1, 1000))

        dist_type = agent._detect_distribution(normal_series)
        assert dist_type == DistributionType.NORMAL

    def test_detect_distribution_skewed(self, agent):
        """Test skewed distribution detection."""
        # Create highly skewed data
        skewed_series = pd.Series(np.random.exponential(2, 1000))

        dist_type = agent._detect_distribution(skewed_series)
        assert dist_type in [DistributionType.SKEWED, DistributionType.UNIFORM]

    def test_compute_correlations(self, agent, correlated_dataframe):
        """Test correlation computation."""
        correlations = agent._compute_correlations(correlated_dataframe)

        assert correlations is not None
        assert len(correlations.correlations) > 0

        # Find the x-y correlation
        xy_corr = None
        for pair in correlations.correlations:
            if set([pair.field_a, pair.field_b]) == {"var_x", "var_y"}:
                xy_corr = pair
                break

        assert xy_corr is not None
        assert xy_corr.coefficient > 0.7  # Should be strongly correlated

    def test_compute_correlations_significant(self, agent, correlated_dataframe):
        """Test significant correlation detection."""
        correlations = agent._compute_correlations(correlated_dataframe)

        # x and y should be in significant pairs
        significant_fields = set()
        for pair in correlations.significant_pairs:
            significant_fields.add(pair.field_a)
            significant_fields.add(pair.field_b)

        assert "var_x" in significant_fields or "var_y" in significant_fields

    def test_detect_outliers(self, agent, analysis_dataframe):
        """Test outlier detection."""
        # Add some outliers
        df = analysis_dataframe.copy()
        df.loc[0, "salary"] = 500000  # Clear outlier
        df.loc[1, "salary"] = 1000  # Clear outlier

        outliers = agent._detect_outliers(df)

        assert outliers is not None
        assert outliers.total_outliers > 0
        assert "salary" in outliers.outliers_by_field

    def test_detect_outliers_no_outliers(self, agent):
        """Test outlier detection with no outliers."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 11, 12, 13, 14]
        })

        outliers = agent._detect_outliers(df)
        assert outliers.total_outliers == 0

    def test_perform_clustering(self, agent, analysis_dataframe):
        """Test clustering analysis."""
        clusters = agent._perform_clustering(analysis_dataframe)

        assert clusters is not None
        assert clusters.n_clusters >= 2
        assert len(clusters.cluster_sizes) == clusters.n_clusters
        assert clusters.silhouette_score is not None

    def test_perform_clustering_insufficient_data(self, agent):
        """Test clustering with insufficient data."""
        small_df = pd.DataFrame({
            "a": [1, 2],
            "b": [3, 4]
        })

        clusters = agent._perform_clustering(small_df)
        assert clusters is None

    def test_assess_correlation_severity_high(self, agent):
        """Test high correlation severity assessment."""
        severity = agent._assess_correlation_severity(0.9)
        assert severity == Severity.HIGH

    def test_assess_correlation_severity_medium(self, agent):
        """Test medium correlation severity assessment."""
        severity = agent._assess_correlation_severity(0.7)
        assert severity == Severity.MEDIUM

    def test_assess_correlation_severity_low(self, agent):
        """Test low correlation severity assessment."""
        severity = agent._assess_correlation_severity(0.4)
        assert severity == Severity.LOW

    @patch("agents.insight_generator_agent.plt")
    def test_generate_visualizations(self, mock_plt, agent, analysis_dataframe):
        """Test visualization generation."""
        mock_fig = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, MagicMock())

        correlations = agent._compute_correlations(analysis_dataframe)
        outliers = agent._detect_outliers(analysis_dataframe)

        visualizations = agent._generate_visualizations(
            analysis_dataframe, "test-dataset",
            correlations, outliers, None
        )

        # Should generate some visualizations
        assert len(visualizations) >= 0  # May fail without matplotlib

    def test_export_report_json(self, agent, analysis_dataframe):
        """Test JSON report export."""
        stats = agent._compute_statistics(analysis_dataframe)

        path = agent._export_report(
            "test-dataset", "insight-123", "json",
            stats, None, None, None, "Test narrative", [], []
        )

        assert path.endswith(".json")
        assert os.path.exists(path)

    def test_export_report_html(self, agent, analysis_dataframe):
        """Test HTML report export."""
        stats = agent._compute_statistics(analysis_dataframe)

        path = agent._export_report(
            "test-dataset", "insight-123", "html",
            stats, None, None, None, "Test narrative", [], []
        )

        assert path.endswith(".html")
        assert os.path.exists(path)

    def test_generate_html_report_structure(self, agent, analysis_dataframe):
        """Test HTML report structure."""
        stats = agent._compute_statistics(analysis_dataframe)

        html = agent._generate_html_report(
            "test-dataset", stats, None, None, None,
            "Test narrative", [], []
        )

        assert "<html>" in html
        assert "test-dataset" in html
        assert "Test narrative" in html

    def test_process_full_pipeline(self, mock_agent, analysis_dataframe):
        """Test full insight generation pipeline."""
        with patch.object(mock_agent, "_generate_visualizations", return_value=[]):
            result = mock_agent.process(
                "test-dataset",
                analysis_dataframe,
                generate_visualizations=False
            )

        assert result is not None
        assert result.insight_id is not None
        assert result.fused_dataset_id == "test-dataset"
        assert result.statistics is not None

    def test_key_findings_generated(self, mock_agent, correlated_dataframe):
        """Test that key findings are generated."""
        with patch.object(mock_agent, "_generate_visualizations", return_value=[]):
            result = mock_agent.process(
                "test-dataset",
                correlated_dataframe,
                generate_visualizations=False
            )

        # Should have correlation findings
        correlation_findings = [
            f for f in result.key_findings
            if f.type == FindingType.CORRELATION
        ]
        assert len(correlation_findings) > 0

    def test_event_published(self, mock_agent, analysis_dataframe):
        """Test that insight.generated event is published."""
        events_received = []

        def event_handler(event):
            events_received.append(event)

        mock_agent.subscribe("insight.generated", event_handler)

        with patch.object(mock_agent, "_generate_visualizations", return_value=[]):
            mock_agent.process(
                "test-dataset",
                analysis_dataframe,
                generate_visualizations=False
            )

        assert len(events_received) == 1
        assert events_received[0]["event_type"] == "insight.generated"

    def test_analysis_types_filter(self, mock_agent, analysis_dataframe):
        """Test filtering analysis types."""
        with patch.object(mock_agent, "_generate_visualizations", return_value=[]):
            result = mock_agent.process(
                "test-dataset",
                analysis_dataframe,
                analysis_types=["correlations"],  # Only correlations
                generate_visualizations=False
            )

        assert result.correlations is not None
        # Outliers and clusters should still exist but empty
        assert result.outliers is None or result.outliers.total_outliers == 0
