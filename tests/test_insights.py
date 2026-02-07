"""Correctness tests for Insight Agent.

Every statistical assertion uses exact values (within floating-point
tolerance), not just "is not None" or "len > 0".
"""

import math

import numpy as np
import pandas as pd
import pytest

from agents.insight_agent import InsightAgent


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _make_agent(**overrides):
    """Create an InsightAgent with test defaults."""
    cfg = {
        "insight": {
            "correlation_method": "pearson",
            "correlation_threshold": 0.5,
            "outlier_iqr_multiplier": 1.5,
            "include_stats": True,
            "include_correlations": True,
            "include_outliers": True,
        }
    }
    cfg["insight"].update(overrides)
    return InsightAgent(config=cfg)


def _stat(result, field_name):
    """Get FieldStatistics for a field by name."""
    for s in result.statistics:
        if s.field_name == field_name:
            return s
    raise KeyError(f"No statistics for field '{field_name}'")


def _corr(result, field_a, field_b):
    """Get CorrelationPair for a field pair."""
    for c in result.correlations:
        if (c.field_a == field_a and c.field_b == field_b) or \
           (c.field_a == field_b and c.field_b == field_a):
            return c
    return None


def _outlier(result, field_name):
    """Get OutlierInfo for a field by name."""
    for o in result.outliers:
        if o.field_name == field_name:
            return o
    return None


# ================================================================== #
#  1. Initialization                                                   #
# ================================================================== #

class TestInsightInit:
    def test_agent_name(self):
        agent = _make_agent()
        assert agent.agent_name == "InsightAgent"

    def test_event_type(self):
        agent = _make_agent()
        assert agent.event_type == "dataset.analyzed"


# ================================================================== #
#  2. Descriptive statistics — exact values                            #
# ================================================================== #

class TestDescriptiveStats:
    def test_simple_sequence(self):
        """[1,2,3,4,5] → known mean, median, std, min, max, quartiles."""
        agent = _make_agent()
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        result = agent.process(df, "test")

        s = _stat(result, "x")
        assert s.count == 5
        assert s.mean == pytest.approx(3.0)
        assert s.median == pytest.approx(3.0)
        assert s.min == pytest.approx(1.0)
        assert s.max == pytest.approx(5.0)
        # std with ddof=1: sqrt(10/4) = sqrt(2.5) ≈ 1.5811
        assert s.std == pytest.approx(math.sqrt(2.5), rel=1e-4)
        assert s.q1 == pytest.approx(2.0)
        assert s.q3 == pytest.approx(4.0)

    def test_single_value(self):
        """All same values → std=0, min=max=mean=median."""
        agent = _make_agent()
        df = pd.DataFrame({"x": [7.0, 7.0, 7.0]})
        result = agent.process(df)

        s = _stat(result, "x")
        assert s.mean == pytest.approx(7.0)
        assert s.median == pytest.approx(7.0)
        assert s.std == pytest.approx(0.0)
        assert s.min == pytest.approx(7.0)
        assert s.max == pytest.approx(7.0)

    def test_null_count(self):
        """Null values are counted but excluded from statistics."""
        agent = _make_agent()
        df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0, 5.0]})
        result = agent.process(df)

        s = _stat(result, "x")
        assert s.null_count == 1
        assert s.count == 4
        assert s.mean == pytest.approx(3.0)

    def test_unique_count(self):
        """Unique count tracks distinct values."""
        agent = _make_agent()
        df = pd.DataFrame({"x": [1, 1, 2, 2, 3]})
        result = agent.process(df)

        s = _stat(result, "x")
        assert s.unique_count == 3

    def test_multiple_columns(self):
        """Stats computed for each numeric column independently."""
        agent = _make_agent()
        df = pd.DataFrame({
            "a": [10, 20, 30],
            "b": [100, 200, 300],
            "name": ["x", "y", "z"],  # non-numeric, should be skipped
        })
        result = agent.process(df)

        assert len(result.statistics) == 2
        assert _stat(result, "a").mean == pytest.approx(20.0)
        assert _stat(result, "b").mean == pytest.approx(200.0)

    def test_record_count_and_field_count(self):
        """InsightResult tracks DataFrame dimensions."""
        agent = _make_agent()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["x", "y", "z"]})
        result = agent.process(df)
        assert result.record_count == 3
        assert result.field_count == 3

    def test_empty_numeric_skipped(self):
        """All-null numeric column is skipped."""
        agent = _make_agent()
        df = pd.DataFrame({"x": [None, None, None]})
        result = agent.process(df)
        assert len(result.statistics) == 0

    def test_skewed_distribution(self):
        """Verify quartiles for a skewed dataset."""
        agent = _make_agent()
        df = pd.DataFrame({"x": [1, 1, 1, 1, 1, 1, 1, 100]})
        result = agent.process(df)

        s = _stat(result, "x")
        assert s.median == pytest.approx(1.0)
        assert s.mean == pytest.approx(13.375)  # (7*1 + 100) / 8
        assert s.max == pytest.approx(100.0)


# ================================================================== #
#  3. Correlation analysis — exact values                              #
# ================================================================== #

class TestCorrelations:
    def test_perfect_positive_correlation(self):
        """x and 2x have Pearson correlation = 1.0."""
        agent = _make_agent(correlation_threshold=0.0)
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
        })
        result = agent.process(df)

        c = _corr(result, "x", "y")
        assert c is not None
        assert c.coefficient == pytest.approx(1.0, abs=1e-6)
        assert c.p_value < 0.01

    def test_perfect_negative_correlation(self):
        """x and -x have Pearson correlation = -1.0."""
        agent = _make_agent(correlation_threshold=0.0)
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [-1, -2, -3, -4, -5],
        })
        result = agent.process(df)

        c = _corr(result, "x", "y")
        assert c is not None
        assert c.coefficient == pytest.approx(-1.0, abs=1e-6)

    def test_zero_correlation(self):
        """Uncorrelated variables should have |coef| < threshold."""
        agent = _make_agent(correlation_threshold=0.5)
        np.random.seed(42)
        df = pd.DataFrame({
            "x": [1, 0, -1, 0, 1, 0, -1, 0],
            "y": [0, 1, 0, -1, 0, 1, 0, -1],
        })
        result = agent.process(df)

        c = _corr(result, "x", "y")
        assert c is None  # Should not appear (below threshold)

    def test_threshold_filtering(self):
        """Correlations below threshold are excluded."""
        agent = _make_agent(correlation_threshold=0.99)
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [1, 2, 3, 4, 6],  # Almost perfectly correlated but not quite
        })
        result = agent.process(df)

        # The correlation is high (~0.99) but might not hit 0.99 threshold
        # The exact value depends on data — this tests the filter works
        for c in result.correlations:
            assert abs(c.coefficient) >= 0.99

    def test_correlation_symmetric(self):
        """Correlation matrix is symmetric (only upper triangle stored)."""
        agent = _make_agent(correlation_threshold=0.0)
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [5, 4, 3, 2, 1],
        })
        result = agent.process(df)

        # Should have 3 pairs: a-b, a-c, b-c
        assert len(result.correlations) == 3

    def test_spearman_method(self):
        """Spearman correlation captures monotonic relationships."""
        agent = _make_agent(correlation_method="spearman", correlation_threshold=0.0)
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [1, 8, 27, 64, 125],  # x^3 - monotonically increasing
        })
        result = agent.process(df)

        c = _corr(result, "x", "y")
        assert c is not None
        assert c.coefficient == pytest.approx(1.0, abs=1e-6)

    def test_single_numeric_column_no_correlations(self):
        """Need at least 2 numeric columns for correlations."""
        agent = _make_agent()
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = agent.process(df)
        assert len(result.correlations) == 0

    def test_constant_column_skipped(self):
        """Constant columns (zero variance) don't produce correlations."""
        agent = _make_agent(correlation_threshold=0.0)
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "const": [1, 1, 1, 1, 1],
        })
        result = agent.process(df)
        assert len(result.correlations) == 0

    def test_with_noise(self):
        """Strong linear relationship with noise → high but imperfect r."""
        agent = _make_agent(correlation_threshold=0.5)
        np.random.seed(42)
        x = np.arange(50, dtype=float)
        noise = np.random.normal(0, 1, 50)
        y = 3 * x + 10 + noise
        df = pd.DataFrame({"x": x, "y": y})
        result = agent.process(df)

        c = _corr(result, "x", "y")
        assert c is not None
        assert c.coefficient > 0.95  # Very high correlation with small noise


# ================================================================== #
#  4. Outlier detection — exact values                                 #
# ================================================================== #

class TestOutlierDetection:
    def test_planted_outlier(self):
        """[1, 1, 1, 100] → 100 is an outlier by IQR."""
        agent = _make_agent()
        df = pd.DataFrame({"x": [1.0, 1.0, 1.0, 100.0]})
        result = agent.process(df)

        o = _outlier(result, "x")
        assert o is not None
        assert o.outlier_count == 1
        assert 3 in o.outlier_indices  # index of 100.0

    def test_no_outliers_in_uniform_data(self):
        """Evenly spaced data has no outliers."""
        agent = _make_agent()
        df = pd.DataFrame({"x": list(range(1, 21))})  # 1..20
        result = agent.process(df)

        o = _outlier(result, "x")
        assert o is None

    def test_iqr_bounds_correct(self):
        """Verify the IQR bounds are computed correctly.

        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100] (n=11)
        Pandas linear interpolation:
          Q1 at position 0.25*10=2.5 → lerp(3,4) = 3.5
          Q3 at position 0.75*10=7.5 → lerp(8,9) = 8.5
        IQR = 5.0
        Lower = 3.5 - 1.5*5 = -4.0
        Upper = 8.5 + 1.5*5 = 16.0
        Outlier: 100 (above 16.0)
        """
        agent = _make_agent()
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]})
        result = agent.process(df)

        o = _outlier(result, "x")
        assert o is not None
        assert o.outlier_count == 1
        assert o.upper_bound == pytest.approx(16.0)
        assert o.lower_bound == pytest.approx(-4.0)

    def test_multiple_outliers(self):
        """Multiple outliers on both sides are detected."""
        agent = _make_agent()
        # Normal range: 10-20. Outliers: -50 and 80
        data = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, -50, 80]
        df = pd.DataFrame({"x": data})
        result = agent.process(df)

        o = _outlier(result, "x")
        assert o is not None
        assert o.outlier_count == 2

    def test_custom_iqr_multiplier(self):
        """Larger IQR multiplier = fewer outliers."""
        # With k=1.5, 100 is an outlier in [1..10, 100]
        # With k=100, 100 is NOT an outlier (bounds expand hugely)
        agent_strict = _make_agent(outlier_iqr_multiplier=1.5)
        agent_lenient = _make_agent(outlier_iqr_multiplier=100.0)

        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]})

        result_strict = agent_strict.process(df)
        result_lenient = agent_lenient.process(df)

        assert _outlier(result_strict, "x") is not None
        assert _outlier(result_lenient, "x") is None

    def test_too_few_values_skipped(self):
        """Fewer than 4 values → skip outlier detection."""
        agent = _make_agent()
        df = pd.DataFrame({"x": [1.0, 2.0, 1000.0]})
        result = agent.process(df)

        o = _outlier(result, "x")
        assert o is None  # Not enough data points

    def test_outlier_per_column(self):
        """Each column is analyzed independently."""
        agent = _make_agent()
        df = pd.DataFrame({
            "a": [1, 1, 1, 1, 1, 1, 1, 100],  # Has outlier
            "b": [1, 2, 3, 4, 5, 6, 7, 8],  # No outlier
        })
        result = agent.process(df)

        assert _outlier(result, "a") is not None
        assert _outlier(result, "b") is None


# ================================================================== #
#  5. Selective analysis                                               #
# ================================================================== #

class TestSelectiveAnalysis:
    def test_stats_only(self):
        """Can run only descriptive statistics."""
        agent = _make_agent()
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
        result = agent.process(
            df, include_stats=True, include_correlations=False, include_outliers=False
        )
        assert len(result.statistics) == 2
        assert len(result.correlations) == 0
        assert len(result.outliers) == 0

    def test_correlations_only(self):
        """Can run only correlation analysis."""
        agent = _make_agent()
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
        result = agent.process(
            df, include_stats=False, include_correlations=True, include_outliers=False
        )
        assert len(result.statistics) == 0
        assert len(result.correlations) >= 1
        assert len(result.outliers) == 0

    def test_outliers_only(self):
        """Can run only outlier detection."""
        agent = _make_agent()
        df = pd.DataFrame({"x": [1, 1, 1, 1, 1, 1, 1, 100]})
        result = agent.process(
            df, include_stats=False, include_correlations=False, include_outliers=True
        )
        assert len(result.statistics) == 0
        assert len(result.correlations) == 0
        assert len(result.outliers) >= 1


# ================================================================== #
#  6. Event publishing                                                 #
# ================================================================== #

class TestEventPublishing:
    def test_event_published(self):
        """dataset.analyzed event is published."""
        events = []
        def handler(event):
            events.append(event)

        agent = _make_agent()
        agent.subscribe("dataset.analyzed", handler)
        agent.process(pd.DataFrame({"x": [1, 2, 3]}))

        assert len(events) == 1
        assert events[0]["event_type"] == "dataset.analyzed"


# ================================================================== #
#  7. Edge cases                                                       #
# ================================================================== #

class TestEdgeCases:
    def test_empty_dataframe(self):
        """Empty DataFrame produces empty results."""
        agent = _make_agent()
        df = pd.DataFrame()
        result = agent.process(df)
        assert result.record_count == 0
        assert len(result.statistics) == 0
        assert len(result.correlations) == 0
        assert len(result.outliers) == 0

    def test_no_numeric_columns(self):
        """Non-numeric DataFrame produces empty statistics."""
        agent = _make_agent()
        df = pd.DataFrame({"name": ["Alice", "Bob"], "city": ["NYC", "LA"]})
        result = agent.process(df)
        assert len(result.statistics) == 0
        assert len(result.correlations) == 0
        assert len(result.outliers) == 0

    def test_large_integers(self):
        """Statistics work with large integer values."""
        agent = _make_agent()
        df = pd.DataFrame({"x": [10**9, 2 * 10**9, 3 * 10**9]})
        result = agent.process(df)
        s = _stat(result, "x")
        assert s.mean == pytest.approx(2e9)

    def test_negative_values(self):
        """Statistics correctly handle negative numbers."""
        agent = _make_agent()
        df = pd.DataFrame({"x": [-10, -5, 0, 5, 10]})
        result = agent.process(df)
        s = _stat(result, "x")
        assert s.mean == pytest.approx(0.0)
        assert s.min == pytest.approx(-10.0)
        assert s.max == pytest.approx(10.0)
