"""Insight Agent for HelixForge.

Computes correctness-validated statistical analysis on DataFrames:
  - Descriptive statistics (mean, median, std, min, max, quartiles)
  - Pearson/Spearman correlation matrix
  - Outlier detection (IQR method)

No LLM dependency. No visualization dependency. Pure numpy/scipy/pandas.
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from agents.base_agent import BaseAgent
from models.schemas import (
    CorrelationPair,
    FieldStatistics,
    InsightConfig,
    InsightResult,
    OutlierInfo,
)


class InsightAgent(BaseAgent):
    """Agent for statistical analysis of datasets.

    Produces descriptive statistics, correlations, and outlier detection
    on numeric columns. All results are validated with exact-value tests.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        super().__init__(config, correlation_id)
        self._config = InsightConfig(**self.config.get("insight", {}))

    @property
    def event_type(self) -> str:
        return "dataset.analyzed"

    def process(
        self,
        df: pd.DataFrame,
        source_description: str = "",
        **kwargs
    ) -> InsightResult:
        """Analyze a DataFrame and produce statistical insights.

        Args:
            df: DataFrame to analyze.
            source_description: Description of the data source.
            **kwargs: Overrides for include_stats, include_correlations,
                      include_outliers.

        Returns:
            InsightResult with statistics, correlations, and outliers.
        """
        self.logger.info(
            f"Starting analysis: {len(df)} rows, {len(df.columns)} columns"
        )

        analysis_id = str(uuid.uuid4())

        include_stats = kwargs.get("include_stats", self._config.include_stats)
        include_correlations = kwargs.get(
            "include_correlations", self._config.include_correlations
        )
        include_outliers = kwargs.get(
            "include_outliers", self._config.include_outliers
        )

        field_stats = []
        correlations = []
        outliers = []

        if include_stats:
            field_stats = self._compute_statistics(df)

        if include_correlations:
            correlations = self._compute_correlations(df)

        if include_outliers:
            outliers = self._detect_outliers(df)

        result = InsightResult(
            analysis_id=analysis_id,
            source_description=source_description,
            record_count=len(df),
            field_count=len(df.columns),
            statistics=field_stats,
            correlations=correlations,
            outliers=outliers,
        )

        self.publish(self.event_type, result.model_dump())

        self.logger.info(
            f"Analysis complete: {len(field_stats)} stats, "
            f"{len(correlations)} correlations, {len(outliers)} outlier reports"
        )

        return result

    # ------------------------------------------------------------------ #
    #  Descriptive statistics                                              #
    # ------------------------------------------------------------------ #

    def _compute_statistics(self, df: pd.DataFrame) -> List[FieldStatistics]:
        """Compute descriptive statistics for all numeric columns."""
        results = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            series = df[col]
            clean = series.dropna()

            if len(clean) == 0:
                continue

            desc = clean.describe()

            results.append(FieldStatistics(
                field_name=col,
                count=int(desc["count"]),
                mean=float(clean.mean()),
                std=float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
                min=float(clean.min()),
                q1=float(clean.quantile(0.25)),
                median=float(clean.median()),
                q3=float(clean.quantile(0.75)),
                max=float(clean.max()),
                null_count=int(series.isna().sum()),
                unique_count=int(series.nunique()),
            ))

        return results

    # ------------------------------------------------------------------ #
    #  Correlation analysis                                                #
    # ------------------------------------------------------------------ #

    def _compute_correlations(self, df: pd.DataFrame) -> List[CorrelationPair]:
        """Compute pairwise correlations for numeric columns.

        Returns pairs with |coefficient| >= threshold, sorted by
        absolute coefficient descending.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return []

        method = self._config.correlation_method
        threshold = self._config.correlation_threshold
        results = []

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col_a = numeric_cols[i]
                col_b = numeric_cols[j]

                # Drop rows where either column is NaN
                pair_df = df[[col_a, col_b]].dropna()

                if len(pair_df) < 3:
                    continue

                a = pair_df[col_a].values
                b = pair_df[col_b].values

                # Check for zero variance
                if np.std(a) == 0 or np.std(b) == 0:
                    continue

                if method == "spearman":
                    coef, p_val = scipy_stats.spearmanr(a, b)
                else:
                    coef, p_val = scipy_stats.pearsonr(a, b)

                if abs(coef) >= threshold:
                    results.append(CorrelationPair(
                        field_a=col_a,
                        field_b=col_b,
                        coefficient=round(float(coef), 6),
                        p_value=round(float(p_val), 6),
                    ))

        # Sort by absolute coefficient descending
        results.sort(key=lambda x: abs(x.coefficient), reverse=True)
        return results

    # ------------------------------------------------------------------ #
    #  Outlier detection (IQR)                                             #
    # ------------------------------------------------------------------ #

    def _detect_outliers(self, df: pd.DataFrame) -> List[OutlierInfo]:
        """Detect outliers using the IQR method.

        A value is an outlier if it is below Q1 - k*IQR or above
        Q3 + k*IQR, where k is the configured IQR multiplier (default 1.5).
        """
        results = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        k = self._config.outlier_iqr_multiplier

        for col in numeric_cols:
            series = df[col].dropna()

            if len(series) < 4:
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            lower = q1 - k * iqr
            upper = q3 + k * iqr

            outlier_mask = (series < lower) | (series > upper)
            outlier_indices = series[outlier_mask].index.tolist()

            if len(outlier_indices) > 0:
                results.append(OutlierInfo(
                    field_name=col,
                    outlier_count=len(outlier_indices),
                    lower_bound=round(lower, 6),
                    upper_bound=round(upper, 6),
                    outlier_indices=outlier_indices,
                ))

        return results
