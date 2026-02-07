"""Insight Agent for HelixForge.

Computes correctness-validated statistical analysis on DataFrames:
  - Descriptive statistics (mean, median, std, min, max, quartiles)
  - Pearson/Spearman correlation matrix
  - Outlier detection (IQR method)
  - K-means clustering with silhouette scoring
  - LLM narrative generation (optional)

Clustering uses scikit-learn. Narrative uses the LLMProvider protocol.
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from agents.base_agent import BaseAgent
from models.schemas import (
    ClusterInfo,
    CorrelationPair,
    FieldStatistics,
    InsightConfig,
    InsightResult,
    OutlierInfo,
)


class InsightAgent(BaseAgent):
    """Agent for statistical analysis of datasets.

    Produces descriptive statistics, correlations, outlier detection,
    clustering, and optional LLM narratives.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        provider=None,
    ):
        super().__init__(config, correlation_id)
        self._config = InsightConfig(**self.config.get("insight", {}))
        self._provider = provider

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
                      include_outliers, include_clustering, include_narrative,
                      n_clusters.

        Returns:
            InsightResult with statistics, correlations, outliers,
            clustering, and optional narrative.
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
        include_clustering = kwargs.get(
            "include_clustering", self._config.include_clustering
        )
        include_narrative = kwargs.get(
            "include_narrative", self._config.include_narrative
        )
        n_clusters = kwargs.get("n_clusters", self._config.n_clusters)

        field_stats = []
        correlations = []
        outliers = []
        clustering = None
        narrative = None

        if include_stats:
            field_stats = self._compute_statistics(df)

        if include_correlations:
            correlations = self._compute_correlations(df)

        if include_outliers:
            outliers = self._detect_outliers(df)

        if include_clustering:
            clustering = self._perform_clustering(df, n_clusters)

        result = InsightResult(
            analysis_id=analysis_id,
            source_description=source_description,
            record_count=len(df),
            field_count=len(df.columns),
            statistics=field_stats,
            correlations=correlations,
            outliers=outliers,
            clustering=clustering,
        )

        if include_narrative and self._provider is not None:
            narrative = self._generate_narrative(result)
            result.narrative = narrative

        self.publish(self.event_type, result.model_dump())

        self.logger.info(
            f"Analysis complete: {len(field_stats)} stats, "
            f"{len(correlations)} correlations, {len(outliers)} outlier reports"
            + (f", {clustering.n_clusters} clusters" if clustering else "")
            + (", narrative generated" if narrative else "")
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

    # ------------------------------------------------------------------ #
    #  K-means clustering                                                  #
    # ------------------------------------------------------------------ #

    def _perform_clustering(
        self, df: pd.DataFrame, n_clusters: int
    ) -> Optional[ClusterInfo]:
        """Cluster rows using K-means on numeric columns.

        Requires at least n_clusters+1 rows and 1 numeric column.
        Columns are standardized (z-score) before clustering.
        Returns None if data is insufficient.
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            self.logger.warning("No numeric columns for clustering")
            return None

        # Use only rows without NaN in numeric columns
        clean = df[numeric_cols].dropna()
        if len(clean) < n_clusters + 1:
            self.logger.warning(
                f"Too few rows ({len(clean)}) for {n_clusters} clusters"
            )
            return None

        # Standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(clean.values)

        # Fit K-means
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(X)

        # Silhouette score (only if >1 unique labels and enough samples)
        silhouette = None
        if len(set(labels)) > 1:
            from sklearn.metrics import silhouette_score
            silhouette = float(silhouette_score(X, labels))

        # Convert centroids back to original scale
        centroids_original = scaler.inverse_transform(km.cluster_centers_)

        return ClusterInfo(
            n_clusters=n_clusters,
            labels=labels.tolist(),
            centroids=[row.tolist() for row in centroids_original],
            silhouette_score=round(silhouette, 6) if silhouette is not None else None,
            inertia=round(float(km.inertia_), 6),
            features_used=numeric_cols,
        )

    # ------------------------------------------------------------------ #
    #  LLM narrative generation                                            #
    # ------------------------------------------------------------------ #

    def _generate_narrative(self, result: InsightResult) -> Optional[str]:
        """Generate a natural-language summary of the analysis results.

        Uses the LLMProvider.complete() protocol. Returns None if no
        provider is configured or the call fails.
        """
        if self._provider is None:
            return None

        # Build a concise prompt from the result
        lines = [
            f"Dataset: {result.source_description or 'unknown'}",
            f"Records: {result.record_count}, Fields: {result.field_count}",
        ]

        if result.statistics:
            lines.append("\nKey statistics:")
            for s in result.statistics[:10]:
                lines.append(
                    f"  {s.field_name}: mean={s.mean:.2f}, std={s.std:.2f}, "
                    f"min={s.min:.2f}, max={s.max:.2f}, nulls={s.null_count}"
                )

        if result.correlations:
            lines.append("\nStrong correlations:")
            for c in result.correlations[:5]:
                lines.append(
                    f"  {c.field_a} <-> {c.field_b}: r={c.coefficient:+.4f}"
                )

        if result.outliers:
            lines.append("\nOutliers detected:")
            for o in result.outliers[:5]:
                lines.append(
                    f"  {o.field_name}: {o.outlier_count} outlier(s)"
                )

        if result.clustering:
            cl = result.clustering
            lines.append(
                f"\nClustering: {cl.n_clusters} clusters, "
                f"silhouette={cl.silhouette_score}"
            )

        prompt_text = "\n".join(lines)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a data analyst. Given the statistical summary below, "
                    "write a concise 2-4 sentence narrative highlighting the most "
                    "important findings. Focus on actionable insights."
                ),
            },
            {"role": "user", "content": prompt_text},
        ]

        try:
            return self._provider.complete(messages, model="gpt-4o", temperature=0.3)
        except Exception as e:
            self.logger.warning(f"Narrative generation failed: {e}")
            return None
