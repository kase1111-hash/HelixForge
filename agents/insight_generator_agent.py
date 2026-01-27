"""Insight Generator Agent for HelixForge.

Analyzes fused datasets to generate statistics, correlations,
outlier detection, clustering, visualizations, and narrative summaries.
"""

import json
import os
import uuid
from html import escape as html_escape
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from agents.base_agent import BaseAgent
from models.schemas import (
    ClusteringResult,
    CorrelationMatrix,
    CorrelationPair,
    DatasetStatistics,
    DistributionType,
    FieldStatistics,
    Finding,
    FindingType,
    InsightConfig,
    InsightResult,
    OutlierReport,
    Severity,
    Visualization,
    VisualizationType,
)


class InsightGeneratorAgent(BaseAgent):
    """Agent for generating insights from fused datasets.

    Performs statistical analysis, correlation detection, outlier
    identification, clustering, and generates visualizations and
    narrative summaries.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        super().__init__(config, correlation_id)
        self._config = InsightConfig(**self.config.get("insights", {}))
        self._openai_client = None

    @property
    def event_type(self) -> str:
        return "insight.generated"

    def _get_openai_client(self):
        """Lazily initialize OpenAI client."""
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI()
        return self._openai_client

    def process(
        self,
        fused_dataset_id: str,
        df: pd.DataFrame,
        **kwargs
    ) -> InsightResult:
        """Generate insights for a fused dataset.

        Args:
            fused_dataset_id: ID of the fused dataset.
            df: Fused DataFrame.
            **kwargs: Additional options.

        Returns:
            InsightResult with analysis results.
        """
        self.logger.info(f"Generating insights for dataset: {fused_dataset_id}")

        insight_id = str(uuid.uuid4())
        key_findings = []
        visualizations = []

        try:
            analysis_types = kwargs.get("analysis_types", ["correlations", "outliers", "clusters"])

            # Compute statistics
            statistics = self._compute_statistics(df)

            # Correlation analysis
            correlations = None
            if "correlations" in analysis_types:
                correlations = self._compute_correlations(df)
                # Add significant correlations to findings
                for pair in correlations.significant_pairs[:5]:
                    finding = Finding(
                        finding_id=str(uuid.uuid4()),
                        type=FindingType.CORRELATION,
                        severity=self._assess_correlation_severity(pair.coefficient),
                        description=f"Strong correlation between {pair.field_a} and {pair.field_b}: r={pair.coefficient:.3f} (p={pair.p_value:.4f})",
                        supporting_data={"coefficient": pair.coefficient, "p_value": pair.p_value}
                    )
                    key_findings.append(finding)

            # Outlier detection
            outliers = None
            if "outliers" in analysis_types:
                outliers = self._detect_outliers(df)
                if outliers.total_outliers > 0:
                    finding = Finding(
                        finding_id=str(uuid.uuid4()),
                        type=FindingType.OUTLIER,
                        severity=Severity.MEDIUM if outliers.total_outliers > 10 else Severity.LOW,
                        description=f"Detected {outliers.total_outliers} outliers across {len(outliers.outliers_by_field)} fields",
                        supporting_data=outliers.outliers_by_field
                    )
                    key_findings.append(finding)

            # Clustering
            clusters = None
            if "clusters" in analysis_types:
                clusters = self._perform_clustering(df)
                if clusters and clusters.n_clusters > 1:
                    finding = Finding(
                        finding_id=str(uuid.uuid4()),
                        type=FindingType.CLUSTER,
                        severity=Severity.MEDIUM,
                        description=f"Identified {clusters.n_clusters} natural clusters in the data (silhouette score: {clusters.silhouette_score:.3f})",
                        supporting_data={"cluster_sizes": clusters.cluster_sizes}
                    )
                    key_findings.append(finding)

            # Generate visualizations
            if kwargs.get("generate_visualizations", True):
                visualizations = self._generate_visualizations(
                    df, fused_dataset_id, correlations, outliers, clusters
                )

            # Generate narrative
            narrative = None
            if kwargs.get("generate_narrative", True):
                narrative = self._generate_narrative(
                    df, statistics, correlations, outliers, clusters, key_findings
                )

            # Export reports
            export_paths = {}
            export_formats = kwargs.get("export_formats", self._config.export_formats)
            for fmt in export_formats:
                path = self._export_report(
                    fused_dataset_id, insight_id, fmt,
                    statistics, correlations, outliers, clusters,
                    narrative, key_findings, visualizations
                )
                export_paths[fmt] = path

            result = InsightResult(
                insight_id=insight_id,
                fused_dataset_id=fused_dataset_id,
                statistics=statistics,
                correlations=correlations,
                outliers=outliers,
                clusters=clusters,
                narrative_summary=narrative,
                key_findings=key_findings,
                visualizations=visualizations,
                export_paths=export_paths
            )

            # Publish event
            self.publish(self.event_type, result.model_dump())

            self.logger.info(f"Insight generation complete: {len(key_findings)} findings")

            return result

        except Exception as e:
            self.handle_error(e, {"fused_dataset_id": fused_dataset_id})
            raise

    def _compute_statistics(self, df: pd.DataFrame) -> DatasetStatistics:
        """Compute descriptive statistics.

        Args:
            df: DataFrame.

        Returns:
            DatasetStatistics.
        """
        field_stats = {}

        for col in df.columns:
            series = df[col]
            stats_dict = FieldStatistics(
                null_count=int(series.isna().sum()),
                unique_count=int(series.nunique())
            )

            if pd.api.types.is_numeric_dtype(series):
                numeric = series.dropna()
                if len(numeric) > 0:
                    stats_dict.mean = float(numeric.mean())
                    stats_dict.std = float(numeric.std())
                    stats_dict.min = float(numeric.min())
                    stats_dict.max = float(numeric.max())
                    stats_dict.median = float(numeric.median())
                    stats_dict.q1 = float(numeric.quantile(0.25))
                    stats_dict.q3 = float(numeric.quantile(0.75))
                    stats_dict.distribution_type = self._detect_distribution(numeric)

            field_stats[col] = stats_dict

        return DatasetStatistics(
            record_count=len(df),
            field_count=len(df.columns),
            field_stats=field_stats
        )

    def _detect_distribution(self, series: pd.Series) -> DistributionType:
        """Detect distribution type of numeric series.

        Args:
            series: Numeric series.

        Returns:
            DistributionType.
        """
        if len(series) < 20:
            return DistributionType.UNIFORM

        # Test for normality
        try:
            _, p_value = stats.normaltest(series)
            if p_value > 0.05:
                return DistributionType.NORMAL
        except Exception:
            pass

        # Check skewness
        skew = series.skew()
        if abs(skew) > 1:
            return DistributionType.SKEWED

        # Check for bimodal (simplified)
        try:
            hist, _ = np.histogram(series, bins=20)
            peaks = np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))[0]
            if len(peaks) >= 2:
                return DistributionType.BIMODAL
        except Exception:
            pass

        return DistributionType.UNIFORM

    def _compute_correlations(self, df: pd.DataFrame) -> CorrelationMatrix:
        """Compute correlation matrix.

        Args:
            df: DataFrame.

        Returns:
            CorrelationMatrix.
        """
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            return CorrelationMatrix(correlations=[], significant_pairs=[])

        correlations = []
        significant = []

        cols = numeric_df.columns.tolist()

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col_a = cols[i]
                col_b = cols[j]

                # Drop NaN rows for this pair
                pair_df = numeric_df[[col_a, col_b]].dropna()
                if len(pair_df) < 3:
                    continue

                try:
                    if self._config.correlation_method == "pearson":
                        coef, p_val = stats.pearsonr(pair_df[col_a], pair_df[col_b])
                    else:
                        coef, p_val = stats.spearmanr(pair_df[col_a], pair_df[col_b])

                    is_significant = p_val < self._config.correlation_significance_threshold

                    pair = CorrelationPair(
                        field_a=col_a,
                        field_b=col_b,
                        coefficient=float(coef),
                        p_value=float(p_val),
                        significant=is_significant
                    )
                    correlations.append(pair)

                    if is_significant and abs(coef) > 0.5:
                        significant.append(pair)

                except Exception:
                    continue

        # Sort significant by absolute coefficient
        significant.sort(key=lambda x: abs(x.coefficient), reverse=True)

        return CorrelationMatrix(
            method=self._config.correlation_method,
            correlations=correlations,
            significant_pairs=significant
        )

    def _detect_outliers(self, df: pd.DataFrame) -> OutlierReport:
        """Detect outliers using IQR method.

        Args:
            df: DataFrame.

        Returns:
            OutlierReport.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        outliers_by_field = {}
        all_outlier_indices = set()

        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) < 4:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            multiplier = self._config.outlier_iqr_multiplier

            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr

            outliers = series[(series < lower_bound) | (series > upper_bound)]

            if len(outliers) > 0:
                outliers_by_field[col] = len(outliers)
                all_outlier_indices.update(outliers.index.tolist())

        return OutlierReport(
            method=f"iqr_{self._config.outlier_iqr_multiplier}x",
            total_outliers=len(all_outlier_indices),
            outliers_by_field=outliers_by_field,
            outlier_records=list(all_outlier_indices)[:100]  # Limit to 100
        )

    def _perform_clustering(self, df: pd.DataFrame) -> Optional[ClusteringResult]:
        """Perform clustering analysis.

        Args:
            df: DataFrame.

        Returns:
            ClusteringResult or None.
        """
        numeric_df = df.select_dtypes(include=[np.number]).dropna()

        if len(numeric_df) < 10 or len(numeric_df.columns) < 2:
            return None

        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        # Find optimal k
        k_range = range(
            self._config.clustering_k_range[0],
            min(self._config.clustering_k_range[1], len(numeric_df) // 2)
        )

        best_k = 2
        best_score = -1

        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(scaled_data)
                score = silhouette_score(scaled_data, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue

        # Final clustering with best k
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_data)

        cluster_sizes = [int((labels == i).sum()) for i in range(best_k)]

        return ClusteringResult(
            algorithm="kmeans",
            n_clusters=best_k,
            cluster_sizes=cluster_sizes,
            silhouette_score=float(best_score),
            cluster_labels=labels.tolist()
        )

    def _assess_correlation_severity(self, coefficient: float) -> Severity:
        """Assess severity of correlation.

        Args:
            coefficient: Correlation coefficient.

        Returns:
            Severity level.
        """
        abs_coef = abs(coefficient)
        if abs_coef >= 0.8:
            return Severity.HIGH
        elif abs_coef >= 0.6:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _generate_visualizations(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        correlations: Optional[CorrelationMatrix],
        outliers: Optional[OutlierReport],
        clusters: Optional[ClusteringResult]
    ) -> List[Visualization]:
        """Generate visualizations.

        Args:
            df: DataFrame.
            dataset_id: Dataset ID.
            correlations: Correlation results.
            outliers: Outlier results.
            clusters: Clustering results.

        Returns:
            List of Visualization objects.
        """
        visualizations = []
        output_dir = self._config.output_path
        os.makedirs(output_dir, exist_ok=True)

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns

            numeric_df = df.select_dtypes(include=[np.number])

            # Correlation heatmap
            if correlations and len(numeric_df.columns) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = numeric_df.corr()
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
                ax.set_title("Correlation Matrix")

                path = os.path.join(output_dir, f"{dataset_id}_correlation.png")
                fig.savefig(path, dpi=self._config.visualization_dpi, bbox_inches='tight')
                plt.close(fig)

                visualizations.append(Visualization(
                    viz_id=str(uuid.uuid4()),
                    type=VisualizationType.CORRELATION_MATRIX,
                    title="Correlation Matrix",
                    file_path=path,
                    format="png"
                ))

            # Distribution histograms for top fields
            for col in list(numeric_df.columns)[:4]:
                fig, ax = plt.subplots(figsize=(8, 6))
                numeric_df[col].hist(bins=30, ax=ax)
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")

                path = os.path.join(output_dir, f"{dataset_id}_{col}_hist.png")
                fig.savefig(path, dpi=self._config.visualization_dpi, bbox_inches='tight')
                plt.close(fig)

                visualizations.append(Visualization(
                    viz_id=str(uuid.uuid4()),
                    type=VisualizationType.HISTOGRAM,
                    title=f"Distribution of {col}",
                    file_path=path,
                    format="png"
                ))

        except Exception as e:
            self.logger.warning(f"Visualization generation failed: {e}")

        return visualizations

    def _generate_narrative(
        self,
        df: pd.DataFrame,
        statistics: DatasetStatistics,
        correlations: Optional[CorrelationMatrix],
        outliers: Optional[OutlierReport],
        clusters: Optional[ClusteringResult],
        findings: List[Finding]
    ) -> str:
        """Generate narrative summary using LLM.

        Args:
            df: DataFrame.
            statistics: Dataset statistics.
            correlations: Correlation results.
            outliers: Outlier results.
            clusters: Clustering results.
            findings: Key findings.

        Returns:
            Narrative string.
        """
        # Build summary for prompt
        stats_summary = f"Records: {statistics.record_count}, Fields: {statistics.field_count}"

        corr_summary = "No significant correlations found."
        if correlations and correlations.significant_pairs:
            pairs = correlations.significant_pairs[:5]
            corr_summary = "\n".join([
                f"- {p.field_a} and {p.field_b}: r={p.coefficient:.3f}"
                for p in pairs
            ])

        outlier_summary = "No outliers detected."
        if outliers and outliers.total_outliers > 0:
            outlier_summary = f"{outliers.total_outliers} outliers in {len(outliers.outliers_by_field)} fields"

        cluster_summary = "No clustering performed."
        if clusters:
            cluster_summary = f"{clusters.n_clusters} clusters (silhouette: {clusters.silhouette_score:.3f})"

        prompt = f"""You are a data analyst summarizing findings from a fused dataset.

Dataset: {statistics.record_count} records, {statistics.field_count} fields

Key Statistics:
{stats_summary}

Significant Correlations:
{corr_summary}

Detected Outliers:
{outlier_summary}

Clusters Found:
{cluster_summary}

Write a clear, professional narrative summary (3-5 paragraphs) highlighting:
1. The most important correlations and their implications
2. Notable outliers or anomalies
3. Any patterns revealed by clustering
4. Recommended next steps for analysis

Use specific numbers and field names. Avoid jargon."""

        try:
            client = self._get_openai_client()
            response = client.chat.completions.create(
                model=self._config.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.warning(f"Narrative generation failed: {e}")
            # Fallback narrative
            return f"""Analysis of {statistics.record_count} records across {statistics.field_count} fields.

{corr_summary if correlations else ''}

{outlier_summary}

{cluster_summary}"""

    def _export_report(
        self,
        dataset_id: str,
        insight_id: str,
        format: str,
        statistics: DatasetStatistics,
        correlations: Optional[CorrelationMatrix],
        outliers: Optional[OutlierReport],
        clusters: Optional[ClusteringResult],
        narrative: Optional[str],
        findings: List[Finding],
        visualizations: List[Visualization]
    ) -> str:
        """Export report in specified format.

        Args:
            Various report data.

        Returns:
            Path to exported report.
        """
        output_dir = self._config.output_path
        os.makedirs(output_dir, exist_ok=True)

        if format == "json":
            path = os.path.join(output_dir, f"{dataset_id}_report.json")
            report_data = {
                "insight_id": insight_id,
                "dataset_id": dataset_id,
                "statistics": statistics.model_dump(),
                "correlations": correlations.model_dump() if correlations else None,
                "outliers": outliers.model_dump() if outliers else None,
                "clusters": clusters.model_dump() if clusters else None,
                "narrative": narrative,
                "findings": [f.model_dump() for f in findings]
            }
            with open(path, "w") as f:
                json.dump(report_data, f, indent=2, default=str)

        elif format == "html":
            path = os.path.join(output_dir, f"{dataset_id}_report.html")
            html = self._generate_html_report(
                dataset_id, statistics, correlations, outliers,
                clusters, narrative, findings, visualizations
            )
            with open(path, "w") as f:
                f.write(html)

        else:
            path = os.path.join(output_dir, f"{dataset_id}_report.txt")
            with open(path, "w") as f:
                f.write(f"Insight Report: {dataset_id}\n")
                f.write("=" * 50 + "\n\n")
                if narrative:
                    f.write(narrative)

        return path

    def _generate_html_report(
        self,
        dataset_id: str,
        statistics: DatasetStatistics,
        correlations: Optional[CorrelationMatrix],
        outliers: Optional[OutlierReport],
        clusters: Optional[ClusteringResult],
        narrative: Optional[str],
        findings: List[Finding],
        visualizations: List[Visualization]
    ) -> str:
        """Generate HTML report.

        Args:
            Various report data.

        Returns:
            HTML string.
        """
        findings_html = "\n".join([
            f"<li><strong>{html_escape(f.type.value)}</strong>: {html_escape(f.description)}</li>"
            for f in findings
        ])

        viz_html = "\n".join([
            f'<img src="{html_escape(v.file_path)}" alt="{html_escape(v.title)}" style="max-width: 100%;">'
            for v in visualizations
        ])

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Insight Report - {html_escape(dataset_id)}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        .section {{ margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 8px; }}
        .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
        .stat-box {{ background: white; padding: 15px; border-radius: 4px; }}
        img {{ margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Insight Report: {html_escape(dataset_id)}</h1>

    <div class="section">
        <h2>Overview</h2>
        <div class="stats">
            <div class="stat-box">
                <strong>Records:</strong> {statistics.record_count}
            </div>
            <div class="stat-box">
                <strong>Fields:</strong> {statistics.field_count}
            </div>
            <div class="stat-box">
                <strong>Outliers:</strong> {outliers.total_outliers if outliers else 0}
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Key Findings</h2>
        <ul>{findings_html}</ul>
    </div>

    <div class="section">
        <h2>Narrative Summary</h2>
        <p>{html_escape(narrative) if narrative else 'No narrative generated.'}</p>
    </div>

    <div class="section">
        <h2>Visualizations</h2>
        {viz_html}
    </div>
</body>
</html>"""
