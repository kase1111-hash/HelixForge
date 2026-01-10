"""Fusion Agent for HelixForge.

Merges aligned datasets using various join strategies,
applies transformations, and handles missing values.
"""

import os
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from agents.base_agent import BaseAgent
from models.schemas import (
    AlignmentResult,
    FieldAlignment,
    FusionConfig,
    FusionResult,
    ImputationMethod,
    ImputationSummary,
    JoinStrategy,
    TransformationLog,
)


# Built-in transformation templates
BUILTIN_TRANSFORMS: Dict[str, Callable] = {
    "celsius_to_fahrenheit": lambda v: v * 9 / 5 + 32,
    "fahrenheit_to_celsius": lambda v: (v - 32) * 5 / 9,
    "kg_to_lb": lambda v: v * 2.20462,
    "lb_to_kg": lambda v: v / 2.20462,
    "m_to_ft": lambda v: v * 3.28084,
    "ft_to_m": lambda v: v / 3.28084,
    "days_to_months": lambda v: v / 30.44,
    "months_to_days": lambda v: v * 30.44,
}


class FusionAgent(BaseAgent):
    """Agent for fusing aligned datasets.

    Supports multiple join strategies, value transformations,
    and intelligent missing value imputation.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        super().__init__(config, correlation_id)
        self._config = FusionConfig(**self.config.get("fusion", {}))
        self._dataframes: Dict[str, pd.DataFrame] = {}

    @property
    def event_type(self) -> str:
        return "dataset.fused"

    def process(
        self,
        dataframes: Dict[str, pd.DataFrame],
        alignment_result: AlignmentResult,
        **kwargs
    ) -> FusionResult:
        """Fuse multiple datasets based on alignments.

        Args:
            dataframes: Dictionary mapping dataset_id to DataFrame.
            alignment_result: Result from ontology alignment.
            **kwargs: Additional options.

        Returns:
            FusionResult with merged dataset info.
        """
        self.logger.info(f"Starting fusion of {len(dataframes)} datasets")

        fused_id = str(uuid.uuid4())
        transformations = []

        try:
            # Get join strategy
            strategy = kwargs.get("join_strategy") or self._config.default_join_strategy
            if isinstance(strategy, str):
                strategy = JoinStrategy(strategy)

            # Start with the first dataset
            dataset_ids = list(dataframes.keys())
            fused_df = dataframes[dataset_ids[0]].copy()
            source_datasets = [dataset_ids[0]]

            # Merge remaining datasets
            for dataset_id in dataset_ids[1:]:
                df_to_merge = dataframes[dataset_id]

                # Find alignments for this dataset pair
                relevant_alignments = [
                    a for a in alignment_result.alignments
                    if (a.source_dataset in source_datasets and a.target_dataset == dataset_id) or
                       (a.target_dataset in source_datasets and a.source_dataset == dataset_id)
                ]

                if relevant_alignments:
                    fused_df, trans = self._merge_datasets(
                        fused_df, df_to_merge, relevant_alignments, strategy
                    )
                    transformations.extend(trans)
                    source_datasets.append(dataset_id)
                else:
                    self.logger.warning(
                        f"No alignments found for {dataset_id}, skipping"
                    )

            # Apply imputation
            imputation_method = kwargs.get("imputation_method") or self._config.imputation_method
            if isinstance(imputation_method, str):
                imputation_method = ImputationMethod(imputation_method)

            fused_df, imputation_summary = self._impute_missing(
                fused_df, imputation_method
            )

            # Save fused dataset
            storage_path = self._save_fused_dataset(fused_df, fused_id)

            result = FusionResult(
                fused_dataset_id=fused_id,
                source_datasets=source_datasets,
                record_count=len(fused_df),
                field_count=len(fused_df.columns),
                merged_fields=list(fused_df.columns),
                join_strategy=strategy,
                transformations_applied=transformations,
                imputation_summary=imputation_summary,
                storage_path=storage_path
            )

            # Store DataFrame
            self._dataframes[fused_id] = fused_df

            # Publish event
            self.publish(self.event_type, result.model_dump())

            self.logger.info(
                f"Fusion complete: {result.record_count} records, "
                f"{result.field_count} fields"
            )

            return result

        except Exception as e:
            self.handle_error(e, {"fused_id": fused_id})
            raise

    def get_fused_dataframe(self, fused_id: str) -> Optional[pd.DataFrame]:
        """Get a fused DataFrame by ID.

        Args:
            fused_id: Fused dataset ID.

        Returns:
            DataFrame or None.
        """
        return self._dataframes.get(fused_id)

    def _merge_datasets(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        alignments: List[FieldAlignment],
        strategy: JoinStrategy
    ) -> Tuple[pd.DataFrame, List[TransformationLog]]:
        """Merge two datasets using alignments.

        Args:
            df_left: Left DataFrame.
            df_right: Right DataFrame.
            alignments: Field alignments between datasets.
            strategy: Join strategy to use.

        Returns:
            Tuple of (merged DataFrame, transformation logs).
        """
        transformations = []

        # Apply transformations to right dataset
        df_right = df_right.copy()
        for alignment in alignments:
            if alignment.transformation_hint:
                df_right, trans = self._apply_transformation(
                    df_right, alignment
                )
                if trans:
                    transformations.append(trans)

        # Identify join columns
        join_cols_left = []
        join_cols_right = []

        for alignment in alignments:
            if alignment.alignment_type.value in ("exact", "synonym"):
                # Determine which side has which column
                if alignment.source_field in df_left.columns:
                    join_cols_left.append(alignment.source_field)
                    join_cols_right.append(alignment.target_field)
                elif alignment.target_field in df_left.columns:
                    join_cols_left.append(alignment.target_field)
                    join_cols_right.append(alignment.source_field)

        if strategy == JoinStrategy.EXACT_KEY and join_cols_left:
            # Standard merge on key columns
            merged = pd.merge(
                df_left, df_right,
                left_on=join_cols_left,
                right_on=join_cols_right,
                how="outer",
                suffixes=("", "_dup")
            )
            # Remove duplicate columns
            merged = merged.loc[:, ~merged.columns.str.endswith("_dup")]

        elif strategy == JoinStrategy.SEMANTIC_SIMILARITY:
            # Semantic join based on similarity
            merged = self._semantic_join(df_left, df_right, alignments)

        elif strategy == JoinStrategy.PROBABILISTIC:
            # Probabilistic matching
            merged = self._probabilistic_join(df_left, df_right, alignments)

        elif strategy == JoinStrategy.TEMPORAL:
            # Time-based alignment
            merged = self._temporal_join(df_left, df_right, alignments)

        else:
            # Default: concatenate
            merged = pd.concat([df_left, df_right], ignore_index=True)

        return merged, transformations

    def _apply_transformation(
        self,
        df: pd.DataFrame,
        alignment: FieldAlignment
    ) -> Tuple[pd.DataFrame, Optional[TransformationLog]]:
        """Apply transformation to a field.

        Args:
            df: DataFrame.
            alignment: Alignment with transformation hint.

        Returns:
            Tuple of (transformed DataFrame, transformation log).
        """
        if not alignment.transformation_hint:
            return df, None

        hint = alignment.transformation_hint

        # Parse transformation hint
        if hint.startswith("unit_conversion:"):
            transform_name = hint.split(":")[1]

            # Handle inverse transforms
            if transform_name.endswith("_inverse"):
                base_name = transform_name.replace("_inverse", "")
                # Find inverse transform
                inverse_map = {
                    "celsius_to_fahrenheit": "fahrenheit_to_celsius",
                    "kg_to_lb": "lb_to_kg",
                    "m_to_ft": "ft_to_m",
                    "days_to_months": "months_to_days",
                }
                transform_name = inverse_map.get(base_name, base_name)

            transform_func = BUILTIN_TRANSFORMS.get(transform_name)

            if transform_func:
                field = alignment.target_field
                if field in df.columns:
                    sample_before = df[field].dropna().head(1).tolist()
                    df[field] = df[field].apply(
                        lambda x: transform_func(x) if pd.notna(x) else x
                    )
                    sample_after = df[field].dropna().head(1).tolist()

                    return df, TransformationLog(
                        field=field,
                        operation="unit_conversion",
                        template_id=transform_name,
                        source_value_sample=sample_before[0] if sample_before else None,
                        target_value_sample=sample_after[0] if sample_after else None,
                        records_affected=df[field].notna().sum()
                    )

        elif hint.startswith("type_cast:"):
            # Handle type casting
            types = hint.split(":")[1].split("->")
            if len(types) == 2:
                target_type = types[1]
                field = alignment.target_field

                if field in df.columns:
                    try:
                        if target_type == "float":
                            df[field] = pd.to_numeric(df[field], errors="coerce")
                        elif target_type == "integer":
                            df[field] = pd.to_numeric(df[field], errors="coerce").astype("Int64")
                        elif target_type == "string":
                            df[field] = df[field].astype(str)
                        elif target_type == "datetime":
                            df[field] = pd.to_datetime(df[field], errors="coerce")

                        return df, TransformationLog(
                            field=field,
                            operation="type_cast",
                            template_id=hint,
                            records_affected=len(df)
                        )
                    except Exception as e:
                        self.logger.warning(f"Type cast failed for {field}: {e}")

        return df, None

    def _semantic_join(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        alignments: List[FieldAlignment]
    ) -> pd.DataFrame:
        """Join datasets based on semantic similarity.

        Args:
            df_left: Left DataFrame.
            df_right: Right DataFrame.
            alignments: Field alignments.

        Returns:
            Merged DataFrame.
        """
        from utils.similarity import record_similarity

        # Build field mapping
        field_map = {}
        for a in alignments:
            if a.source_field in df_left.columns and a.target_field in df_right.columns:
                field_map[a.source_field] = a.target_field

        if not field_map:
            return pd.concat([df_left, df_right], ignore_index=True)

        # Match records
        matched_indices = []
        threshold = self._config.similarity_join_threshold

        for i, left_row in df_left.iterrows():
            best_match = None
            best_sim = threshold

            for j, right_row in df_right.iterrows():
                # Create comparable dicts
                left_dict = {k: left_row[k] for k in field_map.keys() if k in left_row}
                right_dict = {field_map[k]: right_row[field_map[k]] for k in field_map.keys() if field_map[k] in right_row}
                right_dict_renamed = {k: right_dict[field_map[k]] for k in field_map.keys() if field_map[k] in right_dict}

                sim = record_similarity(left_dict, right_dict_renamed)
                if sim > best_sim:
                    best_sim = sim
                    best_match = j

            matched_indices.append((i, best_match))

        # Build merged dataframe
        merged_rows = []
        used_right = set()

        for left_idx, right_idx in matched_indices:
            left_row = df_left.iloc[left_idx].to_dict()
            if right_idx is not None:
                right_row = df_right.iloc[right_idx].to_dict()
                # Merge rows, preferring left values
                merged = {**right_row, **left_row}
                merged_rows.append(merged)
                used_right.add(right_idx)
            else:
                merged_rows.append(left_row)

        # Add unmatched right rows
        for j in range(len(df_right)):
            if j not in used_right:
                merged_rows.append(df_right.iloc[j].to_dict())

        return pd.DataFrame(merged_rows)

    def _probabilistic_join(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        alignments: List[FieldAlignment]
    ) -> pd.DataFrame:
        """Probabilistic record matching.

        Args:
            df_left: Left DataFrame.
            df_right: Right DataFrame.
            alignments: Field alignments.

        Returns:
            Merged DataFrame.
        """
        # Simplified probabilistic join using weighted field similarities
        return self._semantic_join(df_left, df_right, alignments)

    def _temporal_join(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        alignments: List[FieldAlignment]
    ) -> pd.DataFrame:
        """Join based on temporal alignment.

        Args:
            df_left: Left DataFrame.
            df_right: Right DataFrame.
            alignments: Field alignments.

        Returns:
            Merged DataFrame.
        """
        # Find timestamp columns
        timestamp_cols = []
        for a in alignments:
            if "time" in a.source_field.lower() or "date" in a.source_field.lower():
                timestamp_cols.append((a.source_field, a.target_field))

        if not timestamp_cols:
            return pd.concat([df_left, df_right], ignore_index=True)

        left_ts, right_ts = timestamp_cols[0]

        # Convert to datetime
        df_left[left_ts] = pd.to_datetime(df_left[left_ts], errors="coerce")
        df_right[right_ts] = pd.to_datetime(df_right[right_ts], errors="coerce")

        # Use merge_asof for temporal alignment
        df_left_sorted = df_left.sort_values(left_ts)
        df_right_sorted = df_right.sort_values(right_ts)

        tolerance = pd.Timedelta(seconds=self._config.temporal_tolerance_seconds)

        merged = pd.merge_asof(
            df_left_sorted, df_right_sorted,
            left_on=left_ts, right_on=right_ts,
            tolerance=tolerance,
            direction="nearest"
        )

        return merged

    def _impute_missing(
        self,
        df: pd.DataFrame,
        method: ImputationMethod
    ) -> Tuple[pd.DataFrame, ImputationSummary]:
        """Impute missing values.

        Args:
            df: DataFrame with missing values.
            method: Imputation method.

        Returns:
            Tuple of (imputed DataFrame, imputation summary).
        """
        df = df.copy()
        fields_imputed = {}
        total_filled = 0

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Filter columns with acceptable null ratio
        cols_to_impute = []
        for col in numeric_cols:
            null_ratio = df[col].isna().sum() / len(df)
            if 0 < null_ratio <= self._config.max_null_ratio_for_inclusion:
                cols_to_impute.append(col)

        if not cols_to_impute:
            return df, ImputationSummary(
                total_nulls_filled=0,
                fields_imputed={},
                method_used=method
            )

        # Count nulls before
        null_counts_before = {col: df[col].isna().sum() for col in cols_to_impute}

        if method == ImputationMethod.MEAN:
            for col in cols_to_impute:
                df[col] = df[col].fillna(df[col].mean())

        elif method == ImputationMethod.MEDIAN:
            for col in cols_to_impute:
                df[col] = df[col].fillna(df[col].median())

        elif method == ImputationMethod.MODE:
            for col in cols_to_impute:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])

        elif method == ImputationMethod.KNN:
            imputer = KNNImputer(n_neighbors=self._config.knn_neighbors)
            df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

        # Count imputed values
        for col in cols_to_impute:
            filled = null_counts_before[col] - df[col].isna().sum()
            if filled > 0:
                fields_imputed[col] = filled
                total_filled += filled

        return df, ImputationSummary(
            total_nulls_filled=total_filled,
            fields_imputed=fields_imputed,
            method_used=method
        )

    def _save_fused_dataset(
        self,
        df: pd.DataFrame,
        fused_id: str
    ) -> str:
        """Save fused dataset to disk.

        Args:
            df: Fused DataFrame.
            fused_id: Fused dataset ID.

        Returns:
            Storage path.
        """
        output_dir = self._config.output_path
        os.makedirs(output_dir, exist_ok=True)

        if self._config.output_format == "parquet":
            path = os.path.join(output_dir, f"{fused_id}.parquet")
            df.to_parquet(path, index=False)
        else:
            path = os.path.join(output_dir, f"{fused_id}.csv")
            df.to_csv(path, index=False)

        return path
