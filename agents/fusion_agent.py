"""Fusion Agent for HelixForge.

Merges aligned datasets using two primary join strategies:
  - exact_key: join on a shared key column (the simple, reliable case)
  - semantic_similarity: join on record-level similarity (the
    differentiating case that makes HelixForge unique)

Probabilistic and temporal strategies are gated behind the
experimental_strategies config flag.
"""

import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from models.schemas import (
    AlignmentResult,
    AlignmentType,
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

    Supports exact_key and semantic_similarity join strategies,
    value transformations, and missing value imputation.
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
            **kwargs: Additional options:
                join_strategy: Override default strategy.
                key_column: Column name to use as join key (for exact_key).
                imputation_method: Override default imputation.

        Returns:
            FusionResult with merged dataset info.
        """
        self.logger.info(f"Starting fusion of {len(dataframes)} datasets")

        fused_id = str(uuid.uuid4())
        transformations = []
        key_column = kwargs.get("key_column")
        effective_strategy = None  # Track what strategy was actually used

        try:
            # Resolve strategy
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
                    # Resolve auto strategy per merge
                    effective_strategy = strategy
                    if strategy == JoinStrategy.AUTO:
                        effective_strategy = self._detect_strategy(
                            relevant_alignments, fused_df, df_to_merge, key_column
                        )

                    fused_df, trans = self._merge_datasets(
                        fused_df, df_to_merge, relevant_alignments,
                        effective_strategy, key_column
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

            # Report the strategy that was actually used
            report_strategy = effective_strategy if effective_strategy else strategy
            if report_strategy == JoinStrategy.AUTO:
                report_strategy = JoinStrategy.EXACT_KEY

            result = FusionResult(
                fused_dataset_id=fused_id,
                source_datasets=source_datasets,
                record_count=len(fused_df),
                field_count=len(fused_df.columns),
                merged_fields=list(fused_df.columns),
                join_strategy=report_strategy,
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
        """Get a fused DataFrame by ID."""
        return self._dataframes.get(fused_id)

    # ------------------------------------------------------------------ #
    #  Strategy detection                                                  #
    # ------------------------------------------------------------------ #

    def _detect_strategy(
        self,
        alignments: List[FieldAlignment],
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        key_column: Optional[str] = None,
    ) -> JoinStrategy:
        """Auto-detect the best join strategy.

        Uses exact_key when a shared key column is detected (either
        user-specified or from an exact/synonym alignment), otherwise
        falls back to semantic_similarity.
        """
        if key_column:
            return JoinStrategy.EXACT_KEY

        # Check if any alignment is exact or synonym with a column in both DFs
        for a in alignments:
            if a.alignment_type in (AlignmentType.EXACT, AlignmentType.SYNONYM):
                if a.source_field in df_left.columns and a.target_field in df_right.columns:
                    return JoinStrategy.EXACT_KEY
                if a.target_field in df_left.columns and a.source_field in df_right.columns:
                    return JoinStrategy.EXACT_KEY

        return JoinStrategy.SEMANTIC_SIMILARITY

    # ------------------------------------------------------------------ #
    #  Merge logic                                                         #
    # ------------------------------------------------------------------ #

    def _merge_datasets(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        alignments: List[FieldAlignment],
        strategy: JoinStrategy,
        key_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, List[TransformationLog]]:
        """Merge two datasets using alignments.

        Args:
            df_left: Left DataFrame.
            df_right: Right DataFrame.
            alignments: Field alignments between datasets.
            strategy: Join strategy to use.
            key_column: Optional user-specified key column name.

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

        # Build column mapping: right_col -> left_col
        col_map = self._build_column_map(df_left, df_right, alignments)

        if strategy == JoinStrategy.EXACT_KEY:
            merged = self._exact_key_merge(
                df_left, df_right, alignments, col_map, key_column
            )

        elif strategy == JoinStrategy.SEMANTIC_SIMILARITY:
            merged = self._semantic_join(df_left, df_right, alignments)

        elif strategy == JoinStrategy.PROBABILISTIC:
            if not self._config.experimental_strategies:
                self.logger.warning(
                    "Probabilistic strategy is experimental and disabled. "
                    "Falling back to semantic_similarity."
                )
                merged = self._semantic_join(df_left, df_right, alignments)
            else:
                merged = self._probabilistic_join(df_left, df_right, alignments)

        elif strategy == JoinStrategy.TEMPORAL:
            if not self._config.experimental_strategies:
                self.logger.warning(
                    "Temporal strategy is experimental and disabled. "
                    "Falling back to semantic_similarity."
                )
                merged = self._semantic_join(df_left, df_right, alignments)
            else:
                merged = self._temporal_join(df_left, df_right, alignments)

        else:
            merged = pd.concat([df_left, df_right], ignore_index=True)

        return merged, transformations

    def _build_column_map(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        alignments: List[FieldAlignment],
    ) -> Dict[str, str]:
        """Build mapping from right column names to left column names.

        Returns:
            Dict where key=right_col, value=left_col.
        """
        col_map: Dict[str, str] = {}
        for a in alignments:
            if a.source_field in df_left.columns and a.target_field in df_right.columns:
                col_map[a.target_field] = a.source_field
            elif a.target_field in df_left.columns and a.source_field in df_right.columns:
                col_map[a.source_field] = a.target_field
        return col_map

    def _exact_key_merge(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        alignments: List[FieldAlignment],
        col_map: Dict[str, str],
        key_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Merge on a shared key column, unifying aligned columns.

        Steps:
          1. Identify the key column (user-specified or auto-detected)
          2. Rename right columns to match left column names (via alignment)
          3. Merge on the shared key
          4. Coalesce overlapping columns (prefer left non-null, then right)
        """
        # 1. Determine key column
        key_left, key_right = self._find_key_columns(
            df_left, df_right, alignments, col_map, key_column
        )

        if key_left is None:
            # No key found, fall back to concat
            self.logger.warning("No key column found for exact_key merge, falling back to concat")
            return pd.concat([df_left, df_right], ignore_index=True)

        # 2. Rename right columns to match left names
        rename_map = {}
        for right_col, left_col in col_map.items():
            if right_col != key_right or left_col != key_left:
                # For the key column, we rename it to match left key
                # For value columns, we rename to match left names
                rename_map[right_col] = left_col
            else:
                # Key column: rename to match left key name
                rename_map[right_col] = key_left

        df_right_renamed = df_right.rename(columns=rename_map)

        # 3. Merge on the key
        merged = pd.merge(
            df_left, df_right_renamed,
            on=key_left, how="outer",
            suffixes=("", "_right")
        )

        # 4. Coalesce overlapping columns (prefer left, fill with right)
        for col in list(merged.columns):
            if col.endswith("_right"):
                base_col = col[:-6]  # Remove "_right"
                if base_col in merged.columns:
                    merged[base_col] = merged[base_col].fillna(merged[col])
                    merged = merged.drop(columns=[col])
                else:
                    # No base column, just rename
                    merged = merged.rename(columns={col: base_col})

        return merged

    def _find_key_columns(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        alignments: List[FieldAlignment],
        col_map: Dict[str, str],
        key_column: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Find the key column pair for exact_key merge.

        Returns:
            (left_key_name, right_key_name) or (None, None).
        """
        if key_column:
            # User specified the key - find corresponding right column
            if key_column in df_left.columns:
                # Find the right-side equivalent
                for right_col, left_col in col_map.items():
                    if left_col == key_column:
                        return key_column, right_col
                # Same name in both datasets
                if key_column in df_right.columns:
                    return key_column, key_column
            return None, None

        # Auto-detect: prefer exact/synonym alignments
        for a in alignments:
            if a.alignment_type in (AlignmentType.EXACT, AlignmentType.SYNONYM):
                if a.source_field in df_left.columns and a.target_field in df_right.columns:
                    return a.source_field, a.target_field
                if a.target_field in df_left.columns and a.source_field in df_right.columns:
                    return a.target_field, a.source_field

        # Fall back to first alignment with columns in both DFs
        for a in alignments:
            if a.source_field in df_left.columns and a.target_field in df_right.columns:
                return a.source_field, a.target_field
            if a.target_field in df_left.columns and a.source_field in df_right.columns:
                return a.target_field, a.source_field

        return None, None

    # ------------------------------------------------------------------ #
    #  Transformations                                                     #
    # ------------------------------------------------------------------ #

    def _apply_transformation(
        self,
        df: pd.DataFrame,
        alignment: FieldAlignment
    ) -> Tuple[pd.DataFrame, Optional[TransformationLog]]:
        """Apply transformation to a field."""
        if not alignment.transformation_hint:
            return df, None

        hint = alignment.transformation_hint

        if hint.startswith("unit_conversion:"):
            transform_name = hint.split(":")[1]

            # Handle inverse transforms
            if transform_name.endswith("_inverse"):
                base_name = transform_name.replace("_inverse", "")
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

    # ------------------------------------------------------------------ #
    #  Semantic join                                                       #
    # ------------------------------------------------------------------ #

    def _semantic_join(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        alignments: List[FieldAlignment]
    ) -> pd.DataFrame:
        """Join datasets based on record-level semantic similarity."""
        from utils.similarity import record_similarity

        # Build field mapping: left_col -> right_col
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
                left_dict = {k: left_row[k] for k in field_map.keys() if k in left_row}
                right_dict = {field_map[k]: right_row[field_map[k]] for k in field_map.keys() if field_map[k] in right_row}
                right_dict_renamed = {k: right_dict[field_map[k]] for k in field_map.keys() if field_map[k] in right_dict}

                sim = record_similarity(left_dict, right_dict_renamed)
                if sim > best_sim:
                    best_sim = sim
                    best_match = j

            matched_indices.append((i, best_match))

        # Build merged DataFrame
        merged_rows = []
        used_right = set()

        for left_idx, right_idx in matched_indices:
            left_row = df_left.loc[left_idx].to_dict()
            if right_idx is not None:
                right_row = df_right.loc[right_idx].to_dict()
                # Rename right columns to left names where aligned
                renamed_right = {}
                reverse_map = {v: k for k, v in field_map.items()}
                for col, val in right_row.items():
                    if col in reverse_map:
                        renamed_right[reverse_map[col]] = val
                    else:
                        renamed_right[col] = val
                # Merge: prefer left non-null, then right
                merged = {**renamed_right, **{k: v for k, v in left_row.items() if pd.notna(v)}}
                # Fill remaining nulls from right
                for k, v in renamed_right.items():
                    if k in merged and pd.isna(merged[k]) and pd.notna(v):
                        merged[k] = v
                merged_rows.append(merged)
                used_right.add(right_idx)
            else:
                merged_rows.append(left_row)

        # Add unmatched right rows (renamed)
        reverse_map = {v: k for k, v in field_map.items()}
        for j in range(len(df_right)):
            if j not in used_right:
                right_row = df_right.iloc[j].to_dict()
                renamed = {}
                for col, val in right_row.items():
                    if col in reverse_map:
                        renamed[reverse_map[col]] = val
                    else:
                        renamed[col] = val
                merged_rows.append(renamed)

        return pd.DataFrame(merged_rows)

    # ------------------------------------------------------------------ #
    #  Probabilistic join (experimental)                                   #
    # ------------------------------------------------------------------ #

    def _probabilistic_join(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        alignments: List[FieldAlignment],
    ) -> pd.DataFrame:
        """Join records using weighted multi-column fuzzy matching.

        For each left row, finds the best-matching right row by computing
        a weighted similarity across all aligned column pairs. Numeric
        columns use normalized distance; string columns use fuzzy ratio.
        """
        from fuzzywuzzy import fuzz

        # Build field mapping
        field_map = {}
        for a in alignments:
            if a.source_field in df_left.columns and a.target_field in df_right.columns:
                field_map[a.source_field] = a.target_field

        if not field_map:
            return pd.concat([df_left, df_right], ignore_index=True)

        threshold = self._config.probabilistic_match_threshold
        weight = 1.0 / len(field_map) if field_map else 1.0

        matched_indices = []
        used_right = set()

        for i in range(len(df_left)):
            left_row = df_left.iloc[i]
            best_match = None
            best_score = threshold

            for j in range(len(df_right)):
                if j in used_right:
                    continue
                right_row = df_right.iloc[j]

                score = 0.0
                for left_col, right_col in field_map.items():
                    lv = left_row[left_col]
                    rv = right_row[right_col]

                    if pd.isna(lv) or pd.isna(rv):
                        continue

                    # Numeric comparison: 1 - normalized distance
                    try:
                        lf = float(lv)
                        rf = float(rv)
                        max_val = max(abs(lf), abs(rf), 1e-9)
                        col_sim = 1.0 - abs(lf - rf) / max_val
                    except (ValueError, TypeError):
                        # String comparison: fuzzy ratio
                        col_sim = fuzz.ratio(str(lv), str(rv)) / 100.0

                    score += col_sim * weight

                if score > best_score:
                    best_score = score
                    best_match = j

            matched_indices.append((i, best_match))
            if best_match is not None:
                used_right.add(best_match)

        # Build merged DataFrame
        merged_rows = []
        reverse_map = {v: k for k, v in field_map.items()}

        for left_idx, right_idx in matched_indices:
            left_row = df_left.iloc[left_idx].to_dict()
            if right_idx is not None:
                right_row = df_right.iloc[right_idx].to_dict()
                renamed_right = {}
                for col, val in right_row.items():
                    renamed_right[reverse_map.get(col, col)] = val
                merged = {**renamed_right, **{k: v for k, v in left_row.items() if pd.notna(v)}}
                for k, v in renamed_right.items():
                    if k in merged and pd.isna(merged[k]) and pd.notna(v):
                        merged[k] = v
                merged_rows.append(merged)
            else:
                merged_rows.append(left_row)

        # Add unmatched right rows
        for j in range(len(df_right)):
            if j not in used_right:
                right_row = df_right.iloc[j].to_dict()
                renamed = {}
                for col, val in right_row.items():
                    renamed[reverse_map.get(col, col)] = val
                merged_rows.append(renamed)

        return pd.DataFrame(merged_rows)

    # ------------------------------------------------------------------ #
    #  Temporal join (experimental)                                        #
    # ------------------------------------------------------------------ #

    def _temporal_join(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        alignments: List[FieldAlignment]
    ) -> pd.DataFrame:
        """Join based on temporal alignment (experimental)."""
        timestamp_cols = []
        for a in alignments:
            if "time" in a.source_field.lower() or "date" in a.source_field.lower():
                timestamp_cols.append((a.source_field, a.target_field))

        if not timestamp_cols:
            return pd.concat([df_left, df_right], ignore_index=True)

        left_ts, right_ts = timestamp_cols[0]

        df_left = df_left.copy()
        df_right = df_right.copy()
        df_left[left_ts] = pd.to_datetime(df_left[left_ts], errors="coerce")
        df_right[right_ts] = pd.to_datetime(df_right[right_ts], errors="coerce")

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

    # ------------------------------------------------------------------ #
    #  Imputation                                                          #
    # ------------------------------------------------------------------ #

    def _impute_missing(
        self,
        df: pd.DataFrame,
        method: ImputationMethod
    ) -> Tuple[pd.DataFrame, ImputationSummary]:
        """Impute missing values in numeric columns."""
        df = df.copy()
        fields_imputed = {}
        total_filled = 0

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Filter columns with acceptable null ratio
        cols_to_impute = []
        for col in numeric_cols:
            null_ratio = df[col].isna().sum() / len(df) if len(df) > 0 else 0
            if 0 < null_ratio <= self._config.max_null_ratio_for_inclusion:
                cols_to_impute.append(col)

        if not cols_to_impute:
            return df, ImputationSummary(
                total_nulls_filled=0,
                fields_imputed={},
                method_used=method
            )

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
            from sklearn.impute import KNNImputer
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

    # ------------------------------------------------------------------ #
    #  Storage                                                             #
    # ------------------------------------------------------------------ #

    def _save_fused_dataset(
        self,
        df: pd.DataFrame,
        fused_id: str
    ) -> str:
        """Save fused dataset to disk."""
        output_dir = self._config.output_path
        os.makedirs(output_dir, exist_ok=True)

        if self._config.output_format == "parquet":
            path = os.path.join(output_dir, f"{fused_id}.parquet")
            df.to_parquet(path, index=False)
        else:
            path = os.path.join(output_dir, f"{fused_id}.csv")
            df.to_csv(path, index=False)

        return path
