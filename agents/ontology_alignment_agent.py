"""Ontology Alignment Agent for HelixForge.

Identifies semantic relationships between fields across
different datasets using embedding similarity.
"""

import uuid
from typing import Any, Dict, List, Optional


from agents.base_agent import BaseAgent
from models.schemas import (
    AlignmentConfig,
    AlignmentResult,
    AlignmentType,
    DatasetMetadata,
    FieldAlignment,
    FieldMetadata,
)
from utils.embeddings import cosine_similarity


class OntologyAlignmentAgent(BaseAgent):
    """Agent for aligning ontologies across datasets.

    Computes semantic similarity between fields to identify
    matching columns across different data sources.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        super().__init__(config, correlation_id)
        self._config = AlignmentConfig(**self.config.get("alignment", {}))

    @property
    def event_type(self) -> str:
        return "ontology.aligned"

    def process(
        self,
        metadata_list: List[DatasetMetadata],
        **kwargs
    ) -> AlignmentResult:
        """Align fields across multiple datasets.

        Args:
            metadata_list: List of DatasetMetadata objects.
            **kwargs: Additional options.

        Returns:
            AlignmentResult with field mappings.
        """
        self.logger.info(f"Starting alignment for {len(metadata_list)} datasets")

        job_id = str(uuid.uuid4())
        all_alignments = []
        unmatched_fields = []

        try:
            # Compare each pair of datasets
            for i in range(len(metadata_list)):
                for j in range(i + 1, len(metadata_list)):
                    meta_a = metadata_list[i]
                    meta_b = metadata_list[j]

                    alignments = self._align_datasets(meta_a, meta_b)
                    all_alignments.extend(alignments)

                    self.logger.debug(
                        f"Found {len(alignments)} alignments between "
                        f"{meta_a.dataset_id} and {meta_b.dataset_id}"
                    )

            # Warn if no alignments were found
            if not all_alignments:
                self.logger.warning(
                    f"No alignments found between {len(metadata_list)} datasets. "
                    f"This may indicate datasets have no semantically similar fields, "
                    f"or the similarity threshold ({self._config.similarity_threshold}) is too high."
                )

            # Find unmatched fields
            matched_fields = set()
            for a in all_alignments:
                matched_fields.add(f"{a.source_dataset}.{a.source_field}")
                matched_fields.add(f"{a.target_dataset}.{a.target_field}")

            for meta in metadata_list:
                for field in meta.fields:
                    field_key = f"{meta.dataset_id}.{field.field_name}"
                    if field_key not in matched_fields:
                        unmatched_fields.append(field_key)

            result = AlignmentResult(
                alignment_job_id=job_id,
                datasets_aligned=[m.dataset_id for m in metadata_list],
                alignments=all_alignments,
                unmatched_fields=unmatched_fields,
            )

            # Publish event
            self.publish(self.event_type, result.model_dump())

            self.logger.info(
                f"Alignment complete: {len(all_alignments)} alignments, "
                f"{len(unmatched_fields)} unmatched fields"
            )

            return result

        except Exception as e:
            self.handle_error(e, {"job_id": job_id})
            raise

    def _align_datasets(
        self,
        meta_a: DatasetMetadata,
        meta_b: DatasetMetadata
    ) -> List[FieldAlignment]:
        """Align fields between two datasets.

        Args:
            meta_a: First dataset metadata.
            meta_b: Second dataset metadata.

        Returns:
            List of field alignments.
        """
        alignments = []

        for field_a in meta_a.fields:
            field_alignments = []

            for field_b in meta_b.fields:
                similarity = self._compute_similarity(field_a, field_b)

                if similarity >= self._config.similarity_threshold:
                    alignment_type = self._classify_alignment(similarity)
                    transformation_hint = self._suggest_transformation(field_a, field_b)

                    alignment = FieldAlignment(
                        alignment_id=str(uuid.uuid4()),
                        source_dataset=meta_a.dataset_id,
                        source_field=field_a.field_name,
                        target_dataset=meta_b.dataset_id,
                        target_field=field_b.field_name,
                        similarity=similarity,
                        alignment_type=alignment_type,
                        transformation_hint=transformation_hint
                    )
                    field_alignments.append(alignment)

            # Keep top N alignments per field
            field_alignments.sort(key=lambda x: x.similarity, reverse=True)
            alignments.extend(field_alignments[:self._config.max_alignments_per_field])

        # Resolve conflicts
        alignments = self._resolve_conflicts(alignments)

        return alignments

    def _compute_similarity(
        self,
        field_a: FieldMetadata,
        field_b: FieldMetadata
    ) -> float:
        """Compute similarity between two fields.

        Args:
            field_a: First field metadata.
            field_b: Second field metadata.

        Returns:
            Similarity score between 0 and 1.
        """
        # Use embeddings if available
        if field_a.embedding and field_b.embedding:
            try:
                return min(cosine_similarity(field_a.embedding, field_b.embedding), 1.0)
            except ValueError:
                pass

        # Fallback to name and label similarity
        from utils.similarity import string_similarity

        name_sim = string_similarity(
            field_a.field_name.lower(),
            field_b.field_name.lower(),
            method="token_set"
        )

        label_sim = string_similarity(
            field_a.semantic_label.lower(),
            field_b.semantic_label.lower(),
            method="token_set"
        )

        # Weight by confidence
        avg_confidence = (field_a.confidence + field_b.confidence) / 2

        # Combine similarities
        combined = (name_sim * 0.3 + label_sim * 0.5 + avg_confidence * 0.2)

        # Boost if same semantic type
        if field_a.semantic_type == field_b.semantic_type:
            combined = min(combined * 1.1, 1.0)

        # Penalize if different data types
        if field_a.data_type != field_b.data_type:
            combined *= 0.9

        return min(combined, 1.0)

    def _classify_alignment(self, similarity: float) -> AlignmentType:
        """Classify alignment type based on similarity.

        Args:
            similarity: Similarity score.

        Returns:
            AlignmentType enum value.
        """
        if similarity >= self._config.exact_match_threshold:
            return AlignmentType.EXACT
        elif similarity >= self._config.synonym_threshold:
            return AlignmentType.SYNONYM
        elif similarity >= 0.80:
            return AlignmentType.RELATED
        elif similarity >= self._config.similarity_threshold:
            return AlignmentType.PARTIAL
        else:
            return AlignmentType.NONE

    def _suggest_transformation(
        self,
        field_a: FieldMetadata,
        field_b: FieldMetadata
    ) -> Optional[str]:
        """Suggest transformation needed for alignment.

        Args:
            field_a: Source field.
            field_b: Target field.

        Returns:
            Transformation hint or None.
        """
        # Type cast needed
        if field_a.data_type != field_b.data_type:
            return f"type_cast:{field_a.data_type.value}->{field_b.data_type.value}"

        # Check for unit conversion patterns
        name_a = field_a.field_name.lower()
        name_b = field_b.field_name.lower()

        unit_patterns = [
            (["celsius", "temp_c"], ["fahrenheit", "temp_f"], "celsius_to_fahrenheit"),
            (["kg", "kilogram"], ["lb", "pound"], "kg_to_lb"),
            (["meter", "metres"], ["feet", "ft"], "m_to_ft"),
            (["days"], ["months"], "days_to_months"),
        ]

        for pattern_a, pattern_b, transform in unit_patterns:
            if any(p in name_a for p in pattern_a) and any(p in name_b for p in pattern_b):
                return f"unit_conversion:{transform}"
            if any(p in name_b for p in pattern_a) and any(p in name_a for p in pattern_b):
                return f"unit_conversion:{transform}_inverse"

        return None

    def _resolve_conflicts(
        self,
        alignments: List[FieldAlignment]
    ) -> List[FieldAlignment]:
        """Resolve conflicting alignments.

        Args:
            alignments: List of alignments with potential conflicts.

        Returns:
            Resolved alignments.
        """
        if self._config.conflict_resolution != "highest_similarity":
            return alignments

        # Group by target field
        target_groups: Dict[str, List[FieldAlignment]] = {}
        for a in alignments:
            key = f"{a.target_dataset}.{a.target_field}"
            if key not in target_groups:
                target_groups[key] = []
            target_groups[key].append(a)

        # Keep only highest similarity for each target
        resolved = []
        for group in target_groups.values():
            best = max(group, key=lambda x: x.similarity)
            resolved.append(best)

        return resolved

