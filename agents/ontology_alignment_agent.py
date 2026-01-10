"""Ontology Alignment Agent for HelixForge.

Identifies semantic relationships between fields across
different datasets using embedding similarity and graph construction.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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

    Computes semantic similarity between fields and builds
    a unified ontology graph representing relationships.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        super().__init__(config, correlation_id)
        self._config = AlignmentConfig(**self.config.get("alignment", {}))
        self._graph_client = None

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

            # Build ontology graph
            graph_uri = self._build_ontology_graph(metadata_list, all_alignments)

            result = AlignmentResult(
                alignment_job_id=job_id,
                datasets_aligned=[m.dataset_id for m in metadata_list],
                alignments=all_alignments,
                unmatched_fields=unmatched_fields,
                ontology_graph_uri=graph_uri
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
                return cosine_similarity(field_a.embedding, field_b.embedding)
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

        return combined

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
        elif similarity >= 0.85:
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

    def _build_ontology_graph(
        self,
        metadata_list: List[DatasetMetadata],
        alignments: List[FieldAlignment]
    ) -> Optional[str]:
        """Build ontology graph in Neo4j.

        Args:
            metadata_list: Dataset metadata.
            alignments: Field alignments.

        Returns:
            Graph URI or None if graph store unavailable.
        """
        try:
            from neo4j import GraphDatabase

            uri = self._config.graph_uri
            driver = GraphDatabase.driver(uri)

            with driver.session() as session:
                # Create dataset nodes
                for meta in metadata_list:
                    session.run(
                        """
                        MERGE (d:Dataset {id: $id})
                        SET d.name = $name, d.domain = $domain
                        """,
                        id=meta.dataset_id,
                        name=meta.dataset_id,
                        domain=meta.domain_tags[0] if meta.domain_tags else "unknown"
                    )

                    # Create field nodes
                    for field in meta.fields:
                        session.run(
                            """
                            MERGE (f:Field {id: $id})
                            SET f.name = $name, f.semantic_label = $label,
                                f.dataset_id = $dataset_id
                            WITH f
                            MATCH (d:Dataset {id: $dataset_id})
                            MERGE (f)-[:BELONGS_TO]->(d)
                            """,
                            id=f"{meta.dataset_id}.{field.field_name}",
                            name=field.field_name,
                            label=field.semantic_label,
                            dataset_id=meta.dataset_id
                        )

                # Create alignment relationships
                for alignment in alignments:
                    session.run(
                        """
                        MATCH (a:Field {id: $source_id})
                        MATCH (b:Field {id: $target_id})
                        MERGE (a)-[r:ALIGNS_WITH]->(b)
                        SET r.similarity = $similarity, r.type = $type
                        """,
                        source_id=f"{alignment.source_dataset}.{alignment.source_field}",
                        target_id=f"{alignment.target_dataset}.{alignment.target_field}",
                        similarity=alignment.similarity,
                        type=alignment.alignment_type.value
                    )

            driver.close()
            return uri

        except Exception as e:
            self.logger.warning(f"Failed to build ontology graph: {e}")
            return None

    def validate_alignment(
        self,
        alignment_id: str,
        validated: bool
    ) -> bool:
        """Mark an alignment as validated.

        Args:
            alignment_id: ID of the alignment.
            validated: Validation status.

        Returns:
            True if successful.
        """
        # This would update the alignment in storage
        self.logger.info(f"Alignment {alignment_id} validated: {validated}")
        return True
