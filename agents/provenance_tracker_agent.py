"""Provenance Tracker Agent for HelixForge.

Tracks data lineage and transformations throughout the
pipeline, building provenance graphs and generating reports.
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from models.schemas import (
    FieldOrigin,
    IngestResult,
    ProvenanceConfig,
    ProvenanceOperation,
    ProvenanceReport,
    ProvenanceTrace,
    TransformationRecord,
)


class ProvenanceTrackerAgent(BaseAgent):
    """Agent for tracking data provenance.

    Records all data transformations and builds a lineage
    graph that can trace any field back to its original source.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        super().__init__(config, correlation_id)
        self._config = ProvenanceConfig(**self.config.get("provenance", {}))
        self._traces: Dict[str, ProvenanceTrace] = {}
        self._transformations: List[TransformationRecord] = []
        self._graph_driver = None

    @property
    def event_type(self) -> str:
        return "trace.updated"

    def process(
        self,
        fused_dataset_id: str,
        **kwargs
    ) -> ProvenanceReport:
        """Generate provenance report for a fused dataset.

        Args:
            fused_dataset_id: ID of the fused dataset.
            **kwargs: Additional options.

        Returns:
            ProvenanceReport.
        """
        self.logger.info(f"Generating provenance report for: {fused_dataset_id}")

        try:
            # Collect all traces for this dataset
            dataset_traces = [
                trace for trace in self._traces.values()
                if trace.fused_dataset_id == fused_dataset_id
            ]

            # Calculate coverage
            total_fields = len(dataset_traces)
            complete_traces = len([t for t in dataset_traces if t.origins])

            report = ProvenanceReport(
                report_id=str(uuid.uuid4()),
                fused_dataset_id=fused_dataset_id,
                total_fields=total_fields,
                fields_with_complete_provenance=complete_traces,
                coverage_percentage=complete_traces / total_fields if total_fields > 0 else 0,
                traces=dataset_traces,
                format=self._config.report_format
            )

            # Export report
            self._export_report(report)

            # Publish event
            self.publish(self.event_type, report.model_dump())

            self.logger.info(
                f"Provenance report complete: {complete_traces}/{total_fields} fields traced"
            )

            return report

        except Exception as e:
            self.handle_error(e, {"fused_dataset_id": fused_dataset_id})
            raise

    def record_ingestion(
        self,
        ingest_result: IngestResult,
        **kwargs
    ) -> None:
        """Record data ingestion.

        Args:
            ingest_result: Result from data ingestion.
        """
        self.logger.debug(f"Recording ingestion: {ingest_result.dataset_id}")

        for i, field_name in enumerate(ingest_result.schema_fields):
            origin = FieldOrigin(
                source_file=ingest_result.source,
                source_column=field_name,
                source_column_index=i,
                dataset_id=ingest_result.dataset_id,
                ingested_at=ingest_result.ingested_at,
                content_hash=ingest_result.content_hash
            )

            trace = ProvenanceTrace(
                trace_id=str(uuid.uuid4()),
                field=field_name,
                fused_dataset_id=ingest_result.dataset_id,
                lineage_depth=0,
                origins=[origin],
                transformations=[],
                confidence=1.0
            )

            trace_key = f"{ingest_result.dataset_id}.{field_name}"
            self._traces[trace_key] = trace

            # Record in graph
            self._record_to_graph("ingestion", trace)

        transformation = TransformationRecord(
            step_id=str(uuid.uuid4()),
            operation=ProvenanceOperation.INGEST,
            input_fields=[],
            output_field=ingest_result.dataset_id,
            parameters={
                "source": ingest_result.source,
                "source_type": ingest_result.source_type.value,
                "row_count": ingest_result.row_count
            },
            agent="DataIngestorAgent"
        )
        self._transformations.append(transformation)

    def record_transformation(
        self,
        source_fields: List[str],
        target_field: str,
        operation: str,
        parameters: Dict[str, Any],
        fused_dataset_id: str,
        **kwargs
    ) -> None:
        """Record a field transformation.

        Args:
            source_fields: Source field identifiers.
            target_field: Target field name.
            operation: Type of operation.
            parameters: Transformation parameters.
            fused_dataset_id: Fused dataset ID.
        """
        self.logger.debug(f"Recording transformation: {source_fields} -> {target_field}")

        # Get source traces
        source_origins = []
        source_transforms = []
        max_depth = 0

        for source_field in source_fields:
            if source_field in self._traces:
                source_trace = self._traces[source_field]
                source_origins.extend(source_trace.origins)
                source_transforms.extend(source_trace.transformations)
                max_depth = max(max_depth, source_trace.lineage_depth)

        # Create transformation record
        transformation = TransformationRecord(
            step_id=str(uuid.uuid4()),
            operation=ProvenanceOperation(operation) if operation in [e.value for e in ProvenanceOperation] else ProvenanceOperation.TRANSFORM,
            input_fields=source_fields,
            output_field=target_field,
            parameters=parameters,
            agent=kwargs.get("agent", "Unknown"),
            confidence_delta=-self._config.confidence_decay_per_step
        )

        self._transformations.append(transformation)

        # Calculate confidence with decay
        base_confidence = 1.0
        for t in source_transforms:
            base_confidence += t.confidence_delta
        confidence = max(0.0, min(1.0, base_confidence + transformation.confidence_delta))

        # Create or update trace
        trace_key = f"{fused_dataset_id}.{target_field}"
        self._traces[trace_key] = ProvenanceTrace(
            trace_id=str(uuid.uuid4()),
            field=target_field,
            fused_dataset_id=fused_dataset_id,
            lineage_depth=max_depth + 1,
            origins=source_origins,
            transformations=source_transforms + [transformation],
            confidence=confidence
        )

        # Record in graph
        self._record_to_graph("transformation", self._traces[trace_key])

    def record_alignment(
        self,
        source_dataset: str,
        source_field: str,
        target_dataset: str,
        target_field: str,
        similarity: float,
        **kwargs
    ) -> None:
        """Record field alignment.

        Args:
            source_dataset: Source dataset ID.
            source_field: Source field name.
            target_dataset: Target dataset ID.
            target_field: Target field name.
            similarity: Alignment similarity score.
        """
        self.logger.debug(
            f"Recording alignment: {source_dataset}.{source_field} -> "
            f"{target_dataset}.{target_field}"
        )

        transformation = TransformationRecord(
            step_id=str(uuid.uuid4()),
            operation=ProvenanceOperation.ALIGN,
            input_fields=[f"{source_dataset}.{source_field}"],
            output_field=f"{target_dataset}.{target_field}",
            parameters={"similarity": similarity},
            agent="OntologyAlignmentAgent",
            confidence_delta=-(1 - similarity) * 0.1  # Penalize low similarity
        )

        self._transformations.append(transformation)

    def record_fusion(
        self,
        source_datasets: List[str],
        fused_dataset_id: str,
        join_strategy: str,
        field_mappings: Dict[str, List[str]],
        **kwargs
    ) -> None:
        """Record dataset fusion.

        Args:
            source_datasets: Source dataset IDs.
            fused_dataset_id: Fused dataset ID.
            join_strategy: Join strategy used.
            field_mappings: Mapping of fused fields to source fields.
        """
        self.logger.debug(f"Recording fusion: {source_datasets} -> {fused_dataset_id}")

        for fused_field, source_fields in field_mappings.items():
            self.record_transformation(
                source_fields=source_fields,
                target_field=fused_field,
                operation="fuse",
                parameters={"join_strategy": join_strategy},
                fused_dataset_id=fused_dataset_id,
                agent="FusionAgent"
            )

    def query_lineage(
        self,
        dataset_id: str,
        field: str
    ) -> Optional[ProvenanceTrace]:
        """Query lineage for a specific field.

        Args:
            dataset_id: Dataset ID.
            field: Field name.

        Returns:
            ProvenanceTrace or None.
        """
        trace_key = f"{dataset_id}.{field}"
        return self._traces.get(trace_key)

    def build_lineage_graph(
        self,
        fused_dataset_id: str
    ) -> Dict[str, Any]:
        """Build a lineage graph structure.

        Args:
            fused_dataset_id: Fused dataset ID.

        Returns:
            Graph structure as dictionary.
        """
        graph: Dict[str, List[Dict[str, Any]]] = {
            "nodes": [],
            "edges": []
        }

        # Add nodes for all traced fields
        for trace_key, trace in self._traces.items():
            if trace.fused_dataset_id == fused_dataset_id:
                # Add fused field node
                graph["nodes"].append({
                    "id": trace_key,
                    "type": "fused_field",
                    "label": trace.field,
                    "confidence": trace.confidence
                })

                # Add origin nodes and edges
                for origin in trace.origins:
                    origin_id = f"{origin.dataset_id}.{origin.source_column}"
                    if not any(n["id"] == origin_id for n in graph["nodes"]):
                        graph["nodes"].append({
                            "id": origin_id,
                            "type": "source_field",
                            "label": origin.source_column,
                            "source_file": origin.source_file
                        })

                    # Add edge
                    graph["edges"].append({
                        "source": origin_id,
                        "target": trace_key,
                        "transformations": len(trace.transformations)
                    })

        return graph

    def _record_to_graph(
        self,
        record_type: str,
        trace: ProvenanceTrace
    ) -> None:
        """Record provenance to Neo4j graph.

        Args:
            record_type: Type of record.
            trace: Provenance trace.
        """
        try:
            if self._graph_driver is None:
                from neo4j import GraphDatabase
                self._graph_driver = GraphDatabase.driver(
                    self._config.graph_uri,
                    auth=(self._config.graph_user, self._config.graph_password)
                    if self._config.graph_password else None
                )

            driver = self._graph_driver
            assert driver is not None
            with driver.session() as session:
                if record_type == "ingestion":
                    for origin in trace.origins:
                        session.run(
                            """
                            MERGE (s:RawSource {file_path: $file_path})
                            SET s.content_hash = $hash, s.ingested_at = $ingested_at

                            MERGE (f:SourceField {id: $field_id})
                            SET f.name = $name, f.column_index = $col_idx,
                                f.dataset_id = $dataset_id

                            MERGE (f)-[:EXTRACTED_FROM]->(s)
                            """,
                            file_path=origin.source_file,
                            hash=origin.content_hash,
                            ingested_at=origin.ingested_at.isoformat(),
                            field_id=f"{origin.dataset_id}.{origin.source_column}",
                            name=origin.source_column,
                            col_idx=origin.source_column_index,
                            dataset_id=origin.dataset_id
                        )

                elif record_type == "transformation":
                    # Create fused field node
                    session.run(
                        """
                        MERGE (f:FusedField {id: $field_id})
                        SET f.name = $name, f.fused_dataset_id = $dataset_id,
                            f.confidence = $confidence
                        """,
                        field_id=f"{trace.fused_dataset_id}.{trace.field}",
                        name=trace.field,
                        dataset_id=trace.fused_dataset_id,
                        confidence=trace.confidence
                    )

                    # Create edges from sources
                    for origin in trace.origins:
                        session.run(
                            """
                            MATCH (s:SourceField {id: $source_id})
                            MATCH (f:FusedField {id: $target_id})
                            MERGE (f)-[:MERGED_FROM {depth: $depth}]->(s)
                            """,
                            source_id=f"{origin.dataset_id}.{origin.source_column}",
                            target_id=f"{trace.fused_dataset_id}.{trace.field}",
                            depth=trace.lineage_depth
                        )

        except Exception as e:
            self.logger.warning(f"Failed to record to graph: {e}")

    def _export_report(self, report: ProvenanceReport) -> str:
        """Export provenance report.

        Args:
            report: ProvenanceReport to export.

        Returns:
            Path to exported report.
        """
        output_dir = self._config.report_output_path
        os.makedirs(output_dir, exist_ok=True)

        if self._config.report_format == "json":
            path = os.path.join(output_dir, f"{report.fused_dataset_id}_provenance.json")
            with open(path, "w") as f:
                json.dump(report.model_dump(), f, indent=2, default=str)

        elif self._config.report_format == "html":
            path = os.path.join(output_dir, f"{report.fused_dataset_id}_provenance.html")
            html = self._generate_html_report(report)
            with open(path, "w") as f:
                f.write(html)

        else:
            path = os.path.join(output_dir, f"{report.fused_dataset_id}_provenance.txt")
            with open(path, "w") as f:
                f.write(f"Provenance Report: {report.fused_dataset_id}\n")
                f.write(f"Coverage: {report.coverage_percentage:.1%}\n\n")
                for trace in report.traces:
                    f.write(f"\nField: {trace.field}\n")
                    f.write(f"  Confidence: {trace.confidence:.2f}\n")
                    f.write(f"  Lineage Depth: {trace.lineage_depth}\n")
                    for origin in trace.origins:
                        f.write(f"  Origin: {origin.source_file}:{origin.source_column}\n")

        return path

    def _generate_html_report(self, report: ProvenanceReport) -> str:
        """Generate HTML provenance report.

        Args:
            report: ProvenanceReport.

        Returns:
            HTML string.
        """
        traces_html = ""
        for trace in report.traces:
            origins_html = "\n".join([
                f"<li>{o.source_file} : {o.source_column} (col {o.source_column_index})</li>"
                for o in trace.origins
            ])

            transforms_html = "\n".join([
                f"<li>{t.operation.value}: {t.input_fields} -> {t.output_field}</li>"
                for t in trace.transformations
            ])

            traces_html += f"""
            <div class="trace">
                <h3>{trace.field}</h3>
                <p><strong>Confidence:</strong> {trace.confidence:.2%}</p>
                <p><strong>Lineage Depth:</strong> {trace.lineage_depth}</p>
                <h4>Origins</h4>
                <ul>{origins_html}</ul>
                <h4>Transformations</h4>
                <ul>{transforms_html if transforms_html else '<li>None</li>'}</ul>
            </div>
            """

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Provenance Report - {report.fused_dataset_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f0f0f0; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .trace {{ background: white; border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 4px; }}
        .trace h3 {{ color: #0066cc; margin-top: 0; }}
    </style>
</head>
<body>
    <h1>Provenance Report</h1>
    <p>Dataset: {report.fused_dataset_id}</p>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Fields:</strong> {report.total_fields}</p>
        <p><strong>Fields with Complete Provenance:</strong> {report.fields_with_complete_provenance}</p>
        <p><strong>Coverage:</strong> {report.coverage_percentage:.1%}</p>
        <p><strong>Generated:</strong> {report.generated_at}</p>
    </div>

    <h2>Field Lineage</h2>
    {traces_html}
</body>
</html>"""

    def close(self) -> None:
        """Close graph database connection."""
        if self._graph_driver:
            self._graph_driver.close()
            self._graph_driver = None
