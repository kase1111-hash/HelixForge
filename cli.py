"""HelixForge CLI - Cross-Dataset Insight Synthesizer.

Usage:
    python cli.py describe <file> [--format json|table] [--provider mock|openai]
    python cli.py align <file1> <file2> [--format json|table] [--provider mock|openai]
    python cli.py ingest <file> [--format json|table]
"""

import argparse
import json
import sys
from typing import Optional

from agents.data_ingestor_agent import DataIngestorAgent
from agents.metadata_interpreter_agent import MetadataInterpreterAgent
from utils.llm import LLMProvider, MockProvider, OpenAIProvider


def get_provider(provider_name: str) -> LLMProvider:
    """Create an LLM provider by name."""
    if provider_name == "mock":
        return MockProvider()
    elif provider_name == "openai":
        return OpenAIProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Use 'mock' or 'openai'.")


def cmd_describe(args):
    """Describe a dataset's fields with semantic labels."""
    file_path = args.file
    output_format = args.format
    provider = get_provider(args.provider)

    # Ingest the file
    ingestor_config = {
        "ingestor": {
            "max_file_size_mb": 500,
            "sample_size": 10,
            "temp_storage_path": "/tmp/helixforge",
        }
    }
    ingestor = DataIngestorAgent(config=ingestor_config)

    try:
        result = ingestor.ingest_file(file_path)
    except Exception as e:
        print(f"Error ingesting {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

    df = ingestor.get_dataframe(result.dataset_id)

    # Interpret metadata
    interpreter_config = {
        "interpreter": {
            "embedding_model": "text-embedding-3-large",
            "embedding_dimensions": 1536,
            "llm_model": "gpt-4o",
            "llm_temperature": 0.2,
            "max_sample_values": 10,
            "batch_size": 50,
        }
    }
    interpreter = MetadataInterpreterAgent(
        config=interpreter_config, provider=provider
    )
    metadata = interpreter.process(result.dataset_id, df)

    if output_format == "json":
        _print_json(metadata, result)
    else:
        _print_table(metadata, result)


def _print_json(metadata, ingest_result):
    """Print metadata as JSON."""
    output = {
        "dataset_id": metadata.dataset_id,
        "description": metadata.dataset_description,
        "domain_tags": metadata.domain_tags,
        "row_count": ingest_result.row_count,
        "fields": [
            {
                "field_name": f.field_name,
                "data_type": f.data_type.value,
                "semantic_type": f.semantic_type.value,
                "semantic_label": f.semantic_label,
                "description": f.description,
                "null_ratio": round(f.null_ratio, 4),
                "unique_ratio": round(f.unique_ratio, 4),
                "confidence": round(f.confidence, 2),
                "sample_values": f.sample_values[:3],
            }
            for f in metadata.fields
        ],
    }
    print(json.dumps(output, indent=2, default=str))


def _print_table(metadata, ingest_result):
    """Print metadata as a formatted table."""
    print(f"\n  Dataset: {metadata.dataset_id}")
    print(f"  Description: {metadata.dataset_description}")
    print(f"  Rows: {ingest_result.row_count}  |  Domains: {', '.join(metadata.domain_tags)}")
    print()

    # Column headers
    headers = ["Field", "Type", "Semantic", "Label", "Null%", "Uniq%", "Conf", "Samples"]
    widths = [20, 10, 12, 20, 6, 6, 5, 30]

    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(f"  {header_line}")
    print(f"  {'─' * len(header_line)}")

    for f in metadata.fields:
        samples_str = str(f.sample_values[:3])
        if len(samples_str) > 30:
            samples_str = samples_str[:27] + "..."

        row = [
            f.field_name[:20],
            f.data_type.value[:10],
            f.semantic_type.value[:12],
            f.semantic_label[:20],
            f"{f.null_ratio:.0%}",
            f"{f.unique_ratio:.0%}",
            f"{f.confidence:.0%}",
            samples_str,
        ]
        row_line = "  ".join(val.ljust(w) for val, w in zip(row, widths))
        print(f"  {row_line}")

    print()


def cmd_align(args):
    """Align two datasets and print field mappings."""
    provider = get_provider(args.provider)
    output_format = args.format

    ingestor_config = {
        "ingestor": {
            "max_file_size_mb": 500,
            "sample_size": 10,
            "temp_storage_path": "/tmp/helixforge",
        }
    }
    interpreter_config = {
        "interpreter": {
            "embedding_model": "text-embedding-3-large",
            "embedding_dimensions": 1536,
            "llm_model": "gpt-4o",
            "llm_temperature": 0.2,
            "max_sample_values": 10,
            "batch_size": 50,
        }
    }

    ingestor = DataIngestorAgent(config=ingestor_config)
    interpreter = MetadataInterpreterAgent(
        config=interpreter_config, provider=provider
    )

    # Ingest and interpret both files
    metadata_list = []
    for file_path in [args.file1, args.file2]:
        try:
            result = ingestor.ingest_file(file_path)
        except Exception as e:
            print(f"Error ingesting {file_path}: {e}", file=sys.stderr)
            sys.exit(1)

        df = ingestor.get_dataframe(result.dataset_id)
        metadata = interpreter.process(result.dataset_id, df)
        metadata_list.append(metadata)

    # Run alignment
    from agents.ontology_alignment_agent import OntologyAlignmentAgent

    alignment_config = {
        "alignment": {
            "similarity_threshold": args.threshold,
        }
    }
    aligner = OntologyAlignmentAgent(config=alignment_config)
    alignment_result = aligner.process(metadata_list)

    if output_format == "json":
        _print_align_json(alignment_result, metadata_list)
    else:
        _print_align_table(alignment_result, metadata_list)


def _print_align_json(alignment_result, metadata_list):
    """Print alignment result as JSON."""
    output = {
        "job_id": alignment_result.alignment_job_id,
        "datasets": alignment_result.datasets_aligned,
        "alignments": [
            {
                "source": f"{a.source_dataset}.{a.source_field}",
                "target": f"{a.target_dataset}.{a.target_field}",
                "similarity": round(a.similarity, 4),
                "type": a.alignment_type.value,
                "transformation": a.transformation_hint,
            }
            for a in alignment_result.alignments
        ],
        "unmatched_fields": alignment_result.unmatched_fields,
        "total_alignments": len(alignment_result.alignments),
        "total_unmatched": len(alignment_result.unmatched_fields),
    }
    print(json.dumps(output, indent=2, default=str))


def _print_align_table(alignment_result, metadata_list):
    """Print alignment result as a formatted table."""
    ds_a, ds_b = alignment_result.datasets_aligned[:2]
    n_a = len(metadata_list[0].fields) if metadata_list else 0
    n_b = len(metadata_list[1].fields) if len(metadata_list) > 1 else 0

    print(f"\n  Alignment: {ds_a} ({n_a} fields) <-> {ds_b} ({n_b} fields)")
    print(f"  Found {len(alignment_result.alignments)} alignment(s), "
          f"{len(alignment_result.unmatched_fields)} unmatched field(s)")
    print()

    if alignment_result.alignments:
        headers = ["Source", "Target", "Score", "Type", "Transform"]
        widths = [25, 25, 7, 10, 30]

        header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
        print(f"  {header_line}")
        print(f"  {'─' * len(header_line)}")

        for a in sorted(alignment_result.alignments, key=lambda x: x.similarity, reverse=True):
            transform = a.transformation_hint or "—"
            row = [
                f"{a.source_field}"[:25],
                f"{a.target_field}"[:25],
                f"{a.similarity:.2f}",
                a.alignment_type.value[:10],
                transform[:30],
            ]
            row_line = "  ".join(val.ljust(w) for val, w in zip(row, widths))
            print(f"  {row_line}")
        print()

    if alignment_result.unmatched_fields:
        print("  Unmatched:")
        for field in alignment_result.unmatched_fields:
            print(f"    - {field}")
        print()


def cmd_ingest(args):
    """Ingest a dataset and print a JSON summary."""
    file_path = args.file
    output_format = args.format

    ingestor_config = {
        "ingestor": {
            "max_file_size_mb": 500,
            "sample_size": 10,
            "temp_storage_path": "/tmp/helixforge",
        }
    }
    ingestor = DataIngestorAgent(config=ingestor_config)

    try:
        result = ingestor.ingest_file(file_path, dataset_id=args.dataset_id)
    except Exception as e:
        print(f"Error ingesting {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

    output = {
        "dataset_id": result.dataset_id,
        "source": result.source,
        "source_type": result.source_type.value,
        "row_count": result.row_count,
        "columns": result.schema_fields,
        "dtypes": result.dtypes,
        "content_hash": result.content_hash,
        "encoding": result.encoding,
        "storage_path": result.storage_path,
    }

    if output_format == "json":
        print(json.dumps(output, indent=2, default=str))
    else:
        print(f"\n  Dataset ID: {result.dataset_id}")
        print(f"  Source:     {result.source}")
        print(f"  Format:     {result.source_type.value}")
        print(f"  Rows:       {result.row_count}")
        print(f"  Columns:    {', '.join(result.schema_fields)}")
        print(f"  Hash:       {result.content_hash[:16]}...")
        if result.encoding:
            print(f"  Encoding:   {result.encoding}")
        print(f"  Stored at:  {result.storage_path}")
        print()


def main():
    parser = argparse.ArgumentParser(
        prog="helixforge",
        description="HelixForge - Cross-Dataset Insight Synthesizer",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest command
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest a dataset file and print summary"
    )
    ingest_parser.add_argument("file", help="Path to the dataset file (csv, json, parquet, xlsx)")
    ingest_parser.add_argument(
        "--format", choices=["json", "table"], default="json",
        help="Output format (default: json)"
    )
    ingest_parser.add_argument(
        "--dataset-id", default=None,
        help="Custom dataset ID (default: auto-generated UUID)"
    )

    # align command
    align_parser = subparsers.add_parser(
        "align", help="Align two datasets and show field mappings"
    )
    align_parser.add_argument("file1", help="Path to the first dataset file")
    align_parser.add_argument("file2", help="Path to the second dataset file")
    align_parser.add_argument(
        "--format", choices=["json", "table"], default="table",
        help="Output format (default: table)"
    )
    align_parser.add_argument(
        "--provider", choices=["mock", "openai"], default="mock",
        help="LLM provider for embeddings (default: mock)"
    )
    align_parser.add_argument(
        "--threshold", type=float, default=0.50,
        help="Similarity threshold for alignment (default: 0.50)"
    )

    # describe command
    describe_parser = subparsers.add_parser(
        "describe", help="Describe a dataset's fields with semantic labels"
    )
    describe_parser.add_argument("file", help="Path to the dataset file (csv, json, parquet)")
    describe_parser.add_argument(
        "--format", choices=["json", "table"], default="table",
        help="Output format (default: table)"
    )
    describe_parser.add_argument(
        "--provider", choices=["mock", "openai"], default="mock",
        help="LLM provider (default: mock, use 'openai' for real API calls)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "describe":
        cmd_describe(args)
    elif args.command == "align":
        cmd_align(args)


if __name__ == "__main__":
    main()
