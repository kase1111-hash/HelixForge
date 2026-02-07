"""HelixForge CLI - Cross-Dataset Insight Synthesizer.

Usage:
    python cli.py describe <file> [--format json|table] [--provider mock|openai]
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


if __name__ == "__main__":
    main()
