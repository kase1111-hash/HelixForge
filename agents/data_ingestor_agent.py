"""Data Ingestor Agent for HelixForge.

Handles ingestion of data from various sources including
CSV, Parquet, JSON files, SQL databases, and REST APIs.
"""

import hashlib
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import chardet
import pandas as pd
import pyarrow.parquet as pq
import requests
from sqlalchemy import create_engine, text

from agents.base_agent import BaseAgent
from models.schemas import IngestorConfig, IngestResult, SourceType
from utils.validation import ValidationError, validate_file_path, validate_url


class IngestionError(Exception):
    """Raised when data ingestion fails."""
    pass


class DataIngestorAgent(BaseAgent):
    """Agent for ingesting data from various sources.

    Supports CSV, Parquet, JSON, Excel files, SQL databases, and REST APIs.
    Performs automatic encoding detection, delimiter detection, and type inference.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        super().__init__(config, correlation_id)
        self._config = IngestorConfig(**self.config.get("ingestor", {}))
        self._dataframes: Dict[str, pd.DataFrame] = {}

    @property
    def event_type(self) -> str:
        return "data.ingested"

    def process(
        self,
        source: str,
        source_type: Optional[str] = None,
        **kwargs
    ) -> IngestResult:
        """Ingest data from the specified source.

        Args:
            source: Path to file, database connection string, or URL.
            source_type: Type of source ('csv', 'parquet', 'json', 'sql', 'rest').
                        Auto-detected if not provided.
            **kwargs: Additional arguments for specific source types.

        Returns:
            IngestResult with metadata about the ingested data.

        Raises:
            IngestionError: If ingestion fails.
        """
        self.logger.info(f"Starting ingestion from: {source}")

        try:
            # Auto-detect source type if not provided
            if source_type is None:
                source_type = self._detect_source_type(source)

            source_type_enum = SourceType(source_type.lower())

            # Route to appropriate handler
            handlers = {
                SourceType.CSV: self._ingest_csv,
                SourceType.PARQUET: self._ingest_parquet,
                SourceType.JSON: self._ingest_json,
                SourceType.XLSX: self._ingest_excel,
                SourceType.SQL: self._ingest_sql,
                SourceType.REST: self._ingest_rest,
            }

            handler = handlers.get(source_type_enum)
            if handler is None:
                raise IngestionError(f"Unsupported source type: {source_type}")

            df, metadata = handler(source, **kwargs)

            # Generate dataset ID
            dataset_id = kwargs.get("dataset_id") or str(uuid.uuid4())

            # Compute content hash
            content_hash = self._compute_hash(df)

            # Store DataFrame
            storage_path = self._store_dataframe(df, dataset_id)

            # Get sample data
            sample_size = min(self._config.sample_size, len(df))
            sample_data = df.head(sample_size).to_dict(orient="records")

            # Build result
            result = IngestResult(
                dataset_id=dataset_id,
                source=source,
                source_type=source_type_enum,
                schema=list(df.columns),
                dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
                row_count=len(df),
                sample_rows=sample_size,
                sample_data=sample_data,
                content_hash=content_hash,
                encoding=metadata.get("encoding"),
                storage_path=storage_path
            )

            # Publish event
            self.publish(self.event_type, result.model_dump())

            self.logger.info(
                f"Ingestion complete: {result.row_count} rows, "
                f"{len(result.schema_fields)} columns"
            )

            return result

        except Exception as e:
            self.handle_error(e, {"source": source, "source_type": source_type})
            raise IngestionError(f"Failed to ingest from {source}: {e}") from e

    def ingest_file(self, file_path: str, **kwargs) -> IngestResult:
        """Convenience method to ingest a file.

        Args:
            file_path: Path to the file.
            **kwargs: Additional arguments.

        Returns:
            IngestResult.
        """
        return self.process(file_path, **kwargs)

    def ingest_sql(
        self,
        connection_string: str,
        query: str,
        **kwargs
    ) -> IngestResult:
        """Convenience method to ingest from SQL.

        Args:
            connection_string: Database connection string.
            query: SQL query to execute.
            **kwargs: Additional arguments.

        Returns:
            IngestResult.
        """
        return self.process(
            connection_string,
            source_type="sql",
            query=query,
            **kwargs
        )

    def ingest_rest(self, url: str, **kwargs) -> IngestResult:
        """Convenience method to ingest from REST API.

        Args:
            url: API endpoint URL.
            **kwargs: Additional arguments (headers, auth, etc.).

        Returns:
            IngestResult.
        """
        return self.process(url, source_type="rest", **kwargs)

    def get_dataframe(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Get a stored DataFrame by dataset ID.

        Args:
            dataset_id: Dataset identifier.

        Returns:
            DataFrame or None if not found.
        """
        return self._dataframes.get(dataset_id)

    def _detect_source_type(self, source: str) -> str:
        """Auto-detect source type from source string.

        Args:
            source: Source path, URL, or connection string.

        Returns:
            Detected source type.
        """
        source_lower = source.lower()

        # Check file extensions
        if source_lower.endswith(".csv"):
            return "csv"
        elif source_lower.endswith(".parquet"):
            return "parquet"
        elif source_lower.endswith((".json", ".jsonl")):
            return "json"
        elif source_lower.endswith((".xlsx", ".xls")):
            return "xlsx"

        # Check for database connection strings
        if any(db in source_lower for db in ["postgresql://", "mysql://", "sqlite://"]):
            return "sql"

        # Check for URLs
        if source_lower.startswith(("http://", "https://")):
            return "rest"

        raise IngestionError(f"Cannot auto-detect source type for: {source}")

    def _ingest_csv(
        self,
        file_path: str,
        **kwargs
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Ingest a CSV file.

        Args:
            file_path: Path to CSV file.
            **kwargs: pandas.read_csv arguments.

        Returns:
            Tuple of (DataFrame, metadata).
        """
        validate_file_path(
            file_path,
            allowed_extensions={".csv"},
            max_size_mb=self._config.max_file_size_mb
        )

        # Detect encoding
        encoding = kwargs.pop("encoding", None)
        if encoding is None:
            encoding = self._detect_encoding(file_path)

        # Detect delimiter
        delimiter = kwargs.pop("delimiter", None) or kwargs.pop("sep", None)
        if delimiter is None:
            delimiter = self._detect_delimiter(file_path, encoding)

        df = pd.read_csv(
            file_path,
            encoding=encoding,
            sep=delimiter,
            **kwargs
        )

        return df, {"encoding": encoding, "delimiter": delimiter}

    def _ingest_parquet(
        self,
        file_path: str,
        **kwargs
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Ingest a Parquet file.

        Args:
            file_path: Path to Parquet file.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (DataFrame, metadata).
        """
        validate_file_path(
            file_path,
            allowed_extensions={".parquet"},
            max_size_mb=self._config.max_file_size_mb
        )

        table = pq.read_table(file_path)
        df = table.to_pandas()

        return df, {"format": "parquet"}

    def _ingest_json(
        self,
        file_path: str,
        **kwargs
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Ingest a JSON file.

        Args:
            file_path: Path to JSON file.
            **kwargs: pandas.read_json arguments.

        Returns:
            Tuple of (DataFrame, metadata).
        """
        validate_file_path(
            file_path,
            allowed_extensions={".json", ".jsonl"},
            max_size_mb=self._config.max_file_size_mb
        )

        # Check if JSON Lines format
        lines = kwargs.pop("lines", None)
        if lines is None:
            lines = file_path.lower().endswith(".jsonl")

        df = pd.read_json(file_path, lines=lines, **kwargs)

        return df, {"format": "json", "lines": lines}

    def _ingest_excel(
        self,
        file_path: str,
        **kwargs
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Ingest an Excel file.

        Args:
            file_path: Path to Excel file.
            **kwargs: pandas.read_excel arguments.

        Returns:
            Tuple of (DataFrame, metadata).
        """
        validate_file_path(
            file_path,
            allowed_extensions={".xlsx", ".xls"},
            max_size_mb=self._config.max_file_size_mb
        )

        sheet_name = kwargs.pop("sheet_name", 0)
        df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)

        return df, {"format": "excel", "sheet": sheet_name}

    def _ingest_sql(
        self,
        connection_string: str,
        **kwargs
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Ingest from SQL database.

        Args:
            connection_string: Database connection string.
            **kwargs: Must include 'query'.

        Returns:
            Tuple of (DataFrame, metadata).
        """
        query = kwargs.pop("query", None)
        if not query:
            raise IngestionError("SQL ingestion requires 'query' parameter")

        timeout = kwargs.pop("timeout", self._config.sql_timeout_seconds)

        engine = create_engine(connection_string)
        with engine.connect() as conn:
            conn = conn.execution_options(timeout=timeout)
            df = pd.read_sql(text(query), conn)

        return df, {"format": "sql", "query": query}

    def _ingest_rest(
        self,
        url: str,
        **kwargs
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Ingest from REST API.

        Args:
            url: API endpoint URL.
            **kwargs: requests.get arguments.

        Returns:
            Tuple of (DataFrame, metadata).
        """
        validate_url(url)

        headers = kwargs.pop("headers", {})
        auth = kwargs.pop("auth", None)
        timeout = kwargs.pop("timeout", self._config.rest_timeout_seconds)

        response = requests.get(
            url,
            headers=headers,
            auth=auth,
            timeout=timeout,
            **kwargs
        )
        response.raise_for_status()

        data = response.json()

        # Handle different JSON structures
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Try to find a list in the response
            for key, value in data.items():
                if isinstance(value, list) and value:
                    df = pd.DataFrame(value)
                    break
            else:
                df = pd.DataFrame([data])
        else:
            raise IngestionError(f"Unexpected response format: {type(data)}")

        return df, {"format": "rest", "url": url}

    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding.

        Args:
            file_path: Path to file.

        Returns:
            Detected encoding name.
        """
        sample_size = self._config.encoding_detection_sample_bytes

        with open(file_path, "rb") as f:
            sample = f.read(sample_size)

        result = chardet.detect(sample)
        encoding = result.get("encoding", "utf-8")

        self.logger.debug(f"Detected encoding: {encoding} (confidence: {result.get('confidence')})")

        return encoding or "utf-8"

    def _detect_delimiter(self, file_path: str, encoding: str) -> str:
        """Detect CSV delimiter.

        Args:
            file_path: Path to CSV file.
            encoding: File encoding.

        Returns:
            Detected delimiter.
        """
        delimiters = [",", "\t", ";", "|"]

        with open(file_path, "r", encoding=encoding) as f:
            sample = f.read(8192)

        # Count occurrences of each delimiter
        counts = {d: sample.count(d) for d in delimiters}

        # Return the most common delimiter
        best_delimiter = max(counts, key=counts.get)

        self.logger.debug(f"Detected delimiter: {repr(best_delimiter)}")

        return best_delimiter

    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute SHA-256 hash of DataFrame content.

        Args:
            df: DataFrame to hash.

        Returns:
            Hex digest of hash.
        """
        # Sort columns for consistency
        df_sorted = df.reindex(sorted(df.columns), axis=1)

        # Convert to bytes
        content = df_sorted.to_csv(index=False).encode("utf-8")

        return hashlib.sha256(content).hexdigest()

    def _store_dataframe(self, df: pd.DataFrame, dataset_id: str) -> str:
        """Store DataFrame and return storage path.

        Args:
            df: DataFrame to store.
            dataset_id: Dataset identifier.

        Returns:
            Storage path.
        """
        # Store in memory
        self._dataframes[dataset_id] = df

        # Also save to disk
        storage_dir = self._config.temp_storage_path
        os.makedirs(storage_dir, exist_ok=True)
        storage_path = os.path.join(storage_dir, f"{dataset_id}.parquet")
        df.to_parquet(storage_path, index=False)

        return storage_path
