"""Metadata Interpreter Agent for HelixForge.

Generates semantic understanding of dataset fields using
LLM-powered inference and embedding generation.
"""

import json
from typing import Any, Dict, List, Optional

import pandas as pd

from agents.base_agent import BaseAgent
from models.schemas import (
    DatasetMetadata,
    DataType,
    FieldMetadata,
    InterpreterConfig,
    SemanticType,
)
from utils.llm import LLMProvider, OpenAIProvider


class MetadataInterpreterAgent(BaseAgent):
    """Agent for interpreting dataset metadata.

    Generates semantic labels, descriptions, and embeddings
    for each field in a dataset using LLM inference.

    Accepts an LLMProvider for dependency injection, defaulting
    to OpenAIProvider if none is supplied.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
    ):
        super().__init__(config, correlation_id)
        self._config = InterpreterConfig(**self.config.get("interpreter", {}))
        self._provider = provider
        # Legacy attribute kept for backward compat with tests that set
        # agent._openai_client directly.
        self._openai_client = None

    @property
    def event_type(self) -> str:
        return "metadata.ready"

    def _get_provider(self) -> LLMProvider:
        """Return the configured LLM provider, lazily initializing if needed."""
        if self._provider is not None:
            return self._provider
        # Legacy path: if _openai_client was set directly (e.g. by tests),
        # wrap it in an OpenAIProvider.
        if self._openai_client is not None:
            self._provider = OpenAIProvider(client=self._openai_client)
            return self._provider
        # Default: create a new OpenAIProvider
        self._provider = OpenAIProvider()
        return self._provider

    def process(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        **kwargs
    ) -> DatasetMetadata:
        """Interpret metadata for a dataset.

        Args:
            dataset_id: Dataset identifier.
            df: DataFrame to analyze.
            **kwargs: Additional options.

        Returns:
            DatasetMetadata with semantic labels and embeddings.
        """
        self.logger.info(f"Starting metadata interpretation for dataset: {dataset_id}")

        try:
            fields_metadata = []

            # Process each field
            for col in df.columns:
                field_meta = self._interpret_field(dataset_id, col, df[col])
                fields_metadata.append(field_meta)
                self.logger.debug(f"Interpreted field: {col} -> {field_meta.semantic_label}")

            # Generate embeddings in batch
            field_names = [f.field_name for f in fields_metadata]
            embeddings = self._generate_embeddings(field_names, df)

            for i, field_meta in enumerate(fields_metadata):
                field_meta.embedding = embeddings[i]

            # Generate dataset description
            dataset_description = self._generate_dataset_description(
                dataset_id, df, fields_metadata
            )

            # Infer domain tags
            domain_tags = self._infer_domain_tags(fields_metadata, dataset_description)

            result = DatasetMetadata(
                dataset_id=dataset_id,
                fields=fields_metadata,
                dataset_description=dataset_description,
                domain_tags=domain_tags
            )

            # Publish event
            self.publish(self.event_type, result.model_dump())

            self.logger.info(
                f"Metadata interpretation complete: {len(fields_metadata)} fields, "
                f"domains: {domain_tags}"
            )

            return result

        except Exception as e:
            self.handle_error(e, {"dataset_id": dataset_id})
            raise

    def _interpret_field(
        self,
        dataset_id: str,
        field_name: str,
        series: pd.Series
    ) -> FieldMetadata:
        """Interpret a single field.

        Args:
            dataset_id: Dataset identifier.
            field_name: Name of the field.
            series: Pandas Series with field data.

        Returns:
            FieldMetadata for the field.
        """
        # Compute basic statistics
        null_ratio = series.isna().sum() / len(series) if len(series) > 0 else 0
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0

        # Get sample values
        non_null = series.dropna()
        sample_values = non_null.head(self._config.max_sample_values).tolist()

        # Infer data type
        data_type = self._infer_data_type(series)

        # Use LLM to infer semantics
        semantic_info = self._llm_infer_semantics(
            field_name, data_type, sample_values, null_ratio, unique_ratio
        )

        return FieldMetadata(
            dataset_id=dataset_id,
            field_name=field_name,
            semantic_label=semantic_info.get("semantic_label", field_name),
            description=semantic_info.get("description", f"Field: {field_name}"),
            data_type=data_type,
            semantic_type=SemanticType(semantic_info.get("semantic_type", "unknown")),
            sample_values=sample_values[:5],
            null_ratio=null_ratio,
            unique_ratio=unique_ratio,
            confidence=semantic_info.get("confidence", 0.5)
        )

    def _infer_data_type(self, series: pd.Series) -> DataType:
        """Infer data type from pandas series.

        Handles both classic object dtype and pandas 2.x StringDtype.

        Args:
            series: Pandas Series.

        Returns:
            DataType enum value.
        """
        dtype = series.dtype

        if pd.api.types.is_integer_dtype(dtype):
            return DataType.INTEGER
        elif pd.api.types.is_float_dtype(dtype):
            return DataType.FLOAT
        elif pd.api.types.is_bool_dtype(dtype):
            return DataType.BOOLEAN
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return DataType.DATETIME
        elif pd.api.types.is_string_dtype(dtype):
            # Covers object, StringDtype, and str dtype (pandas 2.x+)
            sample = series.dropna().head(10)
            try:
                pd.to_datetime(sample)
                return DataType.DATETIME
            except (ValueError, TypeError):
                pass
            return DataType.STRING
        else:
            return DataType.OBJECT

    def _llm_infer_semantics(
        self,
        field_name: str,
        data_type: DataType,
        sample_values: List[Any],
        null_ratio: float,
        unique_ratio: float
    ) -> Dict[str, Any]:
        """Use LLM to infer semantic meaning.

        Args:
            field_name: Name of the field.
            data_type: Inferred data type.
            sample_values: Sample values from the field.
            null_ratio: Proportion of null values.
            unique_ratio: Cardinality ratio.

        Returns:
            Dictionary with semantic_label, description, semantic_type, confidence.
        """
        prompt = f"""Given the following field information from a dataset:

Field Name: {field_name}
Data Type: {data_type.value}
Sample Values: {sample_values[:10]}
Null Ratio: {null_ratio:.2%}
Unique Ratio: {unique_ratio:.2%}

Infer the semantic meaning of this field. Respond with:
1. semantic_label: A concise label (2-4 words)
2. description: A one-sentence explanation
3. semantic_type: One of [identifier, metric, category, timestamp, text, unknown]
4. confidence: Your certainty (0.0-1.0)

Respond in JSON format only."""

        try:
            provider = self._get_provider()
            response_text = provider.complete(
                messages=[
                    {"role": "system", "content": "You are a data analyst expert at understanding dataset schemas."},
                    {"role": "user", "content": prompt}
                ],
                model=self._config.llm_model,
                temperature=self._config.llm_temperature,
                response_format={"type": "json_object"},
            )

            result = json.loads(response_text)

            # Validate semantic_type
            valid_types = [t.value for t in SemanticType]
            if result.get("semantic_type") not in valid_types:
                result["semantic_type"] = "unknown"

            return result

        except Exception as e:
            self.logger.warning(f"LLM inference failed for {field_name}: {e}")
            # Fallback to heuristics
            return self._heuristic_inference(field_name, data_type, unique_ratio)

    def _heuristic_inference(
        self,
        field_name: str,
        data_type: DataType,
        unique_ratio: float
    ) -> Dict[str, Any]:
        """Fallback heuristic-based inference.

        Args:
            field_name: Name of the field.
            data_type: Data type.
            unique_ratio: Cardinality ratio.

        Returns:
            Semantic inference dictionary.
        """
        name_lower = field_name.lower()

        # Identifier patterns
        if any(p in name_lower for p in ["id", "key", "code", "uuid"]):
            return {
                "semantic_label": "Identifier",
                "description": f"Unique identifier field: {field_name}",
                "semantic_type": "identifier",
                "confidence": 0.7
            }

        # Timestamp patterns
        if any(p in name_lower for p in ["date", "time", "created", "updated", "timestamp"]):
            return {
                "semantic_label": "Timestamp",
                "description": f"Date/time field: {field_name}",
                "semantic_type": "timestamp",
                "confidence": 0.7
            }

        # Metric patterns
        if data_type in (DataType.INTEGER, DataType.FLOAT):
            if any(p in name_lower for p in ["count", "amount", "total", "sum", "avg", "price", "cost"]):
                return {
                    "semantic_label": "Numeric Metric",
                    "description": f"Numeric measurement: {field_name}",
                    "semantic_type": "metric",
                    "confidence": 0.6
                }

        # Category patterns
        if unique_ratio < 0.1:
            return {
                "semantic_label": "Category",
                "description": f"Categorical field: {field_name}",
                "semantic_type": "category",
                "confidence": 0.5
            }

        # Default to text
        return {
            "semantic_label": field_name.replace("_", " ").title(),
            "description": f"Field: {field_name}",
            "semantic_type": "text" if data_type == DataType.STRING else "unknown",
            "confidence": 0.3
        }

    def _generate_embeddings(
        self,
        field_names: List[str],
        df: pd.DataFrame
    ) -> List[List[float]]:
        """Generate embeddings for fields.

        Args:
            field_names: List of field names.
            df: DataFrame with data.

        Returns:
            List of embedding vectors.
        """
        # Create text representations for each field
        texts = []
        for name in field_names:
            series = df[name]
            samples = series.dropna().head(5).astype(str).tolist()
            text = f"{name}: {', '.join(samples)}"
            texts.append(text)

        try:
            provider = self._get_provider()
            return provider.embed(
                texts,
                model=self._config.embedding_model,
                batch_size=self._config.batch_size,
                dimensions=self._config.embedding_dimensions,
            )
        except Exception as e:
            self.logger.warning(f"Embedding generation failed: {e}")
            dim = self._config.embedding_dimensions
            return [[0.0] * dim for _ in field_names]

    def _generate_dataset_description(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        fields: List[FieldMetadata]
    ) -> str:
        """Generate a natural language description of the dataset.

        Args:
            dataset_id: Dataset identifier.
            df: DataFrame.
            fields: List of field metadata.

        Returns:
            Dataset description string.
        """
        field_summary = ", ".join([f.semantic_label for f in fields[:10]])

        prompt = f"""Describe this dataset in 2-3 sentences:

Dataset ID: {dataset_id}
Rows: {len(df)}
Fields: {field_summary}

Field details:
{chr(10).join([f"- {f.field_name}: {f.description}" for f in fields[:15]])}

Write a concise description focusing on what data this dataset contains and its potential use."""

        try:
            provider = self._get_provider()
            return provider.complete(
                messages=[
                    {"role": "system", "content": "You are a data documentation expert."},
                    {"role": "user", "content": prompt}
                ],
                model=self._config.llm_model,
                temperature=self._config.llm_temperature,
                max_tokens=200,
            ).strip()
        except Exception as e:
            self.logger.warning(f"Dataset description generation failed: {e}")
            return f"Dataset with {len(df)} rows and {len(fields)} fields including: {field_summary}"

    def _infer_domain_tags(
        self,
        fields: List[FieldMetadata],
        description: str
    ) -> List[str]:
        """Infer domain tags from field metadata.

        Args:
            fields: List of field metadata.
            description: Dataset description.

        Returns:
            List of domain tags.
        """
        # Combine field names and labels
        text = " ".join([f.field_name + " " + f.semantic_label for f in fields])
        text += " " + description

        text_lower = text.lower()

        domains = []

        # Domain detection rules
        domain_keywords = {
            "healthcare": ["patient", "diagnosis", "treatment", "medical", "hospital", "drug", "clinical"],
            "genomics": ["gene", "dna", "rna", "sequence", "mutation", "variant", "expression"],
            "finance": ["price", "revenue", "profit", "transaction", "account", "payment", "currency"],
            "ecommerce": ["product", "order", "customer", "cart", "shipping", "inventory"],
            "marketing": ["campaign", "conversion", "click", "impression", "engagement"],
            "hr": ["employee", "salary", "department", "hire", "performance"],
            "logistics": ["shipment", "delivery", "warehouse", "route", "tracking"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in text_lower for kw in keywords):
                domains.append(domain)

        return domains[:3] if domains else ["general"]
