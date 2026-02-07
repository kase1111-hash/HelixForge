"""LLM Provider abstraction for HelixForge.

Defines a protocol for LLM providers and implementations for
OpenAI (production) and a deterministic mock (testing).
"""

import hashlib
import json
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers.

    Any object implementing embed() and complete() can be used
    as an LLM provider throughout HelixForge.
    """

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed.
            **kwargs: Provider-specific options (model, batch_size, etc.).

        Returns:
            List of embedding vectors, one per input text.
        """
        ...

    def complete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Provider-specific options (model, temperature, etc.).

        Returns:
            The assistant's response text.
        """
        ...


class OpenAIProvider:
    """OpenAI LLM provider.

    Wraps the OpenAI Python client for embeddings and chat completions.
    Handles batching for embedding requests.
    """

    def __init__(self, client=None):
        self._client = client

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def embed(
        self,
        texts: List[str],
        model: str = "text-embedding-3-large",
        batch_size: int = 100,
        dimensions: int = 1536,
        **kwargs,
    ) -> List[List[float]]:
        """Generate embeddings via OpenAI API with batching."""
        if not texts:
            raise ValueError("Texts list cannot be empty")

        client = self._get_client()
        all_embeddings: List[Optional[List[float]]] = [None] * len(texts)

        valid_indices = []
        valid_texts = []
        for i, text in enumerate(texts):
            stripped = text.strip() if text else ""
            if stripped:
                valid_indices.append(i)
                valid_texts.append(stripped)

        for batch_start in range(0, len(valid_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_texts))
            batch = valid_texts[batch_start:batch_end]
            batch_indices = valid_indices[batch_start:batch_end]

            response = client.embeddings.create(input=batch, model=model)
            sorted_data = sorted(response.data, key=lambda x: x.index)

            for j, data in enumerate(sorted_data):
                original_idx = batch_indices[j]
                all_embeddings[original_idx] = data.embedding

        zero_vector = [0.0] * dimensions
        for i in range(len(all_embeddings)):
            if all_embeddings[i] is None:
                all_embeddings[i] = zero_vector

        return all_embeddings

    def complete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate chat completion via OpenAI API."""
        client = self._get_client()
        response = client.chat.completions.create(messages=messages, **kwargs)
        return response.choices[0].message.content


class MockProvider:
    """Deterministic mock LLM provider for testing.

    Generates embeddings via word-level hashing so that semantically
    related field names (sharing words) produce similar embeddings.
    Completions use heuristic rules instead of an LLM.
    """

    def __init__(self, dimensions: int = 1536):
        self._dimensions = dimensions
        self._word_vectors: Dict[str, np.ndarray] = {}

    def _get_word_vector(self, word: str) -> np.ndarray:
        """Get a deterministic unit vector for a word."""
        if word not in self._word_vectors:
            seed = int(hashlib.md5(word.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._dimensions)
            vec = vec / np.linalg.norm(vec)
            self._word_vectors[word] = vec
        return self._word_vectors[word]

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate deterministic embeddings based on word content.

        Words are tokenized by splitting on whitespace, underscores,
        and hyphens. Each word maps to a fixed random unit vector.
        The embedding is the normalized sum of word vectors.

        This means:
        - "employee_name" and "worker_name" share the "name" component
        - "employee_id" and "employee_name" share the "employee" component
        - Identical inputs always produce identical outputs
        """
        embeddings = []
        for text in texts:
            words = self._tokenize(text)
            if not words:
                embeddings.append([0.0] * self._dimensions)
                continue

            vec = np.zeros(self._dimensions)
            for word in words:
                vec += self._get_word_vector(word)

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            embeddings.append(vec.tolist())
        return embeddings

    def complete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Return deterministic completions using heuristic rules.

        Parses the user message to extract field information and
        returns a JSON response mimicking LLM semantic inference.
        """
        user_msg = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")

        # Try to extract field name from the prompt
        field_name = self._extract_field_name(user_msg)

        if field_name:
            result = self._heuristic_semantics(field_name, user_msg)
            return json.dumps(result)

        # Fallback: return a generic dataset description
        return "Dataset containing structured records for analysis."

    def _tokenize(self, text: str) -> List[str]:
        """Split text into normalized word tokens."""
        text = text.lower()
        for sep in ["_", "-", ":", ",", "."]:
            text = text.replace(sep, " ")
        return [w for w in text.split() if len(w) > 0]

    def _extract_field_name(self, prompt: str) -> Optional[str]:
        """Extract field name from an LLM prompt."""
        for line in prompt.split("\n"):
            line = line.strip()
            if line.startswith("Field Name:"):
                return line.split(":", 1)[1].strip()
        return None

    def _heuristic_semantics(self, field_name: str, prompt: str) -> Dict[str, Any]:
        """Infer semantic meaning from field name using rules."""
        name_lower = field_name.lower()

        # Extract data type from prompt if available
        data_type = "string"
        for line in prompt.split("\n"):
            if line.strip().startswith("Data Type:"):
                data_type = line.split(":", 1)[1].strip()

        # Identifier patterns
        id_keywords = ["id", "key", "code", "uuid", "identifier", "pk"]
        if any(kw in name_lower.split("_") for kw in id_keywords):
            return {
                "semantic_label": self._humanize(field_name) + " Identifier",
                "description": f"Unique identifier: {field_name}",
                "semantic_type": "identifier",
                "confidence": 0.85,
            }

        # Timestamp patterns
        time_keywords = ["date", "time", "created", "updated", "timestamp", "dob",
                         "born", "hired", "started", "ended", "expires"]
        if any(kw in name_lower for kw in time_keywords):
            return {
                "semantic_label": self._humanize(field_name),
                "description": f"Date/time field: {field_name}",
                "semantic_type": "timestamp",
                "confidence": 0.80,
            }

        # Metric patterns
        metric_keywords = ["count", "amount", "total", "sum", "avg", "price",
                           "cost", "salary", "revenue", "profit", "score",
                           "rate", "ratio", "percent", "weight", "height",
                           "age", "quantity", "budget", "income", "pay",
                           "compensation", "balance", "fee", "tax", "value"]
        if data_type in ("float", "integer") or any(kw in name_lower for kw in metric_keywords):
            if any(kw in name_lower for kw in metric_keywords):
                return {
                    "semantic_label": self._humanize(field_name),
                    "description": f"Numeric measurement: {field_name}",
                    "semantic_type": "metric",
                    "confidence": 0.75,
                }

        # Category patterns
        cat_keywords = ["type", "status", "category", "group", "class", "level",
                        "tier", "dept", "department", "team", "role", "gender",
                        "country", "state", "city", "region"]
        if any(kw in name_lower for kw in cat_keywords):
            return {
                "semantic_label": self._humanize(field_name),
                "description": f"Categorical field: {field_name}",
                "semantic_type": "category",
                "confidence": 0.70,
            }

        # Name / text patterns
        name_keywords = ["name", "title", "label", "description", "comment",
                         "note", "text", "email", "address", "phone", "url"]
        if any(kw in name_lower for kw in name_keywords):
            return {
                "semantic_label": self._humanize(field_name),
                "description": f"Text field: {field_name}",
                "semantic_type": "text",
                "confidence": 0.70,
            }

        # Default
        return {
            "semantic_label": self._humanize(field_name),
            "description": f"Field: {field_name}",
            "semantic_type": "unknown",
            "confidence": 0.40,
        }

    @staticmethod
    def _humanize(name: str) -> str:
        """Convert field_name to Human Name."""
        return name.replace("_", " ").replace("-", " ").title()
