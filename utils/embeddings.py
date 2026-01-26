"""Embeddings utility for HelixForge.

Provides functions for generating and working with text embeddings
using OpenAI's embedding models.
"""

import math
from typing import List, Optional, Tuple

import numpy as np

# OpenAI client will be lazily initialized
_openai_client = None


def _get_openai_client():
    """Lazily initialize OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client


def get_embedding(
    text: str,
    model: str = "text-embedding-3-large"
) -> List[float]:
    """Generate embedding for a single text.

    Args:
        text: Text to embed.
        model: Embedding model to use.

    Returns:
        List of floats representing the embedding vector.

    Raises:
        ValueError: If text is empty.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    client = _get_openai_client()
    response = client.embeddings.create(
        input=text.strip(),
        model=model
    )
    return response.data[0].embedding


def batch_embed(
    texts: List[str],
    model: str = "text-embedding-3-large",
    batch_size: int = 100,
    embedding_dimensions: int = 3072
) -> List[List[float]]:
    """Generate embeddings for multiple texts.

    Args:
        texts: List of texts to embed.
        model: Embedding model to use.
        batch_size: Maximum texts per API call.
        embedding_dimensions: Dimension of embeddings (for zero vectors on empty texts).

    Returns:
        List of embedding vectors (same length as input texts).
        Empty/whitespace-only texts will have zero vectors.

    Raises:
        ValueError: If texts list is empty.
    """
    if not texts:
        raise ValueError("Texts list cannot be empty")

    client = _get_openai_client()

    # Initialize results with None placeholders to maintain index alignment
    all_embeddings: List[Optional[List[float]]] = [None] * len(texts)

    # Track which indices have valid (non-empty) texts
    valid_indices = []
    valid_texts = []
    for i, text in enumerate(texts):
        stripped = text.strip() if text else ""
        if stripped:
            valid_indices.append(i)
            valid_texts.append(stripped)

    # Process valid texts in batches
    for batch_start in range(0, len(valid_texts), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_texts))
        batch = valid_texts[batch_start:batch_end]
        batch_indices = valid_indices[batch_start:batch_end]

        response = client.embeddings.create(
            input=batch,
            model=model
        )

        # Sort by index to maintain order within batch
        sorted_data = sorted(response.data, key=lambda x: x.index)

        # Map embeddings back to original indices
        for j, data in enumerate(sorted_data):
            original_idx = batch_indices[j]
            all_embeddings[original_idx] = data.embedding

    # Fill remaining None entries with zero vectors (for empty/whitespace texts)
    zero_vector = [0.0] * embedding_dimensions
    for i in range(len(all_embeddings)):
        if all_embeddings[i] is None:
            all_embeddings[i] = zero_vector

    return all_embeddings


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity score between -1.0 and 1.0.

    Raises:
        ValueError: If vectors have different dimensions or are zero vectors.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vectors must have same dimensions: {len(vec_a)} != {len(vec_b)}"
        )

    # Convert to numpy for efficiency
    a = np.array(vec_a)
    b = np.array(vec_b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cannot compute similarity for zero vectors")

    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean_distance(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute Euclidean distance between two vectors.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Euclidean distance (non-negative float).

    Raises:
        ValueError: If vectors have different dimensions.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vectors must have same dimensions: {len(vec_a)} != {len(vec_b)}"
        )

    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.linalg.norm(a - b))


def find_similar(
    query_vec: List[float],
    corpus: List[List[float]],
    top_k: int = 5,
    threshold: Optional[float] = None
) -> List[Tuple[int, float]]:
    """Find most similar vectors in a corpus.

    Args:
        query_vec: Query embedding vector.
        corpus: List of embedding vectors to search.
        top_k: Number of results to return.
        threshold: Optional minimum similarity threshold.

    Returns:
        List of (index, similarity) tuples, sorted by similarity descending.
    """
    if not corpus:
        return []

    results = []
    for i, vec in enumerate(corpus):
        try:
            sim = cosine_similarity(query_vec, vec)
            if threshold is None or sim >= threshold:
                results.append((i, sim))
        except ValueError:
            continue

    # Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def normalize_vector(vec: List[float]) -> List[float]:
    """Normalize a vector to unit length.

    Args:
        vec: Input vector.

    Returns:
        Normalized vector with L2 norm of 1.

    Raises:
        ValueError: If vector is zero vector.
    """
    arr = np.array(vec)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Cannot normalize zero vector")
    return (arr / norm).tolist()


def average_embeddings(embeddings: List[List[float]]) -> List[float]:
    """Compute average of multiple embeddings.

    Args:
        embeddings: List of embedding vectors.

    Returns:
        Average embedding vector.

    Raises:
        ValueError: If embeddings list is empty or dimensions don't match.
    """
    if not embeddings:
        raise ValueError("Embeddings list cannot be empty")

    first_dim = len(embeddings[0])
    for i, emb in enumerate(embeddings[1:], 1):
        if len(emb) != first_dim:
            raise ValueError(f"Embedding {i} has different dimension: {len(emb)} != {first_dim}")

    arr = np.array(embeddings)
    return np.mean(arr, axis=0).tolist()


def combine_embeddings(
    embeddings: List[List[float]],
    weights: Optional[List[float]] = None
) -> List[float]:
    """Combine multiple embeddings with optional weights.

    Args:
        embeddings: List of embedding vectors.
        weights: Optional weights for each embedding (must sum to 1).

    Returns:
        Combined embedding vector.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not embeddings:
        raise ValueError("Embeddings list cannot be empty")

    if weights is None:
        weights = [1.0 / len(embeddings)] * len(embeddings)

    if len(weights) != len(embeddings):
        raise ValueError("Weights must match number of embeddings")

    if not math.isclose(sum(weights), 1.0, rel_tol=1e-5):
        raise ValueError("Weights must sum to 1")

    first_dim = len(embeddings[0])
    result = np.zeros(first_dim)

    for emb, weight in zip(embeddings, weights):
        if len(emb) != first_dim:
            raise ValueError("All embeddings must have same dimension")
        result += np.array(emb) * weight

    return result.tolist()
