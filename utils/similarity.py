"""Similarity utility for HelixForge.

Provides functions for computing various types of similarity
between strings, records, and semantic entities.
"""

from typing import Any, Dict, List, Optional, Tuple

from fuzzywuzzy import fuzz

from utils.embeddings import cosine_similarity


def string_similarity(
    a: str,
    b: str,
    method: str = "levenshtein"
) -> float:
    """Compute string similarity using various methods.

    Args:
        a: First string.
        b: Second string.
        method: Similarity method ('levenshtein', 'token_sort', 'token_set', 'partial').

    Returns:
        Similarity score between 0.0 and 1.0.

    Raises:
        ValueError: If method is unknown.
    """
    if not a or not b:
        return 0.0

    a = a.strip().lower()
    b = b.strip().lower()

    if a == b:
        return 1.0

    methods = {
        "levenshtein": fuzz.ratio,
        "token_sort": fuzz.token_sort_ratio,
        "token_set": fuzz.token_set_ratio,
        "partial": fuzz.partial_ratio,
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")

    # fuzz returns 0-100, normalize to 0-1
    return methods[method](a, b) / 100.0


def semantic_similarity(
    a: str,
    b: str,
    embeddings_cache: Optional[Dict[str, List[float]]] = None
) -> float:
    """Compute semantic similarity using embeddings.

    Args:
        a: First text.
        b: Second text.
        embeddings_cache: Optional pre-computed embeddings cache.

    Returns:
        Cosine similarity score between -1.0 and 1.0 (typically 0.0 to 1.0).
    """
    from utils.embeddings import get_embedding

    if embeddings_cache is None:
        embeddings_cache = {}

    # Get or compute embeddings
    if a not in embeddings_cache:
        embeddings_cache[a] = get_embedding(a)
    if b not in embeddings_cache:
        embeddings_cache[b] = get_embedding(b)

    return cosine_similarity(embeddings_cache[a], embeddings_cache[b])


def record_similarity(
    row_a: Dict[str, Any],
    row_b: Dict[str, Any],
    field_weights: Optional[Dict[str, float]] = None,
    method: str = "levenshtein"
) -> float:
    """Compute similarity between two records.

    Args:
        row_a: First record (dict of field -> value).
        row_b: Second record.
        field_weights: Optional weights for each field.
        method: String similarity method for text fields.

    Returns:
        Weighted average similarity score between 0.0 and 1.0.
    """
    # Find common fields
    common_fields = set(row_a.keys()) & set(row_b.keys())
    if not common_fields:
        return 0.0

    # Default to equal weights
    if field_weights is None:
        field_weights = {f: 1.0 / len(common_fields) for f in common_fields}
    else:
        # Filter to common fields and normalize
        field_weights = {f: w for f, w in field_weights.items() if f in common_fields}
        total = sum(field_weights.values())
        if total > 0:
            field_weights = {f: w / total for f, w in field_weights.items()}
        else:
            field_weights = {f: 1.0 / len(common_fields) for f in common_fields}

    total_sim = 0.0
    total_weight = 0.0

    for field in common_fields:
        weight = field_weights.get(field, 0.0)
        if weight == 0:
            continue

        val_a = row_a[field]
        val_b = row_b[field]

        # Handle None values
        if val_a is None or val_b is None:
            if val_a is None and val_b is None:
                sim = 1.0
            else:
                sim = 0.0
        # Compare by type
        elif isinstance(val_a, str) and isinstance(val_b, str):
            sim = string_similarity(val_a, val_b, method)
        elif isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            # Numeric similarity based on relative difference
            if val_a == val_b:
                sim = 1.0
            elif val_a == 0 and val_b == 0:
                sim = 1.0
            else:
                # Use epsilon to prevent division by zero for very small values
                max_val = max(abs(val_a), abs(val_b), 1e-10)
                sim = 1.0 - min(abs(val_a - val_b) / max_val, 1.0)
        elif isinstance(val_a, bool) and isinstance(val_b, bool):
            sim = 1.0 if val_a == val_b else 0.0
        else:
            # Convert to string and compare
            sim = string_similarity(str(val_a), str(val_b), method)

        total_sim += sim * weight
        total_weight += weight

    return total_sim / total_weight if total_weight > 0 else 0.0


def find_best_match(
    record: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    threshold: float = 0.7,
    field_weights: Optional[Dict[str, float]] = None
) -> Optional[Tuple[int, Dict[str, Any], float]]:
    """Find the best matching record from candidates.

    Args:
        record: Record to match.
        candidates: List of candidate records.
        threshold: Minimum similarity threshold.
        field_weights: Optional weights for each field.

    Returns:
        Tuple of (index, matched_record, similarity) or None if no match.
    """
    best_match = None
    best_sim = threshold
    best_idx = -1

    for i, candidate in enumerate(candidates):
        sim = record_similarity(record, candidate, field_weights)
        if sim > best_sim:
            best_sim = sim
            best_match = candidate
            best_idx = i

    if best_match is not None:
        return (best_idx, best_match, best_sim)
    return None


def find_all_matches(
    record: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    threshold: float = 0.7,
    max_matches: int = 5,
    field_weights: Optional[Dict[str, float]] = None
) -> List[Tuple[int, Dict[str, Any], float]]:
    """Find all matching records above threshold.

    Args:
        record: Record to match.
        candidates: List of candidate records.
        threshold: Minimum similarity threshold.
        max_matches: Maximum number of matches to return.
        field_weights: Optional weights for each field.

    Returns:
        List of (index, matched_record, similarity) tuples, sorted by similarity.
    """
    matches = []

    for i, candidate in enumerate(candidates):
        sim = record_similarity(record, candidate, field_weights)
        if sim >= threshold:
            matches.append((i, candidate, sim))

    # Sort by similarity descending
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches[:max_matches]


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets.

    Args:
        set_a: First set.
        set_b: Second set.

    Returns:
        Jaccard similarity between 0.0 and 1.0.
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def dice_coefficient(set_a: set, set_b: set) -> float:
    """Compute Dice coefficient between two sets.

    Args:
        set_a: First set.
        set_b: Second set.

    Returns:
        Dice coefficient between 0.0 and 1.0.
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    return 2 * intersection / (len(set_a) + len(set_b))


def ngram_similarity(
    a: str,
    b: str,
    n: int = 2
) -> float:
    """Compute n-gram similarity between two strings.

    Args:
        a: First string.
        b: Second string.
        n: Size of n-grams.

    Returns:
        Jaccard similarity of n-gram sets.
    """
    def get_ngrams(s: str, n: int) -> set:
        s = s.lower().strip()
        if len(s) < n:
            return {s}
        return {s[i:i+n] for i in range(len(s) - n + 1)}

    ngrams_a = get_ngrams(a, n)
    ngrams_b = get_ngrams(b, n)
    return jaccard_similarity(ngrams_a, ngrams_b)
