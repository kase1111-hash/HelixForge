"""Unit tests for utility modules."""

import math
import os
import tempfile

import pytest

from utils.validation import (
    ValidationError,
    sanitize_string,
    validate_dataset_id,
    validate_field_name,
    validate_file_path,
    validate_list_length,
    validate_numeric_range,
    validate_sql_identifier,
    validate_url,
)


class TestValidation:
    """Tests for validation utilities."""

    def test_validate_file_path_valid(self, tmp_path):
        """Test valid file path validation."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("test content")

        result = validate_file_path(
            str(test_file),
            allowed_extensions={".csv"},
            must_exist=True
        )
        assert result == str(test_file.absolute())

    def test_validate_file_path_invalid_extension(self, tmp_path):
        """Test file path with invalid extension."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(ValidationError, match="extension.*not allowed"):
            validate_file_path(
                str(test_file),
                allowed_extensions={".csv", ".json"},
                must_exist=True
            )

    def test_validate_file_path_not_exists(self):
        """Test file path that doesn't exist."""
        with pytest.raises(ValidationError, match="does not exist"):
            validate_file_path("/nonexistent/file.csv", must_exist=True)

    def test_validate_file_path_traversal_blocked(self):
        """Test that path traversal is blocked."""
        with pytest.raises(ValidationError, match="traversal"):
            validate_file_path("../../../etc/passwd", must_exist=False)

    def test_validate_file_path_empty(self):
        """Test empty file path."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_file_path("")

    def test_validate_url_valid(self):
        """Test valid URL validation."""
        url = "https://api.example.com/data"
        result = validate_url(url)
        assert result == url

    def test_validate_url_invalid_scheme(self):
        """Test URL with invalid scheme."""
        with pytest.raises(ValidationError, match="scheme.*not allowed"):
            validate_url("ftp://example.com")

    def test_validate_url_localhost_blocked(self):
        """Test that localhost URLs are blocked."""
        with pytest.raises(ValidationError, match="Internal hosts"):
            validate_url("http://localhost:8080")

    def test_validate_url_empty(self):
        """Test empty URL."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_url("")

    def test_validate_sql_identifier_valid(self):
        """Test valid SQL identifier."""
        assert validate_sql_identifier("column_name") == "column_name"
        assert validate_sql_identifier("Table1") == "Table1"
        assert validate_sql_identifier("_private") == "_private"

    def test_validate_sql_identifier_invalid_chars(self):
        """Test SQL identifier with invalid characters."""
        with pytest.raises(ValidationError, match="Invalid SQL identifier"):
            validate_sql_identifier("column-name")

        with pytest.raises(ValidationError, match="Invalid SQL identifier"):
            validate_sql_identifier("123column")

    def test_validate_sql_identifier_keyword(self):
        """Test SQL keyword is rejected."""
        with pytest.raises(ValidationError, match="keyword"):
            validate_sql_identifier("SELECT")

    def test_sanitize_string_basic(self):
        """Test basic string sanitization."""
        result = sanitize_string("  Hello World  ")
        assert result == "Hello World"

    def test_sanitize_string_html_escape(self):
        """Test HTML character escaping."""
        result = sanitize_string("<script>alert('xss')</script>", allow_html=False)
        assert "<" not in result
        assert ">" not in result

    def test_sanitize_string_max_length(self):
        """Test string truncation."""
        long_string = "a" * 2000
        result = sanitize_string(long_string, max_length=100)
        assert len(result) == 100

    def test_sanitize_string_null_bytes(self):
        """Test null byte removal."""
        result = sanitize_string("hello\x00world")
        assert "\x00" not in result

    def test_validate_dataset_id_valid(self):
        """Test valid dataset ID."""
        assert validate_dataset_id("dataset-001") == "dataset-001"
        assert validate_dataset_id("my_dataset_v2") == "my_dataset_v2"

    def test_validate_dataset_id_invalid(self):
        """Test invalid dataset ID."""
        with pytest.raises(ValidationError):
            validate_dataset_id("")

        with pytest.raises(ValidationError):
            validate_dataset_id("a" * 100)  # Too long

    def test_validate_field_name_valid(self):
        """Test valid field name."""
        assert validate_field_name("column_name") == "column_name"
        assert validate_field_name("Column With Spaces") == "Column With Spaces"

    def test_validate_field_name_empty(self):
        """Test empty field name."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_field_name("")

    def test_validate_numeric_range_valid(self):
        """Test valid numeric range."""
        assert validate_numeric_range(5.0, min_val=0, max_val=10) == 5.0
        assert validate_numeric_range(0.0, min_val=0) == 0.0

    def test_validate_numeric_range_invalid(self):
        """Test invalid numeric range."""
        with pytest.raises(ValidationError, match="must be >="):
            validate_numeric_range(-1, min_val=0)

        with pytest.raises(ValidationError, match="must be <="):
            validate_numeric_range(15, max_val=10)

    def test_validate_list_length_valid(self):
        """Test valid list length."""
        items = [1, 2, 3]
        result = validate_list_length(items, min_length=1, max_length=5)
        assert result == items

    def test_validate_list_length_invalid(self):
        """Test invalid list length."""
        with pytest.raises(ValidationError, match="at least"):
            validate_list_length([], min_length=1)

        with pytest.raises(ValidationError, match="cannot have more"):
            validate_list_length([1, 2, 3], max_length=2)


class TestSimilarity:
    """Tests for similarity utilities."""

    def test_string_similarity_identical(self):
        """Test similarity of identical strings."""
        from utils.similarity import string_similarity

        result = string_similarity("hello", "hello")
        assert result == 1.0

    def test_string_similarity_different(self):
        """Test similarity of different strings."""
        from utils.similarity import string_similarity

        result = string_similarity("hello", "world")
        assert 0 <= result < 1

    def test_string_similarity_similar(self):
        """Test similarity of similar strings."""
        from utils.similarity import string_similarity

        result = string_similarity("employee_id", "emp_id")
        assert result > 0.5

    def test_string_similarity_empty(self):
        """Test similarity with empty strings."""
        from utils.similarity import string_similarity

        result = string_similarity("", "hello")
        assert result == 0.0

    def test_string_similarity_methods(self):
        """Test different similarity methods."""
        from utils.similarity import string_similarity

        s1 = "annual salary"
        s2 = "salary annual"

        levenshtein = string_similarity(s1, s2, method="levenshtein")
        token_sort = string_similarity(s1, s2, method="token_sort")

        # token_sort should be higher for reordered words
        assert token_sort > levenshtein

    def test_record_similarity_identical(self):
        """Test similarity of identical records."""
        from utils.similarity import record_similarity

        record = {"name": "Alice", "age": 30}
        result = record_similarity(record, record)
        assert result == 1.0

    def test_record_similarity_partial(self):
        """Test similarity of partially matching records."""
        from utils.similarity import record_similarity

        r1 = {"name": "Alice", "age": 30, "city": "NYC"}
        r2 = {"name": "Alice", "age": 31, "city": "LA"}

        result = record_similarity(r1, r2)
        assert 0 < result < 1

    def test_find_best_match(self):
        """Test finding best matching record."""
        from utils.similarity import find_best_match

        record = {"name": "Alice", "age": 30}
        candidates = [
            {"name": "Bob", "age": 25},
            {"name": "Alice", "age": 30},
            {"name": "Charlie", "age": 35}
        ]

        result = find_best_match(record, candidates, threshold=0.5)
        assert result is not None
        assert result[0] == 1  # Index of best match
        assert result[2] == 1.0  # Perfect match

    def test_jaccard_similarity(self):
        """Test Jaccard similarity."""
        from utils.similarity import jaccard_similarity

        set_a = {1, 2, 3, 4}
        set_b = {3, 4, 5, 6}

        result = jaccard_similarity(set_a, set_b)
        assert result == 2 / 6  # 2 common elements, 6 total unique

    def test_ngram_similarity(self):
        """Test n-gram similarity."""
        from utils.similarity import ngram_similarity

        result = ngram_similarity("hello", "hallo")
        assert 0 < result < 1


class TestEmbeddings:
    """Tests for embedding utilities."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        from utils.embeddings import cosine_similarity

        vec = [1.0, 0.0, 0.0]
        result = cosine_similarity(vec, vec)
        assert math.isclose(result, 1.0)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        from utils.embeddings import cosine_similarity

        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]

        result = cosine_similarity(vec_a, vec_b)
        assert math.isclose(result, 0.0)

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        from utils.embeddings import cosine_similarity

        vec_a = [1.0, 0.0]
        vec_b = [-1.0, 0.0]

        result = cosine_similarity(vec_a, vec_b)
        assert math.isclose(result, -1.0)

    def test_cosine_similarity_different_dimensions(self):
        """Test cosine similarity with different dimensions raises error."""
        from utils.embeddings import cosine_similarity

        with pytest.raises(ValueError, match="same dimensions"):
            cosine_similarity([1, 2, 3], [1, 2])

    def test_normalize_vector(self):
        """Test vector normalization."""
        from utils.embeddings import normalize_vector
        import numpy as np

        vec = [3.0, 4.0]
        result = normalize_vector(vec)

        # Check unit length
        assert math.isclose(np.linalg.norm(result), 1.0)

    def test_average_embeddings(self):
        """Test averaging embeddings."""
        from utils.embeddings import average_embeddings

        embeddings = [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0]
        ]

        result = average_embeddings(embeddings)
        assert result == [2.0, 3.0, 4.0]

    def test_find_similar(self):
        """Test finding similar vectors."""
        from utils.embeddings import find_similar

        query = [1.0, 0.0, 0.0]
        corpus = [
            [1.0, 0.0, 0.0],  # Same
            [0.9, 0.1, 0.0],  # Similar
            [0.0, 1.0, 0.0],  # Different
        ]

        results = find_similar(query, corpus, top_k=2)
        assert len(results) == 2
        assert results[0][0] == 0  # First result is identical vector
        assert results[0][1] == 1.0  # Perfect similarity


class TestLogging:
    """Tests for logging utilities."""

    def test_get_logger(self):
        """Test logger creation."""
        from utils.logging import get_logger

        logger = get_logger("test_module")
        assert logger is not None
        assert logger.name == "test_module"

    def test_get_correlated_logger(self):
        """Test correlated logger creation."""
        from utils.logging import get_correlated_logger

        logger = get_correlated_logger("test", "corr-123", "TestAgent")
        assert logger is not None

    def test_log_event(self, caplog):
        """Test event logging."""
        from utils.logging import get_logger, log_event

        logger = get_logger("test_event", json_format=False)
        log_event(logger, "test.event", {"key": "value"}, "corr-123")

        # Check that event was logged
        assert "test.event" in caplog.text or len(caplog.records) > 0
