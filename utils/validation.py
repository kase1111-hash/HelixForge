"""Validation utility for HelixForge.

Provides input validation and sanitization functions
for security and data integrity.
"""

import os
import re
from typing import Any, List, Optional, Set
from urllib.parse import urlparse


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_file_path(
    path: str,
    allowed_extensions: Optional[Set[str]] = None,
    max_size_mb: Optional[int] = None,
    must_exist: bool = True
) -> str:
    """Validate a file path for security and existence.

    Args:
        path: File path to validate.
        allowed_extensions: Set of allowed file extensions (e.g., {'.csv', '.json'}).
        max_size_mb: Maximum file size in megabytes.
        must_exist: If True, file must exist.

    Returns:
        Validated absolute path.

    Raises:
        ValidationError: If validation fails.
    """
    if not path:
        raise ValidationError("File path cannot be empty")

    # Check for path traversal attempts BEFORE normalization
    # This prevents bypass via paths like "foo/../../../etc/passwd"
    if ".." in path:
        raise ValidationError("Path traversal not allowed")

    # Normalize path
    path = os.path.normpath(path)

    # Double-check after normalization (belt and suspenders)
    if ".." in path:
        raise ValidationError("Path traversal not allowed")

    # Check extension
    if allowed_extensions:
        ext = os.path.splitext(path)[1].lower()
        if ext not in allowed_extensions:
            raise ValidationError(
                f"File extension '{ext}' not allowed. Allowed: {allowed_extensions}"
            )

    # Check existence
    if must_exist and not os.path.exists(path):
        raise ValidationError(f"File does not exist: {path}")

    # Check size
    if must_exist and max_size_mb and os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValidationError(
                f"File too large: {size_mb:.2f}MB > {max_size_mb}MB"
            )

    return os.path.abspath(path)


def validate_url(
    url: str,
    allowed_schemes: Optional[Set[str]] = None,
    allowed_hosts: Optional[Set[str]] = None
) -> str:
    """Validate a URL for security.

    Args:
        url: URL to validate.
        allowed_schemes: Allowed URL schemes (default: {'http', 'https'}).
        allowed_hosts: Optional whitelist of allowed hosts.

    Returns:
        Validated URL string.

    Raises:
        ValidationError: If validation fails.
    """
    if not url:
        raise ValidationError("URL cannot be empty")

    if allowed_schemes is None:
        allowed_schemes = {"http", "https"}

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValidationError(f"Invalid URL format: {e}")

    if not parsed.scheme:
        raise ValidationError("URL must have a scheme (http/https)")

    if parsed.scheme not in allowed_schemes:
        raise ValidationError(
            f"URL scheme '{parsed.scheme}' not allowed. Allowed: {allowed_schemes}"
        )

    if not parsed.netloc:
        raise ValidationError("URL must have a host")

    if allowed_hosts and parsed.netloc not in allowed_hosts:
        raise ValidationError(f"Host '{parsed.netloc}' not in allowed list")

    # Block localhost/internal IPs in production
    host = parsed.netloc.split(":")[0].lower()
    blocked_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
    if host in blocked_hosts:
        raise ValidationError(f"Internal hosts not allowed: {host}")

    return url


def validate_sql_identifier(identifier: str) -> str:
    """Validate a SQL identifier (table/column name).

    Args:
        identifier: SQL identifier to validate.

    Returns:
        Validated identifier.

    Raises:
        ValidationError: If identifier is invalid.
    """
    if not identifier:
        raise ValidationError("SQL identifier cannot be empty")

    # Only allow alphanumeric and underscore
    pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
    if not re.match(pattern, identifier):
        raise ValidationError(
            f"Invalid SQL identifier: '{identifier}'. "
            "Must start with letter/underscore, contain only alphanumeric/underscore."
        )

    # Check length
    if len(identifier) > 128:
        raise ValidationError("SQL identifier too long (max 128 characters)")

    # Check for SQL keywords
    sql_keywords = {
        "select", "insert", "update", "delete", "drop", "create", "alter",
        "truncate", "exec", "execute", "union", "join", "where", "from",
        "table", "database", "index", "grant", "revoke"
    }
    if identifier.lower() in sql_keywords:
        raise ValidationError(f"SQL keyword not allowed as identifier: {identifier}")

    return identifier


def sanitize_string(
    value: str,
    max_length: int = 1000,
    allow_html: bool = False,
    allow_newlines: bool = True
) -> str:
    """Sanitize a string input.

    Args:
        value: String to sanitize.
        max_length: Maximum allowed length.
        allow_html: If False, escape HTML characters.
        allow_newlines: If False, replace newlines with spaces.

    Returns:
        Sanitized string.
    """
    if not value:
        return ""

    # Truncate
    if len(value) > max_length:
        value = value[:max_length]

    # Remove null bytes
    value = value.replace("\x00", "")

    # Handle newlines
    if not allow_newlines:
        value = re.sub(r"[\r\n]+", " ", value)

    # Escape HTML if needed
    if not allow_html:
        html_escape = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#x27;",
        }
        for char, escape in html_escape.items():
            value = value.replace(char, escape)

    return value.strip()


def validate_dataset_id(dataset_id: str) -> str:
    """Validate a dataset ID.

    Args:
        dataset_id: Dataset ID to validate.

    Returns:
        Validated dataset ID.

    Raises:
        ValidationError: If invalid.
    """
    if not dataset_id:
        raise ValidationError("Dataset ID cannot be empty")

    # Allow UUID format or alphanumeric with dashes/underscores
    pattern = r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$"
    if not re.match(pattern, dataset_id):
        raise ValidationError(
            f"Invalid dataset ID: '{dataset_id}'. "
            "Must be alphanumeric with dashes/underscores, max 64 chars."
        )

    return dataset_id


def validate_field_name(field_name: str) -> str:
    """Validate a field/column name.

    Args:
        field_name: Field name to validate.

    Returns:
        Validated field name.

    Raises:
        ValidationError: If invalid.
    """
    if not field_name:
        raise ValidationError("Field name cannot be empty")

    # Allow most characters but limit length
    if len(field_name) > 256:
        raise ValidationError("Field name too long (max 256 characters)")

    # Remove control characters
    field_name = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", field_name)

    return field_name.strip()


def validate_numeric_range(
    value: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "value"
) -> float:
    """Validate a numeric value is within range.

    Args:
        value: Value to validate.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.
        name: Name of the value for error messages.

    Returns:
        Validated value.

    Raises:
        ValidationError: If out of range.
    """
    if min_val is not None and value < min_val:
        raise ValidationError(f"{name} must be >= {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValidationError(f"{name} must be <= {max_val}, got {value}")
    return value


def validate_list_length(
    items: List[Any],
    min_length: int = 0,
    max_length: Optional[int] = None,
    name: str = "list"
) -> List[Any]:
    """Validate list length.

    Args:
        items: List to validate.
        min_length: Minimum required length.
        max_length: Maximum allowed length.
        name: Name of the list for error messages.

    Returns:
        Validated list.

    Raises:
        ValidationError: If length invalid.
    """
    if len(items) < min_length:
        raise ValidationError(f"{name} must have at least {min_length} items")
    if max_length is not None and len(items) > max_length:
        raise ValidationError(f"{name} cannot have more than {max_length} items")
    return items
