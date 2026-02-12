# HelixForge Coding Conventions & Style Guide

## Overview

This document defines coding standards for the HelixForge project to ensure consistency, readability, and maintainability.

---

## Python Version

- **Minimum Version:** Python 3.10+
- **Target Version:** Python 3.11

---

## Code Style

### General Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code style
- Maximum line length: **120 characters** (configured in `pyproject.toml`)
- Use **4 spaces** for indentation (no tabs)
- Use **UTF-8** encoding for all source files

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Modules | lowercase_with_underscores | `data_ingestor_agent.py` |
| Classes | PascalCase | `DataIngestorAgent` |
| Functions | lowercase_with_underscores | `ingest_file()` |
| Variables | lowercase_with_underscores | `dataset_id` |
| Constants | UPPERCASE_WITH_UNDERSCORES | `MAX_FILE_SIZE_MB` |
| Private | Leading underscore | `_internal_method()` |
| Type Variables | PascalCase | `DatasetT` |

### Imports

Order imports in three groups, separated by blank lines:
1. Standard library imports
2. Third-party imports
3. Local application imports

```python
# Standard library
import os
from datetime import datetime
from typing import Dict, List, Optional

# Third-party
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Local
from agents.base_agent import BaseAgent
from utils.logging import get_logger
```

Use absolute imports for clarity:
```python
# Good
from agents.data_ingestor_agent import DataIngestorAgent

# Avoid
from .data_ingestor_agent import DataIngestorAgent
```

---

## Type Hints

### Requirements

- All public functions **must** have type hints
- All class attributes **must** have type annotations
- Use `typing` module for complex types

### Examples

```python
from typing import Dict, List, Optional, Union
from datetime import datetime

def ingest_file(
    file_path: str,
    encoding: Optional[str] = None,
    sample_size: int = 10
) -> Dict[str, Any]:
    """Ingest a file and return metadata."""
    ...

class IngestResult:
    dataset_id: str
    source: str
    row_count: int
    ingested_at: datetime
```

### Pydantic Models

Use Pydantic for all data schemas:

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class IngestResult(BaseModel):
    dataset_id: str = Field(..., description="Unique identifier")
    source: str = Field(..., description="Origin path/URL")
    source_type: str = Field(..., pattern="^(csv|parquet|json|sql|rest)$")
    schema_fields: List[str] = Field(..., alias="schema")
    row_count: int = Field(..., ge=0)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
```

---

## Documentation

### Docstrings

Use Google-style docstrings for all public modules, classes, and functions:

```python
def compute_similarity(
    vec_a: List[float],
    vec_b: List[float]
) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity score between 0.0 and 1.0.

    Raises:
        ValueError: If vectors have different dimensions.

    Example:
        >>> compute_similarity([1, 0], [0, 1])
        0.0
        >>> compute_similarity([1, 0], [1, 0])
        1.0
    """
    ...
```

### Module Docstrings

Every module should have a docstring explaining its purpose:

```python
"""Data Ingestor Agent for HelixForge.

This module handles ingestion of data from various sources including
CSV, Parquet, JSON files, SQL databases, and REST APIs.

Classes:
    DataIngestorAgent: Main agent class for data ingestion.
    IngestorConfig: Configuration settings for the ingestor.

Functions:
    detect_encoding: Auto-detect file encoding.
    detect_delimiter: Auto-detect CSV delimiter.
"""
```

---

## Error Handling

### Custom Exceptions

Define domain-specific exceptions in `utils/errors.py`:

```python
class HelixForgeError(Exception):
    """Base exception for HelixForge."""
    pass

class IngestionError(HelixForgeError):
    """Raised when data ingestion fails."""
    pass

class AlignmentError(HelixForgeError):
    """Raised when ontology alignment fails."""
    pass

class FusionError(HelixForgeError):
    """Raised when data fusion fails."""
    pass
```

### Error Handling Patterns

```python
# Good - specific exceptions with context
try:
    df = pd.read_csv(file_path, encoding=encoding)
except UnicodeDecodeError as e:
    raise IngestionError(f"Failed to decode {file_path}: {e}") from e
except FileNotFoundError:
    raise IngestionError(f"File not found: {file_path}")

# Avoid - catching all exceptions
try:
    df = pd.read_csv(file_path)
except Exception:
    return None  # Bad: silently fails
```

---

## Logging

### Logger Setup

Use structured logging with correlation IDs:

```python
from utils.logging import get_logger

logger = get_logger(__name__)

def ingest_file(file_path: str, correlation_id: str) -> IngestResult:
    logger.info(
        "Starting file ingestion",
        extra={
            "correlation_id": correlation_id,
            "file_path": file_path
        }
    )
    ...
```

### Log Levels

| Level | Use Case |
|-------|----------|
| DEBUG | Detailed diagnostic information |
| INFO | General operational events |
| WARNING | Unexpected but handled situations |
| ERROR | Failures that prevent operation |
| CRITICAL | System-wide failures |

---

## Testing

### Test File Naming

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<function_name>_<scenario>`

### Test Structure

```python
import pytest
from agents.data_ingestor_agent import DataIngestorAgent

class TestDataIngestorAgent:
    """Tests for DataIngestorAgent."""

    @pytest.fixture
    def agent(self):
        """Create agent instance for testing."""
        return DataIngestorAgent()

    def test_ingest_csv_success(self, agent, tmp_path):
        """Test successful CSV ingestion."""
        # Arrange
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n")

        # Act
        result = agent.ingest_file(str(csv_file))

        # Assert
        assert result.row_count == 1
        assert result.source_type == "csv"

    def test_ingest_csv_invalid_encoding(self, agent, tmp_path):
        """Test CSV ingestion with invalid encoding raises error."""
        # Arrange
        csv_file = tmp_path / "test.csv"
        csv_file.write_bytes(b"\xff\xfe invalid")

        # Act & Assert
        with pytest.raises(IngestionError):
            agent.ingest_file(str(csv_file))
```

### Coverage Requirements

- Minimum coverage: **80%**
- Agent modules: **90%**
- Utility modules: **85%**

---

## API Design

### REST Conventions

- Use plural nouns for resources: `/datasets`, `/alignments`
- Use HTTP methods correctly:
  - GET: Retrieve resources
  - POST: Create resources
  - PUT: Update resources (full replacement)
  - PATCH: Partial update
  - DELETE: Remove resources
- Return appropriate status codes:
  - 200: Success
  - 201: Created
  - 400: Bad Request
  - 404: Not Found
  - 500: Internal Server Error

### FastAPI Route Pattern

```python
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_dataset(file: UploadFile) -> IngestResult:
    """Upload and ingest a new dataset."""
    try:
        result = await ingestor.ingest_file(file)
        return result
    except IngestionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
```

---

## Configuration

### Environment Variables

- Use environment variables for secrets
- Prefix all env vars with `HELIXFORGE_`
- Document all required env vars

```python
import os

OPENAI_API_KEY = os.environ.get("HELIXFORGE_OPENAI_API_KEY")
DATABASE_URL = os.environ.get("HELIXFORGE_DATABASE_URL")
```

### Configuration Loading

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    database_url: str
    log_level: str = "INFO"

    class Config:
        env_prefix = "HELIXFORGE_"
        env_file = ".env"
```

---

## Git Conventions

### Branch Naming

- Feature: `feature/<description>`
- Bugfix: `bugfix/<description>`
- Hotfix: `hotfix/<description>`

### Commit Messages

Follow conventional commits:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

Example:
```
feat(ingestor): add Parquet file support

- Implement PyArrow-based Parquet reader
- Add type inference for nested columns
- Update IngestResult schema

Closes #42
```

---

## Tools

### Required Development Tools

```bash
# Install development dependencies
make install-dev

# Linting
make lint

# Formatting
make format

# Type checking
make typecheck

# Testing
make test          # All tests
make test-cov      # With coverage report
```

### Pre-commit Hooks

Configure `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic]
```

---

## Security Guidelines

1. **Never commit secrets** - Use environment variables
2. **Validate all inputs** - Sanitize user-provided data
3. **Use parameterized queries** - Prevent SQL injection
4. **Escape output** - Prevent XSS in reports
5. **Log securely** - Never log credentials or PII
6. **Keep dependencies updated** - Regular security audits
