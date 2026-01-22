# Contributing to HelixForge

Thank you for your interest in contributing to HelixForge! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/HelixForge.git
   cd HelixForge
   ```
3. Add the upstream repository as a remote:
   ```bash
   git remote add upstream https://github.com/kase1111-hash/HelixForge.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Docker & Docker Compose (for full stack testing)
- OpenAI API key (for LLM-powered features)

### Environment Setup

1. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export NEO4J_PASSWORD="password"
   export DB_PASSWORD="password"
   ```

4. **Start infrastructure services** (optional, for integration tests)
   ```bash
   docker-compose up -d postgres neo4j weaviate
   ```

### Using the Makefile

The project includes a Makefile with common development tasks:

```bash
make help          # Show all available commands
make install       # Install dependencies
make test          # Run all tests
make test-unit     # Run unit tests only
make lint          # Run linter
make format        # Format code
make typecheck     # Run type checker
```

## Making Changes

### Branch Naming

Create a descriptive branch name for your changes:

- `feature/add-new-insight-type` - For new features
- `fix/alignment-null-handling` - For bug fixes
- `docs/update-api-reference` - For documentation
- `refactor/simplify-fusion-logic` - For refactoring

### Commit Messages

Write clear, descriptive commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Keep the first line under 72 characters
- Reference issues when applicable

**Examples:**
```
Add semantic similarity threshold configuration

Fix null pointer exception in ontology alignment

Update API documentation for /fuse endpoint

Refactor insight generator to use async processing

Closes #123
```

## Code Standards

### Style Guide

Follow the project's [Style Guide](docs/STYLE_GUIDE.md). Key points:

- **Line length**: 120 characters maximum
- **Imports**: Use absolute imports, sorted with isort
- **Type hints**: Use type annotations for function signatures
- **Docstrings**: Use Google-style docstrings for public functions

### Code Quality Tools

Before submitting, ensure your code passes all checks:

```bash
# Linting
ruff check .

# Formatting
ruff format .

# Type checking
mypy .
```

Or run all checks at once:
```bash
make lint && make format && make typecheck
```

### Architecture Guidelines

- **Agents**: Core processing logic goes in `agents/`. Each agent should inherit from `BaseAgent`.
- **API Routes**: HTTP endpoints go in `api/routes/`. Keep route handlers thin.
- **Models**: Pydantic schemas go in `models/schemas.py`.
- **Utilities**: Shared utilities go in `utils/`.

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test categories
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m acceptance     # Acceptance tests only

# Run tests for a specific module
pytest tests/test_fusion.py
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_<module>.py`
- Name test functions `test_<description>`
- Use fixtures from `conftest.py` where appropriate
- Add appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.)

**Example test:**
```python
import pytest
from agents.fusion_agent import FusionAgent

@pytest.mark.unit
def test_fusion_agent_merge_datasets(mock_datasets):
    """Test that fusion agent correctly merges two datasets."""
    agent = FusionAgent()
    result = agent.merge(mock_datasets)

    assert result.record_count == 100
    assert "merged_field" in result.columns
```

### Test Coverage

Aim for meaningful test coverage:
- Unit tests for business logic
- Integration tests for component interactions
- Acceptance tests for user stories (see `docs/USER_STORIES.md`)

## Submitting Changes

### Pull Request Process

1. **Update your branch** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push your branch** to your fork:
   ```bash
   git push origin your-branch-name
   ```

3. **Create a Pull Request** on GitHub with:
   - A clear title describing the change
   - A description of what changed and why
   - Reference to any related issues
   - Screenshots for UI changes (if applicable)

4. **Address review feedback** by pushing additional commits

5. **Squash commits** if requested before merging

### Pull Request Checklist

Before submitting, verify:

- [ ] Code follows the style guide
- [ ] All tests pass (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] Type checking passes (`make typecheck`)
- [ ] New code has appropriate tests
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear and descriptive

## Reporting Issues

### Bug Reports

When reporting bugs, include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment details (Python version, OS, etc.)
- Relevant logs or error messages

### Feature Requests

When requesting features, include:

- A clear description of the feature
- The problem it solves or use case it addresses
- Any proposed implementation approach (optional)

## Questions?

If you have questions about contributing:

- Check the [FAQ](docs/FAQ.md)
- Review existing issues and discussions
- Open a new issue with your question

---

Thank you for contributing to HelixForge!
