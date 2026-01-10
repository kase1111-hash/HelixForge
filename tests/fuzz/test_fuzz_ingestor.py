"""
Fuzz Testing for Data Ingestor Agent

Uses Hypothesis for property-based testing and fuzzing.
These tests generate random inputs to find edge cases and crashes.

Run with:
    pytest tests/fuzz/ -v --hypothesis-seed=0
    pytest tests/fuzz/ -v --hypothesis-show-statistics
"""

import json
import os
import sys
import tempfile
from io import StringIO

import pytest
from hypothesis import assume, given, settings, HealthCheck
from hypothesis import strategies as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.data_ingestor_agent import DataIngestorAgent


class TestFuzzDataIngestorAgent:
    """Fuzz tests for DataIngestorAgent."""

    @pytest.fixture
    def ingestor(self):
        """Create ingestor instance."""
        return DataIngestorAgent()

    # Strategy for generating CSV-like content
    csv_content = st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "Zs"),
            whitelist_characters=",\n\t",
        ),
        min_size=0,
        max_size=10000,
    )

    # Strategy for generating JSON-like structures
    json_values = st.recursive(
        st.none() | st.booleans() | st.floats(allow_nan=False) | st.text(max_size=100),
        lambda children: st.lists(children, max_size=10) | st.dictionaries(
            st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
            children,
            max_size=10,
        ),
        max_leaves=50,
    )

    @given(content=csv_content)
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_fuzz_csv_parsing_no_crash(self, ingestor, content):
        """Fuzz test: CSV parsing should not crash on arbitrary input."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(content)
            f.flush()

            try:
                # Should not raise unhandled exception
                result = ingestor.ingest_file(f.name)
                # Result can be None for invalid input, that's OK
            except ValueError:
                # Expected for invalid CSV
                pass
            except Exception as e:
                # Unexpected exceptions should be reported
                if "decode" not in str(e).lower():
                    raise
            finally:
                os.unlink(f.name)

    @given(data=json_values)
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_fuzz_json_parsing_no_crash(self, ingestor, data):
        """Fuzz test: JSON parsing should not crash on arbitrary structures."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            try:
                json.dump(data, f)
                f.flush()

                # Should not raise unhandled exception
                result = ingestor.ingest_file(f.name)
            except (ValueError, TypeError):
                # Expected for some structures
                pass
            except json.JSONDecodeError:
                # Expected for invalid JSON
                pass
            finally:
                os.unlink(f.name)

    @given(
        rows=st.lists(
            st.lists(st.text(max_size=50), min_size=1, max_size=10),
            min_size=1,
            max_size=100,
        )
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_fuzz_structured_csv_no_crash(self, ingestor, rows):
        """Fuzz test: Well-formed CSV with random cell values."""
        assume(len(rows) > 0 and all(len(row) > 0 for row in rows))

        # Normalize row lengths
        max_cols = max(len(row) for row in rows)
        normalized = [row + [""] * (max_cols - len(row)) for row in rows]

        csv_content = "\n".join(
            ",".join(
                '"' + cell.replace('"', '""').replace("\n", " ") + '"'
                for cell in row
            )
            for row in normalized
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv_content)
            f.flush()

            try:
                result = ingestor.ingest_file(f.name)
            except ValueError:
                pass
            finally:
                os.unlink(f.name)

    @given(
        num_cols=st.integers(min_value=1, max_value=100),
        num_rows=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=20)
    def test_fuzz_large_csv_dimensions(self, ingestor, num_cols, num_rows):
        """Fuzz test: CSV with varying dimensions."""
        headers = [f"col_{i}" for i in range(num_cols)]
        rows = [",".join(headers)]

        for i in range(num_rows):
            row = [str(i * num_cols + j) for j in range(num_cols)]
            rows.append(",".join(row))

        csv_content = "\n".join(rows)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv_content)
            f.flush()

            try:
                result = ingestor.ingest_file(f.name)
                if result is not None:
                    # Verify basic structure
                    assert hasattr(result, "records") or hasattr(result, "metadata")
            except ValueError:
                pass
            finally:
                os.unlink(f.name)

    @given(
        encoding_bytes=st.binary(min_size=0, max_size=1000)
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_fuzz_binary_content_no_crash(self, ingestor, encoding_bytes):
        """Fuzz test: Binary content should not crash the parser."""
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".csv", delete=False
        ) as f:
            f.write(encoding_bytes)
            f.flush()

            try:
                result = ingestor.ingest_file(f.name)
            except (ValueError, UnicodeDecodeError):
                # Expected for binary content
                pass
            finally:
                os.unlink(f.name)

    @given(
        path=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P"),
                blacklist_characters="/\\",
            ),
            min_size=1,
            max_size=100,
        )
    )
    @settings(max_examples=50)
    def test_fuzz_file_path_no_crash(self, ingestor, path):
        """Fuzz test: Invalid file paths should raise appropriate errors."""
        try:
            result = ingestor.ingest_file(path)
        except (FileNotFoundError, ValueError, OSError):
            # Expected for invalid paths
            pass

    @given(
        nested_depth=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=20)
    def test_fuzz_deeply_nested_json(self, ingestor, nested_depth):
        """Fuzz test: Deeply nested JSON structures."""
        # Build nested structure
        data = {"value": 1}
        for _ in range(nested_depth):
            data = {"nested": data}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            f.flush()

            try:
                result = ingestor.ingest_file(f.name)
            except (ValueError, RecursionError):
                # Expected for very deep nesting
                pass
            finally:
                os.unlink(f.name)


class TestFuzzValidation:
    """Fuzz tests for input validation."""

    @given(
        field_name=st.text(max_size=100),
        field_value=st.one_of(
            st.text(max_size=1000),
            st.integers(),
            st.floats(allow_nan=False),
            st.booleans(),
            st.none(),
        ),
    )
    @settings(max_examples=100)
    def test_fuzz_field_values(self, field_name, field_value):
        """Fuzz test: Various field name/value combinations."""
        from models.schemas import FieldMetadata

        try:
            # Try to create field metadata with fuzzed values
            metadata = FieldMetadata(
                name=field_name if field_name.strip() else "default",
                data_type=type(field_value).__name__,
                sample_values=[str(field_value)] if field_value is not None else [],
            )
            assert metadata.name
        except (ValueError, TypeError):
            # Expected for invalid combinations
            pass


class TestFuzzSchemaDetection:
    """Fuzz tests for schema detection."""

    @given(
        values=st.lists(
            st.one_of(
                st.text(max_size=50),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans(),
            ),
            min_size=1,
            max_size=100,
        )
    )
    @settings(max_examples=100)
    def test_fuzz_type_inference(self, values):
        """Fuzz test: Type inference with mixed values."""
        from utils.helpers import infer_data_type

        try:
            # Convert all to strings as they would appear in CSV
            str_values = [str(v) for v in values]
            inferred_type = infer_data_type(str_values)

            # Type should be one of the known types
            assert inferred_type in [
                "string", "integer", "float", "boolean",
                "date", "datetime", "mixed", "unknown",
            ]
        except (ValueError, TypeError):
            pass
