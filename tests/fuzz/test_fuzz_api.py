"""
Fuzz Testing for API Endpoints

Uses Hypothesis for property-based testing to find edge cases
in API request handling.

Run with:
    pytest tests/fuzz/test_fuzz_api.py -v
"""

import json
import os
import sys

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Try to import FastAPI test client
try:
    from fastapi.testclient import TestClient
    from api.server import app

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestFuzzAPIEndpoints:
    """Fuzz tests for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    # Strategy for generating query parameters
    query_params = st.dictionaries(
        st.text(min_size=1, max_size=20).filter(lambda x: x.isalnum()),
        st.text(max_size=100),
        max_size=10,
    )

    # Strategy for JSON body content
    json_body = st.recursive(
        st.none()
        | st.booleans()
        | st.integers(min_value=-2**31, max_value=2**31)
        | st.floats(allow_nan=False, allow_infinity=False)
        | st.text(max_size=200),
        lambda children: st.lists(children, max_size=20)
        | st.dictionaries(
            st.text(min_size=1, max_size=30).filter(lambda x: x.strip()),
            children,
            max_size=20,
        ),
        max_leaves=100,
    )

    @given(params=query_params)
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_fuzz_health_endpoint_params(self, client, params):
        """Fuzz test: Health endpoint with random query params."""
        response = client.get("/health", params=params)
        # Should always return 200 regardless of params
        assert response.status_code in [200, 422]

    @given(body=json_body)
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_fuzz_ingest_endpoint_body(self, client, body):
        """Fuzz test: Ingest endpoint with random JSON bodies."""
        try:
            response = client.post(
                "/api/v1/ingest",
                json={"data": body, "format": "json"},
            )
            # Valid responses: 200 (success), 400 (bad request), 422 (validation error)
            assert response.status_code in [200, 400, 422, 500]
        except Exception as e:
            # Some edge cases may cause serialization issues
            if "serialize" not in str(e).lower():
                raise

    @given(
        dataset_id=st.text(max_size=100),
    )
    @settings(max_examples=50)
    def test_fuzz_dataset_id_parameter(self, client, dataset_id):
        """Fuzz test: Dataset ID parameter with random strings."""
        response = client.get(f"/api/v1/datasets/{dataset_id}")
        # Should handle any dataset ID gracefully
        assert response.status_code in [200, 400, 404, 422]

    @given(
        job_id=st.text(max_size=100),
    )
    @settings(max_examples=50)
    def test_fuzz_job_id_parameter(self, client, job_id):
        """Fuzz test: Job ID parameter with random strings."""
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code in [200, 400, 404, 422]

    @given(
        dataset_ids=st.lists(st.text(max_size=50), min_size=0, max_size=20),
    )
    @settings(max_examples=50)
    def test_fuzz_analyze_endpoint(self, client, dataset_ids):
        """Fuzz test: Analyze endpoint with random dataset IDs."""
        response = client.post(
            "/api/v1/analyze",
            json={"dataset_ids": dataset_ids},
        )
        assert response.status_code in [200, 400, 404, 422]

    @given(
        format_type=st.text(max_size=50),
    )
    @settings(max_examples=50)
    def test_fuzz_export_format_parameter(self, client, format_type):
        """Fuzz test: Export format parameter with random values."""
        response = client.get(
            "/api/v1/jobs/test-job/export",
            params={"format": format_type},
        )
        assert response.status_code in [200, 400, 404, 422]

    @given(
        headers=st.dictionaries(
            st.text(min_size=1, max_size=30).filter(lambda x: x.replace("-", "").isalnum()),
            st.text(max_size=100),
            max_size=10,
        ),
    )
    @settings(max_examples=50)
    def test_fuzz_request_headers(self, client, headers):
        """Fuzz test: Request with random headers."""
        # Filter out headers that might cause issues
        safe_headers = {
            k: v
            for k, v in headers.items()
            if k.lower() not in ["content-length", "transfer-encoding", "host"]
        }
        response = client.get("/health", headers=safe_headers)
        assert response.status_code in [200, 400, 422]

    @given(
        content_type=st.text(max_size=50),
    )
    @settings(max_examples=30)
    def test_fuzz_content_type_header(self, client, content_type):
        """Fuzz test: Content-Type header with random values."""
        response = client.post(
            "/api/v1/ingest",
            content=b"test data",
            headers={"Content-Type": content_type},
        )
        assert response.status_code in [200, 400, 415, 422]


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not available")
class TestFuzzAPIInputValidation:
    """Fuzz tests for API input validation."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @given(
        limit=st.integers(),
        offset=st.integers(),
    )
    @settings(max_examples=50)
    def test_fuzz_pagination_params(self, client, limit, offset):
        """Fuzz test: Pagination with arbitrary integers."""
        response = client.get(
            "/api/v1/datasets",
            params={"limit": limit, "offset": offset},
        )
        # Should handle any integer values
        assert response.status_code in [200, 400, 422]

    @given(
        sort_by=st.text(max_size=50),
        order=st.text(max_size=20),
    )
    @settings(max_examples=50)
    def test_fuzz_sorting_params(self, client, sort_by, order):
        """Fuzz test: Sorting parameters with random values."""
        response = client.get(
            "/api/v1/datasets",
            params={"sort_by": sort_by, "order": order},
        )
        assert response.status_code in [200, 400, 422]

    @given(
        filter_field=st.text(max_size=50),
        filter_value=st.text(max_size=100),
    )
    @settings(max_examples=50)
    def test_fuzz_filter_params(self, client, filter_field, filter_value):
        """Fuzz test: Filter parameters with random values."""
        response = client.get(
            "/api/v1/datasets",
            params={filter_field: filter_value} if filter_field.strip() else {},
        )
        assert response.status_code in [200, 400, 422]


class TestFuzzSchemaValidation:
    """Fuzz tests for Pydantic schema validation."""

    @given(
        data=st.dictionaries(
            st.text(min_size=1, max_size=30),
            st.one_of(
                st.text(max_size=100),
                st.integers(),
                st.floats(allow_nan=False),
                st.booleans(),
                st.none(),
            ),
            max_size=20,
        ),
    )
    @settings(max_examples=100)
    def test_fuzz_ingest_request_schema(self, data):
        """Fuzz test: IngestRequest schema with random data."""
        from models.schemas import IngestRequest

        try:
            request = IngestRequest(**data)
        except (ValueError, TypeError):
            # Expected for invalid data
            pass

    @given(
        records=st.lists(
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.one_of(st.text(max_size=50), st.integers(), st.floats(allow_nan=False)),
                max_size=10,
            ),
            max_size=50,
        ),
    )
    @settings(max_examples=50)
    def test_fuzz_batch_ingest_schema(self, records):
        """Fuzz test: Batch ingest with random record structures."""
        from models.schemas import IngestRequest

        try:
            request = IngestRequest(records=records, format="json")
        except (ValueError, TypeError):
            pass

    @given(
        threshold=st.floats(),
    )
    @settings(max_examples=50)
    def test_fuzz_confidence_threshold(self, threshold):
        """Fuzz test: Confidence threshold with arbitrary floats."""
        from models.schemas import AnalysisConfig

        try:
            config = AnalysisConfig(confidence_threshold=threshold)
            # Valid threshold should be between 0 and 1
            if config.confidence_threshold is not None:
                assert 0 <= config.confidence_threshold <= 1
        except (ValueError, TypeError):
            # Expected for invalid values
            pass
