"""
HelixForge Performance Testing with Locust

Run with:
    locust -f tests/performance/locustfile.py --host=http://localhost:8000

For headless mode:
    locust -f tests/performance/locustfile.py --host=http://localhost:8000 \
           --users 100 --spawn-rate 10 --run-time 5m --headless
"""

import json
import random
import string
from locust import HttpUser, between, task


class HelixForgeUser(HttpUser):
    """Simulates a user interacting with the HelixForge API."""

    wait_time = between(1, 3)

    def on_start(self):
        """Initialize user session."""
        self.dataset_ids = []
        self.job_ids = []

    @task(10)
    def health_check(self):
        """Check API health endpoint."""
        self.client.get("/health")

    @task(5)
    def get_metrics(self):
        """Fetch Prometheus metrics."""
        self.client.get("/metrics")

    @task(8)
    def ingest_csv_data(self):
        """Test CSV data ingestion endpoint."""
        csv_content = self._generate_csv_data()
        files = {"file": ("test_data.csv", csv_content, "text/csv")}

        with self.client.post(
            "/api/v1/ingest",
            files=files,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "dataset_id" in data:
                    self.dataset_ids.append(data["dataset_id"])
                response.success()
            elif response.status_code == 422:
                response.success()  # Validation error is expected for some inputs
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(6)
    def ingest_json_data(self):
        """Test JSON data ingestion endpoint."""
        json_data = self._generate_json_data()

        with self.client.post(
            "/api/v1/ingest",
            json={"data": json_data, "format": "json"},
            catch_response=True,
        ) as response:
            if response.status_code in [200, 422]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(4)
    def list_datasets(self):
        """List all datasets."""
        self.client.get("/api/v1/datasets")

    @task(3)
    def get_dataset_details(self):
        """Get details of a specific dataset."""
        if self.dataset_ids:
            dataset_id = random.choice(self.dataset_ids)
            self.client.get(f"/api/v1/datasets/{dataset_id}")

    @task(5)
    def start_analysis_job(self):
        """Start a cross-dataset analysis job."""
        if len(self.dataset_ids) >= 2:
            datasets = random.sample(self.dataset_ids, min(2, len(self.dataset_ids)))
            with self.client.post(
                "/api/v1/analyze",
                json={"dataset_ids": datasets},
                catch_response=True,
            ) as response:
                if response.status_code == 200:
                    data = response.json()
                    if "job_id" in data:
                        self.job_ids.append(data["job_id"])
                    response.success()
                elif response.status_code in [400, 404, 422]:
                    response.success()  # Expected for invalid inputs
                else:
                    response.failure(f"Unexpected status: {response.status_code}")

    @task(4)
    def get_job_status(self):
        """Check job status."""
        if self.job_ids:
            job_id = random.choice(self.job_ids)
            self.client.get(f"/api/v1/jobs/{job_id}")

    @task(3)
    def get_insights(self):
        """Get generated insights."""
        if self.job_ids:
            job_id = random.choice(self.job_ids)
            self.client.get(f"/api/v1/jobs/{job_id}/insights")

    @task(2)
    def get_provenance(self):
        """Get provenance information for a dataset."""
        if self.dataset_ids:
            dataset_id = random.choice(self.dataset_ids)
            self.client.get(f"/api/v1/datasets/{dataset_id}/provenance")

    @task(1)
    def export_report(self):
        """Export analysis report."""
        if self.job_ids:
            job_id = random.choice(self.job_ids)
            self.client.get(
                f"/api/v1/jobs/{job_id}/export",
                params={"format": "json"},
            )

    def _generate_csv_data(self, rows: int = 100) -> str:
        """Generate random CSV data for testing."""
        headers = ["id", "name", "value", "category", "timestamp"]
        lines = [",".join(headers)]

        for i in range(rows):
            name = "".join(random.choices(string.ascii_lowercase, k=8))
            value = random.uniform(0, 1000)
            category = random.choice(["A", "B", "C", "D"])
            timestamp = f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
            lines.append(f"{i},{name},{value:.2f},{category},{timestamp}")

        return "\n".join(lines)

    def _generate_json_data(self, records: int = 50) -> list:
        """Generate random JSON data for testing."""
        return [
            {
                "id": i,
                "name": "".join(random.choices(string.ascii_lowercase, k=8)),
                "metrics": {
                    "value": random.uniform(0, 1000),
                    "count": random.randint(0, 100),
                },
                "tags": random.sample(["alpha", "beta", "gamma", "delta"], k=2),
            }
            for i in range(records)
        ]


class StressTestUser(HttpUser):
    """High-frequency user for stress testing."""

    wait_time = between(0.1, 0.5)

    @task
    def rapid_health_check(self):
        """Rapid health checks for stress testing."""
        self.client.get("/health")

    @task
    def rapid_ingest(self):
        """Rapid data ingestion for stress testing."""
        data = {"records": [{"id": i, "value": i * 10} for i in range(10)]}
        self.client.post("/api/v1/ingest", json={"data": data, "format": "json"})
