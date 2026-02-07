"""Golden dataset tests for semantic labeling quality.

Validates that the MockProvider's heuristic-based labeling achieves
>= 80% accuracy on a curated set of field names with known expected
semantic types. This serves as a regression guard.
"""

import json
from pathlib import Path

import pytest

from utils.llm import MockProvider

GOLDEN_PATH = Path(__file__).parent / "golden_labels.json"


@pytest.fixture
def golden_dataset():
    """Load the golden label dataset."""
    with open(GOLDEN_PATH) as f:
        return json.load(f)


@pytest.fixture
def provider():
    return MockProvider(dimensions=1536)


class TestGoldenLabels:
    """Validate semantic labeling accuracy against a golden dataset."""

    def test_golden_dataset_loads(self, golden_dataset):
        """Verify golden dataset file exists and is non-empty."""
        assert len(golden_dataset) >= 40
        for entry in golden_dataset:
            assert "field_name" in entry
            assert "data_type" in entry
            assert "expected_semantic_type" in entry

    def test_labeling_accuracy_at_least_80_percent(self, golden_dataset, provider):
        """MockProvider should correctly label >= 80% of golden dataset fields."""
        correct = 0
        total = len(golden_dataset)
        mismatches = []

        for entry in golden_dataset:
            field_name = entry["field_name"]
            data_type = entry["data_type"]
            expected = entry["expected_semantic_type"]

            # Build the same prompt that MetadataInterpreterAgent sends
            prompt = f"Field Name: {field_name}\nData Type: {data_type}"
            result_text = provider.complete(
                messages=[{"role": "user", "content": prompt}]
            )
            result = json.loads(result_text)
            predicted = result.get("semantic_type", "unknown")

            if predicted == expected:
                correct += 1
            else:
                mismatches.append(
                    f"  {field_name}: expected={expected}, got={predicted}"
                )

        accuracy = correct / total
        detail = f"\nAccuracy: {correct}/{total} = {accuracy:.1%}"
        if mismatches:
            detail += "\nMismatches:\n" + "\n".join(mismatches)

        assert accuracy >= 0.80, (
            f"Labeling accuracy {accuracy:.1%} is below 80% threshold.{detail}"
        )

    def test_all_identifier_fields_correct(self, golden_dataset, provider):
        """All identifier fields should be classified correctly."""
        id_fields = [e for e in golden_dataset if e["expected_semantic_type"] == "identifier"]
        for entry in id_fields:
            result = json.loads(provider.complete(
                messages=[{"role": "user", "content": f"Field Name: {entry['field_name']}\nData Type: {entry['data_type']}"}]
            ))
            assert result["semantic_type"] == "identifier", \
                f"{entry['field_name']} should be identifier, got {result['semantic_type']}"

    def test_all_timestamp_fields_correct(self, golden_dataset, provider):
        """All timestamp fields should be classified correctly."""
        ts_fields = [e for e in golden_dataset if e["expected_semantic_type"] == "timestamp"]
        for entry in ts_fields:
            result = json.loads(provider.complete(
                messages=[{"role": "user", "content": f"Field Name: {entry['field_name']}\nData Type: {entry['data_type']}"}]
            ))
            assert result["semantic_type"] == "timestamp", \
                f"{entry['field_name']} should be timestamp, got {result['semantic_type']}"

    def test_all_metric_fields_correct(self, golden_dataset, provider):
        """All metric fields should be classified correctly."""
        metric_fields = [e for e in golden_dataset if e["expected_semantic_type"] == "metric"]
        for entry in metric_fields:
            result = json.loads(provider.complete(
                messages=[{"role": "user", "content": f"Field Name: {entry['field_name']}\nData Type: {entry['data_type']}"}]
            ))
            assert result["semantic_type"] == "metric", \
                f"{entry['field_name']} should be metric, got {result['semantic_type']}"
