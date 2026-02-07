"""Alignment benchmark tests.

Runs the alignment agent against 10 curated dataset pairs with
manually-verified expected alignments. Computes precision, recall,
and F1 for each pair and in aggregate. Fails if aggregate F1 < 0.75.

Embeddings are hand-crafted so that semantically similar fields
have high cosine similarity and unrelated fields have low similarity.
This isolates the alignment algorithm from the embedding model.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pytest

from agents.ontology_alignment_agent import OntologyAlignmentAgent
from models.schemas import (
    AlignmentResult,
    DatasetMetadata,
    DataType,
    FieldMetadata,
    SemanticType,
)

MANIFEST_PATH = Path(__file__).parent / "benchmarks" / "manifest.json"

# Dimension for hand-crafted embeddings
DIM = 1536
THIRD = DIM // 3


def _make_emb(*parts: float) -> List[float]:
    """Build a DIM-length embedding from 3 equal-length segments.

    Each part value fills THIRD slots. Useful for creating orthogonal
    or semi-orthogonal vectors for testing cosine similarity.
    """
    assert len(parts) == 3
    vec = []
    for p in parts:
        vec.extend([p] * THIRD)
    return vec


def _make_field(
    dataset_id: str,
    name: str,
    label: str,
    data_type: DataType,
    semantic_type: SemanticType,
    embedding: List[float],
    null_ratio: float = 0.0,
    unique_ratio: float = 0.9,
    confidence: float = 0.9,
) -> FieldMetadata:
    return FieldMetadata(
        dataset_id=dataset_id,
        field_name=name,
        semantic_label=label,
        description=f"Field {name}",
        data_type=data_type,
        semantic_type=semantic_type,
        embedding=embedding,
        sample_values=[],
        null_ratio=null_ratio,
        unique_ratio=unique_ratio,
        confidence=confidence,
    )


def _make_metadata(dataset_id: str, fields: List[FieldMetadata]) -> DatasetMetadata:
    return DatasetMetadata(
        dataset_id=dataset_id,
        fields=fields,
        dataset_description=f"Dataset {dataset_id}",
        domain_tags=[],
    )


# ------------------------------------------------------------------ #
#  10 benchmark dataset pairs                                        #
# ------------------------------------------------------------------ #

def _pair_exact_names():
    """Pair 1: Exact field name matches."""
    # Orthogonal embeddings for distinct fields
    emb_name = _make_emb(1.0, 0.0, 0.0)
    emb_dept = _make_emb(0.0, 1.0, 0.0)
    emb_salary = _make_emb(0.0, 0.0, 1.0)

    meta_a = _make_metadata("ds-a", [
        _make_field("ds-a", "name", "Name", DataType.STRING, SemanticType.TEXT, emb_name),
        _make_field("ds-a", "department", "Department", DataType.STRING, SemanticType.CATEGORY, emb_dept),
        _make_field("ds-a", "salary", "Salary", DataType.FLOAT, SemanticType.METRIC, emb_salary),
    ])
    meta_b = _make_metadata("ds-b", [
        _make_field("ds-b", "name", "Name", DataType.STRING, SemanticType.TEXT, emb_name),
        _make_field("ds-b", "department", "Department", DataType.STRING, SemanticType.CATEGORY, emb_dept),
        _make_field("ds-b", "salary", "Salary", DataType.FLOAT, SemanticType.METRIC, emb_salary),
    ])
    return meta_a, meta_b


def _pair_synonym_fields():
    """Pair 2: Synonym field names."""
    emb_id = _make_emb(1.0, 0.0, 0.0)
    emb_salary = _make_emb(0.0, 1.0, 0.0)
    emb_date = _make_emb(0.0, 0.0, 1.0)

    meta_a = _make_metadata("ds-a", [
        _make_field("ds-a", "emp_id", "Employee ID", DataType.INTEGER, SemanticType.IDENTIFIER, emb_id, unique_ratio=1.0),
        _make_field("ds-a", "annual_salary", "Annual Salary", DataType.FLOAT, SemanticType.METRIC, emb_salary),
        _make_field("ds-a", "hire_date", "Hire Date", DataType.DATETIME, SemanticType.TIMESTAMP, emb_date),
    ])
    meta_b = _make_metadata("ds-b", [
        _make_field("ds-b", "employee_id", "Employee ID", DataType.INTEGER, SemanticType.IDENTIFIER, emb_id, unique_ratio=1.0),
        _make_field("ds-b", "yearly_salary", "Yearly Salary", DataType.FLOAT, SemanticType.METRIC, emb_salary),
        _make_field("ds-b", "start_date", "Start Date", DataType.DATETIME, SemanticType.TIMESTAMP, emb_date),
    ])
    return meta_a, meta_b


def _pair_no_overlap():
    """Pair 3: Non-overlapping schemas (expect 0 alignments).

    Uses completely different domains (customer vs environment) with
    no shared tokens in field names, different types, different semantic
    types, and fully orthogonal embeddings.
    """
    meta_a = _make_metadata("ds-a", [
        _make_field("ds-a", "customer_id", "Customer ID", DataType.INTEGER, SemanticType.IDENTIFIER, _make_emb(1.0, 0.0, 0.0), unique_ratio=1.0),
        _make_field("ds-a", "email", "Email Address", DataType.STRING, SemanticType.TEXT, _make_emb(0.0, 1.0, 0.0)),
    ])
    meta_b = _make_metadata("ds-b", [
        _make_field("ds-b", "temperature", "Temperature Reading", DataType.FLOAT, SemanticType.METRIC, _make_emb(0.0, 0.0, 1.0)),
        _make_field("ds-b", "color", "Color Code", DataType.STRING, SemanticType.CATEGORY, _make_emb(0.0, 0.0, 1.0), unique_ratio=0.05),
    ])
    return meta_a, meta_b


def _pair_type_conflict():
    """Pair 4: Type-incompatible fields should NOT align."""
    emb_id = _make_emb(1.0, 0.0, 0.0)
    # created_date (DATETIME) vs active_flag (BOOLEAN) - similar embeddings but incompatible types
    emb_similar = _make_emb(0.0, 1.0, 0.0)

    meta_a = _make_metadata("ds-a", [
        _make_field("ds-a", "record_id", "Record ID", DataType.INTEGER, SemanticType.IDENTIFIER, emb_id, unique_ratio=1.0),
        _make_field("ds-a", "created_date", "Created Date", DataType.DATETIME, SemanticType.TIMESTAMP, emb_similar),
    ])
    meta_b = _make_metadata("ds-b", [
        _make_field("ds-b", "record_id", "Record ID", DataType.INTEGER, SemanticType.IDENTIFIER, emb_id, unique_ratio=1.0),
        _make_field("ds-b", "active_flag", "Active Flag", DataType.BOOLEAN, SemanticType.CATEGORY, emb_similar),
    ])
    return meta_a, meta_b


def _pair_partial_overlap():
    """Pair 5: Some fields match, some don't."""
    emb_id = _make_emb(1.0, 0.0, 0.0)
    emb_dept = _make_emb(0.0, 1.0, 0.0)
    emb_score = _make_emb(0.0, 0.0, 1.0)
    emb_review = _make_emb(0.3, 0.3, 0.4)

    meta_a = _make_metadata("ds-a", [
        _make_field("ds-a", "employee_id", "Employee ID", DataType.INTEGER, SemanticType.IDENTIFIER, emb_id, unique_ratio=1.0),
        _make_field("ds-a", "department", "Department", DataType.STRING, SemanticType.CATEGORY, emb_dept, unique_ratio=0.1),
        _make_field("ds-a", "salary", "Salary", DataType.FLOAT, SemanticType.METRIC, emb_score),
    ])
    meta_b = _make_metadata("ds-b", [
        _make_field("ds-b", "emp_id", "Employee ID", DataType.INTEGER, SemanticType.IDENTIFIER, emb_id, unique_ratio=1.0),
        _make_field("ds-b", "dept", "Department", DataType.STRING, SemanticType.CATEGORY, emb_dept, unique_ratio=0.1),
        _make_field("ds-b", "review_text", "Review Text", DataType.STRING, SemanticType.TEXT, emb_review, unique_ratio=0.95),
    ])
    return meta_a, meta_b


def _pair_many_to_one():
    """Pair 6: Multiple fields compete for one target."""
    emb_amount = _make_emb(0.0, 1.0, 0.0)
    emb_tax = _make_emb(0.0, 0.7, 0.3)     # similar to amount
    emb_id = _make_emb(1.0, 0.0, 0.0)

    meta_a = _make_metadata("ds-a", [
        _make_field("ds-a", "order_id", "Order ID", DataType.INTEGER, SemanticType.IDENTIFIER, emb_id, unique_ratio=1.0),
        _make_field("ds-a", "total_amount", "Total Amount", DataType.FLOAT, SemanticType.METRIC, emb_amount),
        _make_field("ds-a", "tax_amount", "Tax Amount", DataType.FLOAT, SemanticType.METRIC, emb_tax),
    ])
    meta_b = _make_metadata("ds-b", [
        _make_field("ds-b", "id", "ID", DataType.INTEGER, SemanticType.IDENTIFIER, emb_id, unique_ratio=1.0),
        _make_field("ds-b", "amount", "Amount", DataType.FLOAT, SemanticType.METRIC, emb_amount),
    ])
    return meta_a, meta_b


def _pair_cardinality_mismatch():
    """Pair 7: ID (unique) vs category (low cardinality)."""
    emb_id = _make_emb(1.0, 0.0, 0.0)
    emb_status = _make_emb(0.0, 1.0, 0.0)
    emb_cat = _make_emb(0.0, 0.0, 1.0)

    meta_a = _make_metadata("ds-a", [
        _make_field("ds-a", "user_id", "User ID", DataType.INTEGER, SemanticType.IDENTIFIER, emb_id, unique_ratio=1.0),
        _make_field("ds-a", "status", "Status", DataType.STRING, SemanticType.CATEGORY, emb_status, unique_ratio=0.05),
        _make_field("ds-a", "category", "Category", DataType.STRING, SemanticType.CATEGORY, emb_cat, unique_ratio=0.03),
    ])
    meta_b = _make_metadata("ds-b", [
        _make_field("ds-b", "user_id", "User ID", DataType.INTEGER, SemanticType.IDENTIFIER, emb_id, unique_ratio=1.0),
        _make_field("ds-b", "status", "Status", DataType.STRING, SemanticType.CATEGORY, emb_status, unique_ratio=0.05),
    ])
    return meta_a, meta_b


def _pair_naming_conventions():
    """Pair 8: snake_case vs camelCase naming."""
    emb_first = _make_emb(1.0, 0.0, 0.0)
    emb_last = _make_emb(0.0, 1.0, 0.0)
    emb_email = _make_emb(0.0, 0.0, 1.0)

    meta_a = _make_metadata("ds-a", [
        _make_field("ds-a", "first_name", "First Name", DataType.STRING, SemanticType.TEXT, emb_first),
        _make_field("ds-a", "last_name", "Last Name", DataType.STRING, SemanticType.TEXT, emb_last),
        _make_field("ds-a", "email_address", "Email Address", DataType.STRING, SemanticType.TEXT, emb_email),
    ])
    meta_b = _make_metadata("ds-b", [
        _make_field("ds-b", "firstName", "First Name", DataType.STRING, SemanticType.TEXT, emb_first),
        _make_field("ds-b", "lastName", "Last Name", DataType.STRING, SemanticType.TEXT, emb_last),
        _make_field("ds-b", "emailAddress", "Email Address", DataType.STRING, SemanticType.TEXT, emb_email),
    ])
    return meta_a, meta_b


def _pair_medical_domain():
    """Pair 9: Medical domain synonyms."""
    emb_id = _make_emb(1.0, 0.0, 0.0)
    emb_date = _make_emb(0.0, 1.0, 0.0)
    emb_condition = _make_emb(0.0, 0.0, 1.0)

    meta_a = _make_metadata("ds-a", [
        _make_field("ds-a", "patient_id", "Patient ID", DataType.INTEGER, SemanticType.IDENTIFIER, emb_id, unique_ratio=1.0),
        _make_field("ds-a", "birth_date", "Birth Date", DataType.DATETIME, SemanticType.TIMESTAMP, emb_date),
        _make_field("ds-a", "diagnosis", "Diagnosis", DataType.STRING, SemanticType.TEXT, emb_condition),
    ])
    meta_b = _make_metadata("ds-b", [
        _make_field("ds-b", "record_id", "Record ID", DataType.INTEGER, SemanticType.IDENTIFIER, emb_id, unique_ratio=1.0),
        _make_field("ds-b", "date_of_birth", "Date of Birth", DataType.DATETIME, SemanticType.TIMESTAMP, emb_date),
        _make_field("ds-b", "condition", "Condition", DataType.STRING, SemanticType.TEXT, emb_condition),
    ])
    return meta_a, meta_b


def _pair_financial_domain():
    """Pair 10: Financial transactions vs payments."""
    emb_id = _make_emb(1.0, 0.0, 0.0)
    emb_amount = _make_emb(0.0, 1.0, 0.0)
    emb_ts = _make_emb(0.0, 0.0, 1.0)

    meta_a = _make_metadata("ds-a", [
        _make_field("ds-a", "transaction_id", "Transaction ID", DataType.INTEGER, SemanticType.IDENTIFIER, emb_id, unique_ratio=1.0),
        _make_field("ds-a", "amount", "Amount", DataType.FLOAT, SemanticType.METRIC, emb_amount),
        _make_field("ds-a", "created_at", "Created At", DataType.DATETIME, SemanticType.TIMESTAMP, emb_ts),
    ])
    meta_b = _make_metadata("ds-b", [
        _make_field("ds-b", "payment_id", "Payment ID", DataType.INTEGER, SemanticType.IDENTIFIER, emb_id, unique_ratio=1.0),
        _make_field("ds-b", "total", "Total", DataType.FLOAT, SemanticType.METRIC, emb_amount),
        _make_field("ds-b", "timestamp", "Timestamp", DataType.DATETIME, SemanticType.TIMESTAMP, emb_ts),
    ])
    return meta_a, meta_b


# ------------------------------------------------------------------ #
#  Benchmark infrastructure                                          #
# ------------------------------------------------------------------ #

PAIR_BUILDERS = {
    "exact_names": _pair_exact_names,
    "synonym_fields": _pair_synonym_fields,
    "no_overlap": _pair_no_overlap,
    "type_conflict": _pair_type_conflict,
    "partial_overlap": _pair_partial_overlap,
    "many_to_one": _pair_many_to_one,
    "cardinality_mismatch": _pair_cardinality_mismatch,
    "naming_conventions": _pair_naming_conventions,
    "medical_domain": _pair_medical_domain,
    "financial_domain": _pair_financial_domain,
}


def _compute_f1(
    predicted_pairs: Set[Tuple[str, str]],
    expected_pairs: Set[Tuple[str, str]],
) -> Dict[str, float]:
    """Compute precision, recall, F1 for alignment pairs."""
    if not expected_pairs and not predicted_pairs:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not expected_pairs:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if not predicted_pairs:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

    true_positives = len(predicted_pairs & expected_pairs)
    precision = true_positives / len(predicted_pairs) if predicted_pairs else 0.0
    recall = true_positives / len(expected_pairs) if expected_pairs else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def _extract_pairs(result: AlignmentResult) -> Set[Tuple[str, str]]:
    """Extract (source_field, target_field) pairs from alignment result."""
    return {(a.source_field, a.target_field) for a in result.alignments}


# ------------------------------------------------------------------ #
#  Test class                                                        #
# ------------------------------------------------------------------ #

class TestAlignmentBenchmark:
    """Benchmark tests for alignment accuracy."""

    @pytest.fixture
    def agent(self):
        """Create alignment agent with default config."""
        return OntologyAlignmentAgent(config={
            "alignment": {
                "similarity_threshold": 0.50,
                "exact_match_threshold": 0.95,
                "synonym_threshold": 0.85,
                "max_alignments_per_field": 3,
                "conflict_resolution": "highest_similarity",
            }
        })

    @pytest.fixture
    def manifest(self):
        with open(MANIFEST_PATH) as f:
            return json.load(f)

    def test_manifest_has_10_pairs(self, manifest):
        """Manifest defines 10 benchmark pairs."""
        assert len(manifest) == 10

    def test_all_pairs_have_builders(self, manifest):
        """Every manifest entry has a corresponding pair builder."""
        for entry in manifest:
            assert entry["id"] in PAIR_BUILDERS, f"Missing builder: {entry['id']}"

    @pytest.mark.parametrize("pair_id", list(PAIR_BUILDERS.keys()))
    def test_individual_pair(self, agent, manifest, pair_id):
        """Run alignment on a single pair and verify F1 >= 0.5."""
        meta_a, meta_b = PAIR_BUILDERS[pair_id]()
        result = agent.process([meta_a, meta_b])

        # Get expected from manifest
        entry = next(e for e in manifest if e["id"] == pair_id)
        expected = {(a[0], a[1]) for a in entry["expected_alignments"]}
        predicted = _extract_pairs(result)

        metrics = _compute_f1(predicted, expected)

        # Individual pair F1 should be reasonable
        assert metrics["f1"] >= 0.5, (
            f"Pair '{pair_id}' F1={metrics['f1']:.2f} < 0.5\n"
            f"  Expected: {sorted(expected)}\n"
            f"  Predicted: {sorted(predicted)}\n"
            f"  Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}"
        )

    def test_aggregate_f1_at_least_075(self, agent, manifest):
        """Aggregate F1 across all 10 pairs must be >= 0.75."""
        all_predicted: Set[Tuple[str, str]] = set()
        all_expected: Set[Tuple[str, str]] = set()
        pair_results = []

        for entry in manifest:
            pair_id = entry["id"]
            meta_a, meta_b = PAIR_BUILDERS[pair_id]()
            result = agent.process([meta_a, meta_b])

            expected = {(a[0], a[1]) for a in entry["expected_alignments"]}
            predicted = _extract_pairs(result)

            metrics = _compute_f1(predicted, expected)
            pair_results.append((pair_id, metrics, predicted, expected))

            # Use pair_id prefix to avoid cross-pair collisions
            all_predicted.update((f"{pair_id}:{p[0]}", f"{pair_id}:{p[1]}") for p in predicted)
            all_expected.update((f"{pair_id}:{e[0]}", f"{pair_id}:{e[1]}") for e in expected)

        aggregate = _compute_f1(all_predicted, all_expected)

        # Build detailed report
        report = f"\nAggregate F1: {aggregate['f1']:.2f} (P={aggregate['precision']:.2f}, R={aggregate['recall']:.2f})\n"
        for pair_id, metrics, predicted, expected in pair_results:
            status = "PASS" if metrics["f1"] >= 0.5 else "FAIL"
            report += f"  [{status}] {pair_id}: F1={metrics['f1']:.2f} (P={metrics['precision']:.2f}, R={metrics['recall']:.2f})"
            if predicted != expected:
                report += f" predicted={sorted(predicted)} expected={sorted(expected)}"
            report += "\n"

        assert aggregate["f1"] >= 0.75, (
            f"Aggregate F1 {aggregate['f1']:.2f} is below 0.75 threshold.{report}"
        )

    def test_type_incompatible_never_aligned(self, agent):
        """DATETIME<->BOOLEAN fields are never aligned, regardless of embeddings."""
        emb_same = _make_emb(1.0, 0.0, 0.0)  # Identical embeddings!

        meta_a = _make_metadata("ds-a", [
            _make_field("ds-a", "created_at", "Created At", DataType.DATETIME, SemanticType.TIMESTAMP, emb_same),
        ])
        meta_b = _make_metadata("ds-b", [
            _make_field("ds-b", "is_active", "Is Active", DataType.BOOLEAN, SemanticType.CATEGORY, emb_same),
        ])

        result = agent.process([meta_a, meta_b])
        assert len(result.alignments) == 0, "DATETIME<->BOOLEAN should never align"

    def test_datetime_float_incompatible(self, agent):
        """DATETIME<->FLOAT fields are never aligned."""
        emb_same = _make_emb(1.0, 0.0, 0.0)

        meta_a = _make_metadata("ds-a", [
            _make_field("ds-a", "event_time", "Event Time", DataType.DATETIME, SemanticType.TIMESTAMP, emb_same),
        ])
        meta_b = _make_metadata("ds-b", [
            _make_field("ds-b", "amount", "Amount", DataType.FLOAT, SemanticType.METRIC, emb_same),
        ])

        result = agent.process([meta_a, meta_b])
        assert len(result.alignments) == 0, "DATETIME<->FLOAT should never align"

    def test_integer_float_compatible(self, agent):
        """INTEGER<->FLOAT are compatible and can align."""
        emb_same = _make_emb(1.0, 0.0, 0.0)

        meta_a = _make_metadata("ds-a", [
            _make_field("ds-a", "count", "Count", DataType.INTEGER, SemanticType.METRIC, emb_same),
        ])
        meta_b = _make_metadata("ds-b", [
            _make_field("ds-b", "count", "Count", DataType.FLOAT, SemanticType.METRIC, emb_same),
        ])

        result = agent.process([meta_a, meta_b])
        assert len(result.alignments) == 1, "INTEGER<->FLOAT should be compatible"
