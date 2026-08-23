import copy
import json
from pathlib import Path

from vifinqa.g3b.builder import (
    _choose_splits,
    review_subject,
    validate_corpus,
)
from vifinqa.g3b.common import read_jsonl
from vifinqa.g3b.views import validate_views


ROOT = Path(__file__).resolve().parents[1]


def _candidate(family: str, fact_id: str) -> dict:
    return {
        "family": family,
        "fact_ids": [fact_id],
        "program_key": fact_id,
    }


def test_g3b_split_selection_is_deterministic_and_fact_disjoint():
    candidates = [
        _candidate("lookup", f"fact-{index}") for index in range(12)
    ]
    config = {
        "seed": 9,
        "allocation": {
            "hard": {"lookup": 2},
            "primary_locked": {"lookup": 3},
            "primary_tune": {"lookup": 4},
        },
    }
    first = _choose_splits(copy.deepcopy(candidates), config)
    second = _choose_splits(copy.deepcopy(candidates), config)
    assert first == second
    sets = {
        split: {
            fact_id
            for row in rows
            for fact_id in row["fact_ids"]
        }
        for split, rows in first.items()
    }
    assert not sets["hard"] & sets["primary_locked"]
    assert not sets["hard"] & sets["primary_tune"]
    assert not sets["primary_locked"] & sets["primary_tune"]


def test_review_subject_binds_program_and_retrieval_gold():
    record = read_jsonl(ROOT / "data/g3b_v1/g3b_corpus.jsonl")[0]
    baseline = review_subject(record)
    changed = copy.deepcopy(record)
    changed["typed_program"]["output_type"] = "number"
    assert review_subject(changed) != baseline
    changed = copy.deepcopy(record)
    changed["relevant_tables"] = ["changed|1"]
    assert review_subject(changed) != baseline


def test_g3b_bundle_and_ood_views_are_strictly_valid():
    result = validate_corpus(
        ROOT / "data/g3a_extension_v1",
        ROOT / "data/g3b_v1",
        ROOT / "configs/g3b_evaluation_v1.json",
        g3a_v1_dir=ROOT / "data/g3a_v1",
    )
    assert result["valid"]
    assert result["questions"] == 109
    documents = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((ROOT / "data/g3b_v1/views").glob("*.json"))
    ]
    validate_views(documents)
    assert {row["kind"].lower() for row in documents} >= {
        "loto",
        "loyo",
        "loro",
        "lomo",
        "composition",
        "scope_period_stress",
    }


def test_gold_reuses_selection_v2_semantic_surface():
    records = read_jsonl(ROOT / "data/g3b_v1/g3b_corpus.jsonl")
    assert records
    for row in records:
        program = row["typed_program"]
        assert program["schema_version"] == 2
        assert set(program) == {
            "schema_version",
            "output_type",
            "facts",
            "bindings",
            "root",
        }
        assert row["leaf_specs"]
        assert len(row["leaf_specs"]) == len(row["fact_ids"])
