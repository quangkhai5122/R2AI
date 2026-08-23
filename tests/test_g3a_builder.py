import copy

from vifinqa.g3a.builder import _choose_splits, _review_subject


def _candidate(name: str) -> dict:
    return {
        "operator": "lookup",
        "fact_ids": [name],
    }


def test_choose_splits_is_deterministic_and_fact_disjoint():
    candidates = [_candidate(f"fact-{index}") for index in range(12)]
    config = {
        "seed": 7,
        "allocation": {
            "hard": {"lookup": 2},
            "primary_locked": {"lookup": 3},
            "primary_tune": {"lookup": 4},
        },
    }
    first = _choose_splits(copy.deepcopy(candidates), config)
    second = _choose_splits(copy.deepcopy(candidates), config)
    assert first == second
    fact_sets = {
        split: {
            fact
            for row in rows
            for fact in row["fact_ids"]
        }
        for split, rows in first.items()
    }
    assert not (fact_sets["hard"] & fact_sets["primary_locked"])
    assert not (fact_sets["hard"] & fact_sets["primary_tune"])
    assert not (fact_sets["primary_locked"] & fact_sets["primary_tune"])


def test_hard_review_subject_changes_when_gold_changes():
    gold = {
        "id": 1,
        "question": "q",
        "operator": "lookup",
        "answer": 1.0,
        "output_type": "money_million_vnd",
        "tolerance": 0.01,
        "relevant_docs": ["doc"],
        "relevant_tables": ["doc|1"],
        "evidence": [{"fact_id": "f"}],
        "program": {"operator": "lookup", "fact_ids": ["f"]},
    }
    original = _review_subject(gold)
    changed = copy.deepcopy(gold)
    changed["answer"] = 2.0
    assert _review_subject(changed) != original
