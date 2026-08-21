import importlib.util
import json
from pathlib import Path

import pytest


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "60_run_clean_b0_v2.py"
    spec = importlib.util.spec_from_file_location("clean_b0_v2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, route: dict) -> None:
    path.write_text(
        json.dumps({"id": 5, "question": "q", "route": route, "candidates": []}) + "\n",
        encoding="utf-8",
    )


def test_empty_metric_keys_are_a_valid_canonical_miss(tmp_path):
    runner = _load_runner()
    path = tmp_path / "retrieval.jsonl"
    _write(path, {
        "clean_profile": "clean",
        "metric_keys": [],
        "metric_variants": ["chi phi phat"],
        "retrieval_config_sha256": "a" * 64,
    })
    records, misses = runner.validate_clean_retrieval(path)
    assert len(records) == 1
    assert misses == 1


def test_missing_metric_keys_field_is_rejected(tmp_path):
    runner = _load_runner()
    path = tmp_path / "retrieval.jsonl"
    _write(path, {
        "clean_profile": "clean",
        "metric_variants": ["chi phi phat"],
        "retrieval_config_sha256": "a" * 64,
    })
    with pytest.raises(SystemExit, match="metric_keys"):
        runner.validate_clean_retrieval(path)
