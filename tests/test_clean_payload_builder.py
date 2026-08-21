import importlib.util
import json
from pathlib import Path

import pytest


def _load_builder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "59_make_clean_payload.py"
    spec = importlib.util.spec_from_file_location("clean_payload_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_clean_retrieval_validator_rejects_non_clean_record(tmp_path):
    builder = _load_builder()
    path = tmp_path / "retrieval.jsonl"
    path.write_text(json.dumps({"route": {"metric_keys": ["net_revenue"]}}) + "\n")
    with pytest.raises(SystemExit, match="not clean-profile"):
        builder._validate_clean_retrieval(path)


def test_clean_retrieval_validator_accepts_single_fingerprint(tmp_path):
    builder = _load_builder()
    path = tmp_path / "retrieval.jsonl"
    route = {
        "clean_profile": "clean",
        "metric_keys": ["net_revenue"],
        "retrieval_config_sha256": "a" * 64,
    }
    path.write_text(json.dumps({"route": route}) + "\n")
    assert builder._validate_clean_retrieval(path) == (1, "a" * 64)
