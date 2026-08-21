import importlib.util
import json
from pathlib import Path


def _load_builder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "59_make_clean_payload_v2.py"
    spec = importlib.util.spec_from_file_location("clean_payload_v2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_payload_v2_accepts_empty_metric_keys_with_lexical_fallback(tmp_path):
    builder = _load_builder()
    path = tmp_path / "retrieval.jsonl"
    route = {
        "clean_profile": "clean",
        "metric_keys": [],
        "metric_variants": ["chi phi phat"],
        "retrieval_config_sha256": "a" * 64,
    }
    path.write_text(
        json.dumps({"id": 5, "route": route, "candidates": []}) + "\n",
        encoding="utf-8",
    )
    assert builder.validate_clean_retrieval(path) == (1, "a" * 64)
