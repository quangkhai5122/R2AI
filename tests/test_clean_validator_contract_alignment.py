import importlib.util
from pathlib import Path

from vifinqa.codegen import selection_v2


def test_clean_validator_matches_selection_v2_producer_contract():
    root = Path(__file__).resolve().parents[1]
    path = root / "kaggle" / "validate_clean_codegen_v2.py"
    spec = importlib.util.spec_from_file_location("clean_validator_contract", path)
    validator = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(validator)
    assert validator.TRACE_SCHEMA_VERSION == selection_v2.SCHEMA_VERSION == 2
    assert validator.TRACE_MODE == "select_v2"
