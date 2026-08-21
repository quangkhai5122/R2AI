import hashlib
import importlib.util
import json
from pathlib import Path


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_built_payload_has_corrected_selection_v2_validator():
    root = Path(__file__).resolve().parents[1]
    payload = root / "artifacts" / "clean_v1" / "kaggle_payload"
    manifest = json.loads((payload / "payload-manifest.json").read_text(encoding="utf-8"))
    packaged = payload / "code" / "validate_clean_codegen.py"
    source = root / "kaggle" / "validate_clean_codegen_v2.py"
    assert manifest["validation_profile"] == "clean-codegen-select-v2-v2"
    assert _sha256(packaged) == _sha256(source)
    assert _sha256(packaged) == manifest["files"]["code/validate_clean_codegen.py"]

    spec = importlib.util.spec_from_file_location("packaged_validator_v2", packaged)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.TRACE_SCHEMA_VERSION == 2
    assert module.TRACE_MODE == "select_v2"
