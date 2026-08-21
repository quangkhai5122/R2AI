import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_built_payload_uses_side_by_side_nf4_launcher():
    root = Path(__file__).resolve().parents[1]
    payload = root / "artifacts" / "clean_v1" / "kaggle_payload"
    manifest = json.loads((payload / "payload-manifest.json").read_text(encoding="utf-8"))
    canonical = payload / "code" / "kaggle_clean_codegen.py"
    nf4 = payload / "code" / "kaggle_clean_codegen_nf4.py"
    validator = payload / "code" / "validate_clean_codegen.py"
    assert manifest["runtime_profile"] == "hf-bitsandbytes-nf4-v1"
    assert manifest["runtime_launcher"] == "code/kaggle_clean_codegen_nf4.py"
    assert _sha256(canonical) == manifest["files"]["code/kaggle_clean_codegen.py"]
    assert _sha256(nf4) == manifest["files"]["code/kaggle_clean_codegen_nf4.py"]
    assert _sha256(validator) == manifest["files"]["code/validate_clean_codegen.py"]
    # This payload is the immutable input of the completed run. Canonical source
    # may advance for the next payload, so internal manifest hashes control here.


def test_packaged_nf4_launcher_imports_canonical_verifier():
    root = Path(__file__).resolve().parents[1]
    code = root / "artifacts" / "clean_v1" / "kaggle_payload" / "code"
    sys.path.insert(0, str(code))
    try:
        path = code / "kaggle_clean_codegen_nf4.py"
        spec = importlib.util.spec_from_file_location("packaged_clean_nf4", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        assert callable(module.clean_v1.verify_clean_payload)
        assert Path(module.clean_v1.__file__).name == "kaggle_clean_codegen.py"
        module.clean_v1.verify_clean_payload(code.parent, code)
    finally:
        sys.path.remove(str(code))
