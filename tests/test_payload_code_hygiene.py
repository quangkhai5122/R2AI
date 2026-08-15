import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "payload_builder_hygiene", ROOT / "scripts" / "04_make_kaggle_payload.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_payload_code_copy_excludes_editor_and_patch_backups():
    names = [
        "module.py", "module.pyc", "module.py.orig", "notes.rej",
        "hotfix.patch", "__pycache__",
    ]
    ignored = set(MODULE.CODE_COPY_IGNORE("unused", names))
    assert ignored == {
        "module.pyc",
        "module.py.orig",
        "notes.rej",
        "hotfix.patch",
        "__pycache__",
    }
