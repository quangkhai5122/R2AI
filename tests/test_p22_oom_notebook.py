import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_oom_tail_notebook_is_retired_fail_fast():
    path = ROOT / "kaggle" / "vifinqa-codegen-p22-oom-tail.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert notebook["nbformat"] == 4
    assert "RETIRED: do not run" in source
    first_code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    first_source = "".join(first_code["source"])
    assert first_source.startswith("raise RuntimeError('RETIRED notebook")
    assert "schema 7" in first_source
