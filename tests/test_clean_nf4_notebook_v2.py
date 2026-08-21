import json
from pathlib import Path


def _notebook():
    root = Path(__file__).resolve().parents[1]
    return json.loads(
        (root / "kaggle" / "vifinqa-clean-canonical-v1-nf4-v2.ipynb").read_text(
            encoding="utf-8"
        )
    )


def test_final_notebook_uses_side_by_side_nf4_launcher_and_guards():
    notebook = _notebook()
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "code/kaggle_clean_codegen_nf4.py" in source
    assert "Qwen/Qwen2.5-Coder-7B-Instruct" in source
    assert "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ" not in source
    assert "--load-4bit" in source
    assert "--require-complete-llm" in source
    assert "model_revision_observed" in source
    assert "DO_NOT_UPLOAD.txt" in source
    assert "submission_sha256" in source
    assert all(cell.get("id") for cell in notebook["cells"])


def test_final_notebook_full_command_resumes_and_smoke_isolated():
    notebook = _notebook()
    smoke_cell = "".join(notebook["cells"][3]["source"])
    full_cell = "".join(notebook["cells"][4]["source"])
    assert "--no-resume" in smoke_cell
    assert "codegen_smoke_nf4.jsonl" not in full_cell
    assert "--no-resume" not in full_cell
