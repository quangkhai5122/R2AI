import json
from pathlib import Path


def _notebook():
    root = Path(__file__).resolve().parents[1]
    return json.loads(
        (
            root
            / "kaggle"
            / "vifinqa-clean-canonical-b1-14b-nf4.ipynb"
        ).read_text(encoding="utf-8")
    )


def test_canonical_notebook_uses_qwen_14b_runtime_nf4_and_full_provenance():
    notebook = _notebook()
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "code/kaggle_clean_codegen_nf4.py" in source
    assert "Qwen/Qwen2.5-Coder-14B-Instruct" in source
    assert "Qwen/Qwen2.5-Coder-7B-Instruct" not in source
    assert "Instruct-AWQ" not in source
    assert "--load-4bit" in source
    assert "--require-complete-llm" in source
    assert "model_revision_observed" in source
    assert "run_config_nf4.json" in source
    assert "run_config_sha256" in source
    assert "elapsed_seconds" in source
    assert "SpecifierSet" in source
    assert "transformers': '==5.0.0'" in source
    assert "DO_NOT_UPLOAD.txt" in source
    assert "submission_sha256" in source
    assert all(cell.get("id") for cell in notebook["cells"])


def test_canonical_notebook_full_command_resumes_and_smoke_isolated():
    notebook = _notebook()
    smoke_cell = "".join(notebook["cells"][3]["source"])
    full_cell = "".join(notebook["cells"][4]["source"])
    assert "--no-resume" in smoke_cell
    assert "codegen_smoke_nf4.jsonl" not in full_cell
    assert "--no-resume" not in full_cell
    assert "full_started" in full_cell
    assert "full_finished" in full_cell
