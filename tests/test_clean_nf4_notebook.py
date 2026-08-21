import json
from pathlib import Path


def test_clean_nf4_notebook_is_fail_closed_and_uses_base_model_nf4():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "kaggle" / "vifinqa-clean-canonical-v1-nf4.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "hf-bitsandbytes-nf4-v1" in source
    assert "Qwen/Qwen2.5-Coder-7B-Instruct" in source
    assert "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ" not in source
    assert "--load-4bit" in source
    assert "--require-complete-llm" in source
    assert "--no-resume" in source  # smoke only
    assert "codegen_results_nf4.jsonl" in source
    assert "DO_NOT_UPLOAD.txt" in source
    assert "submission_sha256" in source


def test_full_command_does_not_disable_resume():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "kaggle" / "vifinqa-clean-canonical-v1-nf4.ipynb").read_text(
            encoding="utf-8"
        )
    )
    full_cell = "".join(notebook["cells"][4]["source"])
    assert "full_cmd" in full_cell
    assert "--no-resume" not in full_cell
