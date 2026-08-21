import importlib.util
import sys
from pathlib import Path

import pytest


def _load_runner():
    root = Path(__file__).resolve().parents[1]
    kaggle_dir = root / "kaggle"
    sys.path.insert(0, str(kaggle_dir))
    try:
        path = kaggle_dir / "kaggle_clean_codegen_nf4.py"
        spec = importlib.util.spec_from_file_location("clean_nf4_runner", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(kaggle_dir))


def test_nf4_runner_adds_frozen_defaults_and_strips_runtime_report():
    runner = _load_runner()
    argv, report = runner.prepare_legacy_argv([
        "--payload", "payload", "--runtime-report", "runtime.json",
    ])
    assert report == "runtime.json"
    assert argv[argv.index("--model") + 1] == runner.DEFAULT_MODEL
    assert "--load-4bit" in argv
    assert argv[argv.index("--backend") + 1] == "hf"
    assert argv[argv.index("--llm-mode") + 1] == "select_v2"
    assert argv[argv.index("--k") + 1] == "0"
    assert "--runtime-report" not in argv


@pytest.mark.parametrize("model", [
    "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
    "someone/model-7B-GPTQ",
])
def test_nf4_runner_rejects_prequantized_models(model):
    runner = _load_runner()
    with pytest.raises(SystemExit, match="forbids AWQ/GPTQ"):
        runner.prepare_legacy_argv([
            "--payload", "payload", "--model", model,
        ])


def test_nf4_runner_enforces_organizer_model_limit():
    runner = _load_runner()
    with pytest.raises(SystemExit, match="15B limit"):
        runner.prepare_legacy_argv([
            "--payload", "payload", "--model", "vendor/model-16B",
        ])
