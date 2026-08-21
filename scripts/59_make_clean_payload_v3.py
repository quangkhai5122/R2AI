"""Build schema-9 clean payload with the frozen HF+NF4 Kaggle runner.

The schema remains 9 because the clean data/retrieval contract is unchanged.
Runner and validator bytes are added to the manifest, so this must be uploaded
as a new Kaggle dataset version rather than patched inside a notebook.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from tqdm.auto import tqdm


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    root = scripts_dir.parent
    runner_source = root / "kaggle" / "kaggle_clean_codegen_nf4.py"
    validator_source = root / "kaggle" / "validate_clean_codegen.py"
    missing = [str(path) for path in (runner_source, validator_source) if not path.is_file()]
    if missing:
        raise SystemExit(f"NF4 payload sources are missing: {missing}")

    v2 = _load(scripts_dir / "59_make_clean_payload_v2.py", "clean_payload_v2")
    builder = v2._load_builder()
    builder._validate_clean_retrieval = v2.validate_clean_retrieval

    original_copy2 = builder.shutil.copy2

    def copy2_with_nf4(source, destination, *args, **kwargs):
        source_path = Path(source).resolve()
        canonical_runner = (root / "kaggle" / "kaggle_clean_codegen.py").resolve()
        if source_path == canonical_runner:
            copied = original_copy2(runner_source, destination, *args, **kwargs)
            destination_dir = Path(destination).parent
            original_copy2(
                validator_source,
                destination_dir / "validate_clean_codegen.py",
            )
            return copied
        return original_copy2(source, destination, *args, **kwargs)

    builder.shutil.copy2 = copy2_with_nf4
    original_manifest = builder._build_manifest

    def build_manifest(*args, **kwargs):
        manifest = original_manifest(*args, **kwargs)
        manifest["runtime_profile"] = "hf-bitsandbytes-nf4-v1"
        manifest["default_model"] = "Qwen/Qwen2.5-Coder-7B-Instruct"
        manifest["quantization"] = {
            "backend": "bitsandbytes",
            "bits": 4,
            "quant_type": "nf4",
            "compute_dtype": "float16",
            "double_quant": True,
        }
        return manifest

    builder._build_manifest = build_manifest
    original_sha256 = builder._sha256
    hash_progress = tqdm(desc="payload hashing", unit="file", dynamic_ncols=True)

    def sha256_with_progress(path: Path) -> str:
        digest = original_sha256(path)
        hash_progress.update(1)
        hash_progress.set_postfix(file=path.name[:32], refresh=False)
        return digest

    builder._sha256 = sha256_with_progress
    try:
        builder.main()
    finally:
        hash_progress.close()


if __name__ == "__main__":
    main()
