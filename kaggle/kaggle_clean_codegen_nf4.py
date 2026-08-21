"""Schema-9 clean runner using the base Qwen checkpoint with NF4.

This launcher deliberately forbids pre-quantized AWQ/GPTQ checkpoints.  The
Kaggle runtime has changed quantization backends more than once; loading the
base checkpoint through bitsandbytes keeps the clean baseline on one explicit,
auditable path instead of depending on gptqmodel/autoawq compatibility.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import kaggle_clean_codegen as clean_v1
import kaggle_codegen as legacy

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
PAYLOAD_SCHEMA_VERSION = 9
RUNTIME_PROFILE = "hf-bitsandbytes-nf4-v1"
RUNTIME_PACKAGES = (
    "torch", "transformers", "accelerate", "bitsandbytes", "tokenizers",
    "numpy", "pandas", "pyarrow", "tqdm",
)


def _has_option(argv: list[str], name: str) -> bool:
    return any(value == name or value.startswith(name + "=") for value in argv)


def _pop_option(argv: list[str], name: str, default: str) -> tuple[list[str], str]:
    """Remove a launcher-only option before delegating to the legacy parser."""
    cleaned: list[str] = []
    value = default
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == name:
            if index + 1 >= len(argv):
                raise SystemExit(f"{name} requires a value")
            value = argv[index + 1]
            index += 2
            continue
        if item.startswith(name + "="):
            value = item.split("=", 1)[1]
            index += 1
            continue
        cleaned.append(item)
        index += 1
    return cleaned, value


def _inspect_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--backend", default="hf")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--llm-mode", default="select_v2")
    parser.add_argument("--llm-ids-file", default="")
    parser.add_argument("--skip-payload-verification", action="store_true")
    args, _unknown = parser.parse_known_args(argv)
    if args.backend != "hf":
        raise SystemExit(
            f"{RUNTIME_PROFILE} requires --backend hf; got {args.backend!r}"
        )
    lowered = args.model.lower()
    if "awq" in lowered or "gptq" in lowered:
        raise SystemExit(
            "clean NF4 runner forbids AWQ/GPTQ checkpoints; use "
            f"--model {DEFAULT_MODEL} --load-4bit. Installing gptqmodel is "
            "not part of this frozen baseline."
        )
    if args.llm_mode != "select_v2":
        raise SystemExit("clean runner requires --llm-mode select_v2")
    if args.llm_ids_file:
        raise SystemExit("clean runner forbids --llm-ids-file")
    if args.skip_payload_verification:
        raise SystemExit("clean runner forbids --skip-payload-verification")
    match = re.search(r"(?<![0-9.])(\d+(?:\.\d+)?)\s*[bB](?![A-Za-z])", args.model)
    if match and float(match.group(1)) > 15.0:
        raise SystemExit("model exceeds the organizer-confirmed 15B limit")
    return args


def prepare_legacy_argv(argv: list[str]) -> tuple[list[str], str]:
    argv, runtime_report = _pop_option(
        list(argv), "--runtime-report", "/kaggle/working/clean_runtime.json",
    )
    _inspect_args(argv)
    defaults = {
        "--model": DEFAULT_MODEL,
        "--backend": "hf",
        "--llm-mode": "select_v2",
        "--k": "0",
    }
    for option, value in defaults.items():
        if not _has_option(argv, option):
            argv.extend([option, value])
    if not _has_option(argv, "--load-4bit"):
        argv.append("--load-4bit")
    return argv, runtime_report


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in RUNTIME_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def runtime_preflight(model: str) -> dict:
    missing = [
        name for name in ("torch", "transformers", "accelerate", "bitsandbytes")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise SystemExit(
            "missing NF4 runtime packages: " + ", ".join(missing) + ". "
            "Run the notebook dependency preflight before code generation."
        )
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. In Kaggle Notebook settings, select a GPU "
            "accelerator before running the clean baseline."
        )
    gpus = []
    for index in range(torch.cuda.device_count()):
        capability = tuple(int(v) for v in torch.cuda.get_device_capability(index))
        if capability < (6, 0):
            raise SystemExit(
                f"GPU {index} compute capability {capability} is too old for "
                "the frozen bitsandbytes NF4 path"
            )
        props = torch.cuda.get_device_properties(index)
        gpus.append({
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "compute_capability": list(capability),
            "total_memory_bytes": int(props.total_memory),
        })
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_profile": RUNTIME_PROFILE,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "model": model,
        "quantization": {
            "backend": "bitsandbytes",
            "bits": 4,
            "quant_type": "nf4",
            "compute_dtype": "float16",
            "double_quant": True,
        },
        "packages": _package_versions(),
        "cuda": str(torch.version.cuda or ""),
        "gpus": gpus,
    }


def _write_runtime_report(path: str, report: dict, manifest_hash: str) -> None:
    report = dict(report)
    report["payload_manifest_sha256"] = manifest_hash
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"runtime report -> {target}", flush=True)


def main() -> None:
    argv, runtime_report_path = prepare_legacy_argv(list(sys.argv[1:]))
    args = _inspect_args(argv)
    payload = Path(args.payload)
    runtime_code = Path(__file__).resolve().parent
    manifest, manifest_hash = clean_v1.verify_clean_payload(payload, runtime_code)

    report = runtime_preflight(args.model)
    _write_runtime_report(runtime_report_path, report, manifest_hash)
    print(
        f"clean runtime: {RUNTIME_PROFILE} | model={args.model} | "
        "load_4bit=true",
        flush=True,
    )

    def verified(_payload, _runtime_code_dir=None):
        return manifest, manifest_hash

    legacy.PAYLOAD_SCHEMA_VERSION = PAYLOAD_SCHEMA_VERSION
    legacy.verify_payload = verified
    sys.argv = [sys.argv[0], *argv]
    legacy.main()


if __name__ == "__main__":
    main()
