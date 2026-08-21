"""Fail-closed schema-9 launcher for clean canonical baseline v1.

The frozen schema-8 runner remains unchanged.  After verifying the stronger
clean manifest, this launcher delegates generation to that runner's mature
execution/checkpoint code with a manifest hash that includes the clean
contract and registry fingerprints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import kaggle_codegen as legacy

PAYLOAD_SCHEMA_VERSION = 9
CLEAN_PROFILE = "clean"
REQUIRED_FINGERPRINTS = {
    "metric_registry_sha256",
    "operator_registry_sha256",
    "retrieval_sha256",
    "retrieval_config_sha256",
    "store_tree_sha256",
    "code_tree_sha256",
    "environment_sha256",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_clean_payload(payload: Path, runtime_code_dir: Path | None = None):
    manifest_path = payload / "payload-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid clean payload manifest: {exc}") from exc
    if manifest.get("schema_version") != PAYLOAD_SCHEMA_VERSION:
        raise SystemExit(
            f"clean runner requires schema={PAYLOAD_SCHEMA_VERSION}; "
            f"got {manifest.get('schema_version')!r}"
        )
    if manifest.get("profile") != CLEAN_PROFILE:
        raise SystemExit("clean runner requires manifest profile=clean")
    if manifest.get("public_id_masks") is not False:
        raise SystemExit("clean runner requires public_id_masks=false")
    if manifest.get("official_derived_gold") is not False:
        raise SystemExit("clean runner requires official_derived_gold=false")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise SystemExit("clean manifest has no file hashes")
    if any(path == "targets" or path.startswith("targets/") for path in files):
        raise SystemExit("clean payload must not contain target/ID-mask artifacts")
    required = {
        "retrieval.jsonl",
        "store/reports.parquet",
        "environment.json",
        "code/kaggle_codegen.py",
        "code/kaggle_clean_codegen.py",
        "code/vifinqa/clean/profile.py",
        "code/vifinqa/finance/metrics.py",
        "code/vifinqa/finance/operators.py",
        "code/vifinqa/codegen/selection_v2.py",
    }
    missing = sorted(required - set(files))
    if missing:
        raise SystemExit(f"clean manifest misses required files: {missing}")
    fingerprints = manifest.get("fingerprints") or {}
    missing_fp = sorted(REQUIRED_FINGERPRINTS - set(fingerprints))
    if missing_fp:
        raise SystemExit(f"clean manifest misses fingerprints: {missing_fp}")
    for relative, expected in files.items():
        rel_path = Path(relative)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise SystemExit(f"invalid clean manifest path: {relative}")
        source = payload / rel_path
        if not source.is_file() or _sha256(source) != expected:
            raise SystemExit(f"clean payload hash mismatch: {relative}")
    if fingerprints["retrieval_sha256"] != files["retrieval.jsonl"]:
        raise SystemExit("retrieval fingerprint does not match the verified file")
    packaged_code = (payload / "code").resolve()
    if runtime_code_dir is not None and runtime_code_dir.resolve() != packaged_code:
        for relative, expected in files.items():
            if not relative.startswith("code/"):
                continue
            runtime = runtime_code_dir / Path(relative).relative_to("code")
            if not runtime.is_file() or _sha256(runtime) != expected:
                raise SystemExit(f"runtime code differs from clean payload: {relative}")
    stable = json.dumps({
        "schema_version": manifest["schema_version"],
        "profile": manifest["profile"],
        "public_id_masks": manifest["public_id_masks"],
        "official_derived_gold": manifest["official_derived_gold"],
        "fuzzy_scorer": manifest["fuzzy_scorer"],
        "fingerprints": fingerprints,
        "files": files,
    }, sort_keys=True, separators=(",", ":")).encode()
    return manifest, hashlib.sha256(stable).hexdigest()


def _has_option(argv: list[str], name: str) -> bool:
    return any(value == name or value.startswith(name + "=") for value in argv)


def _inspect_clean_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--llm-mode", default="select_v2")
    parser.add_argument("--llm-ids-file", default="")
    parser.add_argument("--skip-payload-verification", action="store_true")
    args, _unknown = parser.parse_known_args(argv)
    if args.llm_mode != "select_v2":
        raise SystemExit("clean runner requires --llm-mode select_v2")
    if args.llm_ids_file:
        raise SystemExit("clean runner forbids --llm-ids-file")
    if args.skip_payload_verification:
        raise SystemExit("clean runner forbids --skip-payload-verification")
    import re
    lowered = args.model.lower()
    if "awq" in lowered or "gptq" in lowered:
        raise SystemExit(
            "clean runner forbids AWQ/GPTQ checkpoints; use runtime NF4"
        )
    match = re.search(r"(?<![0-9.])(\d+(?:\.\d+)?)\s*[bB](?![A-Za-z])", args.model)
    if match and float(match.group(1)) > 15.0:
        raise SystemExit("model exceeds the organizer-confirmed 15B limit")
    return args


def main() -> None:
    argv = list(sys.argv[1:])
    args = _inspect_clean_args(argv)
    if not _has_option(argv, "--model"):
        argv.extend(["--model", "Qwen/Qwen2.5-Coder-14B-Instruct"])
    if not _has_option(argv, "--llm-mode"):
        argv.extend(["--llm-mode", "select_v2"])
    if not _has_option(argv, "--k"):
        argv.extend(["--k", "0"])
    if not _has_option(argv, "--load-4bit"):
        argv.append("--load-4bit")
    payload = Path(args.payload)
    runtime_code = Path(__file__).resolve().parent
    manifest, manifest_hash = verify_clean_payload(payload, runtime_code)

    def verified(_payload, _runtime_code_dir=None):
        return manifest, manifest_hash

    legacy.PAYLOAD_SCHEMA_VERSION = PAYLOAD_SCHEMA_VERSION
    legacy.verify_payload = verified
    sys.argv = [sys.argv[0], *argv]
    legacy.main()


if __name__ == "__main__":
    main()
