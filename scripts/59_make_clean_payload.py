"""Build the fail-closed schema-9 payload for clean canonical baseline v1."""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import __version__ as vifinqa_version
from vifinqa import config
from vifinqa.clean.environment import environment_snapshot
from vifinqa.clean.profile import (
    CLEAN_PROFILE,
    PAYLOAD_SCHEMA_VERSION,
    canonical_json_sha256,
    contract_fingerprints,
)
from vifinqa.utils.io import ensure_dir, setup_stdout
from vifinqa.utils.viet_text import fuzzy_scorer_provenance

MANIFEST_NAME = "payload-manifest.json"
CODE_COPY_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.orig", "*.rej", "*.patch",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_fingerprint(root: Path) -> str:
    rows = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    }
    return canonical_json_sha256(rows)


def _validate_clean_retrieval(path: Path) -> tuple[int, str]:
    count = 0
    config_hashes = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid retrieval JSON at line {line_number}: {exc}") from exc
            route = record.get("route") or {}
            if route.get("clean_profile") != CLEAN_PROFILE:
                raise SystemExit(
                    f"retrieval line {line_number} is not clean-profile canonical retrieval"
                )
            if not route.get("metric_keys"):
                raise SystemExit(f"retrieval line {line_number} has no canonical metric_keys")
            config_hashes.add(str(route.get("retrieval_config_sha256") or ""))
            count += 1
    if not count:
        raise SystemExit("clean retrieval is empty")
    if "" in config_hashes or len(config_hashes) != 1:
        raise SystemExit("clean retrieval mixes or omits retrieval config fingerprints")
    return count, next(iter(config_hashes))


def _safe_output_path(root: Path, raw_path: str | Path) -> Path:
    artifacts = (root / "artifacts").resolve()
    output = Path(raw_path).resolve()
    try:
        relative = output.relative_to(artifacts)
    except ValueError as exc:
        raise SystemExit(f"refusing to replace payload outside {artifacts}: {output}") from exc
    if not relative.parts:
        raise SystemExit("refusing to replace the artifacts root")
    return output


def _build_manifest(output: Path, retrieval_count: int,
                    retrieval_config_sha256: str) -> dict:
    files = [path for path in output.rglob("*") if path.is_file()
             and path.name not in {MANIFEST_NAME, "dataset-metadata.json"}]
    hashes = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in sorted(files, key=lambda item: item.as_posix())
    }
    fingerprints = contract_fingerprints()
    fingerprints.update({
        "retrieval_sha256": hashes["retrieval.jsonl"],
        "retrieval_config_sha256": retrieval_config_sha256,
        "store_tree_sha256": _tree_fingerprint(output / "store"),
        "code_tree_sha256": _tree_fingerprint(output / "code"),
        "environment_sha256": json.loads(
            (output / "environment.json").read_text(encoding="utf-8")
        )["fingerprint_sha256"],
    })
    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "profile": CLEAN_PROFILE,
        "public_id_masks": False,
        "official_derived_gold": False,
        "vifinqa_version": vifinqa_version,
        "fuzzy_scorer": fuzzy_scorer_provenance(),
        "retrieval_records": retrieval_count,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprints": fingerprints,
        "files": hashes,
    }


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", default=str(config.STORE_DIR))
    parser.add_argument(
        "--retrieval",
        default=str(config.ROOT / "artifacts" / "clean_v1" / "retrieval.jsonl"),
    )
    parser.add_argument(
        "--out",
        default=str(config.ROOT / "artifacts" / "clean_v1" / "kaggle_payload"),
    )
    parser.add_argument("--dataset-slug", default="vifinqa-clean-canonical-v1")
    parser.add_argument("--dataset-id", default="YOUR_KAGGLE_USERNAME/vifinqa-clean-canonical-v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1].resolve()
    output = _safe_output_path(root, args.out)
    store = Path(args.store_dir).resolve()
    retrieval = Path(args.retrieval).resolve()
    required = [
        store / "reports.parquet",
        retrieval,
        root / "vifinqa",
        root / "kaggle" / "kaggle_codegen.py",
        root / "kaggle" / "kaggle_clean_codegen.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"clean payload sources are missing: {missing}")
    retrieval_count, retrieval_config_hash = _validate_clean_retrieval(retrieval)
    if args.dry_run:
        print(f"dry-run clean OK: retrieval records={retrieval_count}")
        print(f"dry-run clean OK: retrieval config={retrieval_config_hash}")
        print(f"dry-run clean OK: target masks=forbidden")
        print(f"dry-run clean OK: output={output}")
        return

    if output.exists():
        shutil.rmtree(output)
    ensure_dir(output)
    shutil.copytree(root / "vifinqa", output / "code" / "vifinqa",
                    ignore=CODE_COPY_IGNORE)
    shutil.copy2(root / "kaggle" / "kaggle_codegen.py",
                 output / "code" / "kaggle_codegen.py")
    shutil.copy2(root / "kaggle" / "kaggle_clean_codegen.py",
                 output / "code" / "kaggle_clean_codegen.py")
    shutil.copytree(store, output / "store")
    shutil.copy2(retrieval, output / "retrieval.jsonl")
    (output / "environment.json").write_text(
        json.dumps(environment_snapshot(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "title": args.dataset_slug,
        "id": args.dataset_id,
        "licenses": [{"name": "CC-BY-NC-4.0"}],
    }
    (output / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = _build_manifest(output, retrieval_count, retrieval_config_hash)
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    size_mb = sum(path.stat().st_size for path in output.rglob("*")
                  if path.is_file()) / 1e6
    print(f"clean payload ready: {output} ({size_mb:.0f} MB)")
    print(f"schema={PAYLOAD_SCHEMA_VERSION}; files={len(manifest['files'])}; public_id_masks=false")


if __name__ == "__main__":
    main()
