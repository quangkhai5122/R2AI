"""Step 4 (local): package everything the Kaggle GPU notebook needs.

Creates artifacts/kaggle_payload/ :
    code/vifinqa/...            the python package
    code/kaggle_codegen.py      Kaggle entry script
    store/...                   parquet dual store
    retrieval.jsonl             retrieval output
Upload this FOLDER as a Kaggle Dataset (web UI or `kaggle datasets create`).

  python scripts/04_make_kaggle_payload.py
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa import __version__ as vifinqa_version
from vifinqa.utils.io import setup_stdout, ensure_dir
from vifinqa.utils.viet_text import fuzzy_scorer_provenance

PAYLOAD_SCHEMA_VERSION = 8
MANIFEST_NAME = "payload-manifest.json"
CODE_COPY_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.orig", "*.rej", "*.patch")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _build_manifest(out: Path) -> dict:
    """Fingerprint every runtime input copied to Kaggle.

    Dataset metadata is intentionally excluded because users edit its owner/id
    after packaging and it is not consumed by the runner.
    """
    files = [out / "retrieval.jsonl"]
    for dirname in ("code", "store", "targets"):
        files.extend(p for p in (out / dirname).rglob("*") if p.is_file())
    hashes = {
        p.relative_to(out).as_posix(): _sha256(p)
        for p in sorted(files, key=lambda x: x.as_posix())
    }
    return {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "vifinqa_version": vifinqa_version,
        "fuzzy_scorer": fuzzy_scorer_provenance(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": hashes,
    }


def _safe_output_path(root: Path, raw_path: str | Path) -> Path:
    """Payload creation replaces its output; confine that target to artifacts/."""
    root = root.resolve()
    artifacts_root = (root / "artifacts").resolve()
    out = Path(raw_path).resolve()
    try:
        rel = out.relative_to(artifacts_root)
    except ValueError as e:
        raise SystemExit(
            f"refusing to replace payload outside {artifacts_root}: {out}"
        ) from e
    if not rel.parts:
        raise SystemExit(f"refusing to replace the artifacts root itself: {out}")
    return out


def _existing_metadata(out: Path) -> dict:
    path = out / "dataset-metadata.json"
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _dataset_id(explicit: str, previous: dict, slug: str) -> str:
    dataset_id = explicit.strip() or str(previous.get("id", "")).strip()
    dataset_id = dataset_id or f"YOUR_KAGGLE_USERNAME/{slug}"
    parts = dataset_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise SystemExit("--dataset-id must have the form <username>/<dataset-slug>")
    return dataset_id


def main():
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", default=str(config.STORE_DIR))
    ap.add_argument("--retrieval", default=str(config.RETRIEVAL_JSONL))
    ap.add_argument(
        "--target-dir", default=str(config.ROOT / "artifacts" / "p22_targets"),
        help="optional directory of frozen LLM ID masks copied to payload/targets",
    )
    ap.add_argument("--out", default=str(config.KAGGLE_PAYLOAD_DIR))
    ap.add_argument("--dataset-slug", default="vifinqa-payload")
    ap.add_argument("--dataset-id", default="",
                    help="Kaggle id <username>/<slug>; defaults to the previous "
                         "payload metadata id when rebuilding")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate sources/target/metadata without replacing files")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1].resolve()
    out = _safe_output_path(root, args.out)
    store_dir = Path(args.store_dir).resolve()
    retrieval_path = Path(args.retrieval).resolve()
    target_dir = Path(args.target_dir).resolve() if str(args.target_dir).strip() else None
    if target_dir is not None and not target_dir.is_dir():
        target_dir = None
    code_dir = root / "vifinqa"
    kaggle_entry = root / "kaggle" / "kaggle_codegen.py"
    required = [store_dir / "reports.parquet", retrieval_path,
                code_dir, kaggle_entry]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"payload sources are missing: {missing}")
    if out == store_dir or store_dir in out.parents:
        raise SystemExit("--out must not be the store directory or live inside it")

    previous_meta = _existing_metadata(out)
    dataset_id = _dataset_id(args.dataset_id, previous_meta, args.dataset_slug)
    if args.dry_run:
        print(f"dry-run OK: source store={store_dir}")
        print(f"dry-run OK: retrieval={retrieval_path}")
        print(f"dry-run OK: target masks={target_dir or '(none)'}")
        print(f"dry-run OK: replace target={out}")
        print(f"dry-run OK: dataset id={dataset_id}")
        return

    if out.exists():
        shutil.rmtree(out)
    ensure_dir(out)

    shutil.copytree(code_dir, out / "code" / "vifinqa",
                    ignore=CODE_COPY_IGNORE)
    shutil.copy2(kaggle_entry, out / "code" / "kaggle_codegen.py")
    shutil.copytree(store_dir, out / "store")
    shutil.copy2(retrieval_path, out / "retrieval.jsonl")
    if target_dir is not None:
        shutil.copytree(target_dir, out / "targets")

    meta = {
        "title": args.dataset_slug,
        "id": dataset_id,
        "licenses": [{"name": "CC-BY-NC-4.0"}],
    }
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    manifest = _build_manifest(out)
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"payload ready: {out}  ({size_mb:.0f} MB)")
    print(f"payload schema={PAYLOAD_SCHEMA_VERSION}, verified files="
          f"{len(manifest['files'])}, target masks="
          f"{len(list((out / 'targets').glob('*'))) if (out / 'targets').is_dir() else 0}")
    print("next:")
    if dataset_id.startswith("YOUR_KAGGLE_USERNAME/"):
        print("  1) rerun with --dataset-id <username>/<slug> (recommended), or "
              "edit dataset-metadata.json once")
        print(f"  2) kaggle datasets create -p {out} --dir-mode zip")
    elif str(previous_meta.get("id", "")).strip() == dataset_id:
        print(f"  kaggle datasets version -p {out} -m \"refresh payload\" --dir-mode zip")
    else:
        print(f"  kaggle datasets create -p {out} --dir-mode zip")
    print("     (or upload the folder via kaggle.com/datasets -> New Dataset)")


if __name__ == "__main__":
    main()
