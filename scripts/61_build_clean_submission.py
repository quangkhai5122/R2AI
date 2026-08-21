"""Build a submission only from clean-profile retrieval and complete results."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.submission.build import build_submission


def _assert_clean(path: Path) -> None:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any((row.get("route") or {}).get("clean_profile") != "clean"
                       for row in rows):
        raise SystemExit("submission builder requires clean-profile retrieval")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", default=str(config.ROOT / "artifacts" / "clean_v1" / "retrieval.jsonl"))
    parser.add_argument("--codegen", default=str(config.ROOT / "artifacts" / "clean_v1" / "b1_results.jsonl"))
    parser.add_argument("--store-dir", default=str(config.STORE_DIR))
    parser.add_argument("--out-dir", default=str(config.ROOT / "artifacts" / "clean_v1" / "submission"))
    parser.add_argument("--sub-k", type=int, default=5)
    args = parser.parse_args()
    retrieval = Path(args.retrieval)
    _assert_clean(retrieval)
    build_submission(
        retrieval, Path(args.codegen), Path(args.store_dir), Path(args.out_dir),
        sub_k=args.sub_k, pos_mode="line", expand_docs=False,
    )


if __name__ == "__main__":
    main()
