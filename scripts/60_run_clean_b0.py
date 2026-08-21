"""Run the deterministic B0 answer path on verified clean retrieval."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.clean.profile import CLEAN_PROFILE
from vifinqa.codegen.generate import run_codegen
from vifinqa.codegen.llm_client import NoLLM
from vifinqa.utils.io import setup_stdout


def _validate(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            route = record.get("route") or {}
            if route.get("clean_profile") != CLEAN_PROFILE or not route.get("metric_keys"):
                raise SystemExit(f"line {line_number} is not clean canonical retrieval")
            count += 1
    if not count:
        raise SystemExit("clean retrieval is empty")
    return count


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", default=str(config.ROOT / "artifacts" / "clean_v1" / "retrieval.jsonl"))
    parser.add_argument("--store-dir", default=str(config.STORE_DIR))
    parser.add_argument("--out", default=str(config.ROOT / "artifacts" / "clean_v1" / "b0_results.jsonl"))
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    retrieval = Path(args.retrieval)
    total = _validate(retrieval)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    run_codegen(
        retrieval, Path(args.store_dir), Path(args.out), client=NoLLM(),
        k=args.k, limit=args.limit, use_rule_fallback=True,
        run_signature="clean-b0-deterministic-v1",
    )
    print(f"B0 clean -> {args.out} ({min(total, args.limit) if args.limit else total} records)")


if __name__ == "__main__":
    main()
