"""Deterministic clean B0 runner with tqdm, resume and atomic checkpoints."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm.auto import tqdm

from vifinqa import config
from vifinqa.clean.profile import CLEAN_PROFILE
from vifinqa.codegen.generate import (
    QuestionBundle,
    _empty_result,
    _flush,
    _load_previous,
    _rule_result,
    _srcs,
)
from vifinqa.extraction.build_store import Store
from vifinqa.utils.io import read_jsonl, setup_stdout

RUN_SIGNATURE = "clean-b0-deterministic-v2"


def validate_clean_retrieval(path: Path) -> tuple[list[dict], int]:
    records = []
    empty_keys = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"line {line_number} is invalid JSON: {exc}"
                ) from exc
            route = record.get("route") or {}
            if route.get("clean_profile") != CLEAN_PROFILE:
                raise SystemExit(
                    f"line {line_number} has clean_profile="
                    f"{route.get('clean_profile')!r}, expected 'clean'"
                )
            if "metric_keys" not in route or not isinstance(route["metric_keys"], list):
                raise SystemExit(
                    f"line {line_number} lacks a list-valued metric_keys field"
                )
            if not route["metric_keys"]:
                empty_keys += 1
            if not isinstance(route.get("metric_variants"), list) or not route["metric_variants"]:
                raise SystemExit(
                    f"line {line_number} has no lexical metric fallback"
                )
            if not route.get("retrieval_config_sha256"):
                raise SystemExit(
                    f"line {line_number} has no retrieval config fingerprint"
                )
            records.append(record)
    if not records:
        raise SystemExit("clean retrieval is empty")
    return records, empty_keys


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval",
        default=str(config.ROOT / "artifacts" / "clean_v1" / "retrieval.jsonl"),
    )
    parser.add_argument("--store-dir", default=str(config.STORE_DIR))
    parser.add_argument(
        "--out",
        default=str(config.ROOT / "artifacts" / "clean_v1" / "b0_results.jsonl"),
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.k < 0:
        parser.error("--k must be >= 0")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be >= 1")

    records, empty_keys = validate_clean_retrieval(Path(args.retrieval))
    if args.limit:
        records = records[:args.limit]
    print(
        f"validated {len(records)} clean records; "
        f"canonical misses={empty_keys} (lexical fallback allowed)"
    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = {} if args.no_resume else {
        qid: row for qid, row in _load_previous(output).items()
        if row.get("run_signature") == RUN_SIGNATURE
    }
    if results:
        print(f"resume: {len(results)} completed B0 records")
    store = Store(Path(args.store_dir), cache_size=4)
    progress = tqdm(
        records, desc="clean B0", unit="question", dynamic_ncols=True,
        initial=0,
    )
    processed = 0
    for record in progress:
        qid = record["id"]
        if qid in results:
            continue
        bundle = QuestionBundle(
            record, store, args.k, run_signature=RUN_SIGNATURE,
        )
        if not bundle.tables:
            result = _empty_result(bundle, "no candidate tables")
        else:
            result = _rule_result(bundle, None)
            if result is None:
                result = _empty_result(bundle, "rule found nothing")
        results[qid] = result
        processed += 1
        if processed % args.checkpoint_every == 0:
            _flush(output, records, results)
        progress.set_postfix(
            completed=len(results), source=result["source"], refresh=False,
        )

    _flush(output, records, results)
    print(f"B0 clean complete: {_srcs(records, results)} -> {output}")


if __name__ == "__main__":
    main()
