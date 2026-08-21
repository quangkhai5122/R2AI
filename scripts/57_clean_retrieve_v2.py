"""Clean canonical retrieval v2 with visible progress and atomic output.

An empty ``metric_keys`` list is valid: it means the source-only ontology did
not recognize that phrase.  The route remains clean and falls back to its
normalized lexical metric variants.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm.auto import tqdm

from vifinqa import config
from vifinqa.clean.profile import CLEAN_PROFILE
from vifinqa.clean.retrieval import (
    CleanRetrievalConfig,
    canonicalize_route,
    retrieve_for_route,
)
from vifinqa.extraction.build_store import Store
from vifinqa.router.entities import StockMap
from vifinqa.router.router import route_question
from vifinqa.utils.io import read_jsonl, setup_stdout


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(config.QUESTIONS_JSONL))
    parser.add_argument("--store-dir", default=str(config.STORE_DIR))
    parser.add_argument("--code-stock", default=str(config.CODE_STOCK_CSV))
    parser.add_argument(
        "--out",
        default=str(config.ROOT / "artifacts" / "clean_v1" / "retrieval.jsonl"),
    )
    parser.add_argument(
        "--config",
        default=str(
            config.ROOT / "configs" / "clean_canonical_baseline_v1" / "retrieval.json"
        ),
    )
    parser.add_argument("--depth", type=int, default=config.RETRIEVE_DEPTH)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be >= 1")

    raw_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    retrieval_config = CleanRetrievalConfig(**raw_config)
    retrieval_config.validate()
    questions = read_jsonl(Path(args.questions))
    if args.limit:
        questions = questions[:args.limit]

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(output) + ".partial")
    store = Store(Path(args.store_dir))
    stock = StockMap(Path(args.code_stock))
    unknown_metrics = 0

    try:
        with partial.open("w", encoding="utf-8", newline="\n") as handle:
            progress = tqdm(
                questions, desc="clean retrieval", unit="question", dynamic_ncols=True,
            )
            for index, question in enumerate(progress, 1):
                route = route_question(
                    question["id"], question["question"], stock, store,
                )
                candidates = retrieve_for_route(
                    route, store, args.depth, retrieval_config,
                )
                variants, keys, qualifiers = canonicalize_route(route)
                if not keys:
                    unknown_metrics += 1
                route_dict = route.to_dict()
                route_dict.update({
                    "metric_variants": variants,
                    "metric_keys": keys,
                    "metric_qualifiers": qualifiers,
                    "clean_profile": CLEAN_PROFILE,
                    "retrieval_config_sha256": retrieval_config.fingerprint(),
                })
                record = {
                    "id": question["id"],
                    "question": question["question"],
                    "route": route_dict,
                    "candidates": candidates,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if index % args.checkpoint_every == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
                progress.set_postfix(
                    unknown_metric_keys=unknown_metrics,
                    candidates=len(candidates),
                    refresh=False,
                )
        os.replace(partial, output)
    except BaseException:
        print(f"retrieval interrupted; partial prefix retained at {partial}")
        raise

    print(f"clean retrieval -> {output} ({len(questions)} records)")
    print(
        f"canonical misses={unknown_metrics}; lexical fallback remains active; "
        f"config sha256={retrieval_config.fingerprint()}"
    )


if __name__ == "__main__":
    main()
