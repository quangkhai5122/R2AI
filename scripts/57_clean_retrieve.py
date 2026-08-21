"""Build the source-only canonical retrieval artifact for clean baseline v1."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.clean.retrieval import CleanRetrievalConfig, run_clean_retrieval
from vifinqa.utils.io import setup_stdout


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(config.QUESTIONS_JSONL))
    parser.add_argument("--store-dir", default=str(config.STORE_DIR))
    parser.add_argument("--code-stock", default=str(config.CODE_STOCK_CSV))
    parser.add_argument("--out", default=str(config.ROOT / "artifacts" / "clean_v1" / "retrieval.jsonl"))
    parser.add_argument("--config", default=str(config.ROOT / "configs" / "clean_canonical_baseline_v1" / "retrieval.json"))
    parser.add_argument("--depth", type=int, default=config.RETRIEVE_DEPTH)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    retrieval_config = CleanRetrievalConfig(**raw)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    run_clean_retrieval(
        Path(args.questions), Path(args.store_dir), Path(args.code_stock),
        Path(args.out), args.depth, args.limit, retrieval_config,
    )
    print(f"clean retrieval -> {args.out}")
    print(f"retrieval config sha256={retrieval_config.fingerprint()}")


if __name__ == "__main__":
    main()
