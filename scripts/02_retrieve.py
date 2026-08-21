"""Step 2 (local, CPU): structured lookup + BM25 retrieval -> retrieval.jsonl

  python scripts/02_retrieve.py
  python scripts/02_retrieve.py --limit 30            # smoke test
  python scripts/02_retrieve.py --questions artifacts/validation/val_questions.jsonl \
         --out artifacts/val_retrieval.jsonl          # on the synthetic val set
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.retrieval.retrieve import run_retrieval
from vifinqa.utils.io import setup_stdout


def main():
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(config.QUESTIONS_JSONL))
    ap.add_argument("--store-dir", default=str(config.STORE_DIR))
    ap.add_argument("--code-stock", default=str(config.CODE_STOCK_CSV))
    ap.add_argument("--out", default=str(config.RETRIEVAL_JSONL))
    ap.add_argument("--route-source", default="",
                    help="frozen retrieval JSONL whose routes are reused; only ranking changes")
    ap.add_argument("--freeze-candidate-pool", action="store_true",
                    help="rerank only route-source candidates; requires RRF + route source")
    ap.add_argument("--depth", type=int, default=config.RETRIEVE_DEPTH)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--row-rerank", action="store_true",
                    help="rerank table candidates with row-level shortlist scores")
    ap.add_argument("--row-score-weight", type=float, default=0.18,
                    help="legacy weight for row-level shortlist score")
    ap.add_argument("--retrieval-mode", choices=["legacy", "rrf"], default="legacy",
                    help="legacy preserves the checkpoint ranking; rrf enables "
                         "independent rank fusion")
    ap.add_argument("--use-dense", action="store_true",
                    help="add cached BGE-M3 row-label ranking in rrf mode")
    ap.add_argument("--dense-model", default="BAAI/bge-m3")
    ap.add_argument("--dense-cache-dir", default="",
                    help="default: <store-dir>/label_index")
    ap.add_argument("--dense-device", default=None, help="cuda / cpu / auto")
    ap.add_argument("--dense-required", action="store_true",
                    help="fail instead of falling back when dense cannot load")
    ap.add_argument("--rrf-k", type=float, default=60.0,
                    help="RRF rank constant; only used by --retrieval-mode rrf")
    ap.add_argument("--pool-factor", type=int, default=5,
                    help="candidate pool multiplier before final depth cutoff")
    ap.add_argument("--dense-min-similarity", type=float, default=0.35)
    args = ap.parse_args()

    run_retrieval(Path(args.questions), Path(args.store_dir), Path(args.code_stock),
                  Path(args.out), args.depth, args.limit,
                  route_source_path=Path(args.route_source) if args.route_source else None,
                  freeze_candidate_pool=args.freeze_candidate_pool,
                  row_rerank=args.row_rerank,
                  row_score_weight=args.row_score_weight,
                  retrieval_mode=args.retrieval_mode,
                  use_dense=args.use_dense,
                  dense_model=args.dense_model,
                  dense_cache_dir=Path(args.dense_cache_dir) if args.dense_cache_dir else None,
                  dense_device=args.dense_device,
                  dense_required=args.dense_required,
                  rrf_k=args.rrf_k,
                  pool_factor=args.pool_factor,
                  dense_min_similarity=args.dense_min_similarity)


if __name__ == "__main__":
    main()
