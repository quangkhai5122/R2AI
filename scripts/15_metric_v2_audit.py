"""Audit canonical metric v2 coverage and unresolved phrase clusters."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.finance.metrics_v2 import PROFILE_VERSION, profile_keys
from vifinqa.router.entities import StockMap, parse_question
from vifinqa.utils.io import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(config.QUESTIONS_JSONL))
    parser.add_argument("--code-stock", default=str(config.CODE_STOCK_CSV))
    parser.add_argument("--codegen", default="")
    parser.add_argument("--out", default=str(config.ART_DIR / "metric_v2_audit.json"))
    args = parser.parse_args()

    stock = StockMap(Path(args.code_stock))
    questions = read_jsonl(Path(args.questions))
    source_by_id = {}
    if args.codegen:
        source_by_id = {int(row["id"]): row.get("source", "")
                        for row in read_jsonl(Path(args.codegen))}

    profiles = Counter()
    unresolved = Counter()
    profile_rows = []
    for question in questions:
        parsed = parse_question(question["question"], stock)
        keys = profile_keys([
            parsed.metric_norm, parsed.metric_wide,
            *parsed.metric_variants, question["question"],
        ])
        if keys:
            profiles.update(keys)
        elif not parsed.metric_keys:
            unresolved[parsed.metric_norm or "<empty>"] += 1
        profile_rows.append({
            "id": question["id"], "metric": parsed.metric_norm,
            "v1_keys": parsed.metric_keys, "v2_keys": keys,
            "source": source_by_id.get(int(question["id"]), ""),
        })

    report = {
        "profile_version": PROFILE_VERSION,
        "questions": len(questions),
        "v2_recognized": sum(bool(row["v2_keys"]) for row in profile_rows),
        "v1_or_v2_recognized": sum(bool(row["v1_keys"] or row["v2_keys"])
                                    for row in profile_rows),
        "unrecognized": sum(not row["v1_keys"] and not row["v2_keys"]
                            for row in profile_rows),
        "profile_counts": profiles.most_common(),
        "top_unresolved_phrases": unresolved.most_common(100),
        "rows": profile_rows,
    }
    write_json(Path(args.out), report)
    print(json.dumps({key: report[key] for key in
                      ("profile_version", "questions", "v2_recognized",
                       "v1_or_v2_recognized", "unrecognized")}, indent=2))
    print("top v2 profiles:", profiles.most_common(20))
    print("->", args.out)


if __name__ == "__main__":
    main()
