"""Write complete branch audits for all 21 complex P2.4 tune questions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.devset.p24_semantic_audit_v2 import build_complete_complex_semantic_audit  # noqa: E402
from vifinqa.utils.io import read_jsonl, setup_stdout, write_json  # noqa: E402


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="artifacts/devset_p24/p24_tune_gold.final.jsonl")
    parser.add_argument("--output", default="artifacts/devset_p24/p24_tune_gold.semantic_audit.final.json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    audit = build_complete_complex_semantic_audit(read_jsonl(args.gold))
    write_json(output, audit)
    print(json.dumps({"schema_version": audit["schema_version"],
        "count": audit["count"], "detailed_check_count": audit["detailed_check_count"],
        "metadata_value_columns": audit["metadata_value_columns"],
        "duplicate_invariants": audit["duplicate_invariants"],
        "locked_opened": audit["locked_opened"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
