"""Build strict P2.4 tune gold from compact exact-cell authoring specs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.devset.p24_authoring import build_tune_gold_file  # noqa: E402
from vifinqa.utils.io import setup_stdout  # noqa: E402


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", required=True)
    parser.add_argument("--bundle-dir", default="artifacts/devset_p24")
    parser.add_argument("--store-dir", default="artifacts/store")
    parser.add_argument("--output", default="artifacts/devset_p24/p24_tune_gold.jsonl")
    args = parser.parse_args()
    result = build_tune_gold_file(args.specs, args.bundle_dir, args.output, args.store_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
