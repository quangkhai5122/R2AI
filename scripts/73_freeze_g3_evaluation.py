"""Create or validate the final G3A/G3B evaluation-contract freeze."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.g3b.freeze import (
    create_evaluation_freeze,
    validate_evaluation_freeze,
)
from vifinqa.utils.io import setup_stdout

DEFAULT_OUT = "experiments/g3_evaluation_v1/g3_evaluation_freeze.json"


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--g3a-v1-dir", default="data/g3a_v1")
    parser.add_argument(
        "--extension-dir", default="data/g3a_extension_v1"
    )
    parser.add_argument("--corpus-dir", default="data/g3b_v1")
    parser.add_argument(
        "--config", default="configs/g3b_evaluation_v1.json"
    )
    parser.add_argument("--repo-root", default=".")


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    _common(create)
    create.add_argument("--out", default=DEFAULT_OUT)
    validate = commands.add_parser("validate")
    _common(validate)
    validate.add_argument("--freeze", default=DEFAULT_OUT)
    args = parser.parse_args()
    kwargs = {
        "g3a_v1_dir": args.g3a_v1_dir,
        "extension_dir": args.extension_dir,
        "corpus_dir": args.corpus_dir,
        "config_path": args.config,
        "repo_root": args.repo_root,
    }
    if args.command == "create":
        result = create_evaluation_freeze(
            **kwargs, output_path=args.out
        )
        output = {
            "created": args.out,
            "fingerprint_sha256": result["fingerprint_sha256"],
        }
    else:
        output = validate_evaluation_freeze(
            args.freeze, **kwargs
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
