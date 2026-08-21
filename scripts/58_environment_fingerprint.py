"""Write a reviewable environment fingerprint for clean baseline v1."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.clean.environment import environment_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(config.ROOT / "artifacts" / "clean_v1" / "environment.json"),
    )
    args = parser.parse_args()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = environment_snapshot()
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"environment -> {output}")
    print(f"environment sha256={snapshot['fingerprint_sha256']}")


if __name__ == "__main__":
    main()
