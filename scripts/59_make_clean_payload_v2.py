"""Schema-9 payload builder hotfix for valid canonical misses.

Delegates copying/manifest construction to the original clean builder while
relaxing only the incorrect non-empty ``metric_keys`` requirement.  Progress
is visible during retrieval validation and file hashing.
"""
import importlib.util
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm


def validate_clean_retrieval(path: Path) -> tuple[int, str]:
    count = 0
    empty_keys = 0
    config_hashes = set()
    with path.open(encoding="utf-8") as handle:
        lines = sum(1 for line in handle if line.strip())
    with path.open(encoding="utf-8") as handle:
        progress = tqdm(
            (line for line in handle if line.strip()), total=lines,
            desc="validate clean retrieval", unit="record", dynamic_ncols=True,
        )
        for line_number, line in enumerate(progress, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"invalid retrieval JSON at non-empty line {line_number}: {exc}"
                ) from exc
            route = record.get("route") or {}
            if route.get("clean_profile") != "clean":
                raise SystemExit(
                    f"retrieval record {line_number} is not clean-profile retrieval"
                )
            if "metric_keys" not in route or not isinstance(route["metric_keys"], list):
                raise SystemExit(
                    f"retrieval record {line_number} lacks a list-valued metric_keys field"
                )
            if not route["metric_keys"]:
                empty_keys += 1
            if not isinstance(route.get("metric_variants"), list) or not route["metric_variants"]:
                raise SystemExit(
                    f"retrieval record {line_number} has no lexical metric fallback"
                )
            config_hashes.add(str(route.get("retrieval_config_sha256") or ""))
            count += 1
            progress.set_postfix(canonical_misses=empty_keys, refresh=False)
    if not count:
        raise SystemExit("clean retrieval is empty")
    if "" in config_hashes or len(config_hashes) != 1:
        raise SystemExit("clean retrieval mixes or omits retrieval config fingerprints")
    print(f"canonical misses={empty_keys}; lexical fallback verified")
    return count, next(iter(config_hashes))


def _load_builder():
    path = Path(__file__).with_name("59_make_clean_payload.py")
    spec = importlib.util.spec_from_file_location("clean_payload_builder_v1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    builder = _load_builder()
    builder._validate_clean_retrieval = validate_clean_retrieval
    original_sha256 = builder._sha256
    hash_progress = tqdm(
        desc="payload hashing", unit="file", dynamic_ncols=True,
    )

    def sha256_with_progress(path: Path) -> str:
        digest = original_sha256(path)
        hash_progress.update(1)
        hash_progress.set_postfix(file=path.name[:32], refresh=False)
        return digest

    builder._sha256 = sha256_with_progress
    try:
        builder.main()
    finally:
        hash_progress.close()


if __name__ == "__main__":
    main()
