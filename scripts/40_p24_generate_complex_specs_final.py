"""Generate final complex P2.4 specs with strict value-column selection."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.devset import p24_complex_specs, p24_metrics_v4  # noqa: E402
from vifinqa.devset.p24 import canonical_sha256  # noqa: E402
from vifinqa.devset.p24_metrics import METRICS, MetricDef, _norm  # noqa: E402
from vifinqa.devset.p24_metrics_v6 import StandardMetricResolverV6  # noqa: E402
from vifinqa.utils.io import setup_stdout, write_jsonl  # noqa: E402


def _extend(metric: str, *labels: str) -> None:
    old = METRICS[metric]
    METRICS[metric] = MetricDef(old.labels + tuple(labels), old.codes, old.reject)


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", default="artifacts/store")
    parser.add_argument("--output", default="artifacts/devset_p24/authoring_parts/06_complex_corrected.jsonl")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    _extend("pat", "loi nhuan thuan sau thue", "lo thuan sau thue tndn",
        "loi nhuan lo sau thue tndn")
    _extend("pbt", "loi nhuan ke toan truoc thue", "tong loi nhuan lo ke toan truoc thue")
    _extend("basic_eps", "lai co ban tren moi co phieu")
    _extend("cfo", "luu chuyen tien thuan tu su dung vao hoat dong kinh doanh",
        "luu chuyen tien thuan su dung vao hoat dong kinh doanh",
        "luu chuyen tien te thuan tu hoat dong kinh doanh",
        "luu chuyen tien thuan tu cac hoat dong kinh doanh")
    p24_metrics_v4._compact = lambda value: re.sub(r"[^0-9a-z]+", "", _norm(value))
    p24_complex_specs.StandardMetricResolverV2 = StandardMetricResolverV6
    records = p24_complex_specs.build_complex_tune_specs(args.store_dir)
    write_jsonl(output, records)
    print(json.dumps({"count": len(records), "output": str(output),
        "records_sha256": canonical_sha256(records), "locked_opened": False,
        "resolver": "exact-label-v6+strict-value-column"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
