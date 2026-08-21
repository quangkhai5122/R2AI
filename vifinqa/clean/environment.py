"""Runtime/dependency fingerprinting without importing optional GPU stacks."""
from __future__ import annotations

import importlib.metadata
import platform
import sys

from .profile import canonical_json_sha256

PACKAGES = (
    "numpy", "pandas", "pyarrow", "beautifulsoup4", "lxml",
    "rapidfuzz", "torch", "transformers", "bitsandbytes", "pytest",
)


def environment_snapshot() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    payload = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": versions,
    }
    payload["fingerprint_sha256"] = canonical_json_sha256(payload)
    return payload

