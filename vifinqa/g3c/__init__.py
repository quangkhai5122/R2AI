"""G3C guarded Qwen retrieval ablations.

G3C is intentionally isolated from the frozen clean retriever and G3B
evaluation contract.  The package exposes deterministic, label-blind query
formation plus GPU-only neural backends.
"""

from .common import G3C_SCHEMA, load_config
from .leaves import AtomicLeaf, decompose_atomic_leaves

__all__ = [
    "AtomicLeaf",
    "G3C_SCHEMA",
    "decompose_atomic_leaves",
    "load_config",
]
