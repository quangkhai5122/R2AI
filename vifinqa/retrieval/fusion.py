"""Deterministic rank fusion for multi-channel table retrieval.

Raw BM25, cosine and schema-linking scores are not calibrated to the same
scale. Reciprocal Rank Fusion combines their rankings without pretending the
underlying values are directly comparable.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence


def rank_positions(items: Sequence[Hashable]) -> dict[Hashable, int]:
    """Return one-based ranks, preserving the first occurrence of each item."""
    return {item: rank for rank, item in enumerate(dict.fromkeys(items), start=1)}


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[Hashable]],
    *,
    rank_constant: float = 60.0,
    weights: Mapping[str, float] | None = None,
) -> tuple[dict[Hashable, float], dict[Hashable, dict[str, int]]]:
    """Fuse named rankings and return scores plus rank provenance.

    Missing items contribute zero for a channel. Non-positive weights disable
    a channel, which makes controlled ablations explicit.
    """
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")

    channel_weights = dict(weights or {})
    scores: dict[Hashable, float] = {}
    provenance: dict[Hashable, dict[str, int]] = {}
    for channel, ordered in rankings.items():
        weight = float(channel_weights.get(channel, 1.0))
        if weight <= 0:
            continue
        for item, rank in rank_positions(ordered).items():
            scores[item] = scores.get(item, 0.0) + weight / (rank_constant + rank)
            provenance.setdefault(item, {})[channel] = rank
    return scores, provenance
