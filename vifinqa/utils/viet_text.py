"""Vietnamese text normalization + deterministic fuzzy matching.

The scorer is deliberately pinned to the standard-library implementation.
Previously, merely installing ``rapidfuzz`` changed shortlist scores while the
source code and payload hashes stayed identical.  Kaggle P2.1 used the
``difflib`` path, so keeping that algorithm as an explicit versioned contract
both reproduces the submitted run and removes optional-dependency drift.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

FUZZY_SCORER_BACKEND = "difflib.SequenceMatcher"
FUZZY_SCORER_VERSION = "1"


def fuzzy_scorer_provenance() -> dict[str, str]:
    """Return the semantic scorer contract fingerprinted by Kaggle runs."""
    return {
        "backend": FUZZY_SCORER_BACKEND,
        "version": FUZZY_SCORER_VERSION,
    }

_NON_ALNUM = re.compile(r"[^0-9a-z\s/%]")
_MULTI_WS = re.compile(r"\s+")


def strip_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("đ", "d").replace("Đ", "D")


def norm(s: str) -> str:
    """lowercase, no diacritics, alnum only, collapsed whitespace."""
    if s is None:
        return ""
    s = strip_diacritics(str(s)).lower()
    s = _NON_ALNUM.sub(" ", s)
    return _MULTI_WS.sub(" ", s).strip()


def tokens(s: str) -> list[str]:
    return norm(s).split()


COMPANY_PREFIXES = [
    "cong ty co phan", "ctcp", "cong ty cp", "tong cong ty co phan",
    "tong ctcp", "tong cong ty", "ngan hang thuong mai co phan",
    "ngan hang tmcp", "ngan hang", "tap doan", "cong ty tnhh", "cong ty",
]


def strip_company_prefixes(name_norm: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in COMPANY_PREFIXES:
            if name_norm.startswith(p + " "):
                name_norm = name_norm[len(p):].strip()
                changed = True
    return name_norm


def fuzz_token_set(a: str, b: str) -> float:
    """0..100 token-set similarity on normalized strings."""
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    sa, sb = " ".join(sorted(set(a.split()))), " ".join(sorted(set(b.split())))
    return 100.0 * difflib.SequenceMatcher(None, sa, sb).ratio()


def label_metric_score(label: str, metric: str) -> float:
    """0..100 score for 'does this row label express this metric phrase'.

    token_set_ratio alone gives 100 to any subset ('Chi phi khac' vs
    'chi phi luong va cac khoan khac') -> combine metric-token coverage with an
    order-insensitive full-string ratio.
    """
    lt, mt = set(tokens(label)), set(tokens(metric))
    if not lt or not mt:
        return 0.0
    cov = len(lt & mt) / len(mt)
    sa = " ".join(sorted(norm(label).split()))
    sb = " ".join(sorted(norm(metric).split()))
    sort = 100.0 * difflib.SequenceMatcher(None, sa, sb).ratio()
    return 60.0 * cov + 0.4 * sort


def fuzz_partial(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return 100.0 * difflib.SequenceMatcher(None, a, b).ratio()
