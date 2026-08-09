"""Safely merge two complete codegen runs.

The primary run is authoritative.  A fallback record may replace it only when
the primary record is the exact structural placeholder emitted by codegen
(``source=none``, ``status=failed``, constant-zero query).  In particular, a
successful query whose computed answer is zero is never treated as missing.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path

from ..utils.io import read_jsonl, write_json, write_jsonl
from .semantic import all_dataframe_refs


POLICY = "primary_then_fallback_on_structural_none_v1"


def merge_codegen_hybrid(
    primary_path: Path,
    fallback_path: Path,
    out_path: Path,
    audit_path: Path | None = None,
    *,
    expected_primary_signature: str = "",
    expected_fallback_signature: str = "",
) -> dict:
    """Merge two codegen JSONL files and return the audit payload.

    Both inputs must be complete over the same unique IDs, contain the same
    question text, and have exactly one non-empty ``run_signature`` apiece.
    Records are copied whole from the selected run; only hybrid provenance and
    the derived hybrid ``run_signature`` are added/replaced.
    """
    primary_path = Path(primary_path)
    fallback_path = Path(fallback_path)
    out_path = Path(out_path)
    audit_path = Path(audit_path) if audit_path else out_path.with_suffix(".audit.json")

    primary_rows = _read_unique(primary_path, "primary")
    fallback_rows = _read_unique(fallback_path, "fallback")
    _validate_same_universe(primary_rows, fallback_rows)

    primary_signature = _one_signature(primary_rows, "primary")
    fallback_signature = _one_signature(fallback_rows, "fallback")
    _check_expected_signature(
        primary_signature, expected_primary_signature, "primary"
    )
    _check_expected_signature(
        fallback_signature, expected_fallback_signature, "fallback"
    )

    primary_sha256 = _sha256(primary_path)
    fallback_sha256 = _sha256(fallback_path)
    hybrid_signature = _hybrid_signature(
        primary_signature,
        fallback_signature,
        primary_sha256,
        fallback_sha256,
    )

    output = []
    selected_fallback_ids: list[int] = []
    unresolved_ids: list[int] = []
    counts = Counter()
    fallback_rejected = Counter()

    for qid in sorted(primary_rows):
        primary = primary_rows[qid]
        fallback = fallback_rows[qid]
        primary_missing = is_structural_none(primary)
        selected = primary
        selected_from = "primary"
        fallback_reason = "primary_not_structural_none"

        if primary_missing:
            fallback_reason = _fallback_rejection_reason(fallback)
            if not fallback_reason:
                _validate_executable_record(fallback, "fallback", qid)
                selected = fallback
                selected_from = "fallback"
                selected_fallback_ids.append(qid)
                counts["used_fallback"] += 1
            else:
                unresolved_ids.append(qid)
                fallback_rejected[fallback_reason] += 1
                counts["unresolved"] += 1
        else:
            counts["kept_primary"] += 1

        if selected_from == "primary" and selected.get("status") == "ok":
            _validate_executable_record(selected, "primary", qid)

        row = dict(selected)
        selected_signature = str(selected["run_signature"])
        row["run_signature"] = hybrid_signature
        row["hybrid_provenance"] = {
            "policy": POLICY,
            "selected_from": selected_from,
            "selected_run_signature": selected_signature,
            "primary_run_signature": primary_signature,
            "fallback_run_signature": fallback_signature,
            "primary_status": primary.get("status"),
            "primary_source": primary.get("source"),
            "primary_structural_none": primary_missing,
            "fallback_decision": "selected" if selected_from == "fallback" else fallback_reason,
        }
        output.append(row)

    write_jsonl(out_path, output)
    audit = {
        "schema_version": 1,
        "policy": POLICY,
        "hybrid_run_signature": hybrid_signature,
        "primary": {
            "path": str(primary_path),
            "sha256": primary_sha256,
            "run_signature": primary_signature,
            "rows": len(primary_rows),
        },
        "fallback": {
            "path": str(fallback_path),
            "sha256": fallback_sha256,
            "run_signature": fallback_signature,
            "rows": len(fallback_rows),
        },
        "output": {"path": str(out_path), "rows": len(output)},
        "counts": {
            "total": len(output),
            "kept_primary": counts["kept_primary"],
            "used_fallback": counts["used_fallback"],
            "unresolved": counts["unresolved"],
        },
        "fallback_rejected": dict(sorted(fallback_rejected.items())),
        "selected_fallback_ids": selected_fallback_ids,
        "unresolved_ids": unresolved_ids,
    }
    write_json(audit_path, audit)
    return audit


def is_structural_none(row: dict) -> bool:
    """Return True only for the generator's failed constant-zero placeholder."""
    if str(row.get("source", "")).strip().lower() != "none":
        return False
    if str(row.get("status", "")).strip().lower() != "failed":
        return False
    try:
        answer = float(row.get("answer", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    return math.isfinite(answer) and answer == 0.0 and _is_constant_zero(
        row.get("pandas_query")
    )


def _is_constant_zero(code) -> bool:
    text = str(code or "").strip()
    if not text:
        return True
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        return False
    if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
        return False
    try:
        return float(node.value) == 0.0
    except (TypeError, ValueError):
        return False


def _fallback_rejection_reason(row: dict) -> str:
    if str(row.get("status", "")).strip().lower() != "ok":
        return "fallback_not_ok"
    if str(row.get("source", "")).strip().lower() in ("", "none"):
        return "fallback_source_none"
    try:
        answer = float(row.get("answer"))
    except (TypeError, ValueError):
        return "fallback_answer_invalid"
    if not math.isfinite(answer):
        return "fallback_answer_invalid"
    return ""


def _validate_executable_record(row: dict, label: str, qid: int) -> None:
    code = str(row.get("pandas_query") or "").strip()
    try:
        refs = all_dataframe_refs(code)
    except SyntaxError as exc:
        raise ValueError(
            f"{label} id={qid}: pandas_query has invalid syntax: {exc.msg}"
        ) from exc
    used_vars = row.get("used_vars")
    if not isinstance(used_vars, list):
        raise ValueError(f"{label} id={qid}: used_vars must be a list")
    evidence_vars = {str(item.get("var", "")) for item in used_vars
                     if isinstance(item, dict)}
    missing = refs - evidence_vars
    if missing:
        raise ValueError(
            f"{label} id={qid}: query references variables without evidence: "
            f"{sorted(missing)}"
        )
    if not refs:
        raise ValueError(
            f"{label} id={qid}: status=ok query does not reference a dataframe"
        )


def _read_unique(path: Path, label: str) -> dict[int, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} codegen file not found: {path}")
    out: dict[int, dict] = {}
    duplicates = []
    for row in read_jsonl(path):
        if "id" not in row:
            raise ValueError(f"{label} row has no id")
        qid = int(row["id"])
        if qid in out:
            duplicates.append(qid)
        out[qid] = row
    if duplicates:
        raise ValueError(
            f"{label} contains duplicate ids: {sorted(set(duplicates))[:10]}"
        )
    if not out:
        raise ValueError(f"{label} codegen file is empty: {path}")
    return out


def _validate_same_universe(primary: dict[int, dict], fallback: dict[int, dict]) -> None:
    primary_ids, fallback_ids = set(primary), set(fallback)
    if primary_ids != fallback_ids:
        missing = sorted(primary_ids - fallback_ids)[:10]
        extra = sorted(fallback_ids - primary_ids)[:10]
        raise ValueError(
            "primary/fallback id sets differ: "
            f"missing_in_fallback={missing}, extra_in_fallback={extra}"
        )
    mismatches = []
    for qid in sorted(primary):
        a = _normalise_question(primary[qid].get("question"))
        b = _normalise_question(fallback[qid].get("question"))
        if not a or a != b:
            mismatches.append(qid)
    if mismatches:
        raise ValueError(
            "primary/fallback questions differ or are missing: "
            f"ids={mismatches[:10]}"
        )


def _normalise_question(value) -> str:
    return unicodedata.normalize("NFC", str(value or "")).strip()


def _one_signature(rows: dict[int, dict], label: str) -> str:
    missing = [qid for qid, row in rows.items()
               if not str(row.get("run_signature", "")).strip()]
    if missing:
        raise ValueError(
            f"{label} has rows without run_signature: ids={sorted(missing)[:10]}"
        )
    signatures = {str(row["run_signature"]).strip() for row in rows.values()}
    if len(signatures) != 1:
        raise ValueError(
            f"{label} has multiple run_signatures: {sorted(signatures)}"
        )
    return next(iter(signatures))


def _check_expected_signature(actual: str, expected: str, label: str) -> None:
    if expected and actual != expected:
        raise ValueError(
            f"{label} run_signature mismatch: expected={expected}, actual={actual}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hybrid_signature(
    primary_signature: str,
    fallback_signature: str,
    primary_sha256: str,
    fallback_sha256: str,
) -> str:
    payload = json.dumps(
        {
            "policy": POLICY,
            "primary_run_signature": primary_signature,
            "fallback_run_signature": fallback_signature,
            "primary_sha256": primary_sha256,
            "fallback_sha256": fallback_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
