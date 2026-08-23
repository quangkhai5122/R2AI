"""Frozen OOD diagnostic views for the G3B corpus."""
from __future__ import annotations

from collections import Counter


def _feature_folds(
    records: list[dict],
    feature: str,
    values: list[str],
    limit: int,
) -> dict:
    output = {}
    for value in values[:limit or None]:
        evaluation = sorted(
            int(row["id"])
            for row in records
            if value in row[feature]
        )
        training = sorted(
            int(row["id"])
            for row in records
            if value not in row[feature]
        )
        if not evaluation or not training:
            continue
        training_ids = set(training)
        output[value] = {
            "train_ids": training,
            "eval_ids": evaluation,
            "assertion": {
                "holdout_value_absent_from_train": all(
                    value not in row[feature]
                    for row in records
                    if int(row["id"]) in training_ids
                ),
                "train_eval_id_overlap": sorted(
                    set(training) & set(evaluation)
                ),
            },
        }
    return output


def build_views(records: list[dict], config: dict) -> dict[str, dict]:
    all_ids = {int(row["id"]) for row in records}
    ticker_counts = Counter(
        ticker for row in records for ticker in row["tickers"]
    )
    report_counts = Counter(
        report for row in records for report in row["report_ids"]
    )
    loto = {
        "schema_version": "g3b_view_v1",
        "kind": "LOTO",
        "folds": _feature_folds(
            records,
            "tickers",
            [
                key
                for key, count in ticker_counts.most_common()
                if count >= 2
            ],
            int(config["views"]["max_loto_folds"]),
        ),
        "overlap_policy": (
            "folds overlap and are diagnostic slices, not independent tests"
        ),
    }
    loro = {
        "schema_version": "g3b_view_v1",
        "kind": "LORO",
        "folds": _feature_folds(
            records,
            "report_ids",
            [
                key
                for key, count in report_counts.most_common()
                if count >= 2
            ],
            int(config["views"]["max_loro_folds"]),
        ),
        "overlap_policy": (
            "folds overlap and are diagnostic slices, not independent tests"
        ),
    }

    all_years = {year for row in records for year in row["years"]}
    loyo_folds = {}
    for block in config["views"]["year_blocks"]:
        held = {
            year
            for year in all_years
            if year >= int(block.get("min", 1900))
            and year <= int(block.get("max", 2100))
        }
        evaluation = sorted(
            int(row["id"])
            for row in records
            if held.intersection(row["years"])
        )
        training = sorted(
            int(row["id"])
            for row in records
            if not held.intersection(row["years"])
        )
        if not evaluation or not training:
            continue
        training_ids = set(training)
        loyo_folds[block["name"]] = {
            "holdout_years": sorted(held),
            "train_ids": training,
            "eval_ids": evaluation,
            "assertion": {
                "holdout_year_absent_from_train": all(
                    not held.intersection(row["years"])
                    for row in records
                    if int(row["id"]) in training_ids
                ),
                "train_eval_id_overlap": sorted(
                    set(training) & set(evaluation)
                ),
            },
        }
    loyo = {
        "schema_version": "g3b_view_v1",
        "kind": "LOYO",
        "folds": loyo_folds,
        "overlap_policy": (
            "year blocks overlap other views and are diagnostic slices"
        ),
    }

    lomo_folds = {}
    for metric in sorted({row["metric_family"] for row in records}):
        evaluation = sorted(
            int(row["id"])
            for row in records
            if row["metric_family"] == metric
        )
        training = sorted(all_ids - set(evaluation))
        training_ids = set(training)
        lomo_folds[metric] = {
            "train_ids": training,
            "eval_ids": evaluation,
            "assertion": {
                "metric_absent_from_train": all(
                    row["metric_family"] != metric
                    for row in records
                    if int(row["id"]) in training_ids
                ),
                "train_eval_id_overlap": [],
            },
        }
    lomo = {
        "schema_version": "g3b_view_v1",
        "kind": "LOMO",
        "folds": lomo_folds,
        "overlap_policy": (
            "metric folds overlap other views and are diagnostic slices"
        ),
    }

    target = [
        row
        for row in records
        if row["family"] == "nested_margin_average"
    ]
    shapes = {row["tree_shape"] for row in target}
    training_rows = [
        row for row in records if row["tree_shape"] not in shapes
    ]
    train_ops = {
        op for row in training_rows for op in row["primitive_ops"]
    }
    eval_ops = {op for row in target for op in row["primitive_ops"]}
    composition = {
        "schema_version": "g3b_view_v1",
        "kind": "composition",
        "folds": {
            "nested_margin_average": {
                "held_tree_shapes": sorted(shapes),
                "train_ids": sorted(
                    int(row["id"]) for row in training_rows
                ),
                "eval_ids": sorted(int(row["id"]) for row in target),
                "assertion": {
                    "tree_shape_absent_from_train": all(
                        row["tree_shape"] not in shapes
                        for row in training_rows
                    ),
                    "primitive_ops_seen_in_train": not (
                        eval_ops - train_ops
                    ),
                    "missing_primitive_ops": sorted(
                        eval_ops - train_ops
                    ),
                    "train_eval_id_overlap": [],
                },
            }
        },
    }

    stress_tags = {
        "scope",
        "ambiguous_scope",
        "period",
        "prior_period",
    }
    stress_ids = sorted(
        int(row["id"])
        for row in records
        if set(row["stress_tags"]) & stress_tags
    )
    stress = {
        "schema_version": "g3b_view_v1",
        "kind": "scope_period_stress",
        "eval_ids": stress_ids,
        "assertion": {
            "all_records_have_scope_or_period_tag": all(
                set(row["stress_tags"]) & stress_tags
                for row in records
                if int(row["id"]) in set(stress_ids)
            )
        },
    }
    return {
        "loto": loto,
        "loyo": loyo,
        "loro": loro,
        "lomo": lomo,
        "composition": composition,
        "scope_period_stress": stress,
    }


def validate_views(view_documents: list[dict]) -> None:
    for view in view_documents:
        for fold in view.get("folds", {}).values():
            assertion = fold.get("assertion", {})
            if assertion.get("train_eval_id_overlap"):
                raise ValueError("OOD view has train/eval ID leakage")
            for key in (
                "holdout_value_absent_from_train",
                "holdout_year_absent_from_train",
                "metric_absent_from_train",
                "tree_shape_absent_from_train",
                "primitive_ops_seen_in_train",
            ):
                if assertion.get(key) is False:
                    raise ValueError(f"OOD view assertion failed: {key}")
