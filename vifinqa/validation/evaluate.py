"""Offline evaluation on the synthetic validation set.

Metrics mirror the organizers': macro P/R/F2 on relevant_tables, Answer
Accuracy (|pred-gold| <= 0.01 after 2-decimal rounding), Execution Accuracy
(re-run pandas_query on the evidence CSVs of the submission folder).
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from ..codegen.executor import run_code
from ..utils.io import read_json

TOL = 0.01 + 1e-9


def evaluate(submission_dir: Path, gold_path: Path, json_name: str = "results.json") -> dict:
    submission_dir = Path(submission_dir)
    preds = read_json(submission_dir / json_name)
    gold = read_json(gold_path)

    P = R = F2 = n = 0.0
    n_ans = n_exec = n_run = 0
    for e in preds:
        g = gold.get(str(e["id"]))
        if g is None:
            continue
        n += 1
        gt = set(g["relevant_tables"])
        pt = set(e.get("relevant_tables", []))
        tp = len(gt & pt)
        p = tp / len(pt) if pt else 0.0
        r = tp / len(gt) if gt else 0.0
        f2 = (5 * p * r / (4 * p + r)) if (p + r) else 0.0
        P, R, F2 = P + p, R + r, F2 + f2

        if abs(round(float(e.get("answer", 0.0)), 2) - round(g["answer"], 2)) <= TOL:
            n_ans += 1

        code = e.get("pandas_query") or ""
        dfs = {}
        ok_load = True
        for ev in e.get("evidence", []):
            path = submission_dir / ev["csv_path"]
            try:
                # plain read_csv = what the grader most likely does
                dfs[ev["variable"]] = pd.read_csv(path)
            except Exception:
                ok_load = False
        if code and ok_load:
            n_run += 1
            res = run_code(code, dfs)
            if (res["status"] == "ok"
                    and abs(round(res["value"], 2) - round(g["answer"], 2)) <= TOL):
                n_exec += 1

    n = max(n, 1)
    report = {
        "n": int(n),
        "precision_macro": round(P / n, 4),
        "recall_macro": round(R / n, 4),
        "f2_macro": round(F2 / n, 4),
        "answer_acc": round(n_ans / n, 4),
        "exec_acc": round(n_exec / n, 4),
        "n_query_ran": n_run,
    }
    for k, v in report.items():
        print(f"  {k}: {v}")
    return report
