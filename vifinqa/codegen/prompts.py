"""Prompt construction for Text-to-Pandas (PoT) generation."""
from __future__ import annotations

SYSTEM = """You are an expert Vietnamese financial data analyst who writes pandas code.

You get a Vietnamese question about audited financial statements and one or more
DataFrames (df1, df2, ...). Each DataFrame is a normalized LONG-FORMAT view of one
OCR'd table with columns:
  row(int), label(str: the line-item text), code(str: VAS line code like 10, 60, 270),
  col(int: column index in the original table), col_name(str: original column header,
  often a year like '31/12/2023' or '2023'), value(float: the parsed number),
  unit_scale(float: multiplier converting `value` to VND, e.g. 1e6 if the table is in
  'triệu đồng')

Rules:
1. Use ONLY the provided variables (df1, df2, ...), pd and np. No imports, no I/O, no print.
2. Numbers are already parsed: negatives from '(1.839)' already carry a minus sign.
3. Find the right line item by fuzzy label match (str.contains with a distinctive
   lowercase substring, case=False) and/or by VAS code when obvious
   (e.g. net revenue code '10', profit after tax '60', total assets '270').
4. Pick the right column: match the asked year in col_name ('2023', '31/12/2023',
   'Năm nay'); if col_name is empty, the smallest `col` with data is usually the
   current year, the next one the prior year.
5. Convert units: value_vnd = value * unit_scale. The question asks the answer in a
   specific unit -> answer = round(value_vnd / ANSWER_SCALE, 2). For percentage
   questions, answer = round(ratio * 100, 2).
6. Robustness: ALWAYS pass na=False to .str.contains (labels can be NaN);
   the `code` column may load as numbers -> compare via df['code'].astype(str).
7. End with a line assigning the final float: answer = ...
8. Output ONLY one ```python code block, nothing else."""

DEBUG_SYSTEM = SYSTEM

USER_TMPL = """Question (Vietnamese): {question}

Parsed intent: ticker={tickers}, years={years}, doc_type={doc_type}, \
output_type={output_type}, answer unit = {unit_name} \
(ANSWER_SCALE = {unit_scale:g}), percent={is_percent}

Available DataFrames:
{tables_block}

Write pandas code that computes `answer` as a float in the requested unit \
(rounded to 2 decimals)."""

TABLE_TMPL = """--- {var} | source: {report_id}|{table_pos} | page {page} | \
table unit_scale={unit_scale} ({unit_note}) ---
{preview}
"""

DEBUG_TMPL = """Your previous code failed.

Question: {question}
Previous code:
```python
{code}
```
Error: {error}

Fix the code. Same rules apply (only df1..dfN, pd, np; end with `answer = ...`;
output ONLY one ```python block)."""


def build_user(question: str, route: dict, tables: list[dict]) -> str:
    blocks = []
    for t in tables:
        us = t.get("unit_scale")
        note = "unit unknown, assume VND" if not us else (
            {1.0: "VND", 1e3: "nghìn đồng", 1e6: "triệu đồng", 1e9: "tỷ đồng",
             1e11: "trăm tỷ đồng",
             1e12: "nghìn tỷ đồng"}.get(float(us), f"x{us:g} VND"))
        blocks.append(TABLE_TMPL.format(
            var=t["var"], report_id=t["report_id"], table_pos=t["table_pos"],
            page=t.get("page", "?"), unit_scale=(us or 1.0), unit_note=note,
            preview=t["preview"],
        ))
    return USER_TMPL.format(
        question=question,
        tickers=",".join(route.get("tickers", [])) or "?",
        years=route.get("years", []),
        doc_type=route.get("doc_type", "consolidated"),
        unit_name=route.get("unit_name", "đồng"),
        unit_scale=float(route.get("unit_scale", 1.0)),
        is_percent=route.get("is_percent", False),
        output_type=route.get("output_type", "number"),
        tables_block="\n".join(blocks),
    )


def build_debug(question: str, code: str, error: str) -> str:
    return DEBUG_TMPL.format(question=question, code=code, error=error)
