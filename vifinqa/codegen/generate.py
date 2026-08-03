"""Codegen orchestration: batch prompts -> execute -> majority vote ->
self-debug round(s) -> rule-based fallback.

Batch-first design so vLLM sees big batches (fast on Kaggle T4 x2).
Outputs codegen_results.jsonl records:
    {id, answer, pandas_query, used_vars:[{var, report_id, table_pos}],
     status, source, votes, attempts, detail}
"""
from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path

from ..config import CODEGEN_K
from ..extraction.build_store import Store
from ..retrieval.serialize import tidy_csv_text, df_roundtrip
from ..utils.io import read_jsonl, write_jsonl
from .executor import run_code, extract_code
from .prompts import SYSTEM, build_user
from .rule_codegen import try_rule_answer
from .semantic import (
    all_dataframe_refs,
    answer_dataframe_refs,
    validate_generated_answer,
)


class QuestionBundle:
    def __init__(self, rec: dict, store: Store, k: int = CODEGEN_K,
                 run_signature: str = ""):
        self.id = rec["id"]
        self.question = rec["question"]
        self.route = rec["route"]
        self.run_signature = run_signature
        self.cands = rec["candidates"][:k]
        self.tables: list[dict] = []
        self.dfs: dict = {}
        by_ticker: dict[str, list[dict]] = {}
        for c in self.cands:
            by_ticker.setdefault(c["ticker"], []).append(c)
        meta_lookup = {}
        for ticker, cs in by_ticker.items():
            tdf = store.tables_of(ticker, list({c["report_id"] for c in cs}))
            for m in tdf.to_dict("records"):
                meta_lookup[(m["report_id"], int(m["table_pos"]))] = m
        for i, c in enumerate(self.cands, start=1):
            m = meta_lookup.get((c["report_id"], int(c["table_pos"])))
            if m is None:
                continue
            csv_text = tidy_csv_text(m)
            var = f"df{i}"
            us = m.get("unit_scale")
            us = None if (us is None or us != us) else float(us)
            self.tables.append({
                "var": var, "report_id": c["report_id"],
                "table_pos": int(c["table_pos"]), "page": c.get("page"),
                "unit_scale": us, "unit_source": m.get("unit_source", "none"),
                "report_year": int(m["year"]), "csv_text": csv_text,
            })
            self.dfs[var] = df_roundtrip(csv_text)

    def prompt_messages(self) -> list[dict]:
        from ..retrieval.serialize import preview_for_prompt
        tables = [{**t, "preview": preview_for_prompt(t["csv_text"])} for t in self.tables]
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_user(self.question, self.route, tables)}]

    def used_vars(self, code: str) -> list[dict]:
        # Submission replay must receive every dataframe the program may access,
        # including intermediates that do not flow into the final answer.  The
        # separate semantic guard still requires the answer itself to be grounded.
        try:
            refs = sorted(all_dataframe_refs(code))
        except SyntaxError:
            refs = []
        by_var = {t["var"]: t for t in self.tables}
        used = [by_var[v] for v in refs if v in by_var]
        return [{"var": t["var"], "report_id": t["report_id"],
                 "table_pos": t["table_pos"]} for t in used]

    def debug_messages(self, code: str, error: str) -> list[dict]:
        """Retry with the original route and table previews still in context."""
        messages = self.prompt_messages()
        messages.extend([
            {"role": "assistant", "content": f"```python\n{code}\n```"},
            {"role": "user", "content": (
                "The previous program failed execution or semantic validation.\n"
                f"Error: {error}\n"
                "Return a corrected Python code block grounded in the provided "
                "DataFrames."
            )},
        ])
        return messages


def _vote(samples_results: list[tuple[str, dict]]):
    """samples_results: [(code, exec_result)] -> (answer, code) by majority on
    the rounded value; ties broken by first occurrence."""
    ok = [(c, r) for c, r in samples_results if r["status"] == "ok"]
    if not ok:
        return None
    counts = Counter(round(r["value"], 2) for _c, r in ok)
    winner, n = counts.most_common(1)[0]
    for c, r in ok:
        if round(r["value"], 2) == winner:
            return {"answer": float(winner), "code": c, "votes": int(n),
                    "n_ok": len(ok), "semantic": r.get("semantic")}
    return None


def run_codegen(retrieval_path: Path, store_dir: Path, out_path: Path, client,
                k: int = CODEGEN_K, n_samples: int = 4, temperature: float = 0.7,
                debug_rounds: int = 1, limit: int = 0, use_rule_fallback: bool = True,
                rule_first: bool = False, max_tokens: int = 768,
                checkpoint_every: int = 32, resume: bool = True,
                time_budget_s: float = 0.0, run_signature: str = "") -> None:
    """Crash-safe codegen.

    Order of operations (important on Kaggle where a session can die at any time):
      1. build bundles
      2. compute the deterministic RULE baseline for every question and FLUSH it
         -> out_path is a complete, submittable file from minute ~2 onward
      3. run the LLM in chunks of `checkpoint_every`, overwriting entries with
         better answers, flushing after every chunk
      4. stop cleanly when `time_budget_s` is exceeded (keeps what it has)

    resume=True picks up an existing out_path and skips ids already answered by
    the LLM (source startswith 'llm'). When run_signature is non-empty, only
    records produced by the exact same semantic run configuration are reused.
    """
    t_start = time.time()
    store = Store(store_dir, cache_size=4)
    recs = read_jsonl(retrieval_path)
    if limit:
        recs = recs[:limit]

    bundles, results = [], {}
    print(f"building {len(recs)} bundles...", flush=True)
    for rec in recs:
        b = QuestionBundle(rec, store, k, run_signature=run_signature)
        if not b.tables:
            results[b.id] = _empty_result(b, "no candidate tables")
            continue
        bundles.append(b)

    # ---------- resume: read the previous file BEFORE overwriting it ----------
    prev = _load_previous(out_path) if resume else {}
    already = {
        qid for qid, r in prev.items()
        if str(r.get("source", "")).startswith("llm")
        and (not run_signature or r.get("run_signature") == run_signature)
    }
    if already:
        print(f"resume: {len(already)} questions already solved by the LLM", flush=True)
    elif prev and run_signature:
        print("resume: previous output exists but has a different/missing run "
              "signature; its LLM answers will not be reused", flush=True)

    # ---------- step 2: rule baseline for everything, flush immediately ----------
    print(f"rule baseline over {len(bundles)} questions...", flush=True)
    rule_conf = {}
    for b in bundles:
        if b.id in already:                     # keep the better LLM answer
            results[b.id] = prev[b.id]
            rule_conf[b.id] = 0.0
            continue
        r = _rule_result(b) if use_rule_fallback else None
        results[b.id] = r if r is not None else _empty_result(b, "rule found nothing")
        rule_conf[b.id] = r["detail_conf"] if r else 0.0
    _flush(out_path, recs, results)
    print(f"  baseline written ({_srcs(recs, results)}) -> {out_path}", flush=True)

    if client.name == "none":
        print(f"codegen done (rule only): {_srcs(recs, results)} -> {out_path}")
        return

    llm_todo = [b for b in bundles if b.id not in already
                and not (rule_first and rule_conf.get(b.id, 0) >= 90)]
    skipped = len(bundles) - len(llm_todo) - len(already)
    print(f"LLM queue: {len(llm_todo)} questions "
          f"(rule-first skipped {skipped}, n={n_samples}, T={temperature})", flush=True)

    # ---------- step 3: chunked generation with checkpoints ----------
    chunk = max(1, checkpoint_every)
    n_chunks = (len(llm_todo) + chunk - 1) // chunk
    for ci in range(n_chunks):
        elapsed_before = time.time() - t_start
        if time_budget_s and elapsed_before >= time_budget_s:
            print(f"time budget {time_budget_s/60:.0f}min reached before chunk "
                  f"{ci+1}; keeping the complete checkpoint", flush=True)
            break
        part = llm_todo[ci * chunk:(ci + 1) * chunk]
        t0 = time.time()
        convs = [b.prompt_messages() for b in part]
        try:
            gens = client.chat_batch(convs, n=n_samples, temperature=temperature,
                                     max_tokens=max_tokens)
        except Exception:
            # The rule baseline and all earlier chunks are already durable. Make
            # that guarantee explicit before propagating OOM/backend failures.
            _flush(out_path, recs, results)
            print(f"[chunk {ci+1}/{n_chunks}] generation failed; checkpoint kept "
                  f"at {out_path}", flush=True)
            raise
        if len(gens) != len(part):
            _flush(out_path, recs, results)
            raise RuntimeError(
                f"LLM returned {len(gens)} result groups for {len(part)} prompts"
            )
        debug_queue = []
        for b, samples in zip(part, gens):
            sr = [(extract_code(s), None) for s in samples]
            sr = [(c, _run_validated(b, c)) for c, _ in sr]
            win = _vote(sr)
            if win:
                results[b.id] = _final(b, win["answer"], win["code"], "llm",
                                       votes=win["votes"], n_ok=win["n_ok"],
                                       semantic=win.get("semantic"))
            else:
                err = next((r["error"] for _c, r in sr if r["error"]), "no output")
                debug_queue.append((b, sr[0][0] if sr else "", err))

        # Preserve all successful round-1 answers before the optional debug
        # pass. A debug OOM must not discard useful work from this chunk.
        _flush(out_path, recs, results)

        for rnd in range(debug_rounds):
            if not debug_queue:
                break
            convs = [b.debug_messages(code, err) for b, code, err in debug_queue]
            try:
                gens = client.chat_batch(convs, n=1, temperature=0.2,
                                         max_tokens=max_tokens)
            except Exception:
                _flush(out_path, recs, results)
                print(f"[chunk {ci+1}/{n_chunks}] debug round {rnd+1} failed; "
                      "round-1 checkpoint kept", flush=True)
                raise
            if len(gens) != len(debug_queue):
                _flush(out_path, recs, results)
                raise RuntimeError(
                    f"LLM returned {len(gens)} debug groups for "
                    f"{len(debug_queue)} prompts"
                )
            nxt = []
            for (b, old_code, _err), samples in zip(debug_queue, gens):
                code = extract_code(samples[0]) if samples else ""
                r = _run_validated(b, code)
                if r["status"] == "ok":
                    results[b.id] = _final(
                        b, round(r["value"], 2), code, "llm_debug",
                        semantic=r.get("semantic"),
                    )
                else:
                    nxt.append((b, code or old_code, r["error"] or "no output"))
            debug_queue = nxt
        # questions the LLM could not fix keep their rule-baseline entry

        _flush(out_path, recs, results)
        dt, elapsed = time.time() - t0, time.time() - t_start
        eta = dt * (n_chunks - ci - 1)
        print(f"[chunk {ci+1}/{n_chunks}] {len(part)}q in {dt/60:.1f}min | "
              f"elapsed {elapsed/60:.1f}min | ETA {eta/60:.1f}min | "
              f"{_srcs(recs, results)}", flush=True)

        if time_budget_s and elapsed + dt > time_budget_s:
            print(f"time budget {time_budget_s/60:.0f}min reached -> stopping cleanly "
                  f"(file is complete & submittable; rerun with --resume to continue)",
                  flush=True)
            break

    _flush(out_path, recs, results)
    print(f"codegen done: {_srcs(recs, results)} -> {out_path}")


def _flush(out_path: Path, recs: list[dict], results: dict) -> None:
    """Write every question in input order; atomic-ish via temp file + replace."""
    ordered = [results[rec["id"]] for rec in recs if rec["id"] in results]
    tmp = Path(str(out_path) + ".tmp")
    write_jsonl(tmp, ordered)
    os.replace(tmp, out_path)


def _load_previous(out_path: Path) -> dict:
    if not Path(out_path).exists():
        return {}
    try:
        return {r["id"]: r for r in read_jsonl(out_path)}
    except Exception:  # noqa: BLE001 - corrupt/partial file must not kill the run
        return {}


def _srcs(recs: list[dict], results: dict) -> dict:
    return dict(Counter(results[r["id"]]["source"] for r in recs if r["id"] in results))


def _run_validated(b: QuestionBundle, code: str) -> dict:
    result = run_code(code, b.dfs)
    if result["status"] != "ok":
        return result
    check = validate_generated_answer(
        code, b.dfs.keys(), result["value"], route=getattr(b, "route", {}),
    )
    result["semantic"] = check.to_dict()
    if not check.ok:
        result["status"] = "semantic_error"
        result["error"] = "; ".join(check.errors)
    return result


def _rule_result(b: QuestionBundle) -> dict | None:
    ra = try_rule_answer(b.route, b.tables)
    if not ra.ok:
        return None
    ex = _run_validated(b, ra.pandas_query)
    if ex["status"] != "ok":
        return None
    ans = round(ex["value"], 2)  # answer == what the query reproduces
    out = _final(b, ans, ra.pandas_query, "rule", semantic=ex.get("semantic"))
    out["detail"] = ra.detail
    out["detail_conf"] = ra.confidence
    return out


def _final(b: QuestionBundle, answer: float, code: str, source: str,
           votes: int = 1, n_ok: int = 1, semantic: dict | None = None) -> dict:
    return {"id": b.id, "question": b.question, "answer": float(answer),
            "pandas_query": code, "used_vars": b.used_vars(code),
            "status": "ok", "source": source, "votes": votes, "n_ok": n_ok,
            "detail": "", "detail_conf": 0.0,
            "semantic": semantic or {},
            "run_signature": getattr(b, "run_signature", "")}


def _empty_result(b, reason: str) -> dict:
    return {"id": b.id, "question": b.question, "answer": 0.0,
            "pandas_query": "0.0",
            "used_vars": ([{"var": t["var"], "report_id": t["report_id"],
                            "table_pos": t["table_pos"]} for t in b.tables[:1]]
                          if getattr(b, "tables", None) else []),
            "status": "failed", "source": "none", "votes": 0, "n_ok": 0,
            "detail": reason, "detail_conf": 0.0,
            "semantic": {},
            "run_signature": getattr(b, "run_signature", "")}
