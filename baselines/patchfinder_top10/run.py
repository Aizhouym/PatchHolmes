"""Run PatchHolmes Phase 2 on Favia's PatchFinder_top10 benchmark.

Usage
-----
    # Sequential (1 CVE at a time)
    python -m baselines.patchfinder_top10.run \
        --variant      PatchFinder_top10 \
        --output       logs/phase2/patchfinder_1252/qwen3_235b_openrouter/results.jsonl \
        --num-workers  1

    # Parallel (8 workers — recommended starting point)
    python -m baselines.patchfinder_top10.run \
        --variant      PatchFinder_top10 \
        --output       logs/phase2/patchfinder_1252/qwen3_235b_openrouter/results.jsonl \
        --num-workers  8

What this does
--------------
1. Loads the chosen variant parquet (PatchFinder_top10 / random_10 / a CWE
   pre-filtered variant — see PARQUET_ROOT below).
2. Groups rows by CVE; each CVE is sent as a task to a `multiprocessing.Pool`.
3. Worker processes call `run_phase2_single(data_source=PatchFinderDataSource)`
   so the existing Phase 2 agent / tools / LLM run unchanged. The LLM is
   built per worker via `build_llm()` which reads `PATCHHOLMES_LLM_*` env vars.
4. Parent serializes results to the output jsonl one record at a time.

Resume-friendly: re-running with the same --output skips CVEs already in
the file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Any

# Make the project root importable when run as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


PARQUET_ROOT = Path("./data/patchfinder_top10/cvevc_candidates")
VARIANT_PATHS = {
    "PatchFinder_top10": PARQUET_ROOT / "PatchFinder_top10" / "test-00000-of-00001.parquet",
    "random_10":         PARQUET_ROOT / "random_10"         / "test-00000-of-00001.parquet",
    "CWE_Qwen3-235B":    PARQUET_ROOT / "PatchFinder_top10_CWEReportTool_Qwen3-235B-A22B-Instruct-2507" / "test-00000-of-00001.parquet",
    "CWE_Llama-70B":     PARQUET_ROOT / "PatchFinder_top10_CWEReportTool_Llama-3.3-70B-Instruct"        / "test-00000-of-00001.parquet",
}


# ---------------------------------------------------------------------------
# Worker (runs inside child processes)
# ---------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _worker_init(max_iter: int, read_commit_budget: int, read_file_diff_budget: int) -> None:
    """Per-process init: build the LLM once and stash run-time knobs.

    The LLM picks up `PATCHHOLMES_LLM_MODEL` / `PATCHHOLMES_LLM_API_KEY` /
    `PATCHHOLMES_LLM_BASE_URL` from the environment, which docker-compose
    injects from deploy/.env.
    """
    from patchholmes.phase2.agent import build_llm
    _WORKER["llm"] = build_llm(usage_id=f"patchfinder_worker_{os.getpid()}")
    _WORKER["max_iter"] = max_iter
    _WORKER["read_commit_budget"] = read_commit_budget
    _WORKER["read_file_diff_budget"] = read_file_diff_budget


def _process_one_cve(task: dict[str, Any]) -> dict[str, Any]:
    """Run Phase 2 on one CVE. Task is a self-contained dict so it pickles
    cleanly across process boundaries (no DataFrame slices)."""
    cve_id = task["cve_id"]
    t_start = time.time()
    try:
        import pandas as pd
        from baselines.patchfinder_top10.adapter import build_query_and_candidates
        from baselines.patchfinder_top10.data_source import PatchFinderDataSource
        from patchholmes.phase2.runner import run_phase2_single

        # Reconstruct a DataFrame from the row dicts (cheap; <=10 rows per CVE).
        group_df = pd.DataFrame(task["rows"])
        query, candidates, commit_dict = build_query_and_candidates(group_df)
        ds = PatchFinderDataSource(query, candidates, commit_dict)

        result = run_phase2_single(
            query=query,
            phase1_candidates=candidates,
            repo2commits_root="/dev/null",   # ignored when data_source is given
            llm=_WORKER["llm"],
            top_k=len(candidates),
            max_iteration_per_run=_WORKER["max_iter"],
            read_commit_budget=_WORKER["read_commit_budget"],
            read_file_diff_budget=_WORKER["read_file_diff_budget"],
            data_source=ds,
        )

        fix_set = set(query.fix_commit_ids)
        hit = bool(result.best_commit_id and result.best_commit_id in fix_set)
        return {
            "cve_id": cve_id,
            "owner": query.owner,
            "repo": query.repo,
            "fix_commit_ids": list(query.fix_commit_ids),
            "n_candidates_unique": len(candidates),
            "best_commit_id": result.best_commit_id,
            "reasoning": result.reasoning,
            "hit": hit,
            "patchfinder_rank_of_answer": result.phase1_rank_of_answer,
            "best_rank_in_truth_set": result.best_rank_in_truth_set,
            "stopped_reason": result.stopped_reason,
            "iterations_used": result.iterations_used,
            "commits_inspected": result.commits_inspected,
            "tool_calls": [tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in (result.tool_calls or [])],
            "llm_input_tokens": result.llm_input_tokens,
            "llm_output_tokens": result.llm_output_tokens,
            "estimated_cost_usd": result.estimated_cost_usd,
            "wall_time_sec": result.wall_time_sec,
            "error": result.error,
        }
    except Exception as e:
        return {
            "cve_id": cve_id,
            "owner": task.get("owner", ""),
            "repo": task.get("repo", ""),
            "error": f"{type(e).__name__}: {str(e)[:200]}\n{traceback.format_exc()[-400:]}",
            "wall_time_sec": round(time.time() - t_start, 2),
        }


# ---------------------------------------------------------------------------
# Parent
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run PatchHolmes Phase 2 on PatchFinder_top10 benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant", default="PatchFinder_top10", choices=list(VARIANT_PATHS.keys()))
    p.add_argument(
        "--output",
        default="./logs/phase2/patchfinder_1252/results.jsonl",
    )
    p.add_argument("--num-workers", type=int, default=8,
                   help="Parallel worker processes (set to 1 for sequential).")
    p.add_argument("--max-cves", type=int, default=0,
                   help="Process at most N CVEs (0 = all).")
    p.add_argument("--max-iteration-per-run", type=int, default=15)
    p.add_argument("--read-commit-budget", type=int, default=8000)
    p.add_argument("--read-file-diff-budget", type=int, default=16000)
    p.add_argument("--cve-id", default=None,
                   help="If set, only run this single CVE (for debugging).")
    return p.parse_args()


def load_done_cves(output_path: Path) -> set[str]:
    """Return the set of CVE IDs that have a SUCCESSFUL prior result.

    Errored records are deliberately not counted as "done" so that the next
    run picks them up again — transient provider errors (5xx, 429) often
    succeed on retry. We rewrite the jsonl in-place to drop error records
    before resuming so they don't accumulate.
    """
    if not output_path.exists():
        return set()
    keep: list[dict] = []
    done: set[str] = set()
    with output_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("error"):
                continue
            keep.append(r)
            done.add(r["cve_id"])
    # Rewrite to drop error records (so the next run cleanly resumes)
    with output_path.open("w") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return done


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load parquet ──────────────────────────────────────────────────────
    from baselines.patchfinder_top10.adapter import load_patchfinder_variant
    parquet_path = VARIANT_PATHS[args.variant]
    print(f"Loading {args.variant} from {parquet_path} ...")
    df = load_patchfinder_variant(str(parquet_path))
    cve_ids_in_order = list(df["cve"].drop_duplicates())
    print(f"  {len(df):,} rows  /  {len(cve_ids_in_order):,} CVEs")

    if args.cve_id:
        cve_ids_in_order = [args.cve_id]

    done = load_done_cves(output_path)
    if done:
        print(f"Resume: {len(done):,} CVE already in output, will skip")
    todo_ids = [c for c in cve_ids_in_order if c not in done]
    if args.max_cves > 0:
        todo_ids = todo_ids[: args.max_cves]
    print(f"To process: {len(todo_ids):,} CVEs")
    print(f"Workers:    {args.num_workers}\n")
    if not todo_ids:
        print("Nothing to do.")
        return

    # ── Build self-contained tasks ────────────────────────────────────────
    # Each task is a small dict so workers don't need access to the parent's
    # DataFrame. Pickling a 10-row dict-list is fast.
    grouped = {cve: g for cve, g in df.groupby("cve") if cve in set(todo_ids)}
    tasks: list[dict[str, Any]] = []
    for cve_id in todo_ids:
        g = grouped.get(cve_id)
        if g is None or len(g) == 0:
            continue
        # Convert the rows to plain dicts (pickle-safe)
        rows = g[["cve", "desc", "repo", "commit_id", "commit_message", "diff", "label", "rank"]].to_dict("records")
        tasks.append({
            "cve_id": cve_id,
            "owner": str(g.iloc[0]["repo"]).split("/")[0] if "/" in str(g.iloc[0]["repo"]) else "",
            "repo":  str(g.iloc[0]["repo"]).split("/")[1] if "/" in str(g.iloc[0]["repo"]) else str(g.iloc[0]["repo"]),
            "rows":  rows,
        })

    # ── Pool ──────────────────────────────────────────────────────────────
    t_start = time.time()
    n_done = n_hit = n_err = 0

    init_args = (args.max_iteration_per_run, args.read_commit_budget, args.read_file_diff_budget)

    with output_path.open("a") as out_f:
        if args.num_workers <= 1:
            # Sequential path — easier to debug, no Pool overhead
            _worker_init(*init_args)
            iterator = (_process_one_cve(t) for t in tasks)
        else:
            pool = Pool(
                processes=args.num_workers,
                initializer=_worker_init,
                initargs=init_args,
                maxtasksperchild=50,   # recycle workers periodically (litellm keeps connections)
            )
            iterator = pool.imap_unordered(_process_one_cve, tasks)

        try:
            for i, rec in enumerate(iterator, 1):
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1
                if rec.get("hit"):
                    n_hit += 1
                if rec.get("error"):
                    n_err += 1

                if i % 20 == 0 or i == len(tasks) or i <= 5:
                    elapsed = time.time() - t_start
                    rate = i / max(0.01, elapsed)
                    eta_min = (len(tasks) - i) / max(0.001, rate) / 60
                    mark = "✓" if rec.get("hit") else ("E" if rec.get("error") else "✗")
                    print(
                        f"[{i:>5}/{len(tasks)}] {mark} {rec['cve_id']:<18} "
                        f"hits={n_hit} errs={n_err}  "
                        f"({elapsed/60:.1f}min, {rate*60:.1f}/min, ETA {eta_min:.1f}min)",
                        flush=True,
                    )
        finally:
            if args.num_workers > 1:
                pool.close()
                pool.join()

    elapsed = time.time() - t_start
    print(
        f"\n{'='*70}\n"
        f"Done: {n_done} CVE  /  {n_hit} hit ({n_hit/max(1,n_done):.1%})  /  {n_err} error\n"
        f"Time: {elapsed/60:.1f}min  ({elapsed/max(1,n_done):.2f}s/CVE wall, "
        f"throughput {n_done*60/max(1,elapsed):.1f}/min effective)\n"
        f"Workers: {args.num_workers}\n"
        f"Output:  {output_path}\n"
        f"\nNext: python -m baselines.patchfinder_top10.eval {output_path}"
    )


if __name__ == "__main__":
    main()
