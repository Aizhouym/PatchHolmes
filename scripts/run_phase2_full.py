#!/usr/bin/env python3
"""Full-dataset Phase 2 runner with multiprocessing + resume.

Reads Phase 1 per-CVE jsonl, runs the Phase 2 agent on each CVE in parallel,
streams Phase2Result JSON lines to the output file. Safely resumable: any CVE
whose result line is already present in the output file is skipped.

Usage
-----
python scripts/run_phase2_full.py \\
    --phase1-jsonl   ./logs/phase1/phase1_clean_full_top100.jsonl \\
    --dataset-csv    ./data/ground_truth_queries_clean.csv \\
    --repo2commits   ./data/repo2commits_diff \\
    --output         ./logs/phase2/own_8401/qwen3_coder_30b/results.jsonl \\
    --num-workers    8

Override LLM with:
    --llm-model hosted_vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct
    --llm-base-url http://localhost:8000/v1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Worker (runs inside child processes)
# ---------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}  # per-process state, populated by _worker_init


def _worker_init(
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    descriptions: dict[str, str],
    repo2commits_root: str,
    max_iter: int,
    top_k: int,
) -> None:
    """Per-process initializer — builds the LLM once and stashes config."""
    # Import inside the child so each process loads fresh module state for
    # the SDK's global tool registry (we write to it in each task via
    # register_tool, and we don't want cross-task interference within a
    # process pool that shares globals).
    from patchholmes.phase2.agent import build_llm

    _WORKER["llm"] = build_llm(
        model=llm_model,
        base_url=llm_base_url,
        api_key=llm_api_key,
        usage_id=f"phase2_worker_{os.getpid()}",
    )
    _WORKER["descriptions"] = descriptions
    _WORKER["repo2commits_root"] = repo2commits_root
    _WORKER["max_iter"] = max_iter
    _WORKER["top_k"] = top_k


def _process_one_cve(phase1_rec: dict[str, Any]) -> dict[str, Any]:
    """Run Phase 2 on a single CVE; return a serialised Phase2Result dict.

    Errors at this level are caught and returned as a degraded record so the
    pool keeps going.
    """
    from patchholmes.data_models import CVEQuery, CommitDoc, RankedCandidate
    from patchholmes.phase2.result import Phase2Result
    from patchholmes.phase2.runner import run_phase2_single

    cve_id = phase1_rec["cve_id"]
    descriptions = _WORKER["descriptions"]

    try:
        query = CVEQuery(
            cve_id=cve_id,
            description=descriptions.get(cve_id, ""),
            owner=phase1_rec["owner"],
            repo=phase1_rec["repo"],
            fix_commit_ids=phase1_rec.get("fix_commit_ids") or [],
        )
        candidates = [
            RankedCandidate(
                commit=CommitDoc(
                    commit_id=c["commit_id"],
                    commit_msg="",
                    diff="",
                    owner=phase1_rec["owner"],
                    repo=phase1_rec["repo"],
                    datetime=c.get("datetime", ""),
                ),
                score=float(c.get("score", 0.0)),
                rank=int(c["rank"]),
                source=c.get("source", "rrf"),
                bm25_rank=c.get("bm25_rank"),
                dense_rank=c.get("dense_rank"),
            )
            for c in (phase1_rec.get("candidates") or [])
        ]
        result = run_phase2_single(
            query=query,
            phase1_candidates=candidates,
            repo2commits_root=_WORKER["repo2commits_root"],
            llm=_WORKER["llm"],
            top_k=_WORKER["top_k"],
            max_iteration_per_run=_WORKER["max_iter"],
        )
        return result.to_dict()
    except Exception as e:
        # Degraded record; downstream eval will see error != None and skip.
        return Phase2Result(
            cve_id=cve_id,
            owner=phase1_rec.get("owner", ""),
            repo=phase1_rec.get("repo", ""),
            best_commit_id=None,
            reasoning="",
            fix_commit_ids=phase1_rec.get("fix_commit_ids") or [],
            error=f"{type(e).__name__}: {e}",
        ).to_dict()


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full-dataset Phase 2 runner with parallel workers + resume.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--phase1-jsonl",
        default="./logs/phase1/phase1_clean_full_top100.jsonl",
        help="Phase 1 per-CVE jsonl (with candidates).",
    )
    p.add_argument(
        "--dataset-csv",
        default="./data/ground_truth_queries_clean.csv",
        help="CSV providing CVE descriptions.",
    )
    p.add_argument(
        "--repo2commits",
        default="./data/repo2commits_diff",
        help="Root containing split_<owner>@@<repo>/*.json files.",
    )
    p.add_argument(
        "--output",
        default="./logs/phase2/own_8401/qwen3_coder_30b/results.jsonl",
        help="Output jsonl path. Existing CVE entries are skipped (resume).",
    )
    p.add_argument("--num-workers", type=int, default=8, help="Parallel worker processes.")
    # LLM args default to None so build_llm() reads from env vars
    # (PATCHHOLMES_LLM_{MODEL,BASE_URL,API_KEY}); see .env.example. For local
    # vLLM, leave the env vars unset and build_llm()
    # falls through to the hardcoded vLLM default.
    p.add_argument("--llm-model", default=None,
                   help="LiteLLM model id. Defaults to env var or hosted_vllm fallback.")
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--llm-api-key", default=None)
    p.add_argument("--max-iter", type=int, default=15)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument(
        "--max-cves", type=int, default=0,
        help="Process at most N CVEs (0 = all). Useful for sampling/smoke runs.",
    )
    return p.parse_args()


def load_descriptions(csv_path: Path) -> dict[str, str]:
    descs: dict[str, str] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            cve_id = row.get("cve", "").strip()
            if cve_id:
                descs[cve_id] = (row.get("cve_description") or "").strip()
    return descs


def load_phase1_records(jsonl_path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in jsonl_path.open(encoding="utf-8") if l.strip()]


def load_done_cves(output_path: Path) -> set[str]:
    """Return CVE IDs with a SUCCESSFUL prior result. Errored records are
    dropped from the file so the next run auto-retries them (transient
    provider 5xx / 429 usually succeed on retry).
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
    with output_path.open("w") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return done


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load inputs ────────────────────────────────────────────────────
    print("Loading CVE descriptions ...")
    descriptions = load_descriptions(Path(args.dataset_csv))
    print(f"  {len(descriptions):,} descriptions")

    print("Loading Phase 1 results ...")
    phase1_records = load_phase1_records(Path(args.phase1_jsonl))
    print(f"  {len(phase1_records):,} Phase 1 records")

    # ── 2. Resume: skip already-done CVEs ─────────────────────────────────
    done = load_done_cves(output_path)
    if done:
        print(f"Resume: {len(done):,} CVE already in output, skipping")

    todo = [r for r in phase1_records if r["cve_id"] not in done]
    if args.max_cves > 0:
        todo = todo[: args.max_cves]
    print(f"To process: {len(todo):,} CVE")

    if not todo:
        print("Nothing to do. Exit.")
        return

    # ── 3. Spin up workers ────────────────────────────────────────────────
    print(
        f"\nStarting {args.num_workers} workers"
        f"  (LLM = {args.llm_model or 'env/default'} @ {args.llm_base_url or 'env/default'})"
    )
    print(f"Writing → {output_path}\n")

    init_args = (
        args.llm_model,
        args.llm_base_url,
        args.llm_api_key,
        descriptions,
        args.repo2commits,
        args.max_iter,
        args.top_k,
    )

    n_done = 0
    n_hit = 0
    t_start = time.time()

    with Pool(processes=args.num_workers,
              initializer=_worker_init,
              initargs=init_args) as pool, \
         output_path.open("a") as out_f:

        for result_dict in pool.imap_unordered(_process_one_cve, todo):
            out_f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
            out_f.flush()
            n_done += 1
            if result_dict.get("hit"):
                n_hit += 1
            elapsed = time.time() - t_start
            rate = n_done / max(1.0, elapsed)
            eta_min = (len(todo) - n_done) / max(0.001, rate) / 60
            mark = "✓" if result_dict.get("hit") else "✗"
            err = " [ERROR]" if result_dict.get("error") else ""
            print(
                f"[{n_done:>5}/{len(todo)}] {mark} {result_dict['cve_id']:<18}"
                f" hit={result_dict.get('hit')} "
                f"iter={result_dict.get('iterations_used')} "
                f"truth_rank={result_dict.get('best_rank_in_truth_set')} "
                f"({elapsed/60:.1f}min elapsed, "
                f"{rate*60:.1f} CVE/min, ETA {eta_min:.1f}min){err}",
                flush=True,
            )

    elapsed = time.time() - t_start
    print(
        f"\n{'='*70}\n"
        f"Done: {n_hit}/{n_done} hits = {n_hit/max(1,n_done):.1%} Hit@1\n"
        f"Time: {elapsed/60:.1f}min ({elapsed/max(1,n_done):.1f}s/CVE)\n"
        f"Output: {output_path}\n"
        f"\nRun the metrics script:\n"
        f"  python -m patchholmes.cli.eval_phase2 {output_path} "
        f"--phase1 {args.phase1_jsonl}"
    )


if __name__ == "__main__":
    main()
