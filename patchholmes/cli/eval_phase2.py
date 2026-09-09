#!/usr/bin/env python3
"""Compute Phase 2 evaluation metrics from a phase2 per-CVE JSONL file.

Phase 2 outputs a single `best_commit_id` per CVE, so the primary metric is
**Hit@1**. We also compute:
  - failure breakdown (Phase 1 missed vs Phase 2 picked wrong)
  - agent behaviour stats (iters, commits inspected, stopped reasons)
  - cost stats (input/output tokens, wall time)
  - optional hybrid Recall@K / MRR when a Phase 1 jsonl is supplied
    (agent's answer at rank 1; Phase 1 candidates fill 2..K)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

EVAL_KS = [1, 3, 5, 10, 100]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Phase 2 results: Hit@1, hybrid Recall@K, MRR, cost.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("jsonl", help="Per-CVE JSONL produced by Phase 2.")
    p.add_argument(
        "--phase1",
        default="",
        help="Optional Phase 1 per-query JSONL. If provided, compute hybrid "
             "Recall@K / MRR by composing agent answer (rank 1) + Phase 1 "
             "candidates (rank 2..K).",
    )
    p.add_argument("--output", default="", help="Write JSON summary to this path.")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


# ---------------------------------------------------------------------------
# Phase 2 standalone metrics
# ---------------------------------------------------------------------------

def compute_phase2_only(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    n_with_truth = 0
    n_hit = 0
    n_truth_in_top = 0
    n_p1_missed = 0
    n_p2_picked_wrong = 0

    iters: list[int] = []
    inspected_counts: list[int] = []
    in_tokens: list[int] = []
    out_tokens: list[int] = []
    walls: list[float] = []
    stopped_reasons: Counter[str] = Counter()
    cost = 0.0
    errors = 0

    for rec in records:
        fix_ids = set(rec.get("fix_commit_ids") or [])
        if not fix_ids:
            continue
        n_with_truth += 1
        hit = bool(rec.get("hit"))
        truth_in_top = rec.get("best_rank_in_truth_set") is not None

        if hit:
            n_hit += 1
        if truth_in_top:
            n_truth_in_top += 1
            if not hit:
                n_p2_picked_wrong += 1
        else:
            n_p1_missed += 1

        if rec.get("iterations_used") is not None:
            iters.append(int(rec["iterations_used"]))
        if rec.get("commits_inspected") is not None:
            inspected_counts.append(len(rec["commits_inspected"]))
        if rec.get("llm_input_tokens"):
            in_tokens.append(int(rec["llm_input_tokens"]))
        if rec.get("llm_output_tokens"):
            out_tokens.append(int(rec["llm_output_tokens"]))
        if rec.get("wall_time_sec"):
            walls.append(float(rec["wall_time_sec"]))
        if rec.get("stopped_reason"):
            stopped_reasons[rec["stopped_reason"]] += 1
        if rec.get("estimated_cost_usd"):
            cost += float(rec["estimated_cost_usd"])
        if rec.get("error"):
            errors += 1

    def _avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n_records": n,
        "n_with_truth": n_with_truth,
        "n_hit": n_hit,
        "hit_at_1": n_hit / max(1, n_with_truth),
        # Reachability (upper bound for Phase 2 alone)
        "n_truth_in_top": n_truth_in_top,
        "reachable_rate": n_truth_in_top / max(1, n_with_truth),
        # Phase 2 efficacy on cases where truth WAS in Top-K
        "phase2_efficacy": n_hit / max(1, n_truth_in_top),
        # Failure decomposition
        "n_phase1_missed": n_p1_missed,
        "n_phase2_picked_wrong": n_p2_picked_wrong,
        # Behaviour
        "avg_iterations": _avg(iters),
        "avg_commits_inspected": _avg(inspected_counts),
        # Cost
        "total_input_tokens": sum(in_tokens),
        "total_output_tokens": sum(out_tokens),
        "avg_input_tokens": _avg(in_tokens),
        "avg_output_tokens": _avg(out_tokens),
        "total_cost_usd": round(cost, 4),
        "avg_wall_time_sec": _avg(walls),
        "total_wall_time_sec": sum(walls),
        # Errors / stopped reasons
        "errors": errors,
        "stopped_reasons": dict(stopped_reasons.most_common()),
    }


# ---------------------------------------------------------------------------
# Hybrid Recall@K / MRR (Phase 2 answer ⨯ Phase 1 candidate list)
# ---------------------------------------------------------------------------

def compute_hybrid_metrics(
    phase2_records: list[dict[str, Any]],
    phase1_records: list[dict[str, Any]],
    ks: list[int] = EVAL_KS,
) -> dict[str, Any]:
    """Compose Phase 2's single answer with Phase 1's ranked list.

    For each CVE: rank-1 = agent's answer; rank-2..K = Phase 1 top-(K-1),
    skipping the agent's answer to avoid duplication. We then compute
    Recall@K (any fix ∈ Top-K) and MRR (1 / rank of first fix).
    """
    p1_by_cve = {r["cve_id"]: r for r in phase1_records}

    recall = {k: 0 for k in ks}
    mrr_sum = 0.0
    n_valid = 0
    n_skipped = 0

    for rec in phase2_records:
        cve_id = rec.get("cve_id")
        fix_ids = set(rec.get("fix_commit_ids") or [])
        if not fix_ids or cve_id not in p1_by_cve:
            n_skipped += 1
            continue
        n_valid += 1

        agent_answer = rec.get("best_commit_id")
        p1_cands = p1_by_cve[cve_id].get("candidates") or []

        # Build hybrid ranked list: [agent_answer] + [p1 cands minus agent_answer]
        hybrid: list[str] = []
        if agent_answer:
            hybrid.append(agent_answer)
        for c in p1_cands:
            cid = c["commit_id"]
            if cid != agent_answer and cid not in hybrid:
                hybrid.append(cid)

        # find first relevant rank in the hybrid list
        first_rank = None
        for i, cid in enumerate(hybrid, start=1):
            if cid in fix_ids:
                first_rank = i
                break
        if first_rank is not None:
            mrr_sum += 1.0 / first_rank
            for k in ks:
                if first_rank <= k:
                    recall[k] += 1

    return {
        "n_valid": n_valid,
        "n_skipped": n_skipped,
        "hybrid_MRR": mrr_sum / max(1, n_valid),
        **{f"hybrid_Recall@{k}": recall[k] / max(1, n_valid) for k in ks},
    }


# ---------------------------------------------------------------------------
# Direct Phase 1 vs Phase 2 comparison on the same CVE set
# ---------------------------------------------------------------------------

def compute_comparison(
    phase2_records: list[dict[str, Any]],
    phase1_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare Phase 1's pure ranking to Phase 2's single answer on the same
    CVE set. Used to quantify Phase 2's lift."""
    p1_by_cve = {r["cve_id"]: r for r in phase1_records}

    n = 0
    p1_recall_at_1 = 0
    p2_hit = 0

    for rec in phase2_records:
        cve_id = rec.get("cve_id")
        fix_ids = set(rec.get("fix_commit_ids") or [])
        if not fix_ids or cve_id not in p1_by_cve:
            continue
        n += 1

        # Phase 1 Recall@1: top-1 commit ∈ fixes
        p1_cands = p1_by_cve[cve_id].get("candidates") or []
        if p1_cands and p1_cands[0]["commit_id"] in fix_ids:
            p1_recall_at_1 += 1

        if rec.get("hit"):
            p2_hit += 1

    return {
        "n_aligned": n,
        "phase1_recall_at_1": p1_recall_at_1 / max(1, n),
        "phase2_hit_at_1": p2_hit / max(1, n),
        "absolute_gain": (p2_hit - p1_recall_at_1) / max(1, n),
        "relative_gain_pct": (
            (p2_hit - p1_recall_at_1) / max(1, p1_recall_at_1) * 100
            if p1_recall_at_1 else None
        ),
    }


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(
    p2: dict[str, Any],
    comparison: dict[str, Any] | None = None,
    hybrid: dict[str, Any] | None = None,
) -> None:
    print()
    print("=" * 60)
    print(f"  Phase 2 Evaluation  ({p2['n_with_truth']} valid CVEs)")
    print("=" * 60)
    print(f"  {'Hit@1':<28} {p2['hit_at_1']:.4f}   ({p2['n_hit']}/{p2['n_with_truth']})")
    print(f"  {'Reachable rate (R@K-ceiling)':<28} {p2['reachable_rate']:.4f}   ({p2['n_truth_in_top']}/{p2['n_with_truth']})")
    print(f"  {'Phase-2 efficacy':<28} {p2['phase2_efficacy']:.4f}   (on reachable)")
    print("-" * 60)
    print(f"  Phase 1 missed (truth ∉ Top-K): {p2['n_phase1_missed']}")
    print(f"  Phase 2 picked wrong:            {p2['n_phase2_picked_wrong']}")
    print("-" * 60)
    print(f"  Avg iterations:           {p2['avg_iterations']:.1f}")
    print(f"  Avg commits inspected:    {p2['avg_commits_inspected']:.1f}")
    print(f"  Avg input tokens:         {p2['avg_input_tokens']:,.0f}")
    print(f"  Avg output tokens:        {p2['avg_output_tokens']:,.0f}")
    print(f"  Avg wall time:            {p2['avg_wall_time_sec']:.1f}s")
    print(f"  Total cost (USD):         ${p2['total_cost_usd']}")
    print(f"  Errors:                   {p2['errors']}")
    print(f"  Stopped reasons:          {p2['stopped_reasons']}")

    if comparison is not None:
        print()
        print("=" * 60)
        print(f"  Phase 1 vs Phase 2  ({comparison['n_aligned']} CVEs aligned)")
        print("=" * 60)
        print(f"  Phase 1 Recall@1:  {comparison['phase1_recall_at_1']:.4f}")
        print(f"  Phase 2 Hit@1:     {comparison['phase2_hit_at_1']:.4f}")
        delta = comparison['absolute_gain']
        rel = comparison.get('relative_gain_pct')
        print(f"  Absolute gain:     {delta:+.4f}")
        if rel is not None:
            print(f"  Relative gain:     {rel:+.1f}%")

    if hybrid is not None:
        print()
        print("=" * 60)
        print(f"  Hybrid Phase 1 ⨯ Phase 2  ({hybrid['n_valid']} CVEs)")
        print(f"    (agent's answer at rank 1; Phase 1 fills 2..K)")
        print("=" * 60)
        print(f"  {'hybrid_MRR':<22} {hybrid['hybrid_MRR']:.4f}")
        print("-" * 60)
        for k in EVAL_KS:
            key = f"hybrid_Recall@{k}"
            if key in hybrid:
                print(f"  {key:<22} {hybrid[key]:.4f}")

    print("=" * 60)


def main() -> None:
    args = parse_args()
    path = Path(args.jsonl)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    print(f"Reading {path} ...")
    records = load_jsonl(path)
    print(f"  {len(records)} records loaded")

    p2 = compute_phase2_only(records)

    comparison = None
    hybrid = None
    if args.phase1:
        p1_path = Path(args.phase1)
        if not p1_path.exists():
            raise SystemExit(f"Phase 1 file not found: {p1_path}")
        print(f"Reading {p1_path} for hybrid metrics ...")
        p1_records = load_jsonl(p1_path)
        print(f"  {len(p1_records)} Phase 1 records loaded")
        comparison = compute_comparison(records, p1_records)
        hybrid = compute_hybrid_metrics(records, p1_records)

    print_summary(p2, comparison, hybrid)

    if args.output:
        out = {"phase2_metrics": p2}
        if comparison:
            out["comparison"] = comparison
        if hybrid:
            out["hybrid_metrics"] = hybrid
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nSummary written to {args.output}")


if __name__ == "__main__":
    main()
