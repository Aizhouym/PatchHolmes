#!/usr/bin/env python3
"""CLI for IR-style method evaluation — R@K / NDCG@K / MRR on a CVE sample.

Compares multiple methods (PatchHolmes, IRCoT, Favia, …) side-by-side on
a sample CSV (default: ``data/sample_ground_truth_810.csv``, 809 CVE).
Each method's per-CVE output jsonl is grouped, ranked per the method's
natural output shape, and scored against the truth column.

Usage
-----
    # Three-way comparison (most common):
    python scripts/eval_methods.py \\
        --method patchholmes:logs/phase2/results.jsonl \\
        --method ircot:logs/ircot/results.jsonl \\
        --method favia:logs/favia/results.jsonl \\
        --patchholmes-extend-phase1 logs/phase1/phase1.jsonl

    # Single method:
    python scripts/eval_methods.py \\
        --method patchholmes:logs/phase2/results.jsonl

    # Use the 'full' denominator so errored CVE count as misses (the fairest
    # single number — captures answer quality AND robustness):
    python scripts/eval_methods.py --denominator full \\
        --method patchholmes:logs/phase2/results.jsonl

Method format
-------------
    --method <name>:<path-or-glob>
  where <name> ∈ {patchholmes, ircot, favia} and <path> is a jsonl
  (or glob matching multiple jsonls — sharded outputs are auto-merged;
  multiple paths can also be comma-separated).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patchholmes.eval.metrics import (
    METHODS,
    compute_metrics,
    load_favia_per_pair,
    load_ground_truth,
    load_jsonl_by_cve,
    rank_patchholmes,
    rank_patchholmes_with_phase1,
    rank_favia_per_pair,
    rank_ircot,
)


def build_rankings(
    method: str,
    paths: list[Path],
    phase1_path: str | None = None,
) -> dict[str, list[str]]:
    """Load method output and return {cve_id: ranking}.

    For PatchHolmes, if `phase1_path` is provided, extends agent's inspection
    list with Phase 1's top-100 ordering (for unseen commits). This gives a
    fair fixed-length ranking matching what per-pair methods (Favia) score.
    """
    if method == "favia":
        merged: dict[str, list] = {}
        for p in paths:
            d = load_favia_per_pair(p)
            for cve, recs in d.items():
                merged.setdefault(cve, []).extend(recs)
        return {cve: rank_favia_per_pair(recs) for cve, recs in merged.items()}

    records = load_jsonl_by_cve(*paths)

    if method == "patchholmes" and phase1_path:
        # Load Phase 1 candidates for ranking extension
        phase1 = {}
        with open(phase1_path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    p = json.loads(line)
                    phase1[p["cve_id"]] = [c["commit_id"] for c in (p.get("candidates") or [])]
                except Exception:
                    continue
        return {
            cve: rank_patchholmes_with_phase1(rec, phase1.get(cve, []))
            for cve, rec in records.items()
        }

    ranker = METHODS[method]
    return {cve: ranker(rec) for cve, rec in records.items()}


def parse_method_arg(spec: str) -> tuple[str, list[Path]]:
    """`name:path-or-glob` → (name, [Paths])"""
    if ":" not in spec:
        raise SystemExit(f"--method argument must be name:path[,path,...] (got: {spec})")
    name, paths_str = spec.split(":", 1)
    name = name.strip().lower()
    if name not in METHODS:
        raise SystemExit(f"Unknown method '{name}'. Available: {sorted(METHODS)}")
    paths: list[Path] = []
    for path_part in paths_str.split(","):
        path_part = path_part.strip()
        # Expand globs
        matches = sorted(glob.glob(path_part))
        if not matches:
            raise SystemExit(f"No files match: {path_part}")
        paths.extend(Path(p) for p in matches)
    return name, paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IR metrics (R@K, NDCG@K, MRR) on a CVE sample CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--method", action="append", required=True,
        help="Method to evaluate, format 'name:path-or-glob[,path,...]'. "
             "Can be passed multiple times for side-by-side comparison.",
    )
    p.add_argument(
        "--ground-truth", default="./data/sample_ground_truth_810.csv",
        help="Sample CSV with cve+patch columns.",
    )
    p.add_argument(
        "--denominator", choices=("coverage", "common", "full"), default="common",
        help="Eval denominator. 'coverage' = method's own coverage, "
             "'common' = CVEs covered by ALL methods (apples-to-apples), "
             "'full' = full sample (missing CVE counted as miss).",
    )
    p.add_argument(
        "--ks", type=int, nargs="+", default=[1, 3, 5, 10],
        help="K cutoffs for R@K and NDCG@K.",
    )
    p.add_argument(
        "--summary-json", default=None,
        help="Optional: write summary to this JSON path.",
    )
    p.add_argument(
        "--patchholmes-extend-phase1",
        default=None,
        metavar="PATH",
        help="For 'patchholmes' method, extend ranking past agent's inspection "
             "list with Phase 1 top-100 ordering. Useful for fair R@K vs "
             "Favia (which always ranks 10 candidates). "
             "PATH = phase1 jsonl, e.g., logs/phase1/phase1_clean_full_top100.jsonl",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    gt = load_ground_truth(args.ground_truth)
    print(f"Ground truth: {len(gt)} CVE from {args.ground_truth}\n")

    methods = [parse_method_arg(spec) for spec in args.method]

    # Build rankings for each method
    rankings = {}
    for name, paths in methods:
        phase1 = args.patchholmes_extend_phase1 if name == "patchholmes" else None
        r = build_rankings(name, paths, phase1_path=phase1)
        rankings[name] = r
        suffix = " (+ Phase1 ext)" if name == "patchholmes" and phase1 else ""
        print(f"  [{name:<12}] {len(r):>5} CVE covered from {len(paths)} file(s){suffix}")

    # Determine denominator
    if args.denominator == "common":
        eval_cves = set(gt)
        for r in rankings.values():
            eval_cves &= set(r)
        eval_cves = sorted(eval_cves)
    elif args.denominator == "full":
        eval_cves = sorted(gt)
    else:  # coverage — per-method, computed below
        eval_cves = None

    print(f"\nDenominator: {args.denominator}", end="")
    if eval_cves is not None:
        print(f"  ({len(eval_cves)} CVE)")
    else:
        print(" (per-method coverage)")

    # Compute metrics
    results = {}
    for name, _ in methods:
        if eval_cves is None:
            cves_for_this = sorted(set(rankings[name]) & set(gt))
        else:
            cves_for_this = eval_cves
        results[name] = compute_metrics(rankings[name], gt, cves_for_this, ks=args.ks)

    # Print comparison table
    headers = ["R@" + str(k) for k in args.ks] + ["NDCG@" + str(k) for k in args.ks] + ["MRR"]
    print()
    print("=" * (15 + 11 * len(headers)))
    header_row = f"  {'Method':<13}" + "".join(f"{h:>11}" for h in headers)
    print(header_row)
    print("=" * (15 + 11 * len(headers)))
    for name, _ in methods:
        m = results[name]
        cells = "".join(f"{m[h]*100:>10.2f}%" for h in headers)
        print(f"  {name:<13}{cells}")
    print("=" * (15 + 11 * len(headers)))

    # Persist summary
    if args.summary_json:
        out_path = Path(args.summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "ground_truth": args.ground_truth,
            "denominator": args.denominator,
            "ks": args.ks,
            "n_ground_truth": len(gt),
            "methods": {name: {"paths": [str(p) for p in paths], **results[name]}
                        for name, paths in methods},
        }
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
