"""Compute Hit@1 / Precision / Recall / F1 for a PatchFinder_top10 run.

Reports both reporting cuts:
  - **all CVE** (1,252)             — includes 675 unrecoverable CVE; honest baseline
  - **recoverable CVE** (~577)      — drops CVE whose fix is not in the candidates

For comparison, also computes the trivial baselines (PatchFinder rank=1,
all-yes, all-no) using the source parquet.

Single-pick framing (matches our agent's output, no abstain):
    Per CVE the agent predicts exactly 1 commit as "fix".
    TP = # CVE where predicted commit ∈ fix_set
    FP = # CVE where predicted commit ∉ fix_set         (= n_cve − TP)
    FN = # unique label=1 commits across CVE − TP

Micro-Precision  = TP / (TP + FP)   == Hit@1 by construction
Micro-Recall     = TP / (TP + FN)
Micro-F1         = 2PR / (P+R)

Usage
-----
    python -m baselines.patchfinder_top10.eval logs/phase2/patchfinder_1252/qwen3_235b_openrouter/results.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PARQUET_DEFAULT = "./data/patchfinder_top10/cvevc_candidates/PatchFinder_top10/test-00000-of-00001.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval PatchHolmes on PatchFinder_top10.")
    p.add_argument("results_jsonl",
                   help="Path to the run.py output jsonl.")
    p.add_argument("--parquet",
                   default=PARQUET_DEFAULT,
                   help="The source variant parquet (for ground-truth and baselines).")
    p.add_argument("--summary-out",
                   default=None,
                   help="Optional path to write a JSON summary alongside the printout.")
    return p.parse_args()


def load_results(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.open() if l.strip()]


def compute_metrics(
    preds: dict[str, str | None],
    fix_sets: dict[str, set[str]],
    eval_cves: list[str],
) -> dict[str, Any]:
    """Compute Hit@1 and micro/macro P/R/F1 on the given CVE subset.

    Parameters
    ----------
    preds       : cve_id → predicted commit_id (or None if agent abstained / errored)
    fix_sets    : cve_id → set of label=1 commit_ids (deduped)
    eval_cves   : the subset of CVEs to score (e.g. all 1,252 or just recoverable)
    """
    tp = 0
    fp = 0
    abstain = 0   # for completeness — our agent doesn't abstain
    macro_p_sum = 0.0
    macro_r_sum = 0.0
    macro_f1_sum = 0.0
    macro_n = 0

    total_positives = sum(len(fix_sets.get(c, set())) for c in eval_cves)

    for cve in eval_cves:
        pred = preds.get(cve)
        gold = fix_sets.get(cve, set())

        if pred is None:
            abstain += 1
            # Per-CVE P/R/F1: P undefined (no prediction made), R = 0.
            macro_n += 1
            continue

        if pred in gold:
            tp += 1
            # Per-CVE: TP=1, FP=0, FN=(len(gold)-1)
            p_cve = 1.0
            r_cve = 1.0 / len(gold) if gold else 0.0
        else:
            fp += 1
            # Per-CVE: TP=0, FP=1, FN=len(gold)
            p_cve = 0.0
            r_cve = 0.0

        f1_cve = (2 * p_cve * r_cve / (p_cve + r_cve)) if (p_cve + r_cve) > 0 else 0.0
        macro_p_sum += p_cve
        macro_r_sum += r_cve
        macro_f1_sum += f1_cve
        macro_n += 1

    fn = total_positives - tp
    n_pred = tp + fp
    micro_p = tp / n_pred if n_pred > 0 else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    return {
        "n_cve":           len(eval_cves),
        "n_predicted":     n_pred,
        "n_abstained":     abstain,
        "TP":              tp,
        "FP":              fp,
        "FN":              fn,
        "total_positives": total_positives,
        "hit_at_1":        round(tp / len(eval_cves), 4) if eval_cves else 0.0,
        "micro_precision": round(micro_p, 4),
        "micro_recall":    round(micro_r, 4),
        "micro_f1":        round(micro_f1, 4),
        "macro_precision": round(macro_p_sum / max(1, macro_n), 4),
        "macro_recall":    round(macro_r_sum / max(1, macro_n), 4),
        "macro_f1":        round(macro_f1_sum / max(1, macro_n), 4),
    }


def patchfinder_rank1_baseline(
    df: pd.DataFrame,
    fix_sets: dict[str, set[str]],
    eval_cves: list[str],
) -> dict[str, Any]:
    """What if we just trust PatchFinder rank=1 (no agent)?"""
    rank1_preds: dict[str, str] = {}
    for cve in eval_cves:
        sub = df[df["cve"] == cve].sort_values("rank")
        if len(sub) > 0:
            rank1_preds[cve] = str(sub.iloc[0]["commit_id"])
    return compute_metrics(rank1_preds, fix_sets, eval_cves)


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_jsonl)
    if not results_path.exists():
        raise SystemExit(f"results jsonl not found: {results_path}")

    print(f"Loading results from {results_path} ...")
    results = load_results(results_path)
    print(f"  {len(results):,} records")

    # Build pred dict — agent's pick per CVE.
    preds: dict[str, str | None] = {}
    n_errors = 0
    for r in results:
        if r.get("error"):
            preds[r["cve_id"]] = None
            n_errors += 1
        else:
            preds[r["cve_id"]] = r.get("best_commit_id")
    print(f"  agent errored on {n_errors} CVE")

    # Build fix sets from the source parquet (deduped by commit_id).
    print(f"Loading ground truth from {args.parquet} ...")
    df = pd.read_parquet(args.parquet)
    df_dedup = df.drop_duplicates(subset=["cve", "commit_id"])
    fix_sets: dict[str, set[str]] = {}
    for cve, sub in df_dedup[df_dedup["label"] == 1].groupby("cve"):
        fix_sets[cve] = set(sub["commit_id"].astype(str))

    all_cves = list(df["cve"].drop_duplicates())
    eval_cves_all = [c for c in all_cves if c in preds]
    recoverable_cves = [c for c in eval_cves_all if c in fix_sets and len(fix_sets[c]) > 0]
    print(f"  {len(all_cves):,} CVE total in parquet")
    print(f"  {len(eval_cves_all):,} CVE have agent results")
    print(f"  {len(recoverable_cves):,} CVE are recoverable (fix in candidates)\n")

    metrics_all = compute_metrics(preds, fix_sets, eval_cves_all)
    metrics_rec = compute_metrics(preds, fix_sets, recoverable_cves)
    baseline_all = patchfinder_rank1_baseline(df, fix_sets, eval_cves_all)
    baseline_rec = patchfinder_rank1_baseline(df, fix_sets, recoverable_cves)

    def fmt(m: dict[str, Any]) -> str:
        return (
            f"Hit@1={m['hit_at_1']*100:5.2f}%  "
            f"micro P/R/F1={m['micro_precision']*100:5.2f}% / "
            f"{m['micro_recall']*100:5.2f}% / "
            f"{m['micro_f1']*100:5.2f}%   "
            f"macro F1={m['macro_f1']*100:5.2f}%   "
            f"(TP={m['TP']} FP={m['FP']} FN={m['FN']})"
        )

    print("=" * 100)
    print(f"\nCut 1: ALL CVE (n={metrics_all['n_cve']})")
    print(f"  PatchHolmes:        {fmt(metrics_all)}")
    print(f"  PatchFinder rank=1: {fmt(baseline_all)}")
    print(f"\nCut 2: RECOVERABLE only (n={metrics_rec['n_cve']})")
    print(f"  PatchHolmes:        {fmt(metrics_rec)}")
    print(f"  PatchFinder rank=1: {fmt(baseline_rec)}")
    print("\n" + "=" * 100)

    if args.summary_out:
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "results_jsonl": str(results_path),
            "parquet": args.parquet,
            "patchholmes": {"all": metrics_all, "recoverable": metrics_rec},
            "patchfinder_rank1": {"all": baseline_all, "recoverable": baseline_rec},
            "n_errors": n_errors,
        }, indent=2))
        print(f"\nSummary written to: {out_path}")


if __name__ == "__main__":
    main()
