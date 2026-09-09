"""Same-pool Favia baseline: merge all Favia runs and evaluate on sample_810.

Evaluates Favia's per-pair binary classifier run over PatchHolmes's Phase-1
RRF top-10 pool (identical pool to no-agent / IRCoT-A / PatchHolmes-top10).
Merges the several run-output CSVs (an original partial run, two parallel runs,
and a re-run of the remaining pairs) into one best-record-per-pair set.

Canonical aggregation = rank_favia_per_pair (answer=True by confidence DESC,
then answer=False by confidence ASC) — reproduces the paper's Favia number.
Metric = strict single-truth R@1/3/5/10 on sample_810 + 95% bootstrap CI.

Each source CSV has columns:
    cve, commit_id, repo, label, rank, answer, confidence, failed, error
"""
from __future__ import annotations
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from patchholmes.eval.metrics import load_ground_truth, rank_favia_per_pair, compute_metrics

random.seed(0)
BOOT = 2000


def to_rec(row):
    a = str(row.get("answer", "")).strip()
    ans = True if a == "True" else (False if a == "False" else None)
    try:
        conf = float(row.get("confidence") or 0)
    except ValueError:
        conf = 0.0
    return {"input": {"commit_id": row["commit_id"]},
            "output": {"answer": ans, "confidence": conf}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ground-truth", default="./data/sample_ground_truth_810.csv",
                    help="Ground-truth CSV (cve,patch,...).")
    ap.add_argument("--results-dir", default=None,
                    help="Directory holding the Favia run-output CSVs "
                         "(results_out/A/B/A2/B2.csv). Defaults to "
                         "'favia_samepool_results' next to this script.")
    args = ap.parse_args()

    res_dir = Path(args.results_dir) if args.results_dir else \
        Path(__file__).resolve().parent / "favia_samepool_results"
    srcs = [str(res_dir / f"results_{t}.csv") for t in ("out", "A", "B", "A2", "B2")]

    truth = load_ground_truth(args.ground_truth, augment_from_per_pair=None)
    sample = set(truth)
    # merge: keep the best (valid) record per (cve, commit_id)
    best = {}
    n_failed = 0
    for src in srcs:
        if not Path(src).exists():
            continue
        for r in csv.DictReader(open(src)):
            cve = r.get("cve")
            if cve not in sample:
                continue
            failed = str(r.get("failed", "")).lower() == "true"
            ans = str(r.get("answer", "")).strip() in ("True", "False")
            key = (cve, r["commit_id"])
            # prefer a valid (non-failed, has answer) record over a failed one
            if key not in best or (ans and not failed):
                best[key] = (r, failed and not ans)
    for _, isfail in best.values():
        n_failed += 1 if isfail else 0

    grp = defaultdict(list)
    for (cve, cid), (r, isfail) in best.items():
        if not isfail:
            grp[cve].append(r)

    rankings = {}
    for cve in sample:
        usable = grp.get(cve, [])
        if usable:
            rankings[cve] = rank_favia_per_pair([to_rec(r) for r in usable])

    eval_cves = list(rankings)
    n = len(eval_cves)
    m = compute_metrics(rankings, truth, eval_cves=eval_cves, ks=(1, 3, 5, 10))
    # bootstrap 95% CI on R@1
    hitv = [1 if (rankings[c] and rankings[c][0] in truth[c]) else 0 for c in eval_cves]
    boot = sorted(sum(hitv[random.randrange(n)] for _ in range(n)) / n * 100 for _ in range(BOOT)) if n else [0]
    lo, hi = boot[int(BOOT * 0.025)], boot[int(BOOT * 0.975)]

    def pct(key):
        return m[key] * 100
    print(f"Favia same-pool (Phase-1 top-10, qwen3-235b, max_steps=15, merged 5 sources)")
    print(f"  evaluable CVE = {n}/809")
    print(f"  R@1={pct('R@1'):.2f} R@3={pct('R@3'):.2f} R@5={pct('R@5'):.2f} R@10={pct('R@10'):.2f} "
          f"NDCG@3={pct('NDCG@3'):.2f} NDCG@5={pct('NDCG@5'):.2f} NDCG@10={pct('NDCG@10'):.2f} MRR={pct('MRR'):.2f}")
    print(f"  R@1 95% CI [{lo:.1f}, {hi:.1f}]")
    print("  Markdown row:")
    print(f"  | Favia same-pool (per-pair) | {pct('R@1'):.2f} | {pct('R@3'):.2f} | {pct('R@5'):.2f} | "
          f"{pct('R@10'):.2f} | {pct('NDCG@3'):.2f} | {pct('NDCG@5'):.2f} | {pct('NDCG@10'):.2f} | {pct('MRR'):.2f} |")
    return n, m


if __name__ == "__main__":
    main()
