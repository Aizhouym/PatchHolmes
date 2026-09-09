"""Same-pool Favia R@1 with 95% bootstrap CI.

Favia's per-pair binary classifier run over PatchHolmes's Phase-1 RRF top-10
(identical pool to no-agent / IRCoT-A / PatchHolmes-top10). Uses the same
canonical aggregation as the full eval (`rank_favia_per_pair`: answer=True
ranked first by confidence DESC, then answer=False by confidence ASC) so the
number is apples-to-apples with Favia's original 34.61%.

Input: a single Favia output CSV (cols: cve, commit_id, repo, label, rank,
answer, confidence, failed, error).
Metric: strict single-truth R@1 on the CVEs completed so far, + bootstrap CI.
Failed pairs (git-worktree errors) are reported as a caveat; a CVE whose true
fix pair failed can still miss.
"""
from __future__ import annotations
import argparse
import csv
import random
from collections import defaultdict

from patchholmes.eval.metrics import load_ground_truth, rank_favia_per_pair

random.seed(0)
BOOT = 2000


def to_record(row: dict) -> dict:
    """CSV row -> canonical rank_favia_per_pair record."""
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
    ap.add_argument("--results", default="./data/favia_results.csv",
                    help="Favia output CSV to evaluate.")
    args = ap.parse_args()

    truth = load_ground_truth(args.ground_truth, augment_from_per_pair=None)
    sample = set(truth)
    rows = list(csv.DictReader(open(args.results)))
    grp = defaultdict(list)
    n_failed = 0
    for r in rows:
        if r["cve"] not in sample:
            continue
        if str(r.get("failed", "")).lower() == "true":
            n_failed += 1
        grp[r["cve"]].append(r)

    # a CVE is "evaluable" if it has any non-failed pair to rank
    hits, cves = {}, []
    for cve, rs in grp.items():
        usable = [r for r in rs if str(r.get("failed", "")).lower() != "true"]
        if not usable:
            continue
        ranking = rank_favia_per_pair([to_record(r) for r in usable])
        cves.append(cve)
        hits[cve] = 1 if (ranking and ranking[0] in truth[cve]) else 0

    n = len(cves)
    r1 = sum(hits.values()) / n * 100 if n else 0.0

    # bootstrap 95% CI over the evaluated CVEs
    vals = [hits[c] for c in cves]
    boot = sorted(sum(vals[random.randrange(n)] for _ in range(n)) / n * 100
                  for _ in range(BOOT))
    lo, hi = boot[int(BOOT * 0.025)], boot[int(BOOT * 0.975)]

    print(f"same-pool Favia (Phase-1 top-10, qwen3-235b)")
    print(f"  evaluated CVE = {n} / 809 (partial run)")
    print(f"  failed pairs  = {n_failed} ({n_failed/len(rows)*100:.1f}% of {len(rows)} pairs)")
    print(f"  R@1 (strict, canonical rank_favia_per_pair) = {sum(hits.values())}/{n} "
          f"= {r1:.2f}%  [95% CI {lo:.1f}, {hi:.1f}]")
    # Reference points: same-pool selectors vs Favia's original (different-pool) run.
    print(f"\n  Reference: PatchHolmes-top10 = 59.70% . IRCoT-A = 56.49% . no-agent = 32.63%")
    print(f"        Favia original (different pool) = 34.61%")
    return {"n": n, "r1": r1, "ci": (lo, hi), "n_failed": n_failed}


if __name__ == "__main__":
    main()
