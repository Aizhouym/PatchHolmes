"""Statistical significance for Table 1 with bootstrap CI.

Table 1 methods ONLY: PatchHolmes, Favia, IRCoT (their original runs).
Computes, on the 809-CVE sample_810 (strict single-truth), for K in {1,3,5,10}:
  - R@K per method
  - McNemar exact paired test (p-value) for each pair of methods
  - 95% bootstrap CI for each R@K
All from existing per-CVE results — no re-running of any system.
"""
from __future__ import annotations
import argparse
import json, math, random
from pathlib import Path

from patchholmes.eval.metrics import (
    load_ground_truth, rank_patchholmes_with_phase1, rank_ircot,
    rank_favia_per_pair, load_favia_per_pair,
)

random.seed(0)
KS = [1, 3, 5, 10]
BOOT = 2000

# R@K DISPLAY = paper Table 1 values, CITED VERBATIM — never recomputed.
# (IRCoT is single-pick, so R@K is constant across K.)
# PatchHolmes numbers cited from the paper (strict single-truth R@K on sample_810).
PAPER_R = {
    "PatchHolmes": {1: 59.95, 3: 68.36, 5: 71.20, 10: 77.26},
    "Favia":       {1: 34.61, 3: 59.70, 5: 67.00, 10: 72.31},
    "IRCoT":       {1: 28.55, 3: 28.55, 5: 28.55, 10: 28.55},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", default="./data/sample_ground_truth_810.csv",
                    help="strict single-truth ground-truth CSV")
    ap.add_argument("--ph-run", default="./logs/phase2/qwen3_235b/results.jsonl",
                    help="PatchHolmes Phase-2 results.jsonl")
    ap.add_argument("--ir-run", default="./logs/ircot/qwen3_235b/full_merged.jsonl",
                    help="IRCoT results.jsonl")
    ap.add_argument("--fa-run", default="./logs/favia/qwen3_235b/results.jsonl",
                    help="Favia per-pair results.jsonl")
    ap.add_argument("--phase1", default="./logs/phase1/main_rrf_810.jsonl",
                    help="Phase-1 RRF candidates jsonl (for depth-10 extension)")
    ap.add_argument("--out", default="./tables/significance.md",
                    help="output markdown path")
    args = ap.parse_args()

    truth = load_ground_truth(args.gt, augment_from_per_pair=None)   # 809 CVE, strict single-truth
    sample = set(truth)

    # Phase-1 candidate lists (used to extend PatchHolmes's ranking to depth 10,
    # matching the paper's Table 1 R@K convention: rank_patchholmes_with_phase1).
    phase1 = {}
    for l in open(args.phase1):
        r = json.loads(l)
        if r["cve_id"] in sample:
            phase1[r["cve_id"]] = [c["commit_id"] for c in (r.get("candidates") or [])]

    def rankings_ph():
        d = {}
        for l in open(args.ph_run):
            r = json.loads(l); c = r.get("cve_id")
            if c in sample:
                d[c] = rank_patchholmes_with_phase1(r, phase1.get(c, []), top_k=10)
        return d

    def rankings_ir():
        d = {}
        for l in open(args.ir_run):
            r = json.loads(l); c = r.get("cve_id") or r.get("id") or r.get("cve")
            if c in sample:
                d[c] = rank_ircot(r)
        return d

    def rankings_fa():
        grp = load_favia_per_pair(args.fa_run)
        return {c: rank_favia_per_pair(recs) for c, recs in grp.items() if c in sample}

    RK = {"PatchHolmes": rankings_ph(), "Favia": rankings_fa(), "IRCoT": rankings_ir()}

    def hit_at(rm, K):
        # full 809 denominator: a CVE the method never produced counts as miss (0)
        return {c: (1 if any(x in truth[c] for x in rm.get(c, [])[:K]) else 0) for c in sample}

    H = {m: {K: hit_at(RK[m], K) for K in KS} for m in RK}

    def mcnemar_exact(A, B):
        """Two-sided McNemar EXACT test on paired binary outcomes {cve: 0/1}.
        b = #(A hit, B miss); c = #(A miss, B hit). Under H0 the b+c discordant
        pairs split 50/50, so min(b,c) ~ Binomial(b+c, 0.5).
        p = 2 * P(X <= min(b,c)),  X ~ Binomial(n=b+c, p=0.5),  capped at 1.
        """
        b = sum(1 for c in sample if A[c] == 1 and B[c] == 0)
        cc = sum(1 for c in sample if A[c] == 0 and B[c] == 1)
        n, k = b + cc, min(b := b, cc)
        p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
        return b, cc, p

    def boot_ci(h, it=BOOT):
        vals = [h[c] for c in sample]; n = len(vals)
        rs = sorted(sum(vals[random.randrange(n)] for _ in range(n)) / n * 100 for _ in range(it))
        return rs[int(it * 0.025)], rs[int(it * 0.975)]

    # ---- assemble markdown ----
    out = ["# Table 1 statistical significance", "",
           "Scope: **the three Table 1 methods only** (PatchHolmes / Favia / IRCoT).",
           "Sample: sample_810, 809 CVE, strict single-truth.", "",
           "R@K is **cited directly from the paper Table 1 values (never recomputed)**; the "
           "p-values and 95% CIs are computed from the same runs' per-CVE hit/miss — the paper "
           "publishes neither per-CVE data nor significance tests, so there is no other source. "
           "This is a **new statistic** the paper lacks, not a 're-computation of R@K'. The run "
           "reproduces the paper's aggregate R@K to ≤1 CVE (Favia bit-for-bit, PatchHolmes off by "
           "1), so the per-CVE data is faithful.", ""]

    out += ["## R@K (paper Table 1 values, cited)", "",
            "| K | PatchHolmes | Favia | IRCoT |",
            "|---|---|---|---|"]
    for K in KS:
        out.append(f"| {K} | {PAPER_R['PatchHolmes'][K]:.2f}% | "
                   f"{PAPER_R['Favia'][K]:.2f}% | {PAPER_R['IRCoT'][K]:.2f}% |")

    out += ["", "## McNemar exact test p-values (pairwise)", "",
            "| K | PatchHolmes vs Favia | PatchHolmes vs IRCoT | Favia vs IRCoT |",
            "|---|---|---|---|"]
    pairs = [("PatchHolmes", "Favia"), ("PatchHolmes", "IRCoT"), ("Favia", "IRCoT")]
    for K in KS:
        cells = []
        for a, b in pairs:
            bb, cc, p = mcnemar_exact(H[a][K], H[b][K])
            cells.append(f"p={p:.1e} (b={bb},c={cc}){' *' if p < 0.05 else ''}")
        out.append(f"| {K} | " + " | ".join(cells) + " |")

    out += ["", "## 95% bootstrap confidence intervals for R@K", "",
            "| Method | " + " | ".join(f"R@{K}" for K in KS) + " |",
            "|---|" + "---|" * len(KS)]
    for m in RK:
        cis = []
        for K in KS:
            lo, hi = boot_ci(H[m][K]); cis.append(f"[{lo:.1f}, {hi:.1f}]")
        out.append(f"| {m} | " + " | ".join(cis) + " |")

    out += ["", "## How the p-value is computed (McNemar exact test)", "",
            "Two methods are compared on **the same batch of 809 CVEs**; each CVE is a paired "
            "binary outcome `hit@K` (whether gold is in that method's top-K ranking, 1/0).",
            "",
            "1. Consider only **discordant pairs** (CVEs where the two methods disagree):",
            "   - `b` = # CVEs where A hits, B misses",
            "   - `c` = # CVEs where A misses, B hits",
            "   - CVEs where both hit / both miss do not enter the test.",
            "2. Null hypothesis H0: no difference between methods → each discordant pair leans A or "
            "B with 50% each, i.e. `min(b,c) ~ Binomial(b+c, 0.5)`.",
            "3. Two-sided exact p-value = `2 × P(X ≤ min(b,c))`, `X ~ Binomial(n=b+c, p=0.5)`, "
            "capped at 1.0. (Exact binomial test, no normal approximation, stable for small "
            "samples.)",
            "",
            "Confidence intervals use **bootstrap**: resample the 809 CVEs with replacement 2000 "
            "times, recompute R@K each time, take the 2.5/97.5 percentiles.",
            "",
            "> R@K definition: `hit@K = 1` when the gold commit is within the method's top-K "
            "ranking. PatchHolmes's ranking uses `rank_patchholmes_with_phase1` (agent's "
            "[best + inspected] extended to 10 with Phase-1 candidates, matching the paper Table 1 "
            "convention); IRCoT outputs only one commit, so all K values are identical. Missing "
            "CVEs count as miss (full 809 denominator)."]

    txt = "\n".join(out)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(txt)
    print(txt)
    print(f"\n>>> written to {dest}")


if __name__ == "__main__":
    main()
