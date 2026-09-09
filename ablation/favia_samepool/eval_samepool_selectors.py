"""Same-pool selector comparison: same Phase-1 pool, different selectors.

Isolates the Phase-2 agent's gain. All baselines select over PatchHolmes's
Phase-1 RRF top-10 (sample_810). PatchHolmes's own number is CITED from the
paper (59.95%), never recomputed. New-baseline R@1 IS computed (these runs are
new — no paper number to drift from). Metric = strict single-truth R@1 on
sample_810 (same as the paper's canonical eval).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from patchholmes.eval.metrics import load_ground_truth, rank_favia_per_pair, load_favia_per_pair

# PatchHolmes number cited from the paper (strict single-truth R@1 on sample_810).
PAPER = {"PatchHolmes (paper, top-100)": "59.95%"}


def strict_truth(gt_csv):
    return load_ground_truth(gt_csv, augment_from_per_pair=None)  # {cve: {commit}}, 809


def r_at_k(ranking_by_cve: dict, truth: dict, ks=(1, 5, 10)) -> dict:
    """R@k over the FULL sample_810 denominator (missing CVE = miss)."""
    out = {k: 0 for k in ks}
    for cve, tr in truth.items():
        ranking = ranking_by_cve.get(cve, [])
        for k in ks:
            if any(c in tr for c in ranking[:k]):
                out[k] += 1
    n = len(truth)
    return {f"R@{k}": out[k] / n * 100 for k in ks}, n


def eval_noagent(truth, phase1):
    """No-agent: take RRF rank-1. Also reports Phase-1 R@k ceiling of the pool."""
    ranking = {}
    for l in open(phase1):
        r = json.loads(l)
        if r["cve_id"] in truth:
            ranking[r["cve_id"]] = [c["commit_id"] for c in (r.get("candidates") or [])]
    # no-agent R@1 = rank-1; the pool's R@10 = Phase-1 ceiling for top-10 methods
    top1 = {cve: rk[:1] for cve, rk in ranking.items()}
    m1, n = r_at_k(top1, truth, ks=(1,))
    ceil, _ = r_at_k(ranking, truth, ks=(1, 5, 10))
    return m1["R@1"], ceil, n


def eval_ircot(truth, ircot_path):
    ranking, tin, tout, ncall, nrec = {}, 0, 0, 0, 0
    if not ircot_path.exists():
        return None
    for l in open(ircot_path):
        r = json.loads(l)
        if r["cve_id"] not in truth:
            continue
        nrec += 1
        ranking[r["cve_id"]] = r.get("ranking") or ([r["best_commit_id"]] if r.get("best_commit_id") else [])
        tin += r.get("llm_input_tokens", 0); tout += r.get("llm_output_tokens", 0)
        ncall += r.get("n_llm_calls", 0)
    m, n = r_at_k(ranking, truth, ks=(1,))
    return {"metrics": m, "n_rec": nrec, "tin": tin, "tout": tout, "ncall": ncall}


def eval_favia(truth, favia_path):
    if not favia_path.exists():
        return None
    recs = load_favia_per_pair(str(favia_path))     # groups per CVE
    ranking = {cve: rank_favia_per_pair(rs) for cve, rs in recs.items() if cve in truth}
    m, n = r_at_k(ranking, truth, ks=(1, 5, 10))
    # tokens: favia stores usage in spans, not per-record; count pairs as call proxy
    npairs = sum(len(rs) for cve, rs in recs.items() if cve in truth)
    return {"metrics": m, "n_cve": len(ranking), "n_pairs": npairs}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ground-truth", default="./data/sample_ground_truth_810.csv",
                    help="Ground-truth CSV (cve,patch,...).")
    ap.add_argument("--phase1", default="./logs/phase1/main_rrf_810.jsonl",
                    help="Phase-1 RRF results JSONL.")
    ap.add_argument("--ircot", default="./data/ircot_phase1pool.jsonl",
                    help="IRCoT-A output over the Phase-1 pool (optional).")
    ap.add_argument("--favia", default="./data/favia_phase1pool/favia_phase1_top10.jsonl",
                    help="Favia output over the Phase-1 pool (optional).")
    args = ap.parse_args()

    ircot_path = Path(args.ircot)
    favia_path = Path(args.favia)

    truth = strict_truth(args.ground_truth)
    print(f"sample_810 strict truth: {len(truth)} CVE\n")

    na_r1, ceil, n = eval_noagent(truth, args.phase1)
    print("=== Phase-1 pool ceiling (upper bound for top-10 methods) ===")
    print(f"  Phase-1 R@1/R@5/R@10 = {ceil['R@1']:.2f}% / {ceil['R@5']:.2f}% / {ceil['R@10']:.2f}%")
    print(f"  (fix inside top-10 = R@10 = R@1 upper bound for any top-10 selector)\n")

    print("=== Same Phase-1 pool, different selectors (strict R@1, sample_810) ===")
    print(f"{'method':<34}{'R@1':>8}   note")
    print(f"{'-'*34}{'-'*8}   {'-'*30}")
    print(f"{'no-agent (RRF top-1)':<34}{na_r1:>7.2f}%   take Phase-1 rank-1 directly")

    fav = eval_favia(truth, favia_path)
    if fav:
        m = fav["metrics"]
        print(f"{'Favia (Phase-1 top-10)':<34}{m['R@1']:>7.2f}%   {fav['n_cve']} CVE, {fav['n_pairs']} pairs; R@5={m['R@5']:.1f} R@10={m['R@10']:.1f}")
    else:
        print(f"{'Favia (Phase-1 top-10)':<34}{'—':>8}   (incomplete)")

    irc = eval_ircot(truth, ircot_path)
    if irc:
        m = irc["metrics"]
        print(f"{'IRCoT-A (Phase-1 top-10)':<34}{m['R@1']:>7.2f}%   {irc['n_rec']} CVE; {irc['ncall']} calls; "
              f"tok in {irc['tin']:,}/out {irc['tout']:,}")
    else:
        print(f"{'IRCoT-A (Phase-1 top-10)':<34}{'—':>8}   (incomplete)")

    for k, v in PAPER.items():
        print(f"{k:<34}{v:>8}   paper's reported value (cited, not recomputed)")

    print("\nNote: no-agent/Favia/IRCoT are new runs on the Phase-1 top-10 pool;")
    print("    PatchHolmes 59.95% is the paper value (top-100 pool). The agent's gap over "
          "no-agent/Favia/IRCoT is the absolute gain of the Phase-2 selector.")


if __name__ == "__main__":
    main()
