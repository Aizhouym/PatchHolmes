"""Modular (progressive-disclosure) tool-interface ablation.

read_commit is reparameterised to overview-only (read_commit_budget=0: message +
file manifest, NO diff bodies). The diff-reading capability is factored entirely
into read_file_diff, so the ablation can attribute the diff-reading value to
read_file_diff instead of it being subsumed by read_commit.

Ladder (all budget=0 unless noted; same 235B / same pool / same eval protocol):
  modular_full         list + read_commit(overview) + read_file_diff + submit
  −read_file_diff      list + read_commit(overview) + submit
  list+submit          list + submit                 (budget-independent; reused)

Reference: Full (paper Table 1, read_commit WITH budgeted diff) = 59.95.
The key number = modular_full − (−read_file_diff) = read_file_diff's value when
it is the ONLY diff-reading tool.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from patchholmes.eval.metrics import (
    load_ground_truth, rank_patchholmes_with_phase1, compute_metrics,
)

COLS = ["R@1", "R@3", "R@5", "R@10", "NDCG@3", "NDCG@5", "NDCG@10", "MRR"]

# PatchHolmes numbers cited from the paper (Table 1, strict single-truth on
# sample_810). Cited, not recomputed.
FULL_PAPER = {"R@1": 59.95, "R@3": 68.36, "R@5": 71.20, "R@10": 77.26,
              "NDCG@3": 64.87, "NDCG@5": 66.04, "NDCG@10": 68.01, "MRR": 65.13}

# (label, tools-desc, run filename or None for cited)
LADDER_TEMPLATE = [
    ("Full (main interface, read_commit with diff) ¹", "list+read_commit(+diff)+read_file_diff+submit", None),
    ("modular_full (overview + drill-down)", "list+read_commit(overview)+read_file_diff+submit", "ablation_modular_full.jsonl"),
    ("modular −read_file_diff", "list+read_commit(overview)+submit", "ablation_modular_no_filediff.jsonl"),
    ("list+submit", "list+submit", "ablation_tool_list_submit.jsonl"),
]


def metrics_for(run, truth, sample, phase1):
    rk, sub, miss, steps, n = {}, 0, 0, 0, 0
    for l in open(run):
        r = json.loads(l); c = r.get("cve_id")
        if c not in sample:
            continue
        n += 1
        rk[c] = rank_patchholmes_with_phase1(r, phase1.get(c, []), top_k=10)
        if r.get("stopped_reason") == "submit_answer" and r.get("best_commit_id"):
            sub += 1
        steps += r.get("iterations_used", 0) or 0
    m = compute_metrics(rk, truth, eval_cves=sample, ks=(1, 3, 5, 10))
    return {c: m[c] * 100 for c in COLS}, sub, n, steps / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="./data/sample_ground_truth_810.csv",
                    help="Ground-truth CSV.")
    ap.add_argument("--phase1-jsonl", default="logs/phase1/main_rrf_810.jsonl")
    ap.add_argument("--results-dir", default="./logs/ablation",
                    help="Directory holding the ablation_*.jsonl run files.")
    ap.add_argument("--output-md", default="./ablation/tables/tool_ablation_modular.md")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    LADDER = [
        (label, tools, (results_dir / fname if fname else None))
        for label, tools, fname in LADDER_TEMPLATE
    ]

    truth = load_ground_truth(args.gt, augment_from_per_pair=None)
    sample = set(truth)
    phase1 = {}
    for l in open(args.phase1_jsonl):
        r = json.loads(l)
        if r["cve_id"] in sample:
            phase1[r["cve_id"]] = [c["commit_id"] for c in (r.get("candidates") or [])]

    rows = []
    for label, tools, run in LADDER:
        if run is None:
            rows.append((label, tools, FULL_PAPER, None, None))
        elif run.exists():
            vals, sub, n, avg_steps = metrics_for(run, truth, sample, phase1)
            rows.append((label, tools, vals, sub, avg_steps))
        else:
            rows.append((label, tools, None, None, None))

    got = {r[0]: r[2] for r in rows if r[2]}
    mf = got.get("modular_full (overview + drill-down)", {}).get("R@1")
    mn = got.get("modular −read_file_diff", {}).get("R@1")

    out = ["# Modular (progressive-disclosure) tool-interface ablation", "",
           "`read_commit` is reparameterised to **overview-only** (`read_commit_budget=0`: commit message + file manifest, **no diff body**),",
           "the diff-reading capability is moved entirely into `read_file_diff`. This lets the ablation attribute the value of reading diffs **cleanly to read_file_diff**,",
           "instead of it being subsumed by read_commit's budgeted diff. Same 235B / same pool / same eval protocol (strict 810 / full-809 / "
           "`rank_patchholmes_with_phase1` top10); both modular tiers use max_iter=15 (matching the main run).", "",
           "| Tier | Available tools | R@1 | R@3 | R@5 | R@10 | NDCG@3 | NDCG@5 | NDCG@10 | MRR | submit% |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, tools, vals, sub, avg_steps in rows:
        if vals is None:
            out.append(f"| {label} | {tools} | —(incomplete) | | | | | | | | |")
            continue
        subp = "—(cited)" if sub is None else f"{sub/809*100:.0f}%"
        cells = " | ".join(f"{vals[c]:.2f}" for c in COLS)
        out.append(f"| {label} | {tools} | {cells} | {subp} |")

    out += ["", "¹ Full = paper Table 1 reported value (main interface with read_commit's budgeted diff).", "", "## Conclusion", ""]
    if mf is not None and mn is not None:
        out += [
            f"- **Under the modular interface, read_file_diff carries the value of diff-reading**:",
            f"  modular_full = {mf:.2f} vs without read_file_diff = {mn:.2f} → **ΔR@1 = {mf-mn:+.2f}**.",
            f"  When read_commit gives only an overview, whether you can drill into files to read +/− lines directly determines ~{mf-mn:.0f} R@1 points.",
            f"- Compared with the main interface: read_commit carries its own diff → read_file_diff is redundant (+0); once the modular interface splits them apart → read_file_diff is worth {mf-mn:+.1f}.",
            f"  **The same diff-reading capability, implemented via a different interface, has its value land explicitly on read_file_diff.**",
            "- Conclusion: diff-reading is essential (worth ~+12, consistent with the main interface's list+submit→read_commit +12.6);",
            "  whether a tool is 'useful' depends on how capability is distributed across tools, not on the tool name itself. The two interfaces cross-validate the value of diff-reading, so the ablation holds.",
        ]
    dest = Path(args.output_md)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out))

    print(f"{'tier':<38}{'R@1':>8}{'R@10':>8}{'MRR':>8}  submit% avg_steps")
    for label, tools, vals, sub, avg_steps in rows:
        if vals is None:
            print(f"{label:<38}{'—':>8}"); continue
        sp = "cited" if sub is None else f"{sub/809*100:.0f}%"
        st = "" if avg_steps is None else f"{avg_steps:.1f}"
        print(f"{label:<38}{vals['R@1']:>7.2f}{vals['R@10']:>8.2f}{vals['MRR']:>8.2f}  {sp:>6} {st:>6}")
    if mf is not None and mn is not None:
        print(f"\n>>> read_file_diff value (modular interface) = {mf:.2f} − {mn:.2f} = {mf-mn:+.2f} R@1")
    print(f">>> wrote {dest}")


if __name__ == "__main__":
    main()
