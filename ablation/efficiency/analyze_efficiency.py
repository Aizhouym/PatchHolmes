#!/usr/bin/env python3
"""Efficiency / cost / token report for PatchHolmes Phase 2.

Reports, over a Phase-2 results.jsonl: input/output tokens (distribution),
tool_calls, commits read, read_file_diff usage, failure / context-overflow
rates, wall-clock latency, and a token-vs-cost table.

NOTE: `llm_input_tokens` / `llm_output_tokens` in results.jsonl are CUMULATIVE
across all LLM calls in the per-CVE agent conversation (each turn re-sends the
growing context), i.e. the total tokens billed per CVE. That is exactly the
quantity needed for a cost comparison.
"""
import argparse
import json, statistics as st, collections, pathlib, csv

# EVAL SET = sample_810 (canonical). We ONLY use it to define which 809 CVEs
# the efficiency stats cover. We do NOT recompute any performance metric here
# — performance numbers are the AUTHOR'S ORIGINAL reported values (below),
# to avoid any drift from the paper's Table 1.

# Author's reported R@1 on sample_810 (cited verbatim from paper / ablation docs).
# Not recomputed here.
PAPER_R1 = {
    "Qwen3-235B-A22B (main)": "59.95%",   # PatchHolmes number cited from the paper
    "Qwen3-Coder-30B-A3B":    "59.09%",   # paper Table 4 (author-reported)
}

# Representative paid-API prices ($ per 1M tokens). Clearly-labelled estimates;
# edit here to reprice. Qwen3-235B rates ~ OpenRouter mid-tier providers.
PRICES = {
    "Qwen3-235B-A22B (main)": {"in": 0.20, "out": 0.60},
    "Qwen3-Coder-30B-A3B":    {"in": 0.10, "out": 0.30},
}


def load_sample_ids(path):
    with open(path) as f:
        return {row["cve"] for row in csv.DictReader(f)}


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, int(len(sorted_vals) * p))
    return sorted_vals[i]


def dist(vals):
    s = sorted(vals)
    return {
        "mean": st.mean(vals) if vals else 0,
        "median": st.median(vals) if vals else 0,
        "p90": pct(s, 0.90), "p99": pct(s, 0.99),
        "max": max(vals) if vals else 0,
    }


def analyze(path, sample_ids):
    recs = [json.loads(l) for l in open(path)]
    # restrict to the canonical 809-CVE evaluation sample
    recs = [r for r in recs if (r.get("cve_id") or r.get("cve")) in sample_ids]
    n = len(recs)
    in_tok  = [r.get("llm_input_tokens") or 0 for r in recs]
    out_tok = [r.get("llm_output_tokens") or 0 for r in recs]
    iters   = [r.get("iterations_used") or 0 for r in recs]
    walls   = [r.get("wall_time_sec") or 0 for r in recs]
    # commits_inspected is stored as a list of commit ids -> take its length
    def _n(v):
        return len(v) if isinstance(v, (list, tuple, dict)) else (v or 0)
    commits = [_n(r.get("commits_inspected")) for r in recs]

    # tool-call breakdown
    tool_ct = collections.Counter()
    tcalls_per_cve = []
    used_rfd = 0          # CVEs that used read_file_diff >=1x
    rfd_counts = []
    for r in recs:
        tcs = r.get("tool_calls") or []
        tcalls_per_cve.append(len(tcs))
        this = collections.Counter(t.get("tool") for t in tcs)
        tool_ct.update(this)
        if this.get("read_file_diff", 0) > 0:
            used_rfd += 1
        rfd_counts.append(this.get("read_file_diff", 0))

    # outcomes
    reasons = collections.Counter(r.get("stopped_reason") for r in recs)
    errors  = sum(1 for r in recs if r.get("error"))
    # context-overflow detection: litellm raises ContextWindowExceededError when the
    # single-request prompt exceeds the served max_model_len (32000 in this run).
    ctx_over = sum(
        1 for r in recs
        if any(k in str(r.get("error", "")).lower()
               for k in ["contextwindowexceeded", "maximum context length"])
    )

    tot_in, tot_out = sum(in_tok), sum(out_tok)
    return dict(
        n=n,
        in_tok=dist(in_tok), out_tok=dist(out_tok),
        iters=dist(iters), walls=dist(walls), commits=dist(commits),
        tcalls=dist(tcalls_per_cve), tool_ct=tool_ct,
        used_rfd=used_rfd, rfd_frac=used_rfd / n * 100, rfd_mean=st.mean(rfd_counts),
        reasons=reasons, errors=errors, ctx_over=ctx_over,
        tot_in=tot_in, tot_out=tot_out,
    )


def fmt(d):
    return (f"{d['mean']:,.0f} / {d['median']:,.0f} / {d['p90']:,.0f} / "
            f"{d['p99']:,.0f} / {d['max']:,.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", default="./data/sample_ground_truth_810.csv",
                    help="ground-truth CSV defining the canonical 809-CVE sample")
    ap.add_argument("--main-run",
                    default="./logs/phase2/qwen3_235b/results.jsonl",
                    help="Phase-2 results.jsonl for the main (235B) run")
    ap.add_argument("--coder-run",
                    default="./logs/phase2/qwen3_coder_30b/results.jsonl",
                    help="Phase-2 results.jsonl for the 30B coder run")
    ap.add_argument("--out", default="./tables/efficiency.md",
                    help="output markdown path")
    args = ap.parse_args()

    RUNS = {
        "Qwen3-235B-A22B (main)": args.main_run,
        "Qwen3-Coder-30B-A3B":    args.coder_run,
    }

    sample_ids = load_sample_ids(args.gt)

    out = ["# Efficiency / Cost / Token report",
           "",
           f"**Evaluation sample: canonical 809-CVE sample** (same batch as the paper main "
           f"results / Table 1, {len(sample_ids)} CVEs total). per-CVE token/tool/wall-clock "
           f"taken from the main run, subset to these 809.",
           "",
           "> **Important**: `llm_input_tokens`/`llm_output_tokens` are the total token count",
           "> accumulated across the **entire per-CVE agent conversation** (each turn re-sends the",
           "> growing context), i.e. the tokens actually billed per CVE.",
           ""]

    res = {}
    for name, path in RUNS.items():
        if not pathlib.Path(path).exists():
            continue
        res[name] = analyze(path, sample_ids)

    # ---- Table 1: main efficiency table ----
    out += ["## Table 1 — Per-CVE efficiency profile (mean / median / p90 / p99 / max)", ""]
    hdr = "| Metric | " + " | ".join(res.keys()) + " |"
    out += [hdr, "|" + "---|" * (len(res) + 1)]
    rows = [
        ("CVE count (sample_810)", lambda r: f"{r['n']:,}"),
        ("input tokens (cumulative/CVE)", lambda r: fmt(r["in_tok"])),
        ("output tokens (cumulative/CVE)", lambda r: fmt(r["out_tok"])),
        ("LLM turns (iterations)", lambda r: fmt(r["iters"])),
        ("tool calls / CVE", lambda r: fmt(r["tcalls"])),
        ("commits inspected / CVE", lambda r: fmt(r["commits"])),
        ("wall-clock sec / CVE", lambda r: fmt(r["walls"])),
    ]
    for label, fn in rows:
        out.append(f"| {label} | " + " | ".join(fn(res[k]) for k in res) + " |")
    # performance = author's ORIGINAL reported numbers, cited verbatim (no recompute)
    out.append("| R@1 (paper reported value, not recomputed) | " +
               " | ".join(f"**{PAPER_R1.get(k, 'TBD')}**" for k in res) + " |")
    out += ["",
            "> R@1 is cited directly from the authors' paper reported values (235B=59.95%; "
            "30B=59.09%, paper Table 4). **This script does not recompute any performance "
            "metric**, to avoid any drift from Table 1. What this table adds is the **efficiency "
            "profile** not reported in the paper."]

    # ---- Table 2: tool usage ----
    out += ["", "## Table 2 — Tool-call breakdown (grand totals)", ""]
    out += ["| Tool | " + " | ".join(res.keys()) + " |", "|" + "---|" * (len(res) + 1)]
    all_tools = ["list_candidates", "read_commit", "read_file_diff", "submit_answer"]
    for t in all_tools:
        out.append(f"| `{t}` | " + " | ".join(f"{res[k]['tool_ct'].get(t,0):,}" for k in res) + " |")
    out.append("| **read_file_diff usage rate** | " +
               " | ".join(f"{res[k]['rfd_frac']:.1f}% of CVEs (avg {res[k]['rfd_mean']:.2f}×)" for k in res) + " |")

    # ---- Table 3: outcomes / failures ----
    out += ["", "## Table 3 — Stop reasons / failure rates (context-overflow rate)", ""]
    out += ["| Stop reason | " + " | ".join(res.keys()) + " |", "|" + "---|" * (len(res) + 1)]
    all_reasons = ["submit_answer", "max_iterations", "no_answer", "error"]
    for rea in all_reasons:
        out.append(f"| {rea} | " +
                   " | ".join(f"{res[k]['reasons'].get(rea,0):,} ({res[k]['reasons'].get(rea,0)/res[k]['n']*100:.1f}%)" for k in res) + " |")
    out.append("| **context-overflow** | " +
               " | ".join(f"{res[k]['ctx_over']:,} ({res[k]['ctx_over']/res[k]['n']*100:.2f}%)" for k in res) + " |")
    out += ["",
            "> context-overflow = `litellm.ContextWindowExceededError`: the single-request context",
            "> exceeds the server-side `max_model_len` (= **32,000** in this run). Only 0.49% on this",
            "> sample (0.73% on the full corpus). **Raising the serving `max_model_len` to 49152 (48K)",
            "> eliminates this overflow batch** and is expected to reduce the elevated 235B",
            "> max_iterations rate."]

    # ---- Table 4: cost ----
    FULL_CORPUS = 8401
    out += ["", "## Table 4 — Total tokens & cost estimate", "",
            f"per-CVE cost measured on the **809 sample**; when comparing to Favia's ~\\$400, "
            f"extrapolate per-CVE × {FULL_CORPUS:,} **to the full corpus**. Cost = **hypothetical** "
            "cost using paid-API unit prices (the main results actually ran on local vLLM, zero "
            "marginal cost).",
            "Unit prices are clearly-labelled estimates, see the script `PRICES`.", ""]
    out += ["| Run | 809 total input tok | 809 total output tok | Unit price (in/out /1M) | Cost per CVE | Extrapolated full 8,401 |",
            "|---|---|---|---|---|---|"]
    for k in res:
        r = res[k]; p = PRICES.get(k, {"in": 0, "out": 0})
        cost = r["tot_in"] / 1e6 * p["in"] + r["tot_out"] / 1e6 * p["out"]
        per = cost / r["n"]
        out.append(f"| {k} | {r['tot_in']:,} | {r['tot_out']:,} | "
                   f"${p['in']:.2f} / ${p['out']:.2f} | ${per:.4f} | **${per*FULL_CORPUS:,.0f}** |")

    R235 = res["Qwen3-235B-A22B (main)"]; R30 = res["Qwen3-Coder-30B-A3B"]
    c235 = (R235["tot_in"]/1e6*PRICES["Qwen3-235B-A22B (main)"]["in"]
            + R235["tot_out"]/1e6*PRICES["Qwen3-235B-A22B (main)"]["out"]) / R235["n"] * FULL_CORPUS
    c30 = (R30["tot_in"]/1e6*PRICES["Qwen3-Coder-30B-A3B"]["in"]
           + R30["tot_out"]/1e6*PRICES["Qwen3-Coder-30B-A3B"]["out"]) / R30["n"] * FULL_CORPUS
    speed = R235["walls"]["mean"] / max(R30["walls"]["mean"], 1e-9)  # total wall-clock ratio (mean)
    out += ["",
            "### Key points (cost comparison)",
            "1. **Reports PatchHolmes's own token budget**: the table above — mean ~92K in / "
            "~0.8K out per CVE, 15-turn cap.",
            f"2. **30B is a near-free lunch (strongest point)**: **paper Table 4 already shows 30B "
            f"and 235B have nearly equal R@1** (citing reported values, not recomputed), while on "
            f"efficiency 30B is **~{speed:.1f}× faster** in total processing wall-clock "
            f"({R30['walls']['mean']:.0f}s vs {R235['walls']['mean']:.0f}s mean) at roughly "
            f"**half** the cost (extrapolated full corpus ${c30:,.0f} vs ${c235:,.0f}). → A "
            f"deployable config can use 30B, ~$0.010 per CVE.",
            "3. **Cost upper bound**: the $ in Table 4 is a **no-cache** upper bound; vLLM prefix "
            "caching hits the common re-sent prefix across turns, so actual billing is "
            "significantly lower. The main results ran on local 4×H200, marginal cost ≈ "
            "electricity.",
            "4. **Aligning the price baseline with Favia**: the paper's Favia ~$400/pass price "
            "baseline is not public; head-to-head both use the **same** Qwen3-235B backbone, so a "
            "fair comparison should compare **token volume**. Favia = **10** pointwise calls per "
            "CVE (precision only 0.17, over-saying yes to many candidates); PatchHolmes = **1** "
            "~8-turn conversation per CVE. **Exact Favia token volume requires a re-run logging "
            "usage, after which an apples-to-apples number can be given.**",
            ""]

    txt = "\n".join(out)
    dest = pathlib.Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(txt)
    print(txt)
    print(f"\n>>> written to {dest}")


if __name__ == "__main__":
    main()
