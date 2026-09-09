#!/usr/bin/env python3
"""Tool-interface ablation for the Phase-2 agent.

Same backbone, same Phase-1 pool, same eval — only the set of tools the Phase-2
agent may call changes. Three-rung ladder:

    Full           list + read_commit + read_file_diff + submit   (= main run, cited)
    no_filediff    list + read_commit + submit                    (this script)
    list_submit    list + submit                                  (this script)

The drop between rungs = the marginal value of that tool. Answers "does the
tool interface actually contribute, or would a cheaper interface do as well?".

Built entirely on the existing SDK path: `run_phase2_single(..., tool_names=...)`
restricts the agent to a subset of the already-registered tools (see
patchholmes/phase2/runner.py). We only supply a tier-matched system/user prompt
so the agent isn't told to use a tool it doesn't have. NO new abstractions.

Usage
-----
  python ablation/tool_interface/run_tool_ablation.py --tier no_filediff \
      --output logs/ablation/ablation_tool_no_filediff.jsonl --num-workers 24
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

# Base prompt (the Full-tier prompt). We derive the ablated prompts from its
# task framing + rank prior, swapping only the Tools/Strategy sections.
from patchholmes.phase2.agent import SYSTEM_PROMPT as FULL_SYSTEM_PROMPT  # noqa: E402,F401

# ── Shared preamble: task + rank prior (identical across tiers) ───────────────
_PREAMBLE = """\
You are a security researcher tracing the commit that fixed a specific CVE.

# Task

You are given:
- A CVE identifier and its description.
- A pool of Top-100 candidate commits retrieved by a hybrid BM25+dense pipeline.
  The true fix commit is in this pool with ~90% probability but is **rarely
  ranked first**. Across our calibration set the true fix sits at:
    - rank 1     : ~12% of CVEs
    - rank 2-5   : ~35%
    - rank 6-20  : ~25%
    - rank 21-100: ~15%
  So **never assume rank 1 is the answer just because it scored highest**.

Your job is to identify the single commit that fixes the vulnerability and
submit it as your answer.
"""

# ── Tier: no_filediff (list + read_commit + submit) ──────────────────────────
SP_NO_FILEDIFF = _PREAMBLE + """
# Tools

You have three tools. Use them deliberately — every call costs tokens.

1. `list_candidates()` — Always call this FIRST. It returns a one-line manifest
   for every candidate (commit ID, message first line, file counts per
   category). It gives you a global view before drilling in.

2. `read_commit(commit_id)` — Read one promising candidate in detail.
   **commit_id is a 40-char hex SHA (or an unambiguous prefix, e.g.
   '3bf5eddb89af'), NOT a rank number.** Returns the commit message, file
   manifest, and a budgeted diff render. Real CVE fixes typically add input
   validation, bounds checks, null checks, or correct ordering of operations.

3. `submit_answer(commit_id, reasoning)` — Submit your final single answer.
   **Call this EXACTLY ONCE.** After you call it the task is COMPLETE.

# Strategy guidelines

**Browse before reading.** The manifest already tells you a lot: a doc-only
commit is almost never a CVE fix; a source-file commit matching the description
is much more likely. Eliminate decoys from the manifest first.

**Read broadly, not just the top.** Inspect at least **5 candidates** with
read_commit before submitting. The truth is at rank 1 only ~12% of the time.
Scan the top-20 for source-code commits matching the CVE description.

**Plausibility ≠ proof.** Verify by reading the diff: does the change actually
fix what the CVE description says is broken? Doc-only commits, version bumps,
and formatting changes are almost never the fix.

When confident, call `submit_answer` ONCE with the commit ID and short
evidence-based reasoning."""

# ── Tier: list_submit (list + submit) ────────────────────────────────────────
SP_LIST_SUBMIT = _PREAMBLE + """
# Tools

You have only two tools. You CANNOT read commit diffs — you must decide from
the manifest alone.

1. `list_candidates()` — Call this FIRST. It returns a one-line manifest for
   every candidate: commit ID, the first line of the commit message, and file
   counts per category (source / test / doc / other). This manifest is ALL the
   information you get — there is no diff-reading tool.

2. `submit_answer(commit_id, reasoning)` — Submit your final single answer.
   **Call this EXACTLY ONCE.** commit_id is the 40-char hex SHA (or an
   unambiguous prefix) copied from the manifest, NOT a rank number.

# Strategy guidelines

You must judge purely from each candidate's message line and file-category
counts. Prefer commits whose message mentions the vulnerability, security, a
CVE id, or the affected component in the CVE description, and that touch
**source** files (not doc-only or test-only). Doc-only commits, version bumps,
and formatting changes are almost never the fix. Remember the true fix is at
rank 1 only ~12% of the time — scan the whole manifest, don't default to the
top.

When you have made your best judgement, call `submit_answer` ONCE with the
commit ID and a short reason based on the manifest."""

# ── Modular / progressive-disclosure interface (read_commit_budget=0) ────────
# read_commit is REPARAMETERISED to overview-only (message + file manifest, NO
# diff bodies). The diff-reading capability is factored entirely into
# read_file_diff. This lets the ablation attribute the diff-reading value to
# read_file_diff instead of it being subsumed by read_commit's budgeted diff.
_MODULAR_READ_COMMIT = (
    "`read_commit(commit_id)` — returns ONLY an overview: the commit message and "
    "the file manifest (each file's path, category, and +added/−deleted line "
    "counts). **It does NOT show any diff content — no code lines.**"
)

SP_MODULAR_FULL = _PREAMBLE + f"""
# Tools

You have three tools besides submit. **read_commit does not show code** in this
configuration — only read_file_diff shows the actual +/− lines.

1. `list_candidates()` — Call FIRST. One-line manifest per candidate (commit ID,
   message first line, file counts per category).

2. {_MODULAR_READ_COMMIT}
   To see the actual code changes you MUST call read_file_diff.

3. `read_file_diff(commit_id, file_path)` — Shows the actual diff (the +/− code
   lines) of ONE file. **This is the ONLY way to read code changes.** After
   read_commit shows you which files a commit touched, drill into the most
   promising **source** file(s) with read_file_diff to verify the fix.

4. `submit_answer(commit_id, reasoning)` — Submit your final answer. Call ONCE.

# Strategy guidelines

**You cannot judge a fix from the manifest alone.** The message + file list only
narrow down candidates; the decisive evidence is in the diff. Workflow:
list_candidates → read_commit (see which files/how many lines changed) →
read_file_diff on the promising **source** file(s) (read the actual +/− lines) →
submit. Verify the change actually fixes what the CVE describes before
submitting. Inspect several candidates (truth is at rank 1 only ~12% of the
time). Doc-only commits, version bumps, formatting are almost never the fix."""

SP_MODULAR_NO_FILEDIFF = _PREAMBLE + f"""
# Tools

You have two tools. **There is NO diff-reading tool** — you must judge from the
commit message and file manifest alone.

1. `list_candidates()` — Call FIRST. One-line manifest per candidate.

2. {_MODULAR_READ_COMMIT}
   NOTE: in this configuration there is no way to read diff content — the
   message and file manifest (paths + line counts) are ALL you get.

3. `submit_answer(commit_id, reasoning)` — Submit your final answer. Call ONCE.

# Strategy guidelines

Judge from each candidate's commit message and file manifest (which files, what
category, how many lines added/deleted). Prefer commits whose message mentions
the vulnerability / security / the affected component, and that touch **source**
files with a plausible change size. Doc-only commits, version bumps, and
formatting changes are almost never the fix. Truth is at rank 1 only ~12% of the
time — scan the whole list. When you have your best judgement, submit ONCE."""

SP_MODULAR_NO_COMMIT = _PREAMBLE + """
# Tools

You have two tools besides submit. There is **no read_commit / file-manifest
tool** — the only way to read code is read_file_diff.

1. `list_candidates()` — Call FIRST. One-line manifest per candidate (commit ID,
   message first line, file counts per category). It does NOT list file paths.

2. `read_file_diff(commit_id, file_path)` — Shows the actual diff (+/− code
   lines) of ONE file in a commit. You don't have a file listing, so infer a
   likely path from the CVE description / commit message. **If the path is
   wrong, the tool returns the commit's available files** — use that list to
   pick the correct file, then call again.

3. `submit_answer(commit_id, reasoning)` — Submit your final answer. Call ONCE.

# Strategy guidelines

To verify a candidate you must read its code with read_file_diff. Use the
error-listing behaviour to discover a commit's files: call read_file_diff with a
plausible source path; if not found, the tool lists the real files, then drill
into the promising **source** file. Verify the change fixes what the CVE
describes before submitting. Truth is at rank 1 only ~12% of the time — inspect
several candidates. Doc-only commits, version bumps, formatting are rarely fixes."""

TIERS: dict[str, dict[str, Any]] = {
    "modular_no_commit": {
        "tool_names": ["list_candidates", "read_file_diff", "submit_answer"],
        "system_prompt": SP_MODULAR_NO_COMMIT,
        "user_prompt": (
            "CVE ID: {cve}\nRepository: {owner}/{repo}\n\nCVE Description:\n{desc}\n\n"
            "Identify the single commit from the candidate pool that fixes this "
            "vulnerability. Call list_candidates, then use read_file_diff to read "
            "the actual code of promising commits (there is no read_commit tool — "
            "if a file path is wrong, the tool lists the commit's files), then "
            "submit_answer with your final choice and short reasoning."
        ),
    },
    "modular_full": {
        "tool_names": ["list_candidates", "read_commit", "read_file_diff", "submit_answer"],
        "system_prompt": SP_MODULAR_FULL,
        "user_prompt": (
            "CVE ID: {cve}\nRepository: {owner}/{repo}\n\nCVE Description:\n{desc}\n\n"
            "Identify the single commit from the candidate pool that fixes this "
            "vulnerability. Call list_candidates, then read_commit to see which "
            "files each promising commit changed, then read_file_diff to read the "
            "actual code changes (read_commit does NOT show diffs), then "
            "submit_answer with your final choice and short reasoning."
        ),
    },
    "modular_no_filediff": {
        "tool_names": ["list_candidates", "read_commit", "submit_answer"],
        "system_prompt": SP_MODULAR_NO_FILEDIFF,
        "user_prompt": (
            "CVE ID: {cve}\nRepository: {owner}/{repo}\n\nCVE Description:\n{desc}\n\n"
            "Identify the single commit from the candidate pool that fixes this "
            "vulnerability. Call list_candidates, then read_commit to see each "
            "promising commit's message and file manifest (there is no diff-reading "
            "tool), then submit_answer with your best choice and short reasoning."
        ),
    },
    "no_filediff": {
        "tool_names": ["list_candidates", "read_commit", "submit_answer"],
        "system_prompt": SP_NO_FILEDIFF,
        "user_prompt": (
            "CVE ID: {cve}\nRepository: {owner}/{repo}\n\nCVE Description:\n{desc}\n\n"
            "Identify the single commit from the candidate pool that fixes this "
            "vulnerability. Call list_candidates first, read promising commits "
            "with read_commit, then call submit_answer with your final choice "
            "and a short reasoning."
        ),
    },
    "list_submit": {
        "tool_names": ["list_candidates", "submit_answer"],
        "system_prompt": SP_LIST_SUBMIT,
        "user_prompt": (
            "CVE ID: {cve}\nRepository: {owner}/{repo}\n\nCVE Description:\n{desc}\n\n"
            "Identify the single commit from the candidate pool that fixes this "
            "vulnerability. Call list_candidates to see the manifest, then call "
            "submit_answer with your best choice and a short reasoning based on "
            "the manifest (you cannot read diffs)."
        ),
    },
}

# ── Worker ────────────────────────────────────────────────────────────────────
_W: dict[str, Any] = {}


def _init(llm_model, llm_base_url, llm_api_key, descriptions, repo2commits_root,
          max_iter, top_k, tier, read_commit_budget):
    from patchholmes.phase2.agent import build_llm
    _W["llm"] = build_llm(model=llm_model, base_url=llm_base_url,
                          api_key=llm_api_key, usage_id=f"ablation_{os.getpid()}")
    _W["descriptions"] = descriptions
    _W["repo2commits_root"] = repo2commits_root
    _W["max_iter"] = max_iter
    _W["top_k"] = top_k
    _W["tier"] = TIERS[tier]
    _W["read_commit_budget"] = read_commit_budget


def _process_one(rec):
    from patchholmes.data_models import CVEQuery, CommitDoc, RankedCandidate
    from patchholmes.phase2.result import Phase2Result
    from patchholmes.phase2.runner import run_phase2_single

    cve_id = rec["cve_id"]
    tier = _W["tier"]
    descs = _W["descriptions"]
    try:
        query = CVEQuery(cve_id=cve_id, description=descs.get(cve_id, ""),
                         owner=rec["owner"], repo=rec["repo"],
                         fix_commit_ids=rec.get("fix_commit_ids") or [])
        candidates = [
            RankedCandidate(
                commit=CommitDoc(commit_id=c["commit_id"], commit_msg="", diff="",
                                 owner=rec["owner"], repo=rec["repo"],
                                 datetime=c.get("datetime", "")),
                score=float(c.get("score", 0.0)), rank=int(c["rank"]),
                source=c.get("source", "rrf"),
                bm25_rank=c.get("bm25_rank"), dense_rank=c.get("dense_rank"),
            )
            for c in (rec.get("candidates") or [])
        ]
        up = tier["user_prompt"].format(
            cve=cve_id, owner=rec["owner"], repo=rec["repo"],
            desc=(descs.get(cve_id, "") or "(no description provided)"),
        )
        result = run_phase2_single(
            query=query, phase1_candidates=candidates,
            repo2commits_root=_W["repo2commits_root"], llm=_W["llm"],
            top_k=_W["top_k"], max_iteration_per_run=_W["max_iter"],
            read_commit_budget=_W["read_commit_budget"],
            system_prompt=tier["system_prompt"], user_prompt=up,
            tool_names=tier["tool_names"],
        )
        return result.to_dict()
    except Exception as e:
        return Phase2Result(
            cve_id=cve_id, owner=rec.get("owner", ""), repo=rec.get("repo", ""),
            best_commit_id=None, reasoning="",
            fix_commit_ids=rec.get("fix_commit_ids") or [],
            error=f"{type(e).__name__}: {e}",
        ).to_dict()


def load_descriptions(csv_path):
    d = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            c = (row.get("cve") or "").strip()
            if c:
                d[c] = (row.get("cve_description") or "").strip()
    return d


def load_done(out_path):
    if not Path(out_path).exists():
        return set()
    keep, done = [], set()
    for line in open(out_path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("error"):
            continue
        keep.append(r); done.add(r["cve_id"])
    with open(out_path, "w") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=list(TIERS))
    ap.add_argument("--phase1-jsonl", default="logs/phase1/main_rrf_810.jsonl")
    ap.add_argument("--dataset-csv", default="data/ground_truth_queries_clean.csv")
    ap.add_argument("--repo2commits", default="./data/repo2commits_diff")
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-workers", type=int, default=24)
    ap.add_argument("--llm-model", default="hosted_vllm/qwen3-235b")
    ap.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--llm-api-key", default="EMPTY")
    ap.add_argument("--max-iter", type=int, default=15)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--max-cves", type=int, default=0)
    ap.add_argument("--read-commit-budget", type=int, default=8000,
                    help="Char budget for read_commit's diff render. Set to 0 for "
                         "the modular/overview interface (read_commit = manifest only).")
    args = ap.parse_args()

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    descriptions = load_descriptions(args.dataset_csv)
    recs = [json.loads(l) for l in open(args.phase1_jsonl) if l.strip()]
    done = load_done(out)
    todo = [r for r in recs if r["cve_id"] not in done]
    if args.max_cves > 0:
        todo = todo[: args.max_cves]
    print(f"tier={args.tier} tools={TIERS[args.tier]['tool_names']}")
    print(f"{len(recs)} CVE, {len(done)} done, {len(todo)} to run "
          f"(LLM={args.llm_model} @ {args.llm_base_url})", flush=True)
    if not todo:
        print("Nothing to do."); return

    init = (args.llm_model, args.llm_base_url, args.llm_api_key, descriptions,
            args.repo2commits, args.max_iter, args.top_k, args.tier,
            args.read_commit_budget)
    n = nhit = 0
    t0 = time.time()
    with Pool(args.num_workers, initializer=_init, initargs=init) as pool, \
         out.open("a") as f:
        for rd in pool.imap_unordered(_process_one, todo):
            f.write(json.dumps(rd, ensure_ascii=False) + "\n"); f.flush()
            n += 1; nhit += 1 if rd.get("hit") else 0
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"[{n}/{len(todo)}] hit@1={nhit/n:.1%} "
                      f"({el/60:.1f}min, {n/max(el,1)*60:.1f} CVE/min, "
                      f"ETA {(len(todo)-n)/max(n/max(el,1),1e-6)/60:.1f}min)", flush=True)
    print(f"\nDone tier={args.tier}: {nhit}/{n} raw hit@1={nhit/max(n,1):.1%} → {out}")


if __name__ == "__main__":
    main()
