"""Build a Favia-input parquet from PatchHolmes's Phase-1 RRF top-10 pool.

Runs the existing Favia code on the SAME Phase-1 candidates that PatchHolmes
uses (instead of PatchFinder_top10), so the only variable is the selector.
Output schema matches the PatchFinder_top10 parquet Favia expects:
    cve, desc, repo, commit_id, commit_message, diff, label, rank

Produces roughly 809 CVE x 10 = ~8090 rows.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from patchholmes.phase2.data_source import _load_commits_for_repo

TOP_K = 10


def git_show(sha, meta):
    """Reconstruct a git-show style blob (matches PatchFinder parquet's diff)."""
    author = meta.get("author", "") or ""
    date = meta.get("datetime", "") or ""
    msg = meta.get("commit_msg", "") or ""
    diff = meta.get("diff", "") or ""
    return f"commit {sha}\nAuthor: {author}\nDate:   {date}\n\n    {msg}\n\n{diff}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase1", default="./logs/phase1/main_rrf_810.jsonl",
                    help="Phase-1 RRF results JSONL.")
    ap.add_argument("--desc-csv", default="./data/ground_truth_queries_clean.csv",
                    help="CVE description CSV (cols: cve, cve_description).")
    ap.add_argument("--ground-truth", default="./data/sample_ground_truth_810.csv",
                    help="Ground-truth CSV defining the sample_810 CVE set.")
    ap.add_argument("--repo2commits", default="./data/repo2commits_diff",
                    help="Directory of per-repo commit content JSON.")
    ap.add_argument("--out", default="./data/favia_phase1_top10.parquet",
                    help="Output parquet path.")
    args = ap.parse_args()

    phase1 = Path(args.phase1)
    desc_csv = Path(args.desc_csv)
    gt_csv = Path(args.ground_truth)
    repo2commits = Path(args.repo2commits)
    out = Path(args.out)

    descs = {r["cve"].strip(): (r.get("cve_description") or "").strip()
             for r in csv.DictReader(open(desc_csv)) if r.get("cve")}
    sample = {r["cve"] for r in csv.DictReader(open(gt_csv))}

    recs = [json.loads(l) for l in open(phase1) if json.loads(l)["cve_id"] in sample]
    print(f"sample_810: {len(recs)} CVE", flush=True)

    # group CVEs by repo so each repo's commit JSON is scanned ONCE
    by_repo: dict[tuple, list] = {}
    for r in recs:
        by_repo.setdefault((r["owner"], r["repo"]), []).append(r)
    print(f"{len(by_repo)} unique repos", flush=True)

    rows = []
    for ri, ((owner, repo), rlist) in enumerate(by_repo.items()):
        # all commit_ids needed for this repo across all its CVEs
        need = set()
        for r in rlist:
            for c in (r.get("candidates") or [])[:TOP_K]:
                need.add(c["commit_id"])
        content = _load_commits_for_repo(repo2commits, owner, repo, need)  # one scan per repo
        for r in rlist:
            cve = r["cve_id"]
            fix = set(r.get("fix_commit_ids") or [])
            for c in (r.get("candidates") or [])[:TOP_K]:
                sha = c["commit_id"]
                meta = content.get(sha, {})
                rows.append({
                    "cve": cve,
                    "desc": descs.get(cve, ""),
                    "repo": f"{owner}/{repo}",
                    "commit_id": sha,
                    "commit_message": meta.get("commit_msg", "") or "",
                    "diff": git_show(sha, meta),
                    "label": 1 if sha in fix else 0,
                    "rank": int(c["rank"]),
                })
        if (ri + 1) % 25 == 0:
            print(f"  {ri+1}/{len(by_repo)} repos processed, {len(rows)} rows", flush=True)

    df = pd.DataFrame(rows, columns=["cve", "desc", "repo", "commit_id",
                                     "commit_message", "diff", "label", "rank"])
    df.to_parquet(out, index=False)
    # sanity
    n_cve = df["cve"].nunique()
    n_pos = int(df["label"].sum())
    empty_diff = int((df["diff"].str.len() < 60).sum())
    print(f"\nwrote {out}")
    print(f"  rows={len(df)}  CVE={n_cve}  positives(label=1)={n_pos}  "
          f"(expect ~{n_cve} if each CVE has exactly 1 fix in the pool)")
    print(f"  empty/very-short diff rows: {empty_diff}")
    # how many CVE have their fix in the top-10 pool at all (Phase-1 R@10 ceiling)
    have_fix = df.groupby("cve")["label"].max().sum()
    print(f"  CVE with fix inside top-10 pool: {int(have_fix)}/{n_cve} = "
          f"{have_fix/n_cve*100:.2f}%  (this is Favia's R@10 ceiling on this pool)")


if __name__ == "__main__":
    main()
