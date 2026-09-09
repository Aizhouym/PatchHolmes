"""Shared Phase-1 top-K candidate pool for the controlled selector baselines.

Every baseline (no-agent / Favia / IRCoT) selects over the SAME PatchHolmes
Phase-1 RRF pool so the only variable is the selector (isolates the Phase-2
gain).

Reuses the agent's own data path (Phase2DataSource + diff_render) so the
candidate content each baseline sees is IDENTICAL to what the agent sees.

Paths default to repo-relative locations and can be overridden with the
PATCHHOLMES_PHASE1_JSONL / PATCHHOLMES_DESC_CSV / PATCHHOLMES_GT_CSV /
PATCHHOLMES_REPO2COMMITS environment variables.
"""
from __future__ import annotations
import csv, json, os
from pathlib import Path

from patchholmes.data_models import CVEQuery, CommitDoc, RankedCandidate
from patchholmes.phase2.data_source import Phase2DataSource

PHASE1_JSONL = Path(os.environ.get("PATCHHOLMES_PHASE1_JSONL", "./logs/phase1/main_rrf_810.jsonl"))
DESC_CSV = Path(os.environ.get("PATCHHOLMES_DESC_CSV", "./data/ground_truth_queries_clean.csv"))
GT_CSV = Path(os.environ.get("PATCHHOLMES_GT_CSV", "./data/sample_ground_truth_810.csv"))
REPO2COMMITS = os.environ.get("PATCHHOLMES_REPO2COMMITS", "./data/repo2commits_diff")


# IRCoT-style patch-tracing instruction. Same shape as FlashRAG's
# IRCOT_INSTRUCTION; same termination phrase ("So the answer is:").
PATCH_TRACING_INSTRUCTION = (
    "You are a security analyst. Given a CVE description and a numbered list "
    "of candidate commits from the affected repository, identify which "
    "commit FIXED the vulnerability. Reason step by step over the candidate "
    "diffs. Each step, produce ONE concrete observation about a specific "
    "candidate (e.g. \"[3] is a release-note commit, not the fix\" or \"[7] "
    "adds bounds check in FliDecode.c, matches the CVE\"). As soon as you "
    "can identify the fix, output the line: \"So the answer is: [N]\" where "
    "N is the number of the candidate commit you choose. If after several "
    "steps you cannot tell, commit to your best guess with \"So the answer "
    "is: [N]\".\n"
    "\n"
    "Traps to avoid (NONE of these are the fix commit):\n"
    "  - changelog / release-note commits that mention the CVE\n"
    "  - test commits that only add a regression test\n"
    "  - backports / follow-up cleanups of an earlier fix\n"
    "  - same-module fixes for a DIFFERENT vulnerability\n"
    "\n"
    "The real fix commit usually has: brief technical message; small diff "
    "modifying the vulnerable source file; adds bounds checking, input "
    "validation, output encoding, or similar concrete protection."
)


PATCH_TRACING_EXAMPLE = (
    "[1] escape query in error message\n"
    "Date: 2013-05-29T21:13:32+02:00\n"
    "---\ndiff --git a/view_create.php ...\n"
    "+    \"<i>\" . htmlspecialchars($sql_query) . ...\n"
    "\n"
    "[2] bug #1249239, XSS vulnerability on Create page\n"
    "Date: 2005-08-01T12:38:56+00:00\n"
    "---\ndiff --git a/libraries/common.lib.php ...\n"
    "+ PMA_sanitize($the_query)\n"
    "\n"
    "[3] Update CHANGELOG for 4.0.3 release\n"
    "Date: 2013-06-05T12:00:00+02:00\n"
    "---\ndiff --git a/ChangeLog ...\n"
    "+ - Fix CVE-2013-3742\n"
    "\n"
    "CVE: Cross-site scripting (XSS) vulnerability in view_create.php in "
    "phpMyAdmin 4.x before 4.0.3.\n"
    "Thought: [3] is a CHANGELOG entry, not the fix itself. [2] is from "
    "2005, eight years before the CVE, unrelated. [1] modifies view_create.php "
    "directly by adding htmlspecialchars to escape the user-controlled query. "
    "So the answer is: [1]\n"
    "\n"
)


def load_descriptions(csv_path: Path = DESC_CSV) -> dict[str, str]:
    d = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            cve = (row.get("cve") or "").strip()
            if cve:
                d[cve] = (row.get("cve_description") or "").strip()
    return d


def load_sample_ids(csv_path: Path = GT_CSV) -> set[str]:
    with open(csv_path) as f:
        return {row["cve"] for row in csv.DictReader(f)}


def build_data_source(phase1_rec: dict, descriptions: dict, top_k: int = 10) -> Phase2DataSource:
    """Mirror run_phase2_full.py:_process_one_cve construction."""
    query = CVEQuery(
        cve_id=phase1_rec["cve_id"],
        description=descriptions.get(phase1_rec["cve_id"], ""),
        owner=phase1_rec["owner"],
        repo=phase1_rec["repo"],
        fix_commit_ids=phase1_rec.get("fix_commit_ids") or [],
    )
    candidates = [
        RankedCandidate(
            commit=CommitDoc(
                commit_id=c["commit_id"], commit_msg="", diff="",
                owner=phase1_rec["owner"], repo=phase1_rec["repo"],
                datetime=c.get("datetime", ""),
            ),
            score=float(c.get("score", 0.0)),
            rank=int(c["rank"]),
            source=c.get("source", "rrf"),
            bm25_rank=c.get("bm25_rank"),
            dense_rank=c.get("dense_rank"),
        )
        for c in (phase1_rec.get("candidates") or [])
    ]
    return Phase2DataSource(
        query=query, phase1_candidates=candidates,
        repo2commits_root=REPO2COMMITS, top_k=top_k,
    )


def load_pool(top_k: int = 10):
    """Yield (phase1_rec, Phase2DataSource) for each sample_810 CVE."""
    descriptions = load_descriptions()
    sample = load_sample_ids()
    for line in open(PHASE1_JSONL):
        rec = json.loads(line)
        if rec["cve_id"] in sample:
            yield rec, build_data_source(rec, descriptions, top_k)


if __name__ == "__main__":
    # smoke: render the top-10 for the first CVE
    descriptions = load_descriptions()
    sample = load_sample_ids()
    print(f"sample_810: {len(sample)} CVE, descriptions: {len(descriptions)}")
    rec = next(json.loads(l) for l in open(PHASE1_JSONL) if json.loads(l)["cve_id"] in sample)
    print(f"\nCVE = {rec['cve_id']}  ({rec['owner']}/{rec['repo']})  fix={rec['fix_commit_ids']}")
    print(f"desc = {descriptions.get(rec['cve_id'], '')[:160]}")
    ds = build_data_source(rec, descriptions, top_k=10)
    print("\n===== list_candidates() (the IRCoT [N] list) =====")
    print(ds.list_candidates()[:1500])
    top = rec["candidates"][0]["commit_id"]
    print(f"\n===== render_commit(top-1 = {top[:12]}) (the per-candidate content Favia scores) =====")
    print(ds.render_commit(top)[:1200])
