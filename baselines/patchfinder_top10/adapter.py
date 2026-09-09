"""Convert PatchFinder_top10 parquet rows into the format PatchHolmes's
Phase 2 pipeline expects.

PatchFinder parquet schema (per row, one (cve, candidate) pair):
    cve, desc, repo, commit_id, commit_message, diff, label, rank

The `diff` field is a `git show <SHA>` output:
    commit <SHA>
    Author: ...
    Date:   ...

        <commit message>

    diff --git a/file b/file
    ...

PatchHolmes's Phase2DataSource expects commit content as:
    {
        "commit_id": str,
        "commit_msg": str,
        "diff": str,        # starting with `diff --git`, no commit/Author/Date header
        "datetime": str,
        "author": str,
    }

This module bridges the two.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from patchholmes.data_models import CommitDoc, CVEQuery, RankedCandidate


# Parse `Author:` and `Date:` lines out of the git-show header.
_AUTHOR_RE = re.compile(r"^Author:\s*(.+)$", re.MULTILINE)
_DATE_RE = re.compile(r"^Date:\s+(.+)$", re.MULTILINE)


def patchfinder_row_to_commit(row: Any) -> dict[str, str]:
    """Map one parquet row to the Phase2DataSource commit dict.

    Handles two input shapes uniformly:
      - a pandas Series (`row['diff']`)
      - a plain dict
    """
    raw = str(row["diff"] or "")
    commit_msg = str(row["commit_message"] or "")

    # The diff begins with `commit <SHA>\nAuthor:...\nDate:...\n\n    <msg>\n\ndiff --git ...`
    # We need to strip everything before the first `diff --git` line so that
    # `parse_diff` (which expects raw multi-file diffs) sees a clean input.
    if raw.startswith("commit "):
        idx = raw.find("\ndiff --git ")
        if idx >= 0:
            header = raw[:idx]
            clean_diff = raw[idx + 1:]  # +1 to skip the leading '\n'
        else:
            # Edge case: a 'commit ...' header with no actual diff (rare but
            # PatchFinder occasionally has these for merge-only commits)
            header = raw
            clean_diff = ""
        m_author = _AUTHOR_RE.search(header)
        m_date = _DATE_RE.search(header)
        author = m_author.group(1).strip() if m_author else ""
        datetime = m_date.group(1).strip() if m_date else ""
    else:
        # Already a clean diff (no git-show header). Be permissive.
        clean_diff = raw
        author = ""
        datetime = ""

    return {
        "commit_id": str(row["commit_id"]),
        "commit_msg": commit_msg,
        "diff": clean_diff,
        "datetime": datetime,
        "author": author,
    }


def build_query_and_candidates(
    cve_group: pd.DataFrame,
) -> tuple[CVEQuery, list[RankedCandidate], dict[str, dict[str, str]]]:
    """Convert one CVE's parquet rows (10 candidates) into:
      - CVEQuery (the question)
      - list[RankedCandidate] (Phase 1 output, in rank order)
      - commit_dict: {commit_id: commit_content_dict}  (for the data source)

    The 56 multi-fix-with-duplicate-rows CVEs collapse naturally because
    we dedup by commit_id below — caller should NOT pre-dedup since rank
    information would be lost.
    """
    if cve_group.empty:
        raise ValueError("empty cve_group passed to build_query_and_candidates")

    # All rows share the same CVE metadata; pull from the first row.
    first = cve_group.iloc[0]
    repo_full = str(first["repo"])
    if "/" in repo_full:
        owner, repo = repo_full.split("/", 1)
    else:
        owner, repo = "", repo_full

    # Fix commit IDs (deduped by commit_id; the 56 dup-row CVEs collapse here).
    fix_ids = sorted({
        str(r["commit_id"]) for _, r in cve_group.iterrows() if int(r["label"]) == 1
    })

    query = CVEQuery(
        cve_id=str(first["cve"]),
        description=str(first["desc"] or ""),
        owner=owner,
        repo=repo,
        fix_commit_ids=fix_ids,
    )

    # Build candidates in rank order; dedup by commit_id keeping the lowest rank.
    seen: set[str] = set()
    candidates: list[RankedCandidate] = []
    commit_dict: dict[str, dict[str, str]] = {}

    for _, r in cve_group.sort_values("rank").iterrows():
        cid = str(r["commit_id"])
        if cid in seen:
            continue
        seen.add(cid)

        commit_dict[cid] = patchfinder_row_to_commit(r)

        candidates.append(
            RankedCandidate(
                commit=CommitDoc(
                    commit_id=cid,
                    commit_msg=str(r["commit_message"] or ""),
                    diff="",  # not stored on the candidate; the data source has it
                    owner=owner,
                    repo=repo,
                    datetime=commit_dict[cid].get("datetime", ""),
                ),
                # Inverse rank as a stand-in score; the agent only sees rank,
                # not score, so the exact value doesn't matter.
                score=1.0 / float(r["rank"]),
                rank=int(r["rank"]),
                source="patchfinder",
                bm25_rank=None,
                dense_rank=None,
            )
        )

    return query, candidates, commit_dict


def load_patchfinder_variant(parquet_path: str) -> pd.DataFrame:
    """Load one PatchFinder variant parquet, sanity-check the schema."""
    df = pd.read_parquet(parquet_path)
    required = {"cve", "desc", "repo", "commit_id", "commit_message", "diff", "label", "rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {missing} in {parquet_path}")
    return df
