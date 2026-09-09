"""PatchHolmes — Phase 1 BM25 sparse retrieval.

BM25 is implemented externally via ElasticSearch, which indexes each commit's
message and the changed lines (+/-) of its diff.  It excels at exact keyword
matching — function names, error codes, library names — that frequently appear
in both CVE descriptions and commit messages.

Pre-computed BM25+time scores are stored at:
    ./embeddings/<owner>@@<repo>/bm25_time/result/<CVE-ID>.json

The `new_score` field combines the raw BM25 score with a time-proximity
penalty relative to the CVE's reserve/publish date.

Fields in each BM25 json entry:
    datetime, score, new_score, publish_diff, reserve_time_diff, ...

Memory design
-------------
The BM25 result json already contains ALL commits from the repo (score for
every commit), including their datetime.  Therefore Phase 1 retrieval requires
NO access to repo2commits_diff — all required data (commit_id, score, datetime)
is already in the BM25 json.

We intentionally do NOT load commit_msg or diff from repo2commits_diff during
Phase 1.  For large repos (linux: 1.2 M commits, 5.8 GB of diffs) loading the
full content would exhaust memory.  Instead:

- Phase 1 returns CommitDoc with commit_msg="" and diff="" (lightweight).
- Phase 2 (or Mode B dense encoding) calls load_commit_content() to fetch
  commit_msg + diff ON DEMAND for just the small set of candidates it needs.
- _load_repo_index() builds a lightweight metadata index (no diff) only when
  needed for Dense Mode A (to resolve commit IDs returned by FAISS that may
  not have appeared in the BM25 results).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from patchholmes.data_models import CommitDoc, CVEQuery, RankedCandidate


class BM25Retriever:
    def __init__(
        self,
        feature_root: str | Path,
        repo2commits_root: str | Path,
        combined_csv: str | Path | None = None,
    ) -> None:
        self.feature_root = Path(feature_root)
        self.repo2commits_root = Path(repo2commits_root)

        # Lightweight metadata cache: repo_key → {commit_id: CommitDoc}
        # CommitDoc here has commit_msg and datetime filled, but diff="" (empty).
        # This is populated lazily by _load_repo_index() and used only by
        # Dense Mode A to resolve FAISS results.
        self._meta_cache: dict[str, dict[str, CommitDoc]] = {}

        # Optional ES-based online fallback for CVEs without precomputed JSON.
        self._es_retriever = None
        if combined_csv is not None:
            try:
                from patchholmes.phase1.es_bm25_online import ESBm25Online, load_cve_dates
                cve_dates = load_cve_dates(combined_csv)
                self._es_retriever = ESBm25Online(
                    repo2commits_root=repo2commits_root,
                    cve_dates=cve_dates,
                )
                print(f"  [BM25Retriever] ES online fallback enabled ({len(cve_dates)} CVE dates loaded)", flush=True)
            except Exception as e:
                print(f"  [BM25Retriever] ES fallback unavailable: {e}", flush=True)

    # ------------------------------------------------------------------
    # Primary retrieval — Phase 1
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: CVEQuery,
        top_k: int | None = None,
    ) -> list[RankedCandidate]:
        """Return BM25-ranked candidates for a CVE query.

        All data comes from the pre-computed BM25 json — no repo2commits_diff
        access.  CommitDoc.commit_msg and CommitDoc.diff are left empty; call
        load_commit_content() later if you need the actual text.

        Parameters
        ----------
        query  : the CVE query to retrieve for
        top_k  : maximum number of candidates to return.
                 None (default) means return ALL scored commits — required for
                 correct parallel RRF fusion with the Dense ranking.
        """
        bm25_file = (
            self.feature_root
            / query.repo_key
            / "bm25_time"
            / "result"
            / f"{query.cve_id}.json"
        )
        if not bm25_file.exists():
            if self._es_retriever is not None:
                return self._es_retriever.retrieve(query, top_k=top_k)
            return []

        raw: dict = json.loads(bm25_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not raw:
            return []

        # Sort all commits by new_score descending — no early truncation.
        scored = sorted(
            (
                (
                    cid,
                    float(meta.get("new_score", float("-inf"))) if isinstance(meta, dict) else float("-inf"),
                    str(meta.get("datetime", "")) if isinstance(meta, dict) else "",
                )
                for cid, meta in raw.items()
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        if top_k is not None:
            scored = scored[:top_k]

        results: list[RankedCandidate] = []
        for rank, (cid, score, bm25_dt) in enumerate(scored, start=1):
            # Lightweight CommitDoc — no commit_msg, no diff.
            # Call load_commit_content() downstream if content is needed.
            doc = CommitDoc(
                commit_id=cid,
                commit_msg="",
                diff="",
                owner=query.owner,
                repo=query.repo,
                datetime=bm25_dt,
            )
            results.append(
                RankedCandidate(
                    commit=doc,
                    score=score,
                    rank=rank,
                    source="bm25",
                    bm25_rank=rank,
                )
            )
        return results

    # ------------------------------------------------------------------
    # On-demand content loading — Phase 2 / Mode B
    # ------------------------------------------------------------------

    def load_commit_content(
        self,
        owner: str,
        repo: str,
        commit_ids: list[str],
        fields: tuple[str, ...] = ("commit_msg", "diff"),
    ) -> dict[str, dict[str, str]]:
        """Load commit_msg and/or diff for a specific set of commits.

        Scans the split_<repo> directory and returns only the requested commits.
        Much more memory-efficient than loading the whole repo: a typical Phase 2
        request for top-100 candidates reads ~100 commits instead of 100 000+.

        Parameters
        ----------
        owner, repo  : repository identity
        commit_ids   : list of commit SHAs to look up
        fields       : which fields to return — ("commit_msg",), ("diff",), or both

        Returns
        -------
        dict mapping commit_id → {field: value, ...}
        Only commits actually found in repo2commits_diff are included.
        """
        if not commit_ids:
            return {}

        target = set(commit_ids)
        found: dict[str, dict[str, str]] = {}

        split_dir = self.repo2commits_root / f"split_{owner}@@{repo}"
        if not split_dir.exists():
            return {}

        for fp in glob.glob(str(split_dir / "*.json")):
            if not target:
                break  # all commits found — stop early
            try:
                arr = json.loads(Path(fp).read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(arr, list):
                continue
            for item in arr:
                cid = str(item.get("commit_id", "")).strip()
                if cid not in target:
                    continue
                found[cid] = {f: str(item.get(f, "")) for f in fields}
                target.discard(cid)

        return found

    def enrich_candidates(
        self,
        candidates: list[RankedCandidate],
        fields: tuple[str, ...] = ("commit_msg", "diff"),
    ) -> None:
        """Fill commit_msg / diff in-place for a list of RankedCandidates.

        Groups candidates by repo to minimise file scanning, then fills the
        CommitDoc fields for each candidate that still has empty content.

        Typical call sites:
          - Phase 2: enrich the top-K candidates before passing to LLM/agent
          - Mode B dense encoding: enrich BM25 top-K before embed
        """
        # Group by repo to avoid redundant scans
        by_repo: dict[tuple[str, str], list[RankedCandidate]] = {}
        for c in candidates:
            key = (c.commit.owner, c.commit.repo)
            by_repo.setdefault(key, []).append(c)

        for (owner, repo), group in by_repo.items():
            # Only request commits whose content is actually missing
            need = [c.commit.commit_id for c in group
                    if any(not getattr(c.commit, f, "") for f in fields)]
            if not need:
                continue
            content = self.load_commit_content(owner, repo, need, fields=fields)
            for c in group:
                cid = c.commit.commit_id
                if cid in content:
                    for f in fields:
                        if f in content[cid]:
                            setattr(c.commit, f, content[cid][f])

    # ------------------------------------------------------------------
    # Lightweight metadata index — used by Dense Mode A only
    # ------------------------------------------------------------------

    def _load_repo_index(self, owner: str, repo: str) -> dict[str, CommitDoc]:
        """Load lightweight metadata (commit_id, commit_msg, datetime) for a repo.

        Does NOT load diff — keeping memory usage proportional to commit count
        rather than total diff size.

        Used by Dense Mode A (pipeline._run_parallel) to resolve commit IDs
        returned by FAISS that may not have appeared in the BM25 results.
        """
        repo_key = f"{owner}@@{repo}"
        if repo_key in self._meta_cache:
            return self._meta_cache[repo_key]

        split_dir = self.repo2commits_root / f"split_{repo_key}"
        index: dict[str, CommitDoc] = {}
        if not split_dir.exists():
            self._meta_cache[repo_key] = index
            return index

        for fp in glob.glob(str(split_dir / "*.json")):
            try:
                arr = json.loads(Path(fp).read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(arr, list):
                continue
            for item in arr:
                if not isinstance(item, dict):
                    continue
                cid = str(item.get("commit_id", "")).strip()
                if not cid:
                    continue
                index[cid] = CommitDoc(
                    commit_id=cid,
                    commit_msg=str(item.get("commit_msg", "")),
                    diff="",          # intentionally empty — load on demand
                    owner=owner,
                    repo=repo,
                    datetime=str(item.get("datetime", "")),
                    author=str(item.get("author") or item.get("author_email") or ""),
                )

        self._meta_cache[repo_key] = index
        return index
