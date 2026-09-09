"""Phase2DataSource subclass that takes pre-loaded commit content (from a
PatchFinder parquet) instead of reading `repo2commits_diff/*.json` off disk.

Why subclass instead of patch:
    Phase2DataSource's parent __init__ eagerly calls _load_commits_for_repo(),
    which scans a disk layout we don't have for PatchFinder data. We skip that
    call by overriding __init__ and setting the same 10 instance attributes
    the parent would have set. The rest of the parent class (list_candidates,
    render_commit, render_file_diff, submit_answer, etc.) works unchanged
    because they only read from `self._commit_content`.
"""
from __future__ import annotations

from pathlib import Path

from patchholmes.data_models import CVEQuery, RankedCandidate
from patchholmes.phase2.data_source import Phase2DataSource, SubmittedAnswer


class PatchFinderDataSource(Phase2DataSource):
    """Phase2DataSource that uses pre-loaded commit content instead of disk.

    Parameters
    ----------
    query
        The CVE query (already built from the parquet rows).
    candidates
        Ranked candidates (already built from parquet, in rank order, deduped
        by commit_id).
    commit_dict
        {commit_id → {commit_msg, diff, datetime, author}} dict — usually
        produced by `adapter.build_query_and_candidates`.
    top_k
        Number of candidates the agent sees. Defaults to len(candidates),
        which for PatchFinder_top10 is at most 10 (often fewer after dedup).
    """

    def __init__(
        self,
        query: CVEQuery,
        candidates: list[RankedCandidate],
        commit_dict: dict[str, dict[str, str]],
        top_k: int | None = None,
    ) -> None:
        # Mirror the parent's attribute list, but skip the disk load.
        if top_k is None:
            top_k = len(candidates)

        self.query = query
        self.top_k = top_k
        # repo2commits_root is unused by the methods we expose, but the parent
        # stores a Path here — keep the same type to avoid surprising any code
        # that introspects this attribute.
        self.repo2commits_root = Path("/dev/null")

        self.candidates: list[RankedCandidate] = candidates[:top_k]
        self._cand_by_id: dict[str, RankedCandidate] = {
            c.commit.commit_id: c for c in self.candidates
        }

        # Only keep commit_content entries that correspond to a visible candidate.
        # (Extra entries would be harmless but make the source dict bigger.)
        visible = {c.commit.commit_id for c in self.candidates}
        self._commit_content: dict[str, dict[str, str]] = {
            cid: content for cid, content in commit_dict.items() if cid in visible
        }

        self._parsed_cache: dict = {}
        self._inspected: list[str] = []
        self._inspected_set: set[str] = set()
        self._answer: SubmittedAnswer | None = None
