"""PatchHolmes data models.

Core types shared across Phase 1 (hybrid retrieval) and Phase 2 (agentic reranking):

- CVEQuery        : a single retrieval query — CVE ID, description, target repo,
                    and the (possibly multi-commit) ground-truth patch set.
- CommitDoc       : a candidate commit with message, diff, and timestamp.
- RankedCandidate : a scored, ranked wrapper around CommitDoc.
- Phase1Result    : the output of Phase 1 — ranked candidates + evaluation helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CVEQuery:
    cve_id: str
    description: str
    owner: str
    repo: str
    # A CVE may have multiple ground-truth fix commits (up to 20 in the dataset).
    # Evaluation hits if ANY patch appears in Top-K.
    fix_commit_ids: list[str] = field(default_factory=list)

    @property
    def repo_key(self) -> str:
        return f"{self.owner}@@{self.repo}"

    @property
    def query_key(self) -> str:
        return f"{self.repo_key}::{self.cve_id}"


@dataclass
class CommitDoc:
    commit_id: str
    commit_msg: str
    diff: str
    owner: str = ""
    repo: str = ""
    datetime: str = ""  # ISO-8601 commit timestamp
    author: str = ""


@dataclass
class RankedCandidate:
    commit: CommitDoc
    score: float
    rank: int
    source: str  # "bm25", "dense", "rrf"
    bm25_rank: int | None = None
    dense_rank: int | None = None


@dataclass
class Phase1Result:
    query: CVEQuery
    candidates: list[RankedCandidate] = field(default_factory=list)

    def recall_at_k(self, k: int) -> bool:
        """True if any ground-truth patch appears in the top-k candidates."""
        if not self.query.fix_commit_ids:
            return False
        fix_set = set(self.query.fix_commit_ids)
        return any(c.commit.commit_id in fix_set for c in self.candidates[:k])

    def best_rank(self) -> int | None:
        """Rank of the highest-ranked ground-truth patch (1-indexed), or None."""
        if not self.query.fix_commit_ids:
            return None
        fix_set = set(self.query.fix_commit_ids)
        for c in self.candidates:
            if c.commit.commit_id in fix_set:
                return c.rank
        return None
