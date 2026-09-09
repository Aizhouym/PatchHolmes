"""PatchHolmes — Phase 1 Reciprocal Rank Fusion.

Merges the BM25 sparse ranking and the dense FAISS ranking into a single
unified Top-K list:

    score(c) = 1 / (k + rank_BM25) + 1 / (k + rank_Dense)

The default smoothing constant is k = 60 (standard in the IR literature).

Properties:
- Parameter-free beyond k — no per-feature weight tuning required.
- Depends only on relative rank, not absolute score magnitudes.
- Documents appearing in only one list receive a partial (non-zero) score.
- Consistently outperforms either BM25 or dense retrieval alone on recall@K.
"""
from __future__ import annotations

from patchholmes.data_models import RankedCandidate


def rrf_fuse(
    bm25_results: list[RankedCandidate],
    dense_results: list[RankedCandidate],
    top_k: int = 2000,
    k: int = 60,
) -> list[RankedCandidate]:
    """Fuse two ranked lists with RRF and return the top_k merged results."""
    scores: dict[str, float] = {}
    commit_map: dict[str, RankedCandidate] = {}

    for cand in bm25_results:
        cid = cand.commit.commit_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + cand.rank)
        if cid not in commit_map:
            commit_map[cid] = cand

    for cand in dense_results:
        cid = cand.commit.commit_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + cand.rank)
        # Prefer the richer CommitDoc (one that has msg/diff loaded)
        if cid not in commit_map or (
            not commit_map[cid].commit.commit_msg and cand.commit.commit_msg
        ):
            commit_map[cid] = cand

    bm25_rank_map = {c.commit.commit_id: c.rank for c in bm25_results}
    dense_rank_map = {c.commit.commit_id: c.rank for c in dense_results}

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]

    fused: list[RankedCandidate] = []
    for rank, cid in enumerate(sorted_ids, start=1):
        base = commit_map[cid]
        fused.append(
            RankedCandidate(
                commit=base.commit,
                score=scores[cid],
                rank=rank,
                source="rrf",
                bm25_rank=bm25_rank_map.get(cid),
                dense_rank=dense_rank_map.get(cid),
            )
        )
    return fused
