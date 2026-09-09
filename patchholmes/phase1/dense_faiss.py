"""PatchHolmes — Phase 1 dense retrieval (pre-computed pkl mode).

Loads pre-computed Qwen embeddings from pkl files produced by direct per-repo
encoding, then performs exact inner-product (cosine) search to find the top-K
most similar commits to the CVE query.

Backend priority (auto-selected, or override via PATCHHOLMES_DENSE_BACKEND):
  1. "torch"  — GPU matmul via PyTorch (fastest when CUDA is available)
  2. "faiss"  — CPU IndexFlatIP via faiss-cpu
  3. "chunked"— chunked numpy matmul (fallback, no extra deps)

pkl format (produced by CorpusEncoder):
    (embeddings: np.ndarray shape(N, dim) float32 L2-normalised,
     ids: list[str])

Inner product on L2-normalised vectors == cosine similarity.
"""
from __future__ import annotations

import pickle
import os
from pathlib import Path
from typing import Any

import numpy as np

from patchholmes.data_models import CommitDoc, CVEQuery, RankedCandidate


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _default_backend() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "torch"
    except ImportError:
        pass
    try:
        import faiss  # noqa: F401
        return "faiss"
    except ImportError:
        pass
    return "chunked"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_float32(arr: Any) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"Embedding matrix must be 2D, got shape={out.shape}")
    return out


def _load_pkl(path: Path) -> tuple[np.ndarray, list[Any]]:
    with path.open("rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, tuple) and len(obj) == 2:
        left, right = obj
        if isinstance(left, np.ndarray):
            return _as_float32(left), list(right)
        if isinstance(right, np.ndarray):
            return _as_float32(right), list(left)

    raise ValueError(
        f"Unsupported pkl format in {path}. "
        f"Expected tuple (np.ndarray, list[str])."
    )


def _build_faiss_index(corpus_vecs: np.ndarray):
    import faiss
    index = faiss.IndexFlatIP(corpus_vecs.shape[1])
    index.add(corpus_vecs)
    return index


def _search_exact_chunked(
    query_vec: np.ndarray,
    corpus_vecs: np.ndarray,
    top_k: int,
    chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    best_scores = np.empty((0,), dtype=np.float32)
    best_indices = np.empty((0,), dtype=np.int64)

    q = np.asarray(query_vec, dtype=np.float32)
    for start in range(0, corpus_vecs.shape[0], chunk_size):
        stop = min(start + chunk_size, corpus_vecs.shape[0])
        scores = np.asarray(corpus_vecs[start:stop] @ q, dtype=np.float32)
        idx = np.arange(start, stop, dtype=np.int64)

        if best_scores.size:
            scores = np.concatenate([best_scores, scores])
            idx = np.concatenate([best_indices, idx])

        keep = min(top_k, scores.size)
        if scores.size > keep:
            part = np.argpartition(scores, -keep)[-keep:]
            best_scores = scores[part]
            best_indices = idx[part]
        else:
            best_scores = scores
            best_indices = idx

    order = np.argsort(-best_scores)
    return best_scores[order], best_indices[order]


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------

class DenseRetriever:

    # Max number of repo corpora to keep on GPU at once. Each large repo
    # (e.g. linux, FFmpeg, openssl) holds ~0.5–5 GB on GPU; with 47 GB total
    # we want a low cap. 4 covers the typical "consecutive CVEs share a repo"
    # access pattern with plenty of headroom.
    _GPU_CACHE_MAX_ENTRIES: int = 4

    def __init__(self) -> None:
        self._corpus_cache: dict[Path, tuple[np.ndarray, list]] = {}
        # Use insertion-ordered dict as LRU; key reuse via move-to-end on hit.
        self._gpu_cache: dict[Path, Any] = {}  # corpus_pkl -> torch.Tensor on GPU

    def _load_corpus(self, corpus_pkl: Path) -> tuple[np.ndarray, list]:
        if corpus_pkl not in self._corpus_cache:
            print(f"  [cache miss] loading corpus {corpus_pkl.parent.parent.name} ...", flush=True)
            d_vecs, d_ids = _load_pkl(corpus_pkl)
            self._corpus_cache[corpus_pkl] = (d_vecs, d_ids)
        return self._corpus_cache[corpus_pkl]

    def retrieve_from_pkl(
        self,
        query_pkl: Path,
        corpus_pkl: Path,
        commit_lookup: dict[str, CommitDoc],
        top_k: int = 5000,
        cve_id: str = "",
    ) -> list[RankedCandidate]:
        q_vecs, q_ids = _load_pkl(query_pkl)
        d_vecs, d_ids = self._load_corpus(corpus_pkl)

        if q_vecs.shape[1] != d_vecs.shape[1]:
            raise ValueError(
                f"Dimension mismatch: query dim={q_vecs.shape[1]}, "
                f"corpus dim={d_vecs.shape[1]}"
            )

        if cve_id:
            matches = [i for i, qid in enumerate(q_ids) if str(qid) == cve_id]
            q_idx = matches[0] if matches else 0
        else:
            q_idx = 0

        k = min(top_k, len(d_ids))
        backend = os.environ.get("PATCHHOLMES_DENSE_BACKEND", _default_backend()).lower()

        if backend == "torch":
            import torch
            if corpus_pkl in self._gpu_cache:
                # LRU bump: re-insert to make this entry most-recent
                d_gpu = self._gpu_cache.pop(corpus_pkl)
                self._gpu_cache[corpus_pkl] = d_gpu
            else:
                # Evict oldest entries until we have room for one more
                while len(self._gpu_cache) >= self._GPU_CACHE_MAX_ENTRIES:
                    oldest_key, oldest_tensor = next(iter(self._gpu_cache.items()))
                    del self._gpu_cache[oldest_key]
                    del oldest_tensor  # let Python drop the reference
                    torch.cuda.empty_cache()
                self._gpu_cache[corpus_pkl] = torch.from_numpy(d_vecs).to("cuda")
                d_gpu = self._gpu_cache[corpus_pkl]
            q = torch.from_numpy(q_vecs[q_idx]).to("cuda")
            scores_t = d_gpu @ q
            k_capped = min(k, scores_t.shape[0])
            topk_scores, topk_indices = torch.topk(scores_t, k_capped)
            scores, indices = topk_scores.cpu().numpy(), topk_indices.cpu().numpy()
        elif backend == "faiss":
            index = _build_faiss_index(d_vecs)
            scores, indices = index.search(q_vecs[q_idx : q_idx + 1], k)
            scores = scores[0]
            indices = indices[0]
        else:
            scores, indices = _search_exact_chunked(q_vecs[q_idx], d_vecs, k)

        results: list[RankedCandidate] = []
        for rank, (score, didx) in enumerate(zip(scores, indices), start=1):
            raw_docid = str(d_ids[int(didx)])
            commit_id = raw_docid.rsplit("@@", 1)[-1] if "@@" in raw_docid else raw_docid
            doc = (
                commit_lookup.get(raw_docid)
                or commit_lookup.get(commit_id)
                or CommitDoc(commit_id=commit_id, commit_msg="", diff="")
            )
            results.append(
                RankedCandidate(
                    commit=doc,
                    score=float(score),
                    rank=rank,
                    source="dense",
                    dense_rank=rank,
                )
            )
        return results
