"""PatchHolmes — Phase 1 pipeline (Mode A: parallel full-corpus retrieval).

Both BM25 and Dense independently rank ALL commits in the repository.
RRF then fuses the two full rankings.

    all commits
        ├── BM25 pre-computed scores  → full BM25 ranking
        └── Dense pre-computed pkl   → FAISS full search → full Dense ranking
                                                ↓
                                       RRF fusion → Top-1000

Why this is correct: if the true patch commit is ranked low by BM25 (keyword
mismatch) but high by Dense (semantic match), it still survives into the final
Top-1000. A cascaded pipeline that feeds only BM25 Top-N into Dense would
permanently discard it.

Requires offline embedding pre-computation via scripts/patchholmes.sh encode,
which saves per-repo pkl files under feature_root.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from patchholmes.data_models import CVEQuery, Phase1Result
from patchholmes.phase1.bm25_time import BM25Retriever
from patchholmes.phase1.dense_faiss import DenseRetriever
from patchholmes.phase1.rrf_fusion import rrf_fuse


@dataclass
class Phase1Pipeline:
    bm25_retriever: BM25Retriever
    dense_retriever: DenseRetriever

    # Root directory where direct per-repo encoding saved the pkl files.
    # Layout: <embedding_root>/<owner>@@<repo>/qwen_embedding/corpus.pkl
    #         <embedding_root>/<owner>@@<repo>/qwen_embedding/queries.pkl
    embedding_root: Path

    # Dense FAISS: how many nearest neighbours to retrieve from the full corpus
    dense_top_k: int = 5000

    # Final RRF output size passed to Phase 2
    rrf_top_k: int = 1000

    rrf_k: int = 60  # RRF smoothing constant

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def run(self, query: CVEQuery) -> Phase1Result:
        return self._run_parallel(query)

    def run_batch(self, queries: list[CVEQuery]) -> list[Phase1Result]:
        return [self.run(q) for q in queries]

    # ------------------------------------------------------------------ #
    # Parallel full-corpus retrieval                                       #
    # ------------------------------------------------------------------ #

    def _run_parallel(self, query: CVEQuery) -> Phase1Result:
        """BM25 full ranking + Dense full corpus FAISS → RRF → Top-K."""
        corpus_pkl  = self._pkl_path(query, "corpus.pkl")
        queries_pkl = self._pkl_path(query, "queries.pkl")

        if not corpus_pkl.exists() or not queries_pkl.exists():
            raise FileNotFoundError(
                f"Pre-computed pkl not found for {query.repo_key}.\n"
                f"  Expected: {corpus_pkl}\n"
                f"            {queries_pkl}\n"
                f"  Run scripts/patchholmes.sh encode corpus and scripts/patchholmes.sh encode queries first."
            )

        # Step 1: Dense — FAISS search on full pre-computed corpus.
        # Run this before loading the full BM25 ranking so the large FAISS
        # matrix/index can be released before BM25 objects are materialised.
        commit_lookup = {}
        dense_results = self.dense_retriever.retrieve_from_pkl(
            query_pkl=queries_pkl,
            corpus_pkl=corpus_pkl,
            commit_lookup=commit_lookup,
            top_k=self.dense_top_k,
            cve_id=query.cve_id,
        )

        # Step 2: BM25 — load ALL scored commits (no truncation)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=None)

        # Step 3: RRF fusion → Top-K
        fused = rrf_fuse(bm25_results, dense_results, top_k=self.rrf_top_k, k=self.rrf_k)
        return Phase1Result(query=query, candidates=fused)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _pkl_path(self, query: CVEQuery, filename: str) -> Path:
        return self.embedding_root / query.repo_key / "qwen_embedding" / filename
