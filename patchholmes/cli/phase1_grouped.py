#!/usr/bin/env python3
"""Repo-grouped Phase 1 evaluation CLI."""
from __future__ import annotations

import argparse
import csv
import gc
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from patchholmes.data_models import CVEQuery, CommitDoc, RankedCandidate
from patchholmes.phase1.rrf_fusion import rrf_fuse


EVAL_KS = [10, 100, 500, 1000]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grouped Phase 1 evaluation.")
    p.add_argument("--dataset", default="data/ground_truth_queries_qwen_available.csv")
    p.add_argument("--feature-root", default="./embeddings")
    p.add_argument("--repo2commits-root", default="./data/repo2commits_diff")
    p.add_argument("--embedding-root", required=True)
    p.add_argument("--dense-top-k", type=int, default=5000)
    p.add_argument("--rrf-top-k", type=int, default=1000)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--max-cves", type=int, default=0)
    return p.parse_args()


def load_queries(csv_path: Path, max_cves: int) -> list[CVEQuery]:
    out = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if max_cves > 0 and len(out) >= max_cves:
                break
            fixes = [c.strip() for c in (row.get("fix_commit_ids") or "").split("|") if c.strip()]
            out.append(CVEQuery(
                cve_id=(row.get("cve") or "").strip(),
                description=(row.get("cve_description") or "").strip(),
                owner=(row.get("owner") or "").strip(),
                repo=(row.get("repo") or "").strip(),
                fix_commit_ids=fixes,
            ))
    return [q for q in out if q.cve_id and q.owner and q.repo]


def load_pkl(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def recall_at(candidates: list[RankedCandidate], fixes: list[str], k: int) -> bool:
    targets = set(fixes)
    return any(c.commit.commit_id in targets for c in candidates[:k])


def best_rank(candidates: list[RankedCandidate], fixes: list[str]) -> int | None:
    targets = set(fixes)
    for i, c in enumerate(candidates, start=1):
        if c.commit.commit_id in targets:
            return i
    return None


def main() -> None:
    args = parse_args()
    from patchholmes.phase1.bm25_time import BM25Retriever
    from patchholmes.phase1.dense_faiss import _search_exact_chunked

    queries = load_queries(Path(args.dataset), args.max_cves)
    by_repo: dict[str, list[CVEQuery]] = defaultdict(list)
    for q in queries:
        by_repo[q.repo_key].append(q)

    print(f"Loaded {len(queries)} CVE queries across {len(by_repo)} repos")
    print("Running grouped Phase 1 ...")

    bm25 = BM25Retriever(args.feature_root, args.repo2commits_root)
    embedding_root = Path(args.embedding_root)
    hits = {k: 0 for k in EVAL_KS}
    valid = 0
    first_details: list[str] = []
    processed = 0

    for repo_idx, (repo_key, repo_queries) in enumerate(by_repo.items(), start=1):
        emb_dir = embedding_root / repo_key / "qwen_embedding"
        print(f"  repo {repo_idx}/{len(by_repo)} {repo_key} ({len(repo_queries)} CVEs)")

        d_vecs, d_ids = load_pkl(emb_dir / "corpus.pkl")
        q_vecs, q_ids = load_pkl(emb_dir / "queries.pkl")
        d_vecs = np.asarray(d_vecs, dtype=np.float32)
        q_vecs = np.asarray(q_vecs, dtype=np.float32)
        q_pos = {str(qid): i for i, qid in enumerate(q_ids)}

        for q in repo_queries:
            q_idx = q_pos.get(q.cve_id, 0)
            k = min(args.dense_top_k, len(d_ids))
            scores, indices = _search_exact_chunked(q_vecs[q_idx], d_vecs, k)
            dense_results = []
            for rank, (score, didx) in enumerate(zip(scores, indices), start=1):
                raw_docid = str(d_ids[int(didx)])
                commit_id = raw_docid.rsplit("@@", 1)[-1] if "@@" in raw_docid else raw_docid
                dense_results.append(RankedCandidate(
                    commit=CommitDoc(commit_id=commit_id, commit_msg="", diff="", owner=q.owner, repo=q.repo),
                    score=float(score),
                    rank=rank,
                    source="dense",
                    dense_rank=rank,
                ))

            bm25_results = bm25.retrieve(q, top_k=None)
            fused = rrf_fuse(bm25_results, dense_results, top_k=args.rrf_top_k, k=args.rrf_k)

            if q.fix_commit_ids:
                valid += 1
                for kk in EVAL_KS:
                    if recall_at(fused, q.fix_commit_ids, kk):
                        hits[kk] += 1

            if len(first_details) < 5:
                first_details.append(
                    f"  {q.cve_id} patches={[p[:8] for p in q.fix_commit_ids]} "
                    f"best_rank={best_rank(fused, q.fix_commit_ids)} "
                    f"recall@{args.rrf_top_k}={recall_at(fused, q.fix_commit_ids, args.rrf_top_k)} "
                    f"top3={[c.commit.commit_id[:8] for c in fused[:3]]}"
                )

            processed += 1
            if processed % 100 == 0 or processed == len(queries):
                print(f"    processed {processed}/{len(queries)} queries")
            del dense_results, bm25_results, fused, scores, indices
            gc.collect()

        del d_vecs, d_ids, q_vecs, q_ids
        gc.collect()

    print("\n[ Phase 1 - BM25 + Dense (RRF), grouped ]")
    print(f"{'K':>6}  {'Recall@K':>10}  hits/{valid}")
    print("-" * 36)
    for k in EVAL_KS:
        print(f"{k:>6}  {hits[k] / valid:>10.4f}  {hits[k]}/{valid}")

    print("\n--- Per-query detail (first 5) ---")
    for line in first_details:
        print(line)


if __name__ == "__main__":
    main()
