#!/usr/bin/env python3
"""Phase 1 ablation: produce a top-100 jsonl using ONLY ONE retriever.

Three modes (`--retriever`):
    bm25       : raw BM25 score  (BM25 alone, no time-decay penalty)
    bm25_time  : BM25 + time-decay  (the BM25 leg of our main system)
    dense      : Qwen3-Embedding dense retrieval alone (FAISS IndexFlatIP)

All three modes read pre-computed intermediate data — no re-encoding required:

    bm25*  → ./embeddings/<owner>@@<repo>/bm25_time/result/<CVE-ID>.json
             dict[commit_id -> {score, new_score, datetime, ...}]
                 'score'     = raw BM25
                 'new_score' = BM25 + time penalty

    dense  → ./embeddings/<owner>@@<repo>/qwen_embedding/{corpus,queries}.pkl
             each pkl is a tuple (embeddings: ndarray, ids: list)
             corpus ids are commit_ids; queries ids are CVE-IDs.

Output schema matches `logs/phase1/phase1_clean_full_top100.jsonl`, so the
result can be passed directly to `scripts/run_phase2_full.py` via
`--phase1-jsonl`.

Usage
-----
    # Build top-100 from RAW BM25 only, restricted to the sample_810 CVE list
    python ablation/phase1_retrievers/make_phase1_jsonl.py \
        --retriever     bm25 \
        --queries-csv   ./data/sample_ground_truth_810.csv \
        --feature-root  ./embeddings \
        --top-k         100 \
        --output        ./logs/phase1/ablation_bm25.jsonl

    # Same for BM25+time:
    python ablation/phase1_retrievers/make_phase1_jsonl.py --retriever bm25_time ...

    # Same for Qwen dense:
    python ablation/phase1_retrievers/make_phase1_jsonl.py --retriever dense ...
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


RETRIEVERS = ("bm25", "bm25_time", "dense")


# ---------------------------------------------------------------------------
# BM25 (and BM25+time) — read pre-computed json, sort by chosen score field
# ---------------------------------------------------------------------------


def rank_bm25(
    cve_id: str,
    repo_key: str,
    feature_root: Path,
    score_field: str,            # "score" → raw BM25; "new_score" → BM25+time
    top_k: int,
) -> list[dict]:
    """Return [{commit_id, score, datetime, rank}] for top-k by score_field."""
    score_file = feature_root / repo_key / "bm25_time" / "result" / f"{cve_id}.json"
    if not score_file.exists():
        return []
    raw = json.loads(score_file.read_text())  # dict[commit_id -> {score, new_score, datetime, ...}]
    rows = [
        {"commit_id": cid, "score": float(v.get(score_field, 0.0)),
         "datetime": v.get("datetime", "")}
        for cid, v in raw.items()
    ]
    rows.sort(key=lambda r: r["score"], reverse=True)
    out = []
    for i, r in enumerate(rows[:top_k]):
        r["rank"] = i + 1
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Qwen dense — FAISS top-K against the repo's commit corpus
# ---------------------------------------------------------------------------


def _build_index(corpus_vecs: np.ndarray):
    """CPU FAISS IndexFlatIP; falls back to matmul if faiss unavailable."""
    try:
        import faiss  # noqa: PLC0415
    except ImportError:
        return None
    idx = faiss.IndexFlatIP(corpus_vecs.shape[1])
    idx.add(corpus_vecs.astype(np.float32))
    return idx


def rank_dense(
    cve_id: str,
    repo_key: str,
    feature_root: Path,
    top_k: int,
    _repo_cache: dict | None = None,
) -> list[dict]:
    """FAISS top-k of Qwen embedding for this CVE in this repo."""
    qwen_dir = feature_root / repo_key / "qwen_embedding"
    if not (qwen_dir / "corpus.pkl").exists() or not (qwen_dir / "queries.pkl").exists():
        return []

    # Cache per-repo data across CVEs in the same repo (queries.pkl shared).
    cache_key = str(qwen_dir)
    cached = _repo_cache.get(cache_key) if _repo_cache is not None else None
    if cached is None:
        with (qwen_dir / "corpus.pkl").open("rb") as f:
            corpus_vecs, corpus_ids = pickle.load(f)
        with (qwen_dir / "queries.pkl").open("rb") as f:
            query_vecs, query_ids = pickle.load(f)
        corpus_vecs = np.asarray(corpus_vecs, dtype=np.float32)
        query_vecs = np.asarray(query_vecs, dtype=np.float32)
        cached = {
            "corpus_vecs": corpus_vecs,
            "corpus_ids": list(corpus_ids),
            "query_vecs": query_vecs,
            "query_id_to_row": {qid: i for i, qid in enumerate(query_ids)},
            "index": _build_index(corpus_vecs),
        }
        if _repo_cache is not None:
            _repo_cache[cache_key] = cached

    q_row = cached["query_id_to_row"].get(cve_id)
    if q_row is None:
        return []

    q = cached["query_vecs"][q_row : q_row + 1]
    k = min(top_k, cached["corpus_vecs"].shape[0])

    if cached["index"] is not None:
        scores, idxs = cached["index"].search(q, k)
        scores = scores[0]
        idxs = idxs[0]
    else:
        # matmul fallback (slower but works without faiss)
        full_scores = (cached["corpus_vecs"] @ q.T).ravel()
        idxs = np.argsort(-full_scores)[:k]
        scores = full_scores[idxs]

    out = []
    for rank_i, (s, i) in enumerate(zip(scores, idxs), start=1):
        out.append({
            "commit_id": cached["corpus_ids"][int(i)],
            "score": float(s),
            "datetime": "",       # not stored in pkl; not needed by Phase 2
            "rank": rank_i,
        })
    return out


# ---------------------------------------------------------------------------
# Queries CSV loader — same schema as data/sample_ground_truth_810.csv
# ---------------------------------------------------------------------------


def load_queries(csv_path: Path) -> list[dict]:
    """Return [{cve, owner, repo, fix_commit_ids: [str]}, ...].

    Handles both single-row-per-CVE (sample_810) and multi-row (ground_truth_clean).
    """
    by_cve: dict[str, dict] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            cve = (row.get("cve") or "").strip()
            if not cve:
                continue
            entry = by_cve.setdefault(cve, {
                "cve": cve,
                "owner": row.get("owner", "").strip(),
                "repo": row.get("repo", "").strip(),
                "fix_commit_ids": [],
            })
            patch = (row.get("patch") or "").strip()
            if patch and patch not in entry["fix_commit_ids"]:
                entry["fix_commit_ids"].append(patch)
    return list(by_cve.values())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 ablation: emit a top-K jsonl from ONE retriever.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--retriever", choices=RETRIEVERS, required=True)
    p.add_argument(
        "--queries-csv", required=True, type=Path,
        help="CSV with columns: cve, owner, repo, patch. "
             "Typically data/sample_ground_truth_810.csv.",
    )
    p.add_argument(
        "--feature-root", default=Path("./embeddings"), type=Path,
        help="Directory layout: <feature-root>/<owner>@@<repo>/{bm25_time,qwen_embedding}/",
    )
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    queries = load_queries(args.queries_csv)
    # Group by repo for cache reuse on dense.
    repo_to_cves = defaultdict(list)
    for q in queries:
        repo_to_cves[f"{q['owner']}@@{q['repo']}"].append(q)
    print(f"Queries: {len(queries)} CVE across {len(repo_to_cves)} repos")
    print(f"Retriever: {args.retriever}")

    score_field = {"bm25": "score", "bm25_time": "new_score"}.get(args.retriever)
    dense_cache: dict = {}

    n_written = 0
    n_empty = 0
    with args.output.open("w") as out:
        for repo_key, q_list in sorted(repo_to_cves.items()):
            for q in q_list:
                if args.retriever in ("bm25", "bm25_time"):
                    cands = rank_bm25(
                        q["cve"], repo_key, args.feature_root,
                        score_field=score_field, top_k=args.top_k,
                    )
                else:
                    cands = rank_dense(
                        q["cve"], repo_key, args.feature_root,
                        top_k=args.top_k, _repo_cache=dense_cache,
                    )

                if not cands:
                    n_empty += 1

                # Annotate each candidate with the retriever source label.
                for c in cands:
                    c["source"] = args.retriever

                # Compute basic stats: best_rank of truth, recall@k.
                truth = set(q["fix_commit_ids"])
                fix_ranks = [c["rank"] for c in cands if c["commit_id"] in truth]
                best_rank = min(fix_ranks) if fix_ranks else None
                recall = len(fix_ranks) / len(truth) if truth else 0.0

                rec = {
                    "cve_id": q["cve"],
                    "owner": q["owner"],
                    "repo": q["repo"],
                    "repo_key": repo_key,
                    "fix_commit_ids": q["fix_commit_ids"],
                    "best_rank": best_rank,
                    "recall": recall,
                    "fix_ranks": fix_ranks,
                    "num_candidates": len(cands),
                    "saved_candidates_k": args.top_k,
                    "candidates": cands,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
            # Drop this repo from cache after processing all its CVE (free RAM).
            if args.retriever == "dense":
                qwen_dir = args.feature_root / repo_key / "qwen_embedding"
                dense_cache.pop(str(qwen_dir), None)

    print(f"\n✓ Wrote {n_written} CVE rankings → {args.output}")
    if n_empty:
        print(f"  ⚠ {n_empty} CVE had no retrievable candidates (missing per-repo data)")


if __name__ == "__main__":
    main()
