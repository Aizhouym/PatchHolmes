#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patchholmes.data_models import CVEQuery  # noqa: E402
from patchholmes.phase1.bm25_time import BM25Retriever  # noqa: E402
from patchholmes.phase1.dense_faiss import DenseRetriever  # noqa: E402
from patchholmes.phase1.rrf_fusion import rrf_fuse  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--queries-csv", required=True, type=Path)
    p.add_argument("--feature-root", default=Path("./embeddings"), type=Path)
    p.add_argument("--dense-subdir", default="octen_embedding")
    p.add_argument("--queries-name", default="queries.pkl")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--dense-top-k", type=int, default=5000)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def load_queries(csv_path):
    out = []
    try:
        f = csv_path.open()
    except OSError as e:
        print(f"failed to open {csv_path}: {e}")
        return out
    with f:
        for r in csv.DictReader(f):
            patch = (r.get("patch") or "").strip()
            ids = [s for s in patch.split("|") if s]
            out.append(CVEQuery(
                cve_id=r["cve"].strip(),
                description="",
                owner=r["owner"].strip(),
                repo=r["repo"].strip(),
                fix_commit_ids=ids,
            ))
    return out


def missing_record(q, top_k):
    return {
        "cve_id": q.cve_id,
        "owner": q.owner,
        "repo": q.repo,
        "repo_key": q.repo_key,
        "fix_commit_ids": list(q.fix_commit_ids),
        "best_rank": None,
        "recall": 0.0,
        "fix_ranks": [],
        "num_candidates": 0,
        "saved_candidates_k": top_k,
        "candidates": [],
        "_missing_octen_pkl": True,
    }


def run_dense(dense, q_pkl, c_pkl, k, cve_id):
    try:
        return dense.retrieve_from_pkl(
            query_pkl=q_pkl,
            corpus_pkl=c_pkl,
            commit_lookup={},
            top_k=k,
            cve_id=cve_id,
        )
    except Exception as e:
        print(f"  dense failed for {cve_id}: {e}", flush=True)
        return []


def run_bm25(bm25, q):
    try:
        return bm25.retrieve(q, top_k=None)
    except Exception as e:
        print(f"  bm25 failed for {q.cve_id}: {e}", flush=True)
        return []


def build_record(q, fused, top_k):
    truth = set(q.fix_commit_ids)
    fix_ranks = [c.rank for c in fused if c.commit.commit_id in truth]
    best = min(fix_ranks) if fix_ranks else None
    cands = [
        {
            "rank": c.rank,
            "commit_id": c.commit.commit_id,
            "score": float(c.score),
            "source": "rrf-octen",
            "bm25_rank": c.bm25_rank,
            "dense_rank": c.dense_rank,
            "datetime": c.commit.datetime,
        }
        for c in fused
    ]
    return {
        "cve_id": q.cve_id,
        "owner": q.owner,
        "repo": q.repo,
        "repo_key": q.repo_key,
        "fix_commit_ids": list(q.fix_commit_ids),
        "best_rank": best,
        "recall": len(fix_ranks) / len(truth) if truth else 0.0,
        "fix_ranks": fix_ranks,
        "num_candidates": len(cands),
        "saved_candidates_k": top_k,
        "candidates": cands,
    }


def process_one(q, args, bm25, dense, out):
    c_pkl = args.feature_root / q.repo_key / args.dense_subdir / "corpus.pkl"
    q_pkl = args.feature_root / q.repo_key / args.dense_subdir / args.queries_name
    if not c_pkl.exists() or not q_pkl.exists():
        out.write(json.dumps(missing_record(q, args.top_k), ensure_ascii=False) + "\n")
        return False
    d = run_dense(dense, q_pkl, c_pkl, args.dense_top_k, q.cve_id)
    b = run_bm25(bm25, q)
    fused = rrf_fuse(b, d, top_k=args.top_k, k=args.rrf_k)
    out.write(json.dumps(build_record(q, fused, args.top_k), ensure_ascii=False) + "\n")
    return True


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    qs = load_queries(args.queries_csv)
    n_repos = len(set(q.repo_key for q in qs))
    print(f"Queries: {len(qs)} CVE  ({n_repos} unique repos)")
    print(f"Feature root:  {args.feature_root}")
    print(f"Dense subdir:  {args.dense_subdir}")
    print(f"Queries pkl :  {args.queries_name}")
    print(f"Output:        {args.output}\n")

    bm25 = BM25Retriever(
        feature_root=args.feature_root,
        repo2commits_root=Path("/dev/null"),
    )
    dense = DenseRetriever()

    done = miss = 0
    try:
        out = args.output.open("w")
    except OSError as e:
        print(f"cannot write {args.output}: {e}")
        return
    with out:
        for q in qs:
            ok = process_one(q, args, bm25, dense, out)
            if ok:
                done += 1
            else:
                miss += 1
            if done and done % 50 == 0:
                print(f"  [{done}/{len(qs)}] done", flush=True)

    print(f"\nWrote {done} successful + {miss} missing-pkl records -> {args.output}")


if __name__ == "__main__":
    main()
