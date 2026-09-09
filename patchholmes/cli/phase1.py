#!/usr/bin/env python3
"""Phase 1 evaluation CLI."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

from patchholmes.data_models import CVEQuery, Phase1Result, RankedCandidate


EVAL_KS = [10, 100, 500, 1000]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PatchHolmes Phase 1 evaluation.")
    p.add_argument(
        "--dataset",
        default="data/ground_truth_queries_clean.csv",
        help="Ground truth CSV with cve,cve_description,owner,repo,fix_commit_ids.",
    )
    p.add_argument("--feature-root", default="./embeddings")
    p.add_argument("--repo2commits-root", default="./data/repo2commits_diff")
    p.add_argument(
        "--embedding-root",
        required=True,
        help="Root containing <owner>@@<repo>/qwen_embedding/{corpus,queries}.pkl.",
    )
    p.add_argument(
        "--combined-csv",
        default=None,
        help="Path to combined.csv with reserve_time/publish_time. "
             "Enables ES online BM25 fallback for CVEs without precomputed JSON.",
    )
    p.add_argument("--dense-top-k", type=int, default=5000)
    p.add_argument("--rrf-top-k", type=int, default=1000)
    p.add_argument(
        "--per-query-output",
        default="logs/phase1/phase1_per_query.jsonl",
        help="Write one JSON object per CVE. Use an empty string to disable.",
    )
    p.add_argument(
        "--save-candidates-k",
        type=int,
        default=-1,
        help="-1 saves --rrf-top-k candidates; 0 saves only per-CVE metrics.",
    )
    p.add_argument("--max-cves", type=int, default=0)
    return p.parse_args()


def load_queries(csv_path: Path, max_cves: int) -> list[CVEQuery]:
    queries: list[CVEQuery] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if max_cves > 0 and len(queries) >= max_cves:
                break
            cve = (row.get("cve") or "").strip()
            desc = (row.get("cve_description") or "").strip()
            owner = (row.get("owner") or "").strip()
            repo = (row.get("repo") or "").strip()
            fixes = [c.strip() for c in (row.get("fix_commit_ids") or "").split("|") if c.strip()]
            if cve and owner and repo:
                queries.append(CVEQuery(cve, desc, owner, repo, fixes))
    return queries


def print_eval(label: str, hits: dict[int, int], n: int) -> None:
    if n == 0:
        print("No valid queries.")
        return
    print(f"\n[ {label} ]")
    print(f"{'K':>6}  {'Recall@K':>10}  hits/{n}")
    print("-" * 36)
    for k in EVAL_KS:
        print(f"{k:>6}  {hits[k] / n:>10.4f}  {hits[k]}/{n}")


def candidate_to_record(cand: RankedCandidate) -> dict:
    return {
        "rank": cand.rank,
        "commit_id": cand.commit.commit_id,
        "score": cand.score,
        "source": cand.source,
        "bm25_rank": cand.bm25_rank,
        "dense_rank": cand.dense_rank,
        "datetime": cand.commit.datetime,
    }


def result_to_record(res: Phase1Result, eval_ks: list[int], save_candidates_k: int) -> dict:
    fix_set = set(res.query.fix_commit_ids)
    fix_ranks = [
        {
            "commit_id": cand.commit.commit_id,
            "rank": cand.rank,
            "score": cand.score,
            "bm25_rank": cand.bm25_rank,
            "dense_rank": cand.dense_rank,
        }
        for cand in res.candidates
        if cand.commit.commit_id in fix_set
    ]
    if save_candidates_k < 0:
        save_candidates_k = len(res.candidates)
    return {
        "cve_id": res.query.cve_id,
        "owner": res.query.owner,
        "repo": res.query.repo,
        "repo_key": res.query.repo_key,
        "fix_commit_ids": res.query.fix_commit_ids,
        "best_rank": res.best_rank(),
        "recall": {f"@{k}": res.recall_at_k(k) for k in eval_ks},
        "fix_ranks": fix_ranks,
        "num_candidates": len(res.candidates),
        "saved_candidates_k": min(save_candidates_k, len(res.candidates)),
        "candidates": [
            candidate_to_record(c)
            for c in res.candidates[:save_candidates_k]
        ] if save_candidates_k > 0 else [],
    }


def main() -> None:
    import sys
    import time
    sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()

    print("=" * 60)
    print("PatchHolmes Phase 1")
    print(f"  dataset       : {args.dataset}")
    print(f"  embedding-root: {args.embedding_root}")
    print(f"  feature-root  : {args.feature_root}")
    print(f"  dense-top-k   : {args.dense_top_k}")
    print(f"  rrf-top-k     : {args.rrf_top_k}")
    print("=" * 60)

    print("\n[1/4] Importing pipeline modules ...")
    t0 = time.monotonic()
    from patchholmes.phase1.pipeline import Phase1Pipeline
    from patchholmes.phase1.bm25_time import BM25Retriever
    from patchholmes.phase1.dense_faiss import DenseRetriever, _default_backend
    print(f"      done ({time.monotonic() - t0:.1f}s)")

    print("\n[2/4] Loading queries ...")
    queries = load_queries(Path(args.dataset), args.max_cves)
    print(f"      {len(queries)} CVE queries loaded")

    embedding_root = Path(args.embedding_root)
    missing_corpus = [q for q in queries if not (embedding_root / q.repo_key / "qwen_embedding" / "corpus.pkl").exists()]
    missing_queries = [q for q in queries if not (embedding_root / q.repo_key / "qwen_embedding" / "queries.pkl").exists()]
    if missing_corpus or missing_queries:
        print(f"      WARNING: {len(missing_corpus)} missing corpus.pkl, {len(missing_queries)} missing queries.pkl")
        print("      Run encode corpus / encode queries first.")
    else:
        print("      All pkl files present.")

    print(f"\n[3/4] Initialising pipeline (dense backend: {_default_backend()}) ...")
    t0 = time.monotonic()
    pipeline = Phase1Pipeline(
        bm25_retriever=BM25Retriever(args.feature_root, args.repo2commits_root, combined_csv=args.combined_csv),
        dense_retriever=DenseRetriever(),
        embedding_root=embedding_root,
        dense_top_k=args.dense_top_k,
        rrf_top_k=args.rrf_top_k,
    )
    print(f"      done ({time.monotonic() - t0:.1f}s)")

    print(f"\n[4/4] Running Phase 1 ...")
    hits = {k: 0 for k in EVAL_KS}
    valid = 0
    first_details = []
    out_f = None
    if args.per_query_output:
        out_path = Path(args.per_query_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")
        print(f"  writing per-query results to {out_path}")

    t_start = time.monotonic()
    try:
        for i, query in enumerate(queries, start=1):
            if i == 1 or i % 50 == 0:
                elapsed = time.monotonic() - t_start
                qps = i / elapsed if elapsed > 0 else 0
                eta = (len(queries) - i) / qps if qps > 0 else 0
                r10 = hits[10] / valid if valid else 0
                r1000 = hits[1000] / valid if valid else 0
                print(
                    f"  [{i:>5}/{len(queries)}] {query.repo_key} {query.cve_id}"
                    f"  |  {elapsed/60:.1f}min elapsed  ETA {eta/60:.1f}min"
                    f"  |  R@10={r10:.3f} R@1000={r1000:.3f}"
                )
            res = pipeline.run(query)
            if query.fix_commit_ids:
                valid += 1
                for k in EVAL_KS:
                    if res.recall_at_k(k):
                        hits[k] += 1

            if out_f is not None:
                save_k = args.save_candidates_k if args.save_candidates_k >= 0 else args.rrf_top_k
                out_f.write(json.dumps(result_to_record(res, EVAL_KS, save_k), ensure_ascii=False) + "\n")
                out_f.flush()

            if len(first_details) < 5:
                top3 = [c.commit.commit_id[:8] for c in res.candidates[:3]]
                first_details.append(
                    f"  {res.query.cve_id}"
                    f"  patches={[p[:8] for p in res.query.fix_commit_ids]}"
                    f"  best_rank={res.best_rank()}"
                    f"  recall@{args.rrf_top_k}={res.recall_at_k(args.rrf_top_k)}"
                    f"  top3={top3}"
                )

            del res
            gc.collect()
    finally:
        if out_f is not None:
            out_f.close()

    print_eval("Phase 1 - BM25 + Dense (RRF)", hits, valid)
    print("\n--- Per-query detail (first 5) ---")
    for line in first_details:
        print(line)


if __name__ == "__main__":
    main()
