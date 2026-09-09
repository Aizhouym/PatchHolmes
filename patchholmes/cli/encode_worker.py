#!/usr/bin/env python3
"""Single-shard direct per-repo embedding worker."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from patchholmes.encoding.corpus_encoder import (
    CorpusEncoder,
    load_cve_queries_for_repo,
    load_commits_for_repo,
    load_repo_list,
    save_pkl,
)
from patchholmes.encoding.embedder import build_commit_text
from patchholmes.encoding.embedder import MockEmbedder, QwenEmbedder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-compute per-repo Qwen embeddings for one shard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_argument_group("data source")
    src.add_argument(
        "--gt-csv",
        default="./data/ground_truth_queries_clean.csv",
        help="CSV with cve,cve_description,owner,repo,fix_commit_ids.",
    )
    src.add_argument("--repo2commits-root", default="./data/repo2commits_diff")
    src.add_argument("--output-root", default="./embeddings")

    mdl = p.add_argument_group("model")
    mdl.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    mdl.add_argument("--embedder", choices=["qwen", "mock"], default="qwen")
    mdl.add_argument("--batch-size", type=int, default=256)
    mdl.add_argument("--max-passage-length", type=int, default=2048)
    mdl.add_argument("--max-query-length", type=int, default=512)
    mdl.add_argument("--max-diff-chars", type=int, default=6000)
    mdl.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    mdl.add_argument("--embedding-subdir", default="qwen_embedding",
                     help="Per-repo subdir name for the output pkls. Default 'qwen_embedding' "
                          "matches the main system layout. Set e.g. 'octen_embedding' for an "
                          "alternate-model ablation that should live next to the main pkls.")

    ctl = p.add_argument_group("run control")
    ctl.add_argument("--mode", choices=["both", "corpus", "queries"], default="corpus")
    ctl.add_argument("--max-repos", type=int, default=0)
    ctl.add_argument("--no-skip-existing", action="store_true")
    ctl.add_argument(
        "--repo-keys-file",
        default=None,
        help="Optional file with one owner@@repo per line. When set, process exactly this repo list.",
    )

    shard = p.add_argument_group("multi-GPU sharding")
    shard.add_argument("--num-shards", type=int, default=1)
    shard.add_argument("--shard-id", type=int, default=0)
    shard.add_argument("--device", default=None)
    shard.add_argument("--reverse", action="store_true")

    return p.parse_args()


def load_gt_csv(csv_path: Path) -> tuple[dict[str, str], Path]:
    cve2desc: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cve = (row.get("cve") or "").strip()
            desc = (row.get("cve_description") or "").strip()
            if cve:
                cve2desc[cve] = desc
    return cve2desc, csv_path


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()

    gt_path = Path(args.gt_csv)
    if not gt_path.exists():
        raise SystemExit(f"--gt-csv not found: {gt_path}")

    print(f"Data source: {gt_path}")
    cve2desc, combined_csv = load_gt_csv(gt_path)
    print(f"  {len(cve2desc):,} CVE descriptions loaded")

    all_repos = load_repo_list(combined_csv)
    if args.repo_keys_file:
        allowed = read_repo_keys(Path(args.repo_keys_file))
        repo_map = {f"{owner}@@{repo}": (owner, repo) for owner, repo in all_repos}
        missing = [repo_key for repo_key in allowed if repo_key not in repo_map]
        if missing:
            print(f"WARNING: {len(missing)} repo keys from --repo-keys-file are not in {combined_csv}")
        repos = [repo_map[repo_key] for repo_key in allowed if repo_key in repo_map]
        shard_label = f"assigned shard {args.shard_id}"
    elif args.reverse:
        all_repos = list(reversed(all_repos))
        repos = all_repos[args.shard_id :: args.num_shards] if args.num_shards > 1 else all_repos
        shard_label = f"shard {args.shard_id}/{args.num_shards}" if args.num_shards > 1 else "all repos"
    elif args.num_shards > 1:
        repos = all_repos[args.shard_id :: args.num_shards]
        shard_label = f"shard {args.shard_id}/{args.num_shards}"
    else:
        repos = all_repos
        shard_label = "all repos"

    if args.max_repos > 0:
        repos = repos[: args.max_repos]

    print(f"\n{shard_label}: {len(repos)} repos to process (output -> {args.output_root})")
    if not repos:
        print(json.dumps({"corpus_written": 0, "queries_written": 0, "skipped": 0}, indent=2))
        return

    if args.embedder == "mock":
        print("Using MockEmbedder")
        embedder = MockEmbedder()
    else:
        print(f"Loading QwenEmbedder: {args.model}")
        print(f"  dtype={args.dtype} batch={args.batch_size} max_passage={args.max_passage_length}")
        embedder = QwenEmbedder(
            model_name=args.model,
            batch_size=args.batch_size,
            max_passage_length=args.max_passage_length,
            max_query_length=args.max_query_length,
            dtype=args.dtype,
            device=args.device,
        )

    encoder = CorpusEncoder(
        embedder=embedder,
        output_root=Path(args.output_root),
        max_diff_chars=args.max_diff_chars,
        skip_existing=not args.no_skip_existing,
        embedding_subdir=args.embedding_subdir,
    )

    stats = {"corpus_written": 0, "queries_written": 0, "skipped": 0}
    for i, (owner, repo) in enumerate(repos, start=1):
        repo_key = f"{owner}@@{repo}"
        prefix = f"[{shard_label} {i}/{len(repos)}] {repo_key}"

        if args.mode in ("both", "corpus"):
            result = encode_corpus_with_progress(
                encoder=encoder,
                owner=owner,
                repo=repo,
                repo2commits_root=Path(args.repo2commits_root),
                prefix=prefix,
            )
            if result == "written":
                stats["corpus_written"] += 1
            else:
                stats["skipped"] += 1

        if args.mode in ("both", "queries"):
            result = encode_queries_with_progress(
                encoder=encoder,
                owner=owner,
                repo=repo,
                combined_csv=combined_csv,
                cve2desc=cve2desc,
                prefix=prefix,
            )
            if result == "written":
                stats["queries_written"] += 1
            else:
                stats["skipped"] += 1

    print(json.dumps(stats, indent=2))


def read_repo_keys(path: Path) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        repo_key = line.strip()
        if not repo_key or repo_key.startswith("#") or repo_key in seen:
            continue
        seen.add(repo_key)
        keys.append(repo_key)
    return keys


def encode_corpus_with_progress(
    encoder: CorpusEncoder,
    owner: str,
    repo: str,
    repo2commits_root: Path,
    prefix: str,
) -> str:
    out_pkl = encoder.output_root / f"{owner}@@{repo}" / encoder.embedding_subdir / "corpus.pkl"
    if encoder.skip_existing and out_pkl.exists() and out_pkl.stat().st_size > 0:
        print(f"{prefix} skipped existing corpus -> {out_pkl}")
        return "skipped"

    print(f"{prefix} loading commits ...")
    commits = load_commits_for_repo(repo2commits_root, owner, repo)
    if not commits:
        print(f"{prefix} skipped corpus (no commits found)")
        return "skipped"

    print(f"{prefix} encoding {len(commits):,} commits ...")
    texts = [build_commit_text(msg, diff, encoder.max_diff_chars) for _, msg, diff in commits]
    ids = [cid for cid, _, _ in commits]
    vecs = encoder.embedder.encode_corpus(texts)
    save_pkl(out_pkl, vecs, ids)
    print(f"{prefix} saved corpus -> {out_pkl}")
    return "written"


def encode_queries_with_progress(
    encoder: CorpusEncoder,
    owner: str,
    repo: str,
    combined_csv: Path,
    cve2desc: dict[str, str],
    prefix: str,
) -> str:
    out_pkl = encoder.output_root / f"{owner}@@{repo}" / encoder.embedding_subdir / "queries.pkl"
    if encoder.skip_existing and out_pkl.exists() and out_pkl.stat().st_size > 0:
        print(f"{prefix} skipped existing queries -> {out_pkl}")
        return "skipped"

    cve_pairs = load_cve_queries_for_repo(combined_csv, cve2desc, owner, repo)
    if not cve_pairs:
        print(f"{prefix} skipped queries (no CVEs found)")
        return "skipped"

    print(f"{prefix} encoding {len(cve_pairs):,} CVE queries ...")
    cve_ids = [cve for cve, _ in cve_pairs]
    texts = [desc for _, desc in cve_pairs]
    vecs = encoder.embedder.encode_query(texts)
    save_pkl(out_pkl, vecs, cve_ids)
    print(f"{prefix} saved queries -> {out_pkl}")
    return "written"


if __name__ == "__main__":
    main()
