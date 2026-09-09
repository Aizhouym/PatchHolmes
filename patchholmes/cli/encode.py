#!/usr/bin/env python3
"""Multi-GPU direct per-repo embedding launcher."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = "python3"
DEFAULT_GT_CSV = str(REPO_ROOT / "data" / "ground_truth_queries_clean.csv")
DEFAULT_REPO2COMMITS = "./data/repo2commits_diff"
DEFAULT_FEATURE_ROOT = "./embeddings"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-8B"


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        raise SystemExit(f"{name} must be an integer, got {value!r}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gt-csv", default=DEFAULT_GT_CSV)
    parser.add_argument("--repo2commits-root", default=DEFAULT_REPO2COMMITS)
    parser.add_argument("--output-root", default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--python", default=os.environ.get("PYTHON", DEFAULT_PYTHON))
    parser.add_argument("--dtype", default=os.environ.get("DTYPE", "bfloat16"),
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num-gpus", type=int, default=env_int("NUM_GPUS", 4))
    parser.add_argument("--reverse", action="store_true",
                        default=os.environ.get("REVERSE", "0") == "1")
    parser.add_argument(
        "--static-shards",
        action="store_true",
        help="Use old deterministic repo-index sharding instead of redistributing pending repos.",
    )
    parser.add_argument("--batch-size", type=int, default=env_int("BATCH_SIZE", 128))
    parser.add_argument("--max-passage-length", type=int, default=env_int("MAX_PASSAGE_LEN", 2048))
    parser.add_argument("--max-query-length", type=int, default=env_int("MAX_QUERY_LEN", 512))
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=env_int("PROGRESS_INTERVAL", 30),
        help="Seconds between terminal progress summaries.",
    )


def launch_sharded(args: argparse.Namespace, mode: str, log_name: str) -> int:
    if args.num_gpus < 1:
        raise SystemExit("--num-gpus must be >= 1")

    log_dir = REPO_ROOT / "logs" / log_name
    if args.reverse and mode == "corpus":
        log_dir = REPO_ROOT / "logs" / f"{log_name}_rev"
    log_dir.mkdir(parents=True, exist_ok=True)

    pids: list[tuple[int, subprocess.Popen]] = []
    log_files = []
    progress = build_progress_state(args, mode)
    assignments = build_assignments(args, mode, progress)

    print("========================================")
    print(f"PatchHolmes {mode} encode")
    print(f"  GT CSV       : {args.gt_csv}")
    if mode == "corpus":
        print(f"  repo2commits : {args.repo2commits_root}")
    print(f"  output root  : {args.output_root}")
    print(f"  model        : {args.model}")
    print(f"  dtype        : {args.dtype}")
    print(f"  num GPUs     : {args.num_gpus}")
    print(f"  logs         : {log_dir}/gpu*.log")
    print(f"  already done : {progress['initial_done']}/{progress['total_repos']}")
    print(f"  to encode    : {progress['target_total']}")
    if args.static_shards:
        print("  assignment   : static repo-index shards")
    else:
        print("  assignment   : pending repos redistributed across GPUs")
    print("========================================")

    for gpu_id in range(args.num_gpus):
        if assignments and not read_assignment(assignments[gpu_id]):
            print(f"Skipping GPU {gpu_id}: no pending repos assigned")
            continue

        log_path = log_dir / f"gpu{gpu_id}.log"
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(REPO_ROOT),
        })

        batch_size = args.batch_size
        if mode == "queries" and "BATCH_SIZE" not in os.environ:
            batch_size = 256

        cmd = [
            args.python,
            "-m", "patchholmes.cli.encode_worker",
            "--gt-csv", args.gt_csv,
            "--output-root", args.output_root,
            "--model", args.model,
            "--mode", mode,
            "--batch-size", str(batch_size),
            "--dtype", args.dtype,
            "--device", "cuda:0",
            "--num-shards", str(args.num_gpus),
            "--shard-id", str(gpu_id),
        ]
        if assignments:
            cmd += ["--repo-keys-file", str(assignments[gpu_id])]
        if mode == "corpus":
            cmd += [
                "--repo2commits-root", args.repo2commits_root,
                "--max-passage-length", str(args.max_passage_length),
            ]
            if args.reverse:
                cmd.append("--reverse")
        else:
            cmd += ["--max-query-length", str(args.max_query_length)]

        log_f = log_path.open("w", encoding="utf-8")
        log_files.append(log_f)
        print(f"Launching GPU {gpu_id} -> {log_path}")
        pids.append((gpu_id, subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)))

    print(f"\n{len(pids)}/{args.num_gpus} GPU processes launched.")
    print(f"Monitor: tail -f {log_dir}/gpu*.log\n")

    if not pids:
        print("No pending repos to encode.")
        return 0

    last_report = 0.0
    failed = 0
    try:
        while pids:
            now = time.monotonic()
            if now - last_report >= args.progress_interval:
                print_progress(args, mode, log_dir, progress)
                last_report = now

            remaining: list[tuple[int, subprocess.Popen]] = []
            for gpu_id, proc in pids:
                rc = proc.poll()
                if rc is None:
                    remaining.append((gpu_id, proc))
                elif rc == 0:
                    print(f"GPU {gpu_id} finished OK")
                else:
                    print(f"GPU {gpu_id} FAILED (exit code {rc})", file=sys.stderr)
                    failed += 1
            pids = remaining
            if pids:
                time.sleep(10)
    finally:
        for log_f in log_files:
            log_f.close()

    if failed:
        print(f"WARNING: {failed}/{args.num_gpus} GPU shard(s) failed. Check {log_dir}.", file=sys.stderr)
        return 1
    print("All shards completed successfully.")
    return 0


def build_progress_state(args: argparse.Namespace, mode: str) -> dict[str, int]:
    output_root = Path(args.output_root)
    filename = "corpus.pkl" if mode == "corpus" else "queries.pkl"
    repos = load_repo_keys(Path(args.gt_csv))
    total_repos = len(repos)
    initial_done = count_existing_for_repos(output_root, filename, repos)
    return {
        "total_repos": total_repos,
        "initial_done": initial_done,
        "target_total": max(0, total_repos - initial_done),
    }


def build_assignments(args: argparse.Namespace, mode: str, progress: dict[str, int]) -> list[Path] | None:
    if args.static_shards:
        return None

    filename = "corpus.pkl" if mode == "corpus" else "queries.pkl"
    output_root = Path(args.output_root)
    repos = load_repo_keys(Path(args.gt_csv))
    if args.reverse:
        repos = list(reversed(repos))
    pending = [
        repo_key
        for repo_key in repos
        if not is_nonempty_pkl(output_root / repo_key / "qwen_embedding" / filename)
    ]

    assignment_dir = REPO_ROOT / "logs" / "assignments" / mode
    assignment_dir.mkdir(parents=True, exist_ok=True)
    assignment_paths: list[Path] = []
    for gpu_id in range(args.num_gpus):
        shard_repos = pending[gpu_id :: args.num_gpus]
        path = assignment_dir / f"gpu{gpu_id}.txt"
        path.write_text("\n".join(shard_repos) + ("\n" if shard_repos else ""), encoding="utf-8")
        assignment_paths.append(path)
        print(f"Assigned GPU {gpu_id}: {len(shard_repos)} pending repos")

    assigned_total = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in assignment_paths)
    if assigned_total != progress["target_total"]:
        print(
            f"WARNING: assignment count {assigned_total} differs from target {progress['target_total']}",
            file=sys.stderr,
        )
    return assignment_paths


def read_assignment(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_repo_keys(gt_csv: Path) -> list[str]:
    seen: set[str] = set()
    repos: list[str] = []
    with gt_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            owner = (row.get("owner") or "").strip()
            repo = (row.get("repo") or "").strip()
            repo_key = f"{owner}@@{repo}"
            if owner and repo and repo_key not in seen:
                seen.add(repo_key)
                repos.append(repo_key)
    return repos


def count_existing_for_repos(output_root: Path, filename: str, repos: list[str]) -> int:
    return sum(
        1
        for repo_key in repos
        if is_nonempty_pkl(output_root / repo_key / "qwen_embedding" / filename)
    )


def is_nonempty_pkl(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def print_progress(args: argparse.Namespace, mode: str, log_dir: Path, progress: dict[str, int]) -> None:
    filename = "corpus.pkl" if mode == "corpus" else "queries.pkl"
    done = count_pkls(Path(args.output_root), filename)
    current_done = max(0, done - progress["initial_done"])
    target_total = progress["target_total"]
    total_repos = progress["total_repos"]
    pct_target = (current_done / target_total * 100.0) if target_total else 100.0
    pct_total = (done / total_repos * 100.0) if total_repos else 100.0
    latest = latest_log_lines(log_dir)
    timestamp = time.strftime("%H:%M:%S")

    print(
        f"[{timestamp}] {filename}: "
        f"current {current_done}/{target_total} ({pct_target:.1f}%) | "
        f"total {done}/{total_repos} ({pct_total:.1f}%)"
    )
    if latest:
        for line in latest:
            print(f"    {line}")
    else:
        print("    waiting for worker log output ...")
    print("", flush=True)


def count_pkls(output_root: Path, filename: str) -> int:
    if not output_root.exists():
        return 0
    return sum(
        1
        for path in output_root.glob(f"*/qwen_embedding/{filename}")
        if path.is_file() and path.stat().st_size > 0
    )


def latest_log_lines(log_dir: Path) -> list[str]:
    lines: list[str] = []
    for log_path in sorted(log_dir.glob("gpu*.log")):
        line = tail_last_nonempty_line(log_path)
        if line:
            lines.append(f"{log_path.name}: {line}")
    return lines


def tail_last_nonempty_line(path: Path, max_bytes: int = 8192) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            chunk = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""

    for line in reversed(chunk.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped[-240:]
    return ""


def run_corpus(args: argparse.Namespace) -> int:
    return launch_sharded(args, mode="corpus", log_name="encode")


def run_queries(args: argparse.Namespace) -> int:
    return launch_sharded(args, mode="queries", log_name="encode_queries")


def run_all(args: argparse.Namespace) -> int:
    rc = run_corpus(args)
    if rc != 0:
        return rc
    return run_queries(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PatchHolmes direct per-repo embedding launcher.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("corpus", help="Encode commit corpus pkls per repo.")
    add_common_args(corpus)
    corpus.set_defaults(func=run_corpus)

    queries = sub.add_parser("queries", help="Encode CVE query pkls per repo.")
    add_common_args(queries)
    queries.set_defaults(num_gpus=env_int("NUM_GPUS", 2), func=run_queries)

    all_cmd = sub.add_parser("all", help="Run corpus encoding, then query encoding.")
    add_common_args(all_cmd)
    all_cmd.set_defaults(func=run_all)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
