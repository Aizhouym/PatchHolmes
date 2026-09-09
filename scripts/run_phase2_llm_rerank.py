#!/usr/bin/env python3
"""Apply an LLM listwise reranker as a post-hoc Phase 2 critic.

Wraps `patchholmes.phase2.llm_rerank.LLMCritic` (rank_llm + a GPT-family model)
to re-rank the commits the main agent inspected.

Reads:
  - The main Phase 2 jsonl (`best_commit_id`, `commits_inspected`)
  - The Phase 1 jsonl (for commit_id → candidate metadata)
  - The dataset CSV (for cve_description)
  - The repo2commits_diff root (for raw commit diffs)

For each CVE, re-ranks the commits the main agent inspected and writes a new
jsonl with optional override. Output schema:
  - main_best_commit_id / main_reasoning   ← preserved from main
  - llm_ranking / llm_chose / llm_overrode
  - llm_input_tokens / llm_output_tokens / llm_raw_response
  - best_commit_id / hit / reasoning / stopped_reason  ← updated if override

Usage
-----
python scripts/run_phase2_llm_rerank.py \\
    --main-jsonl     logs/phase2/phase2_clean_full.jsonl \\
    --phase1-jsonl   logs/phase1/phase1_clean_full_top100.jsonl \\
    --dataset-csv    data/ground_truth_queries_clean.csv \\
    --repo2commits   ./data/repo2commits_diff \\
    --output         logs/phase2/phase2_clean_full_llm_rerank.jsonl \\
    --model          gpt-4o-mini \\
    --max-workers    8 \\
    --max-cves       100
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from patchholmes.data_models import CommitDoc, CVEQuery, RankedCandidate
from patchholmes.phase2.data_source import Phase2DataSource
from patchholmes.phase2.llm_rerank import LLMCritic

# Per-1M-token prices (USD). Source: OpenAI public pricing 2026-05.
# Keep this in code so cost is reproducible from the jsonl without external lookup.
MODEL_PRICES = {
    "gpt-4o-mini":     {"input": 0.15, "output": 0.60},
    "gpt-4o":          {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "gpt-4.1":         {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini":    {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano":    {"input": 0.10, "output": 0.40},
}


def cost_for(model: str, in_tok: int, out_tok: int) -> float:
    p = MODEL_PRICES.get(model)
    if not p:
        return 0.0
    return in_tok * p["input"] / 1e6 + out_tok * p["output"] / 1e6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM listwise reranker as post-hoc Phase 2 critic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--main-jsonl",
        default="./logs/phase2/phase2_clean_full.jsonl",
    )
    p.add_argument(
        "--phase1-jsonl",
        default="./logs/phase1/phase1_clean_full_top100.jsonl",
    )
    p.add_argument(
        "--dataset-csv",
        default="./data/ground_truth_queries_clean.csv",
    )
    p.add_argument(
        "--repo2commits",
        default="./data/repo2commits_diff",
    )
    p.add_argument(
        "--output",
        default="./logs/phase2/phase2_clean_full_llm_rerank.jsonl",
        help="Output jsonl path. Resumable: existing CVE entries are skipped.",
    )
    p.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name. Examples: gpt-4o-mini, gpt-4o, gpt-4.1-mini.",
    )
    p.add_argument(
        "--char-budget",
        type=int,
        default=8000,
        help="Per-commit char budget when rendering. Matches main agent's read_commit view.",
    )
    p.add_argument(
        "--max-passage-words",
        type=int,
        default=3000,
        help="rank_llm per-passage word cap. Set > our render to disable secondary truncation.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Concurrent CVE workers (API is IO-bound).",
    )
    p.add_argument(
        "--max-cves",
        type=int,
        default=0,
        help="Process at most N CVEs from the start (0 = all). Mutually exclusive with --sample-size.",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Randomly sample N CVEs (0 = disabled). Use --seed for reproducibility.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --sample-size.",
    )
    p.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Env var holding the OpenAI key.",
    )
    return p.parse_args()


def load_descriptions(csv_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            cve = (row.get("cve") or "").strip()
            if cve:
                out[cve] = (row.get("cve_description") or "").strip()
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def load_done_cves(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with output_path.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["cve_id"])
            except Exception:
                continue
    return done


def build_candidates(rec: dict[str, Any]) -> list[RankedCandidate]:
    out: list[RankedCandidate] = []
    for c in rec.get("candidates") or []:
        out.append(
            RankedCandidate(
                commit=CommitDoc(
                    commit_id=c["commit_id"],
                    commit_msg="",
                    diff="",
                    owner=rec["owner"],
                    repo=rec["repo"],
                    datetime=c.get("datetime", ""),
                ),
                score=float(c.get("score", 0.0)),
                rank=int(c["rank"]),
                source=c.get("source", "rrf"),
                bm25_rank=c.get("bm25_rank"),
                dense_rank=c.get("dense_rank"),
            )
        )
    return out


def process_one(
    main_rec: dict[str, Any],
    phase1_rec: dict[str, Any],
    cve_desc: str,
    repo2commits_root: Path,
    critic: LLMCritic,
    char_budget: int,
    model: str,
) -> dict[str, Any]:
    """Run LLM listwise rerank over main's inspected commits."""
    out = copy.deepcopy(main_rec)
    fix_set = set(main_rec.get("fix_commit_ids") or [])
    main_choice = main_rec.get("best_commit_id")

    # Preserve original main signals BEFORE override, so regression analysis
    # can compare main-vs-llm without recomputing.
    out["main_best_commit_id"] = main_choice
    out["main_reasoning"] = main_rec.get("reasoning", "")
    out["main_hit"] = bool(main_choice and main_choice in fix_set)
    out["phase1_rank_of_main_pick"] = main_rec.get("phase1_rank_of_answer")
    out["llm_model"] = model

    inspected: list[str] = list(main_rec.get("commits_inspected") or [])
    if main_choice and main_choice not in inspected:
        inspected = [main_choice] + inspected
    out["n_inspected"] = len(inspected)

    if not inspected:
        out["n_rendered"] = 0
        out["llm_ranking"] = []
        out["llm_chose"] = None
        out["llm_overrode"] = False
        out["llm_input_tokens"] = 0
        out["llm_output_tokens"] = 0
        out["llm_cost_usd"] = 0.0
        out["llm_raw_response"] = ""
        out["phase1_rank_of_llm_pick"] = None
        out["stopped_reason"] = "llm_skipped:no_inspected"
        return out

    # Render each inspected commit via the same Phase2DataSource main uses.
    query = CVEQuery(
        cve_id=main_rec["cve_id"],
        description=cve_desc,
        owner=main_rec["owner"],
        repo=main_rec["repo"],
        fix_commit_ids=main_rec.get("fix_commit_ids") or [],
    )
    p1_candidates = build_candidates(phase1_rec)
    ds = Phase2DataSource(query, p1_candidates, repo2commits_root, top_k=100)

    rendered: list[tuple[str, str]] = []
    for cid in inspected:
        text = ds.render_commit(cid, char_budget=char_budget)
        if text.startswith("[Error]"):
            continue
        normalised = ds._normalise_cid(cid)[0] or cid
        rendered.append((normalised, text))

    out["n_rendered"] = len(rendered)

    if not rendered:
        out["llm_ranking"] = []
        out["llm_chose"] = None
        out["llm_overrode"] = False
        out["llm_input_tokens"] = 0
        out["llm_output_tokens"] = 0
        out["llm_cost_usd"] = 0.0
        out["llm_raw_response"] = ""
        out["phase1_rank_of_llm_pick"] = None
        out["stopped_reason"] = "llm_skipped:no_renderable"
        return out

    result = critic.rerank(
        cve_id=main_rec["cve_id"],
        cve_description=cve_desc,
        candidates=rendered,
        main_commit_id=main_choice,
    )

    out["llm_ranking"] = result.ranking
    out["llm_chose"] = result.chosen_commit_id
    out["llm_overrode"] = result.overrode_main
    out["llm_input_tokens"] = result.input_tokens
    out["llm_output_tokens"] = result.output_tokens
    out["llm_cost_usd"] = cost_for(model, result.input_tokens, result.output_tokens)
    out["llm_raw_response"] = result.raw_response

    # Phase 1 rank of LLM's chosen commit — always recorded for audit
    rank_map = {c["commit_id"]: int(c["rank"]) for c in phase1_rec.get("candidates") or []}
    out["phase1_rank_of_llm_pick"] = rank_map.get(result.chosen_commit_id)

    if result.overrode_main:
        out["best_commit_id"] = result.chosen_commit_id
        out["reasoning"] = (
            f"[llm override; main picked "
            f"{main_choice[:12] if main_choice else 'none'}] "
            f"{main_rec.get('reasoning', '')}"
        )
        out["stopped_reason"] = (
            f"llm_override:{main_rec.get('stopped_reason', '')}"
        )
        out["hit"] = result.chosen_commit_id in fix_set
        out["phase1_rank_of_answer"] = out["phase1_rank_of_llm_pick"]
    else:
        out["stopped_reason"] = (
            f"llm_confirmed:{main_rec.get('stopped_reason', '')}"
        )

    return out


def main() -> None:
    args = parse_args()

    # rank_llm reads keys from env OR from .env.local; defer to its loader.
    from rank_llm.rerank.api_keys import get_openai_api_key

    if args.api_key_env != "OPENAI_API_KEY" and os.environ.get(args.api_key_env):
        os.environ["OPENAI_API_KEY"] = os.environ[args.api_key_env]
    if not get_openai_api_key():
        sys.exit(
            f"ERROR: no OpenAI key found. Set {args.api_key_env} env var "
            f"or put OPENAI_API_KEY in .env.local."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading CVE descriptions ...")
    descriptions = load_descriptions(Path(args.dataset_csv))
    print(f"  {len(descriptions):,} descriptions")

    print("Loading main Phase 2 results ...")
    main_records = load_jsonl(Path(args.main_jsonl))
    print(f"  {len(main_records):,} main records")

    print("Loading Phase 1 records ...")
    phase1_by_cve = {r["cve_id"]: r for r in load_jsonl(Path(args.phase1_jsonl))}
    print(f"  {len(phase1_by_cve):,} Phase 1 records")

    if args.sample_size > 0 and args.max_cves > 0:
        sys.exit("ERROR: pass either --sample-size OR --max-cves, not both.")

    # Determine the canonical work set BEFORE applying the resume filter, so
    # that re-running with the same --sample-size --seed yields the same set
    # regardless of how much of the output already exists.
    if args.sample_size > 0:
        import random

        rng = random.Random(args.seed)
        n = min(args.sample_size, len(main_records))
        work_set = rng.sample(main_records, n)
        print(
            f"Canonical sample: {n} CVEs from {len(main_records):,} "
            f"(seed={args.seed})"
        )
    elif args.max_cves > 0:
        work_set = main_records[: args.max_cves]
    else:
        work_set = main_records

    done = load_done_cves(output_path)
    if done:
        print(f"Resume: {len(done):,} CVE already in output, skipping")
    todo = [r for r in work_set if r["cve_id"] not in done]
    print(f"To process: {len(todo):,} CVE (of {len(work_set):,} target)\n")

    if not todo:
        print("Nothing to do.")
        return

    print(f"Initialising LLMCritic (model={args.model}, max_workers={args.max_workers}) ...")
    t0 = time.time()
    # Single shared critic across worker threads. SafeOpenai's OpenAI client is
    # thread-safe; key cycling and retry are handled inside rank_llm.
    critic = LLMCritic(
        model=args.model,
        max_passage_words=args.max_passage_words,
    )
    print(f"  ready in {time.time() - t0:.1f}s\n")

    write_lock = Lock()
    t_start = time.time()
    n_done = n_hit = n_override = n_override_hit = 0
    in_tok = out_tok = 0
    total_cost = 0.0
    # Confusion matrix counters (for regression analysis)
    cm = {"both_right": 0, "main_right_llm_broke": 0,
          "main_wrong_llm_fixed": 0, "both_wrong": 0}
    per_cve_tokens: list[tuple[int, int]] = []   # (in_tok, out_tok)

    def _work(rec: dict[str, Any]) -> dict[str, Any] | None:
        cve_id = rec["cve_id"]
        cve_desc = descriptions.get(cve_id, "")
        phase1_rec = phase1_by_cve.get(cve_id)
        if not phase1_rec:
            return None
        return process_one(
            main_rec=rec,
            phase1_rec=phase1_rec,
            cve_desc=cve_desc,
            repo2commits_root=Path(args.repo2commits),
            critic=critic,
            char_budget=args.char_budget,
            model=args.model,
        )

    with output_path.open("a") as out_f, ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as pool:
        futures = {pool.submit(_work, rec): rec for rec in todo}
        for fut in as_completed(futures):
            new_rec = fut.result()
            if new_rec is None:
                continue

            with write_lock:
                out_f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                out_f.flush()

            n_done += 1
            if new_rec.get("hit"):
                n_hit += 1
            if new_rec.get("llm_overrode"):
                n_override += 1
                if new_rec.get("hit"):
                    n_override_hit += 1
            in_tok += new_rec.get("llm_input_tokens", 0)
            out_tok += new_rec.get("llm_output_tokens", 0)
            total_cost += new_rec.get("llm_cost_usd", 0.0)
            per_cve_tokens.append(
                (new_rec.get("llm_input_tokens", 0),
                 new_rec.get("llm_output_tokens", 0))
            )
            # Confusion matrix (regression analysis input)
            mh, lh = new_rec.get("main_hit"), new_rec.get("hit")
            if mh and lh:
                cm["both_right"] += 1
            elif mh and not lh:
                cm["main_right_llm_broke"] += 1
            elif not mh and lh:
                cm["main_wrong_llm_fixed"] += 1
            else:
                cm["both_wrong"] += 1

            if n_done % 50 == 0 or n_done == len(todo):
                elapsed = time.time() - t_start
                rate = n_done / max(0.01, elapsed)
                eta_min = (len(todo) - n_done) / max(0.001, rate) / 60
                mark = "✓" if new_rec.get("hit") else "✗"
                print(
                    f"[{n_done:>5}/{len(todo)}] {mark} {new_rec['cve_id']:<18}"
                    f" override={new_rec.get('llm_overrode')}"
                    f" hit={new_rec.get('hit')}"
                    f" tok_in={in_tok / max(1, n_done):.0f}avg"
                    f" ({elapsed / 60:.1f}min, {rate * 60:.1f}/min, ETA {eta_min:.1f}min)",
                    flush=True,
                )

    elapsed = time.time() - t_start
    main_hits = cm["both_right"] + cm["main_right_llm_broke"]
    llm_hits = cm["both_right"] + cm["main_wrong_llm_fixed"]

    # Distribution stats for cost analysis
    import statistics
    in_toks_only = [t[0] for t in per_cve_tokens]
    in_med = statistics.median(in_toks_only) if in_toks_only else 0
    in_p95 = (
        statistics.quantiles(in_toks_only, n=20)[18]
        if len(in_toks_only) >= 20 else max(in_toks_only or [0])
    )

    summary = {
        "run_metadata": {
            "model": args.model,
            "char_budget": args.char_budget,
            "max_passage_words": args.max_passage_words,
            "max_workers": args.max_workers,
            "sample_size": args.sample_size,
            "seed": args.seed if args.sample_size > 0 else None,
            "main_jsonl": args.main_jsonl,
            "phase1_jsonl": args.phase1_jsonl,
            "output_jsonl": str(output_path),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_time_sec": round(elapsed, 1),
        },
        "metrics": {
            "n_processed": n_done,
            "main_hit_at_1": round(main_hits / max(1, n_done), 4),
            "llm_hit_at_1": round(llm_hits / max(1, n_done), 4),
            "net_change_hits": llm_hits - main_hits,
            "override_rate": round(n_override / max(1, n_done), 4),
            "override_precision": round(n_override_hit / max(1, n_override), 4) if n_override else 0,
            "confusion_matrix": cm,
        },
        "cost": {
            "model_prices_per_1m_tokens_usd": MODEL_PRICES.get(args.model, {}),
            "total_input_tokens": in_tok,
            "total_output_tokens": out_tok,
            "total_cost_usd": round(total_cost, 4),
            "cost_per_cve_usd": round(total_cost / max(1, n_done), 6),
            "input_tokens_per_cve": {
                "median": int(in_med),
                "p95": int(in_p95),
                "max": max(in_toks_only or [0]),
            },
        },
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(
        f"\n{'=' * 70}\n"
        f"Confusion matrix:\n"
        f"  both_right             {cm['both_right']:>5}\n"
        f"  main_right_llm_broke   {cm['main_right_llm_broke']:>5}\n"
        f"  main_wrong_llm_fixed   {cm['main_wrong_llm_fixed']:>5}\n"
        f"  both_wrong             {cm['both_wrong']:>5}\n"
        f"\n"
        f"Main Hit@1: {main_hits}/{n_done} = {main_hits/max(1,n_done):.1%}\n"
        f"LLM Hit@1:  {llm_hits}/{n_done} = {llm_hits/max(1,n_done):.1%}\n"
        f"Net change: {llm_hits - main_hits:+d}\n"
        f"\n"
        f"LLM override rate: {n_override}/{n_done} = {n_override/max(1,n_done):.1%}\n"
        f"LLM override precision: {n_override_hit}/{max(1,n_override)} = "
        f"{n_override_hit/max(1,n_override):.1%}\n"
        f"\n"
        f"Cost: ${total_cost:.4f} total = ${total_cost/max(1,n_done):.5f}/CVE  "
        f"({args.model})\n"
        f"Tokens/CVE input: median={int(in_med)}, p95={int(in_p95)}, max={max(in_toks_only or [0])}\n"
        f"Time: {elapsed/60:.1f}min ({elapsed/max(1,n_done):.2f}s/CVE)\n"
        f"\n"
        f"Per-CVE jsonl: {output_path}\n"
        f"Summary json:  {summary_path}\n"
        f"\n"
        f"Run the metrics script:\n"
        f"  python -m patchholmes.cli.eval_phase2 {output_path} "
        f"--phase1 {args.phase1_jsonl}"
    )


if __name__ == "__main__":
    main()
