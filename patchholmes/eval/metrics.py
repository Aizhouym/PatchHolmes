"""Reusable IR metric computation for the sample_810 evaluation set.

The eval reports per-CVE ranking metrics:
    R@1, R@3, R@5, R@10
    NDCG@1, NDCG@3, NDCG@5, NDCG@10
    MRR

Design
------
Each "method" (PatchHolmes agent, IRCoT, Favia per-pair, ...) is plugged in via a
small adapter function that takes the method's raw record(s) for a given CVE
and returns an ordered list of commit_ids (most-likely-fix first). The metric
computation then operates on this ranking against the ground-truth `patch`
column from sample_ground_truth_810.csv.

Adding a new method
-------------------
1. Write a `rank_<method>(record_or_records)` function in this file or your own
   module — it must return `list[str]` of commit_ids ranked by the method's
   preference, best first.
2. Register it in `METHODS` (or pass it directly to `compute_metrics`).

Single-pick caveat
------------------
Methods that only output one commit (IRCoT, naive baselines) will see
R@1 = R@3 = R@5 = R@10 = MRR = NDCG@K — this is correct, not a bug.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

# ──────────────────────────────────────────────────────────────────────────────
# Ground truth loader
# ──────────────────────────────────────────────────────────────────────────────


def load_ground_truth(
    csv_path: str | Path = "./data/sample_ground_truth_810.csv",
    augment_from_per_pair: str | Path | None = "./data/patchfinder_top10/sample839_final.jsonl",
) -> dict[str, set[str]]:
    """Load the sample CSV. Returns {cve_id: set_of_truth_commit_ids}.

    Many CVEs have a single canonical patch (CSV column), but some have
    multiple fix commits (especially in PatchFinder candidates that contain
    duplicated rows). To match the per-pair eval semantics, we union:
        - CSV's `patch` column (always present)
        - Per-pair file's commit_id where label=1 (if path provided)

    Pass `augment_from_per_pair=None` to use only the CSV (strict single-truth).
    """
    truth: dict[str, set[str]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            truth.setdefault(row["cve"], set()).add(row["patch"])

    if augment_from_per_pair and Path(augment_from_per_pair).exists():
        with open(augment_from_per_pair) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                cve = r.get("input", {}).get("cve")
                if cve and int(r.get("input", {}).get("label", 0)) == 1:
                    truth.setdefault(cve, set()).add(r["input"]["commit_id"])

    return truth


# ──────────────────────────────────────────────────────────────────────────────
# Method adapters: record(s) → ordered list of commit_ids
# ──────────────────────────────────────────────────────────────────────────────


def rank_patchholmes(record: dict) -> list[str]:
    """PatchHolmes Phase 2 agent output.

    Ranking = [best_commit_id] + commits_inspected (deduped, best first).
    Median length 5 commits — gives R@K growth with K but caps around K=10.

    For "ranking past agent's actual inspection" (e.g., to match Favia's
    fixed-10 candidate metric), use `rank_patchholmes_with_phase1` instead.

    Expected record shape:
        {"cve_id": str, "best_commit_id": str, "commits_inspected": list[str], ...}
    """
    best = record.get("best_commit_id")
    inspected = record.get("commits_inspected") or []
    if best:
        rest = [c for c in inspected if c != best]
        return [best] + rest
    return list(inspected)


def rank_patchholmes_with_phase1(
    record: dict,
    phase1_candidates: list[str] | None = None,
    top_k: int | None = 10,
) -> list[str]:
    """Same as `rank_patchholmes` but extends past agent's inspection with the
    Phase 1 retriever's ordering for commits the agent didn't actively read.

    The final ranking is the UNION of:
      1. agent's [best_commit_id, ...commits_inspected...]   (in this order)
      2. Phase 1 candidates that the agent didn't read       (in retriever rank order)
    truncated to `top_k` items (default 10, matching Favia's per-pair output).

    Useful for fair R@K comparison vs methods that score a fixed set of
    candidates (e.g., Favia ranks all 10 PatchFinder candidates).

    Parameters
    ----------
    record
        A single Phase 2 result record.
    phase1_candidates
        Ordered list of Phase 1 retrieval commit_ids (best first). If None,
        falls back to `rank_patchholmes(record)` with no extension.
    top_k
        Truncate the ranking to this length. Default 10 (fair vs Favia top-10).
        Pass `None` to disable truncation (full union, up to 100 if Phase 1
        ran with top-100).
    """
    agent_part = rank_patchholmes(record)
    if phase1_candidates is None:
        return agent_part if top_k is None else agent_part[:top_k]
    seen = set(agent_part)
    extension = [c for c in phase1_candidates if c not in seen]
    full = agent_part + extension
    return full if top_k is None else full[:top_k]


def rank_ircot(record: dict) -> list[str]:
    """IRCoT (FlashRAG batch_ircot_all.py) output — single pick only.

    Expected record shape:
        {"cve_id": str, "pred_commit_id": str, ...}
    """
    pred = record.get("pred_commit_id")
    return [pred] if pred else []


def rank_favia_per_pair(records: list[dict]) -> list[str]:
    """Favia-style per-pair binary output, ranked by (answer, confidence).

    Each per-pair record:
        {"input": {"commit_id": str, ...}, "output": {"answer": bool, "confidence": int}}

    Returns the 10 candidate commit_ids ranked: answer=True first (sorted by
    confidence DESC), then answer=False (sorted by confidence ASC, since
    higher confidence in 'no' = more confident it's not the fix).
    """
    def score(r: dict) -> tuple[int, float]:
        out = r.get("output", {}) or {}
        ans = out.get("answer")
        conf = float(out.get("confidence") or 0)
        if ans is True:
            return (1, conf)
        if ans is False:
            return (0, -conf)
        return (-1, 0.0)

    sorted_recs = sorted(records, key=score, reverse=True)
    return [r["input"]["commit_id"] for r in sorted_recs]


# Registry of known methods. Adding new ones: append here.
METHODS: dict[str, Callable] = {
    "patchholmes": rank_patchholmes,
    "ircot": rank_ircot,
    "favia": rank_favia_per_pair,
}


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────────


def _ndcg_at_k(ranked_labels: list[int], k: int) -> float:
    """Binary-relevance NDCG@K. ranked_labels = [0,1,0,...] in ranking order."""
    dcg = sum(
        (2 ** lbl - 1) / math.log2(i + 2)
        for i, lbl in enumerate(ranked_labels[:k])
    )
    ideal = sorted(ranked_labels, reverse=True)[:k]
    idcg = sum((2 ** lbl - 1) / math.log2(i + 2) for i, lbl in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _truth_rank(ranking: list[str], truth: set[str]) -> int | None:
    """1-indexed rank of the FIRST truth commit in `ranking`, or None if none found.

    `truth` is a set of equally-valid fix commit IDs — common when a CVE has
    multiple fix commits or when PatchFinder candidates duplicate the same fix
    across multiple ranks. We score by the earliest hit (best rank).
    """
    try:
        return next(i + 1 for i, c in enumerate(ranking) if c in truth)
    except StopIteration:
        return None


def compute_metrics(
    rankings: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    eval_cves: Iterable[str] | None = None,
    ks: Iterable[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Compute R@K, NDCG@K, MRR.

    Parameters
    ----------
    rankings
        {cve_id: ranked_commit_ids}
    ground_truth
        {cve_id: truth_commit_id}
    eval_cves
        Subset of CVE IDs to evaluate over. If None, uses keys of
        `rankings ∩ ground_truth` — equivalent to "method's coverage".
        Pass `ground_truth.keys()` for "full sample" denominator
        (CVEs missing from method's output count as missed).
    ks
        Cutoffs for R@K and NDCG@K.

    Returns
    -------
    {
        "n_eval":      int,
        "R@1":         float, ..., "R@10": float,
        "NDCG@1":      float, ..., "NDCG@10": float,
        "MRR":         float,
    }

    Notes
    -----
    For single-pick methods, R@K and NDCG@K collapse to R@1 for all K.
    This is correct behavior, not a bug — single output has no rank ordering.
    """
    if eval_cves is None:
        eval_cves = set(rankings) & set(ground_truth)
    eval_cves = list(eval_cves)
    n = len(eval_cves)
    ks = list(ks)

    R = {k: 0 for k in ks}
    N = {k: 0.0 for k in ks}
    mrr = 0.0

    for cve in eval_cves:
        truth = ground_truth.get(cve)
        if not truth:
            continue
        ranking = rankings.get(cve, [])
        rank = _truth_rank(ranking, truth)
        # Labels: 1 if c is any of the truth commits, 0 otherwise
        labels = [1 if c in truth else 0 for c in ranking]

        for k in ks:
            if rank is not None and rank <= k:
                R[k] += 1
            N[k] += _ndcg_at_k(labels, k)

        if rank is not None:
            mrr += 1.0 / rank

    out: dict[str, float] = {"n_eval": n}
    for k in ks:
        out[f"R@{k}"] = R[k] / n if n else 0.0
        out[f"NDCG@{k}"] = N[k] / n if n else 0.0
    out["MRR"] = mrr / n if n else 0.0
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: load a jsonl into {cve_id: record}
# ──────────────────────────────────────────────────────────────────────────────


def load_jsonl_by_cve(*paths: str | Path) -> dict[str, dict]:
    """Load one or more jsonl files. Returns {cve_id: record}.

    Skips blank/malformed lines and rows without a `cve_id` key.
    If the same CVE appears in multiple files, the last one wins.
    """
    out: dict[str, dict] = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                cve = r.get("cve_id")
                if cve is None:
                    continue
                out[cve] = r
    return out


def load_favia_per_pair(path: str | Path) -> dict[str, list[dict]]:
    """Load Favia-style per-pair jsonl. Returns {cve_id: [10 pair records]}."""
    by_cve: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            cve = r.get("input", {}).get("cve")
            if cve is None:
                continue
            by_cve[cve].append(r)
    return dict(by_cve)
