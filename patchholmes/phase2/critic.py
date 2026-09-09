"""Critic-agent second-opinion pass over the main Phase 2 agent's choice.

Pattern
-------
After the main agent submits an answer, we run a **second** Phase-2-style
agent on a SMALL candidate pool (typically the main agent's chosen commit
plus the next few it actually inspected, plus the top-N from Phase 1).
The critic uses the same 4 tools but a different system prompt that
emphasises head-to-head comparison.

If the critic submits a DIFFERENT commit than the main agent, we treat the
critic's answer as the final output and record both in the trace.

Cost
----
Roughly +50% wall-clock and +25K tokens per CVE (one extra agent loop on a
3-6 commit pool). For the failure modes we identified on Pillow (agent
reads truth but picks a near-neighbour), this is well worth it.

Reference
---------
Inspired by `software-agent-sdk/examples/01_standalone_sdk/34_critic_example.py`
but reuses our existing 4-tool framework instead of `APIBasedCritic` (which
requires the All-Hands LLM proxy).
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from openhands.sdk import LLM

from patchholmes.data_models import CVEQuery, RankedCandidate
from patchholmes.phase2.agent import CRITIC_SYSTEM_PROMPT
from patchholmes.phase2.result import Phase2Result, ToolCallRecord
from patchholmes.phase2.runner import run_phase2_single


# How many extra non-chosen commits to include in the critic's review pool.
# Drawn from main_result.commits_inspected first, then top-N of Phase 1.
DEFAULT_REVIEW_EXTRA = 3
DEFAULT_CRITIC_MAX_ITER = 10  # smaller pool needs fewer iterations


def _build_critic_user_prompt(
    query: CVEQuery, main_answer: str, main_reasoning: str
) -> str:
    desc = (query.description or "").strip() or "(no description provided)"
    main_short = main_answer[:12] if main_answer else "<none>"
    return (
        f"CVE ID: {query.cve_id}\n"
        f"Repository: {query.owner}/{query.repo}\n"
        f"\n"
        f"CVE Description:\n{desc}\n"
        f"\n"
        f"# Primary agent's choice\n"
        f"commit_id: {main_short}\n"
        f"reasoning: {main_reasoning}\n"
        f"\n"
        f"# Your task\n"
        f"Call list_candidates() to see the small review pool. "
        f"Read EVERY candidate (each one matters). Compare them head-to-head, "
        f"focusing on whether the diff actually addresses the CVE described "
        f"above. Then call submit_answer ONCE — either confirming the "
        f"primary's choice or overriding it."
    )


def _resolve_to_full_sha(
    candidate_pool: list[RankedCandidate], partial_or_full: str
) -> str | None:
    """Match an input (full SHA or unambiguous prefix) against the candidate
    pool's commit_ids and return the canonical full SHA, or None."""
    if not partial_or_full:
        return None
    s = partial_or_full.strip().lower()
    # exact match first
    for c in candidate_pool:
        if c.commit.commit_id.lower() == s:
            return c.commit.commit_id
    # unique prefix match (either direction)
    matches = []
    for c in candidate_pool:
        cid = c.commit.commit_id.lower()
        if cid.startswith(s) or s.startswith(cid):
            matches.append(c.commit.commit_id)
    return matches[0] if len(matches) == 1 else None


def _build_review_pool(
    full_candidates: list[RankedCandidate],
    main_answer: str,
    inspected_ids: list[str],
    extra: int,
) -> list[RankedCandidate]:
    """Pick the small pool the critic sees.

    Composition (deduped, preserving Phase 1 rank order in output):
      1. The main agent's submitted answer (if it's a valid candidate).
      2. Up to `extra` commits the main agent inspected (other than the answer).
      3. If still < extra+1 total, top of Phase 1 to round out.
    """
    keep_ids: list[str] = []

    if main_answer:
        resolved = _resolve_to_full_sha(full_candidates, main_answer)
        if resolved:
            keep_ids.append(resolved)

    # add inspected (other than the answer); these are usually full SHAs already
    for cid in inspected_ids:
        resolved = _resolve_to_full_sha(full_candidates, cid)
        if resolved and resolved not in keep_ids:
            keep_ids.append(resolved)
            if len(keep_ids) >= 1 + extra:
                break

    # if still short, add top-of-Phase-1
    if len(keep_ids) < 1 + extra:
        for c in full_candidates:
            if c.commit.commit_id not in keep_ids:
                keep_ids.append(c.commit.commit_id)
                if len(keep_ids) >= 1 + extra:
                    break

    keep_set = set(keep_ids)
    # Preserve original Phase 1 rank order (lowest rank first), re-index 1..N
    review = [c for c in full_candidates if c.commit.commit_id in keep_set]
    review.sort(key=lambda c: c.rank)
    # Re-rank 1..N so the critic sees fresh ranks for its small pool.
    return [
        replace(c, rank=i + 1, source=c.source)
        for i, c in enumerate(review)
    ]


def run_phase2_with_critic(
    query: CVEQuery,
    phase1_candidates: list[RankedCandidate],
    repo2commits_root: str | Path,
    llm: LLM | None = None,
    *,
    top_k: int = 100,
    max_iteration_per_run: int = 15,
    critic_max_iteration: int = DEFAULT_CRITIC_MAX_ITER,
    critic_review_extra: int = DEFAULT_REVIEW_EXTRA,
    **kwargs: Any,
) -> Phase2Result:
    """Run the main Phase 2 agent, then a critic pass to confirm/override.

    Returns a single `Phase2Result`. If the critic overrides the main
    answer, fields `best_commit_id`, `reasoning`, `hit`,
    `phase1_rank_of_answer`, and `tool_calls` reflect the critic's choice
    (with the main agent's trace appended).
    """
    t0 = time.time()

    # ── Pass 1: main agent (top-100) ──────────────────────────────────────
    main = run_phase2_single(
        query=query,
        phase1_candidates=phase1_candidates,
        repo2commits_root=repo2commits_root,
        llm=llm,
        top_k=top_k,
        max_iteration_per_run=max_iteration_per_run,
        **kwargs,
    )

    # Short-circuits: no answer to critique, or fundamentally broken main pass.
    if not main.best_commit_id or main.error:
        return main

    # ── Pass 2: critic on a small review pool ─────────────────────────────
    review_candidates = _build_review_pool(
        full_candidates=phase1_candidates,
        main_answer=main.best_commit_id,
        inspected_ids=main.commits_inspected,
        extra=critic_review_extra,
    )
    if len(review_candidates) < 2:
        # nothing meaningful to review against
        return main

    critic_user_prompt = _build_critic_user_prompt(
        query=query,
        main_answer=main.best_commit_id,
        main_reasoning=main.reasoning,
    )
    critic = run_phase2_single(
        query=query,
        phase1_candidates=review_candidates,
        repo2commits_root=repo2commits_root,
        llm=llm,
        top_k=len(review_candidates),
        max_iteration_per_run=critic_max_iteration,
        system_prompt=CRITIC_SYSTEM_PROMPT,
        user_prompt=critic_user_prompt,
        **kwargs,
    )

    # ── Combine results ───────────────────────────────────────────────────
    wall = time.time() - t0

    # Stitch tool_calls: prefix critic turns with "critic." for clarity
    combined_calls = list(main.tool_calls)
    for tc in critic.tool_calls:
        combined_calls.append(
            ToolCallRecord(
                turn=len(combined_calls) + 1,
                tool=f"critic.{tc.tool}",
                args=tc.args,
            )
        )

    # Sum LLM usage across both passes
    in_tokens = main.llm_input_tokens + critic.llm_input_tokens
    out_tokens = main.llm_output_tokens + critic.llm_output_tokens
    cost = main.estimated_cost_usd + critic.estimated_cost_usd

    # Did the critic override?
    critic_picked = critic.best_commit_id
    if critic_picked and critic_picked != main.best_commit_id:
        # Override path
        fix_set = set(query.fix_commit_ids)
        hit = critic_picked in fix_set
        # rank lookup in the FULL Phase-1 candidate pool, not the reduced critic pool
        p1_rank = None
        for c in phase1_candidates:
            if c.commit.commit_id == critic_picked:
                p1_rank = c.rank
                break
        truth_ranks = [c.rank for c in phase1_candidates[:top_k]
                       if c.commit.commit_id in fix_set]
        truth_best = min(truth_ranks) if truth_ranks else None

        return Phase2Result(
            cve_id=main.cve_id,
            owner=main.owner,
            repo=main.repo,
            best_commit_id=critic_picked,
            reasoning=f"[critic override] {critic.reasoning}",
            fix_commit_ids=main.fix_commit_ids,
            hit=hit,
            phase1_rank_of_answer=p1_rank,
            best_rank_in_truth_set=truth_best,
            stopped_reason=f"critic_override:{critic.stopped_reason}",
            iterations_used=main.iterations_used + critic.iterations_used,
            commits_inspected=list(dict.fromkeys(
                main.commits_inspected + critic.commits_inspected
            )),
            tool_calls=combined_calls,
            llm_input_tokens=in_tokens,
            llm_output_tokens=out_tokens,
            estimated_cost_usd=cost,
            wall_time_sec=round(wall, 3),
            error=critic.error,
        )

    # Critic confirmed (or had no answer); keep main's choice, just record the trace
    return replace(
        main,
        reasoning=f"[critic confirmed] {main.reasoning}",
        iterations_used=main.iterations_used + critic.iterations_used,
        commits_inspected=list(dict.fromkeys(
            main.commits_inspected + critic.commits_inspected
        )),
        tool_calls=combined_calls,
        llm_input_tokens=in_tokens,
        llm_output_tokens=out_tokens,
        estimated_cost_usd=cost,
        wall_time_sec=round(wall, 3),
        stopped_reason=f"critic_confirmed:{main.stopped_reason}",
    )
