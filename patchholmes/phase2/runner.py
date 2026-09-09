"""Orchestrate one Phase 2 agent run per CVE.

`run_phase2_single` is the single-CVE entry point. It:
    1. Builds a Phase2DataSource snapshot of the CVE's Top-K candidates.
    2. Builds the 4 Phase 2 tool instances and registers them with the SDK
       registry (overwriting any registration from a previous CVE).
    3. Builds the LLM and Agent.
    4. Constructs a Conversation with our event callback, sends the user
       prompt, and runs it to completion (or max_iter / stuck).
    5. Extracts the answer, trace, token usage, and cost; returns a
       Phase2Result.
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from openhands.sdk import Conversation, LLM
from openhands.sdk.event import Event
from openhands.sdk.event.llm_convertible import ActionEvent
from openhands.sdk.tool import register_tool

from patchholmes.data_models import CVEQuery, Phase1Result, RankedCandidate
from patchholmes.phase2.agent import build_agent, build_llm
from patchholmes.phase2.data_source import Phase2DataSource
from patchholmes.phase2.result import Phase2Result, ToolCallRecord
from patchholmes.phase2.tools import build_phase2_tools


# ---------------------------------------------------------------------------

def run_phase2_single(
    query: CVEQuery,
    phase1_candidates: list[RankedCandidate],
    repo2commits_root: str | Path,
    llm: LLM | None = None,
    *,
    top_k: int = 100,
    max_iteration_per_run: int = 15,
    read_commit_budget: int = 8000,
    read_file_diff_budget: int = 16000,
    workspace_dir: str | Path | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    data_source: Phase2DataSource | None = None,
    tool_names: list[str] | None = None,
) -> Phase2Result:
    """Run the Phase 2 agent on one CVE and return a Phase2Result.

    Parameters
    ----------
    query
        The CVE query (description, owner, repo, fix_commit_ids).
    phase1_candidates
        Ranked Top-K candidates from Phase 1 (already sorted by rank).
    repo2commits_root
        Root path containing `split_<owner>@@<repo>/*.json`.
    llm
        Pre-built LLM. If None, builds the default local-vLLM Qwen3-Coder.
    top_k
        Number of candidates exposed to the agent (default 100).
    max_iteration_per_run
        Hard cap on agent tool-call turns.
    read_commit_budget, read_file_diff_budget
        Char budgets for the diff renderers.
    workspace_dir
        Workspace for the Conversation. Defaults to a temp dir; the tools
        don't write files there.
    tool_names
        Restrict the agent to a SUBSET of the registered tools (used by the
        tool-interface ablation). None (default) offers all four tools.
    """
    t0 = time.time()

    # ── 1. Data snapshot ───────────────────────────────────────────────────
    # Callers running on alternative data shapes (e.g. PatchFinder_top10
    # parquet) can pre-build a Phase2DataSource subclass and pass it via
    # `data_source` to skip the default disk-backed loader.
    if data_source is not None:
        ds = data_source
    else:
        ds = Phase2DataSource(
            query=query,
            phase1_candidates=phase1_candidates,
            repo2commits_root=repo2commits_root,
            top_k=top_k,
        )

    # ── 2. Tools (register fresh instances bound to this CVE's ds) ─────────
    tools_dict = build_phase2_tools(
        ds,
        read_commit_budget=read_commit_budget,
        read_file_diff_budget=read_file_diff_budget,
    )
    for name, tool in tools_dict.items():
        register_tool(name, tool)

    # ── 3. LLM + Agent ─────────────────────────────────────────────────────
    # `tool_names` restricts the agent to a SUBSET of the registered tools
    # (used by the tool-interface ablation). All tools stay registered so the
    # data source still works; the agent is simply not offered the excluded
    # ones. None = offer all tools (default behaviour).
    if llm is None:
        llm = build_llm()
    selected = (
        list(tools_dict.keys())
        if tool_names is None
        else [n for n in tools_dict if n in tool_names]
    )
    agent = build_agent(llm, selected, system_prompt=system_prompt)

    # ── 4. Event capture ───────────────────────────────────────────────────
    captured_events: list[Event] = []

    def event_callback(event: Event) -> None:
        captured_events.append(event)

    # Throwaway workspace — tools don't write to it. We delete it at the end
    # so 8401-CVE runs don't leave thousands of empty dirs under /tmp.
    auto_workspace = workspace_dir is None
    workspace = Path(workspace_dir) if workspace_dir else Path(tempfile.mkdtemp(prefix="phase2_ws_"))

    # Snapshot LLM cumulative usage so we can compute per-CVE delta.
    prompt_tokens_before = 0
    completion_tokens_before = 0
    cost_before = 0.0
    if llm is not None and getattr(llm, "metrics", None) is not None:
        acc = getattr(llm.metrics, "accumulated_token_usage", None)
        if acc is not None:
            prompt_tokens_before = int(getattr(acc, "prompt_tokens", 0) or 0)
            completion_tokens_before = int(getattr(acc, "completion_tokens", 0) or 0)
        cost_before = float(getattr(llm.metrics, "accumulated_cost", 0.0) or 0.0)

    error_msg: str | None = None
    try:
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[event_callback],
            max_iteration_per_run=max_iteration_per_run,
            stuck_detection=True,
            visualizer=None,  # we capture via callback; no console rendering
        )

        # ── 5. Run ─────────────────────────────────────────────────────────
        prompt = user_prompt if user_prompt is not None else _build_user_prompt(query)
        try:
            conversation.send_message(prompt)
            conversation.run()
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
    finally:
        if auto_workspace:
            shutil.rmtree(workspace, ignore_errors=True)

    wall = time.time() - t0

    # ── 6. Extract trace / answer / cost ───────────────────────────────────
    answer = ds.get_answer()
    best_cid = answer.commit_id if answer else None
    reasoning = answer.reasoning if answer else ""

    tool_calls = _extract_tool_calls(captured_events)
    iterations_used = len(tool_calls)

    if error_msg is not None:
        stopped_reason = "error"
    elif answer is not None:
        stopped_reason = "submit_answer"
    elif iterations_used >= max_iteration_per_run:
        stopped_reason = "max_iterations"
    else:
        # Could be stuck detector or natural end without submit
        stopped_reason = "no_answer"

    fix_set = set(query.fix_commit_ids)
    hit = bool(best_cid and best_cid in fix_set)
    phase1_rank_of_answer = ds.phase1_rank_of(best_cid) if best_cid else None
    best_rank_in_truth = ds.best_rank_in_truth_set()

    # Compute per-CVE token / cost delta against the snapshot taken before run.
    prompt_tokens = 0
    completion_tokens = 0
    cost = 0.0
    metrics = getattr(llm, "metrics", None)
    if metrics is not None:
        acc = getattr(metrics, "accumulated_token_usage", None)
        if acc is not None:
            prompt_tokens = max(0, int(getattr(acc, "prompt_tokens", 0) or 0) - prompt_tokens_before)
            completion_tokens = max(0, int(getattr(acc, "completion_tokens", 0) or 0) - completion_tokens_before)
        cost = max(0.0, float(getattr(metrics, "accumulated_cost", 0.0) or 0.0) - cost_before)

    return Phase2Result(
        cve_id=query.cve_id,
        owner=query.owner,
        repo=query.repo,
        best_commit_id=best_cid,
        reasoning=reasoning,
        fix_commit_ids=list(query.fix_commit_ids),
        hit=hit,
        phase1_rank_of_answer=phase1_rank_of_answer,
        best_rank_in_truth_set=best_rank_in_truth,
        stopped_reason=stopped_reason,
        iterations_used=iterations_used,
        commits_inspected=ds.get_inspected_commits(),
        tool_calls=tool_calls,
        llm_input_tokens=prompt_tokens,
        llm_output_tokens=completion_tokens,
        estimated_cost_usd=cost,
        wall_time_sec=round(wall, 3),
        error=error_msg,
    )


# ---------------------------------------------------------------------------

def run_phase2_from_phase1_result(
    phase1_result: Phase1Result,
    repo2commits_root: str | Path,
    llm: LLM | None = None,
    **kwargs: Any,
) -> Phase2Result:
    """Convenience wrapper when you already have a Phase1Result on hand."""
    return run_phase2_single(
        query=phase1_result.query,
        phase1_candidates=phase1_result.candidates,
        repo2commits_root=repo2commits_root,
        llm=llm,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _build_user_prompt(query: CVEQuery) -> str:
    desc = (query.description or "").strip() or "(no description provided)"
    return (
        f"CVE ID: {query.cve_id}\n"
        f"Repository: {query.owner}/{query.repo}\n"
        f"\n"
        f"CVE Description:\n{desc}\n"
        f"\n"
        f"Identify the single commit from the candidate pool that fixes this "
        f"vulnerability. Call list_candidates first, read promising commits "
        f"with read_commit (drilling in with read_file_diff if needed), then "
        f"call submit_answer with your final choice and a short reasoning."
    )


def _extract_tool_calls(events: list[Event]) -> list[ToolCallRecord]:
    records: list[ToolCallRecord] = []
    turn = 0
    for ev in events:
        if isinstance(ev, ActionEvent):
            turn += 1
            args: dict[str, Any] = {}
            if ev.action is not None:
                try:
                    args = ev.action.model_dump(mode="python")
                except Exception:
                    args = {}
            records.append(
                ToolCallRecord(turn=turn, tool=ev.tool_name, args=args)
            )
    return records
