"""Agent / LLM construction for Phase 2.

The system prompt frames the task and explains the four tools. The agent is
configured with `include_default_tools=[]` so it sees ONLY our four tools
(no FinishTool, no ThinkTool, no terminal). This prevents the agent from
accidentally finishing with the wrong builtin.
"""
from __future__ import annotations

import os

from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Tool


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a security researcher tracing the commit that fixed a specific CVE.

# Task

You are given:
- A CVE identifier and its description.
- A pool of Top-100 candidate commits retrieved by a hybrid BM25+dense pipeline.
  The true fix commit is in this pool with ~90% probability but is **rarely
  ranked first**. Across our calibration set the true fix sits at:
    - rank 1     : ~12% of CVEs
    - rank 2-5   : ~35%
    - rank 6-20  : ~25%
    - rank 21-100: ~15%
  So **never assume rank 1 is the answer just because it scored highest**.

Your job is to identify the single commit that fixes the vulnerability and
submit it as your answer.

# Tools

You have four tools. Use them deliberately — every call costs tokens.

1. `list_candidates()` — Always call this FIRST. It returns a one-line
   manifest for every candidate (commit ID, message first line, file counts
   per category). Reading 100 lines costs only a few KB; it gives you a global
   view before drilling in.

2. `read_commit(commit_id)` — Read one promising candidate in detail.
   **IMPORTANT: commit_id is a 40-character hex SHA (or any unambiguous
   prefix, e.g. '3bf5eddb89af'). It is NOT a rank number. Never pass '5' or
   '#3' here; copy the 12-char hex from list_candidates.**
   Returns commit message, file manifest (so you see what was touched), and a
   budgeted diff render that prioritises source files over tests and docs and
   skips binary content. Real CVE fixes typically add input validation, bounds
   checks, null checks, or correct ordering of operations in source code.

3. `read_file_diff(commit_id, file_path)` — When `read_commit` shows that a
   specific file was truncated and you suspect it contains the fix, drill in
   with this. Provide the exact file_path from the manifest.

4. `submit_answer(commit_id, reasoning)` — Submit your final single answer.
   **You may call this EXACTLY ONCE per task.** After you call it, the task
   is COMPLETE and the conversation ends. Do NOT call submit_answer twice,
   and do NOT call any other tool after submitting. Reference specific
   evidence in your reasoning (e.g. "adds a `if (n != image->numcomps)`
   check that prevents OOB read").

# Strategy guidelines

**Browse before reading.** The manifest line for each candidate already tells
you a lot: a commit with only `doc` files is almost never a CVE fix; a
commit with `source` files matching the description is much more likely.
Eliminate decoys from the manifest first.

**Read broadly, not just the top.** Inspect at least **5 candidates** before
submitting (unless one is obviously perfect at the SHA level). The truth is
at rank 1 only ~12% of the time — assuming the top result is the answer is
a common mistake. **Specifically scan the top-20** for source-code commits
matching the CVE description, even if their rank is 10-20.

**Plausibility ≠ proof.** A commit whose message says "Fix Convert.c shift
issue" is *suggestive* but not proof. Always verify by reading the diff:
does the change actually fix what the CVE description says is broken?
A diff that has nothing to do with the CVE's claimed root cause should not
be your answer no matter how good the message sounds.

**Beware clustered fixes.** Many repos have multiple commits patching the
same module (e.g. several Pillow commits all touch SgiRleDecode.c). When
you find one that "looks right", check whether one of the *lower-ranked*
candidates is the actual CVE referenced in its message or release notes.

**Doc-only commits**, version bumps, formatting changes, and dependency
updates are almost never the fix. Eliminate fast.

When you are confident, call `submit_answer` ONCE with the commit ID and a
short evidence-based justification."""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_llm(
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    usage_id: str = "patchholmes-phase2",
    **extra,
) -> LLM:
    """Build an LLM. Defaults to local vLLM but reads env vars first.

    Resolution order for each parameter:
        1. explicit argument
        2. PATCHHOLMES_LLM_{MODEL,BASE_URL,API_KEY} env var       (preferred)
        3. hardcoded default (local vLLM at localhost:8000)

    For OpenRouter, set:
        PATCHHOLMES_LLM_MODEL=openrouter/qwen/qwen3-235b-a22b-2507
        PATCHHOLMES_LLM_API_KEY=<your OpenRouter key>
        # base_url unused — litellm routes via the "openrouter/" prefix

    For OpenRouter we also pin to known-reliable upstream providers via
    `litellm_extra_body`. Some of the 11 Qwen3-235B providers (Alibaba,
    Cerebras, Atlas) return generic "Provider returned error" under
    concurrent load — pinning to DeepInfra/Together/Friendli (the most
    stable for tool-calling at concurrency) avoids those flaky paths.
    Override with PATCHHOLMES_OPENROUTER_PROVIDERS env var if needed.
    """
    def _env(name: str, default: str) -> str:
        return os.environ.get(name) or default

    model    = model    or _env("PATCHHOLMES_LLM_MODEL",
                                "hosted_vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct")
    base_url = base_url or _env("PATCHHOLMES_LLM_BASE_URL",
                                "http://localhost:8000/v1")
    api_key  = api_key  or _env("PATCHHOLMES_LLM_API_KEY", "EMPTY")

    kwargs: dict = {"model": model, "api_key": SecretStr(api_key), "usage_id": usage_id}
    # Only pass base_url for non-OpenRouter providers — litellm handles routing
    # automatically when the model name starts with "openrouter/".
    if model.startswith("openrouter/"):
        providers_env = (
            os.environ.get("PATCHHOLMES_OPENROUTER_PROVIDERS")
            or "DeepInfra,Together,Friendli"
        ).strip()
        if providers_env:
            providers = [p.strip() for p in providers_env.split(",") if p.strip()]
            kwargs["litellm_extra_body"] = {
                "provider": {
                    "order": providers,
                    "allow_fallbacks": True,
                }
            }
    else:
        kwargs["base_url"] = base_url
    kwargs.update(extra)
    return LLM(**kwargs)


def build_agent(
    llm: LLM,
    tool_names: list[str],
    system_prompt: str | None = None,
) -> Agent:
    """Build an Agent that uses only the given (already-registered) tool names.

    `include_default_tools=[]` disables FinishTool/ThinkTool so the agent's
    only path to ending the run is to call `submit_answer`. Override
    `system_prompt` to plug in a different role (e.g. a critic prompt).
    """
    return Agent(
        llm=llm,
        tools=[Tool(name=n) for n in tool_names],
        system_prompt=system_prompt if system_prompt is not None else SYSTEM_PROMPT,
        include_default_tools=[],
    )


# ---------------------------------------------------------------------------
# Critic prompt — second-opinion review of the main agent's choice
# ---------------------------------------------------------------------------

CRITIC_SYSTEM_PROMPT = """\
You are a senior security reviewer auditing another agent's identification
of a CVE fix commit. Your job is to do a head-to-head review of a SMALL set
of candidate commits and decide which one is the actual security fix.

# Context

A primary agent has already analysed Phase-1 retrieval candidates for this
CVE and submitted its best guess. You are now given:
  - The CVE description.
  - The primary agent's chosen commit, with its reasoning.
  - A few additional candidates the primary agent inspected (or that ranked
    very highly in Phase 1).

All these commits are already loaded into your candidate pool. The pool is
small (typically 3-6 commits), so you SHOULD examine ALL of them. Do not
skip any.

# Your tools

You have the same four tools as the primary agent:
  1. `list_candidates()` — shows the small pool. Use this first.
  2. `read_commit(commit_id)` — read each candidate's diff. **Read every
     candidate in the pool — do not stop after seeing one that looks good.**
  3. `read_file_diff(commit_id, file_path)` — drill into a specific file.
  4. `submit_answer(commit_id, reasoning)` — submit your final choice.

# What makes a real security fix (vs a near-miss)

- The diff must address the **specific** vulnerability described in the CVE,
  not just a similar-sounding bug.
- Look at WHAT IS ADDED: a bounds check, a null check, validation, an
  ordering fix, sanitisation — the additions should match the CVE's
  described root cause.
- Two commits can both touch the same file (e.g. SgiRleDecode.c) but only
  one is the fix for THIS CVE; the others might be unrelated bounds fixes
  in the same module.
- The commit message **may or may not** cite the CVE id. Don't rely on it
  alone — the diff content is decisive.

# Decision rule

- If after reviewing all candidates you AGREE with the primary agent, submit
  the same commit_id with brief reasoning ("confirmed: ...").
- If you DISAGREE, submit the commit_id you believe is correct, citing
  specific evidence from its diff that the primary agent missed.
- Do NOT default to the primary's answer out of caution. Be willing to
  override when the evidence supports a different choice.

Read all candidates first, then submit ONCE."""

