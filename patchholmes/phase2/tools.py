"""ToolDefinition factories for the 4 Phase 2 tools.

Strategy
--------
Each `ToolDefinition` subclass below has a stub `.create()` that is **never
called** in practice — we build the instance directly with
`build_phase2_tools(data_source)` and register that instance via
`register_tool(name, instance)`. The instance-based resolver
(`_resolver_from_instance` in the SDK registry) returns the same instance for
every conversation, which is exactly what we want for a single-CVE Phase 2
run.

Reason
------
Tools must carry a reference to the per-CVE `Phase2DataSource`. The SDK's
`Tool(name=..., params={...})` spec only accepts JSON-serializable params, so
we can't pass the data source through that channel. Registering the instance
directly side-steps the issue.

Per-CVE workflow (runner)
-------------------------
    data_source = Phase2DataSource(query, phase1_candidates, root)
    tools_dict = build_phase2_tools(data_source)
    for name, instance in tools_dict.items():
        register_tool(name, instance)  # overwrites previous CVE's registration
    agent = build_agent(llm, list(tools_dict.keys()))
    Conversation(agent, ...).run()
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

from openhands.sdk.tool import ToolAnnotations, ToolDefinition

from patchholmes.phase2.actions import (
    ListCandidatesAction,
    ReadCommitAction,
    ReadFileDiffAction,
    SubmitAnswerAction,
)
from patchholmes.phase2.data_source import Phase2DataSource
from patchholmes.phase2.executors import (
    ListCandidatesExecutor,
    ReadCommitExecutor,
    ReadFileDiffExecutor,
    SubmitAnswerExecutor,
)
from patchholmes.phase2.observations import (
    ListCandidatesObservation,
    ReadCommitObservation,
    ReadFileDiffObservation,
    SubmitAnswerObservation,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


# ---------------------------------------------------------------------------
# Tool descriptions (these go into the LLM's tool catalogue prompt)
# ---------------------------------------------------------------------------

_LIST_CANDIDATES_DESC = (
    "Return the full Top-K candidate manifest for this CVE — one line per "
    "candidate with rank, short commit ID, the first line of the commit "
    "message, and a per-tag file count (source / test / doc / fixture / "
    "config). Call this FIRST to see what candidates exist. Reading 100 "
    "lines costs only a few KB of tokens; it is much cheaper than "
    "blindly inspecting commits."
)

_READ_COMMIT_DESC = (
    "Read one candidate commit in detail. Returns the commit message, the "
    "full file manifest, and a budgeted diff render (default ~8000 chars). "
    "Source-code files are shown first; binary fixtures are listed but their "
    "content is skipped. If an interesting file is truncated, call "
    "read_file_diff to drill down. Use this on the 3–10 most promising "
    "candidates rather than all 100. "
    "**commit_id MUST be a hex SHA** (40 chars or any unambiguous prefix, "
    "e.g. '3bf5eddb89af' copied from the list_candidates manifest). "
    "DO NOT pass a rank number like '5' or '#3' — those are not valid "
    "commit IDs."
)

_READ_FILE_DIFF_DESC = (
    "Drill into a single file's diff inside a specific commit. Use this when "
    "read_commit truncated a file you suspect contains the fix. Returns up to "
    "~16000 chars; very long file diffs are compressed (context lines dropped, "
    "all +/- lines kept). Provide commit_id and file_path exactly as shown in "
    "the read_commit file manifest."
)

_SUBMIT_ANSWER_DESC = (
    "Submit your final answer. Provide the single commit ID (full SHA or "
    "unambiguous prefix) that you believe is the security fix for this CVE, "
    "plus a short reasoning that references concrete evidence (a function "
    "name added, a validation check inserted, etc.). "
    "**Call this AT MOST ONCE per task.** As soon as you call it, the task "
    "is recorded as complete; calling it again or calling any other tool "
    "afterwards will return an error. Make sure you have inspected enough "
    "commits before submitting — typically at least 5 candidates."
)


# ---------------------------------------------------------------------------
# ToolDefinition subclasses (stub .create() — we build instances directly)
# ---------------------------------------------------------------------------

def _no_create(name: str):
    """Generate a .create() classmethod that errors if called.

    We register pre-built instances, so .create() should never run. If it does,
    that means someone tried to use the bare class via the registry without
    first calling `register_tool(name, instance)`.
    """
    def create(cls, conv_state: "ConversationState | None" = None, **params) -> Sequence[Self]:  # noqa: ARG001
        raise RuntimeError(
            f"{name} must be registered as a pre-built instance via "
            f"build_phase2_tools(data_source). Bare-class registration is not "
            f"supported because the executor needs a Phase2DataSource reference."
        )
    return classmethod(create)


class ListCandidatesTool(ToolDefinition[ListCandidatesAction, ListCandidatesObservation]):
    create = _no_create("ListCandidatesTool")


class ReadCommitTool(ToolDefinition[ReadCommitAction, ReadCommitObservation]):
    create = _no_create("ReadCommitTool")


class ReadFileDiffTool(ToolDefinition[ReadFileDiffAction, ReadFileDiffObservation]):
    create = _no_create("ReadFileDiffTool")


class SubmitAnswerTool(ToolDefinition[SubmitAnswerAction, SubmitAnswerObservation]):
    create = _no_create("SubmitAnswerTool")


# ---------------------------------------------------------------------------
# Per-CVE tool builder
# ---------------------------------------------------------------------------

_READ_ONLY_ANNOTATIONS = ToolAnnotations(
    title=None,
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)

_SUBMIT_ANNOTATIONS = ToolAnnotations(
    title=None,
    readOnlyHint=True,  # doesn't modify external state, just records the answer
    destructiveHint=False,
    idempotentHint=False,  # calling twice "resubmits"
    openWorldHint=False,
)


def build_phase2_tools(
    data_source: Phase2DataSource,
    *,
    read_commit_budget: int = 8000,
    read_file_diff_budget: int = 16000,
) -> dict[str, ToolDefinition]:
    """Construct the 4 Phase 2 tool instances, each bound to `data_source`.

    Returns a name → ToolDefinition mapping. The caller passes each value to
    `register_tool(name, instance)` and then references the names in the
    Agent's `tools=[Tool(name=...), ...]` list.
    """
    tools: dict[str, ToolDefinition] = {}

    tools[ListCandidatesTool.name] = ListCandidatesTool(
        description=_LIST_CANDIDATES_DESC,
        action_type=ListCandidatesAction,
        observation_type=ListCandidatesObservation,
        executor=ListCandidatesExecutor(data_source),
        annotations=_READ_ONLY_ANNOTATIONS,
    )

    tools[ReadCommitTool.name] = ReadCommitTool(
        description=_READ_COMMIT_DESC,
        action_type=ReadCommitAction,
        observation_type=ReadCommitObservation,
        executor=ReadCommitExecutor(data_source, char_budget=read_commit_budget),
        annotations=_READ_ONLY_ANNOTATIONS,
    )

    tools[ReadFileDiffTool.name] = ReadFileDiffTool(
        description=_READ_FILE_DIFF_DESC,
        action_type=ReadFileDiffAction,
        observation_type=ReadFileDiffObservation,
        executor=ReadFileDiffExecutor(data_source, char_budget=read_file_diff_budget),
        annotations=_READ_ONLY_ANNOTATIONS,
    )

    tools[SubmitAnswerTool.name] = SubmitAnswerTool(
        description=_SUBMIT_ANSWER_DESC,
        action_type=SubmitAnswerAction,
        observation_type=SubmitAnswerObservation,
        executor=SubmitAnswerExecutor(data_source),
        annotations=_SUBMIT_ANNOTATIONS,
    )

    return tools
