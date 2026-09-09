"""ToolExecutor subclasses — the actual work behind each Phase 2 tool.

Each executor holds a reference to a `Phase2DataSource` (the per-CVE data
snapshot). When the agent calls a tool, the SDK invokes the executor's
`__call__(action, conversation=None)`.

The executors are intentionally thin: all logic lives in `Phase2DataSource`,
so they can be unit-tested without spinning up the SDK.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from openhands.sdk.tool import ToolExecutor

from patchholmes.phase2.actions import (
    ListCandidatesAction,
    ReadCommitAction,
    ReadFileDiffAction,
    SubmitAnswerAction,
)
from patchholmes.phase2.data_source import Phase2DataSource
from patchholmes.phase2.observations import (
    ListCandidatesObservation,
    ReadCommitObservation,
    ReadFileDiffObservation,
    SubmitAnswerObservation,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import LocalConversation


class ListCandidatesExecutor(ToolExecutor[ListCandidatesAction, ListCandidatesObservation]):
    def __init__(self, data_source: Phase2DataSource) -> None:
        self.ds = data_source

    def __call__(
        self,
        action: ListCandidatesAction,  # noqa: ARG002
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> ListCandidatesObservation:
        text = self.ds.list_candidates()
        return ListCandidatesObservation.from_text(text=text)


class ReadCommitExecutor(ToolExecutor[ReadCommitAction, ReadCommitObservation]):
    def __init__(self, data_source: Phase2DataSource, char_budget: int = 8000) -> None:
        self.ds = data_source
        self.char_budget = char_budget

    def __call__(
        self,
        action: ReadCommitAction,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> ReadCommitObservation:
        text = self.ds.render_commit(action.commit_id, char_budget=self.char_budget)
        is_error = text.startswith("[Error]")
        return ReadCommitObservation.from_text(text=text, is_error=is_error)


class ReadFileDiffExecutor(ToolExecutor[ReadFileDiffAction, ReadFileDiffObservation]):
    def __init__(self, data_source: Phase2DataSource, char_budget: int = 16000) -> None:
        self.ds = data_source
        self.char_budget = char_budget

    def __call__(
        self,
        action: ReadFileDiffAction,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> ReadFileDiffObservation:
        text = self.ds.render_file_diff(
            action.commit_id, action.file_path, char_budget=self.char_budget
        )
        is_error = text.startswith("[Error]")
        return ReadFileDiffObservation.from_text(text=text, is_error=is_error)


class SubmitAnswerExecutor(ToolExecutor[SubmitAnswerAction, SubmitAnswerObservation]):
    def __init__(self, data_source: Phase2DataSource) -> None:
        self.ds = data_source

    def __call__(
        self,
        action: SubmitAnswerAction,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> SubmitAnswerObservation:
        text = self.ds.submit_answer(action.commit_id, action.reasoning)
        return SubmitAnswerObservation.from_text(text=text)
