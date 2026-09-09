"""Pydantic Action models — input schemas for the 4 Phase 2 tools.

Each Action subclasses `openhands.sdk.tool.Action` (which is a Pydantic Schema
with `extra="forbid", frozen=True`). The `Field(description=...)` strings are
what the LLM sees when deciding how to call each tool, so they must be clear.
"""
from __future__ import annotations

from pydantic import Field

from openhands.sdk.tool import Action


class ListCandidatesAction(Action):
    """No arguments — returns the full Top-K candidate manifest."""


class ReadCommitAction(Action):
    commit_id: str = Field(
        description=(
            "Commit ID of the candidate to read. Accepts the full 40-char SHA "
            "or any unambiguous prefix (e.g. 12-char short SHA from list_candidates)."
        ),
    )


class ReadFileDiffAction(Action):
    commit_id: str = Field(
        description="Commit ID (full SHA or unambiguous prefix).",
    )
    file_path: str = Field(
        description=(
            "Path of the file inside the commit, exactly as shown in the "
            "file manifest of read_commit, e.g. 'src/libImaging/Jpeg2KDecode.c'."
        ),
    )


class SubmitAnswerAction(Action):
    commit_id: str = Field(
        description=(
            "The single commit ID you are submitting as the fix for this CVE. "
            "Must be one of the candidates in the Top-K list."
        ),
    )
    reasoning: str = Field(
        description=(
            "A short explanation of why this commit fixes the CVE. Reference "
            "specific code in the diff (function names, added checks, etc.) "
            "to justify the choice."
        ),
    )
