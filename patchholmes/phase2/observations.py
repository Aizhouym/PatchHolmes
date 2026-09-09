"""Observation classes — output of each Phase 2 tool.

We don't need custom fields beyond the base `Observation`: the executors
construct each instance via `<Cls>.from_text(text)`, which wraps the string in
the base class's `content: list[TextContent | ImageContent]` and lets
`to_llm_content` handle the rest.

Subclasses exist mainly so the tool definition can declare a stable
`observation_type` (which the SDK uses for serialization and schema export).
"""
from __future__ import annotations

from openhands.sdk.tool import Observation


class ListCandidatesObservation(Observation):
    """Manifest of every candidate the agent can inspect."""


class ReadCommitObservation(Observation):
    """Full commit details (msg + file manifest + diff render under budget)."""


class ReadFileDiffObservation(Observation):
    """Full diff for a single file inside a commit."""


class SubmitAnswerObservation(Observation):
    """Confirmation that the agent's answer has been recorded."""
