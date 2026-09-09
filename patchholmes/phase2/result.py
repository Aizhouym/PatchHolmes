"""Phase 2 result dataclass — what we write to phase2_*.jsonl.

One record per CVE. The schema captures:
    1. The agent's answer (best_commit_id + reasoning).
    2. Evaluation signals (hit, Phase 1 rank of the answer, ground truth set).
    3. Agent behaviour trace (stopped_reason, tool_calls, commits_inspected).
    4. Cost / latency (tokens, USD, wall time).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    turn: int
    tool: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Phase2Result:
    # ── Identity ───────────────────────────────────────────────────────────
    cve_id: str
    owner: str
    repo: str

    # ── Final answer ───────────────────────────────────────────────────────
    best_commit_id: str | None
    reasoning: str

    # ── Evaluation signals ─────────────────────────────────────────────────
    fix_commit_ids: list[str] = field(default_factory=list)
    hit: bool = False
    phase1_rank_of_answer: int | None = None     # rank of best_commit_id in Phase 1 Top-K
    best_rank_in_truth_set: int | None = None    # best Phase 1 rank among ground-truth commits

    # ── Agent behaviour ────────────────────────────────────────────────────
    stopped_reason: str = "unknown"              # "submit_answer" | "max_iterations" | "stuck" | "error"
    iterations_used: int = 0
    commits_inspected: list[str] = field(default_factory=list)  # commit_ids the agent actually read
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    # ── Cost / latency ─────────────────────────────────────────────────────
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    wall_time_sec: float = 0.0

    # ── Optional error message ─────────────────────────────────────────────
    error: str | None = None

    # ----------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict() already converts ToolCallRecord to dict via dataclass recursion.
        return d

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Phase2Result":
        tool_calls = [ToolCallRecord(**tc) for tc in d.get("tool_calls", [])]
        d2 = {**d, "tool_calls": tool_calls}
        return cls(**d2)
