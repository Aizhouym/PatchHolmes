"""Phase 2 — agentic loop on top of OpenHands software-agent-sdk.

Sub-modules:
    diff_render  — parse raw `git diff` into a structured form an LLM agent can read
    data_source  — wraps Phase 1 output + commit data for one CVE
    result       — Phase2Result dataclass written to phase2_*.jsonl
    actions      — Pydantic Action models (SDK)
    observations — Pydantic Observation models (SDK)
    executors    — ToolExecutor classes (SDK)
    tools        — ToolDefinition.create() factories (SDK)
    agent        — Agent + system prompt builder (SDK)
    runner       — orchestrates a Conversation per CVE
"""
