"""LLM listwise rerank over the main agent's `commits_inspected`.

Thin wrapper around `rank_llm.Reranker` (built on `SafeOpenai`). No custom
reranker class, no custom prompt-rendering — both are required by
`CLAUDE.md` ("Phase 2 LLM Reranker" section). All ranking logic, retry,
window sliding, and output parsing comes from rank_llm.

Usage
-----
>>> critic = LLMCritic(model="gpt-4o-mini")
>>> result = critic.rerank(
...     cve_id="CVE-2021-25287",
...     cve_description="Pillow OOB read in Jpeg2KDecode ...",
...     candidates=[(commit_id, rendered_text), ...],
...     main_commit_id="abc123...",
... )
>>> result.chosen_commit_id, result.overrode_main
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from rank_llm.data import Candidate, Query, Request
from rank_llm.rerank.reranker import Reranker

DEFAULT_TEMPLATE = Path(__file__).parent / "prompts" / "patch_tracing.yaml"


@dataclass
class LLMRerankResult:
    chosen_commit_id: str
    ranking: list[str]                  # full reranked order, top-1 first
    main_commit_id: str | None
    overrode_main: bool
    input_tokens: int = 0
    output_tokens: int = 0
    raw_response: str = ""
    scores_by_id: dict[str, float] = field(default_factory=dict)


class LLMCritic:
    """Listwise GPT reranker over a small (<=20) pool of candidate commits.

    Defaults are tuned for the Phase 2 inspect-list use case:
      - 4-11 commits per CVE
      - rendered text per commit ~8000 chars (~2000 words)
      - GPT-4o family with 128K context
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        prompt_template_path: str | Path | None = None,
        context_size: int = 128_000,
        max_passage_words: int = 3000,
        window_size: int = 20,
        stride: int = 20,
        batch_size: int = 1,
        api_key: str | None = None,
    ) -> None:
        # rank_llm reads the key from env (OPENAI_API_KEY / OPEN_AI_API_KEY)
        # OR from a `.env.local` file in the cwd. Honour an explicit arg.
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        # Defer the no-key check to rank_llm — it knows about dotenv.
        from rank_llm.rerank.api_keys import get_openai_api_key

        if not get_openai_api_key():
            raise RuntimeError(
                "No OpenAI API key found. Set OPENAI_API_KEY env var, "
                "put it in .env.local, or pass api_key=..."
            )

        template_path = Path(prompt_template_path or DEFAULT_TEMPLATE)
        if not template_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {template_path}. Ship "
                "patchholmes/phase2/prompts/patch_tracing.yaml (it is declared as "
                "package data in pyproject.toml) or pass prompt_template_path=..."
            )

        coord = Reranker.create_model_coordinator(
            model_path=model,
            default_model_coordinator=None,
            interactive=False,
            context_size=context_size,
            prompt_template_path=template_path,
            max_passage_words=max_passage_words,
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
        )
        self.reranker = Reranker(model_coordinator=coord)
        self.model = model

    def rerank(
        self,
        cve_id: str,
        cve_description: str,
        candidates: list[tuple[str, str]],
        main_commit_id: str | None = None,
    ) -> LLMRerankResult:
        """Rerank `candidates` (list of (commit_id, rendered_text)) for one CVE."""
        if not candidates:
            return LLMRerankResult(
                chosen_commit_id=main_commit_id or "",
                ranking=[],
                main_commit_id=main_commit_id,
                overrode_main=False,
            )

        request = Request(
            query=Query(text=cve_description, qid=cve_id),
            candidates=[
                Candidate(docid=cid, score=0.0, doc={"text": text})
                for cid, text in candidates
            ],
        )
        result = self.reranker.rerank(
            request,
            rank_start=0,
            rank_end=len(candidates),
            populate_invocations_history=True,
        )

        ranking = [str(c.docid) for c in result.candidates]
        chosen = ranking[0] if ranking else (main_commit_id or "")

        # Score proxy: position in reranked list (top = highest). rank_llm's
        # listwise output is a permutation, not absolute scores, so we record
        # rank-derived scores for analysis.
        n = len(ranking)
        scores_by_id = {cid: (n - i) / n for i, cid in enumerate(ranking)}

        invocations = result.invocations_history or []
        last = invocations[-1] if invocations else None

        return LLMRerankResult(
            chosen_commit_id=chosen,
            ranking=ranking,
            main_commit_id=main_commit_id,
            overrode_main=(
                main_commit_id is not None and chosen != main_commit_id
            ),
            input_tokens=(getattr(last, "input_token_count", 0) if last else 0),
            output_tokens=(getattr(last, "output_token_count", 0) if last else 0),
            raw_response=(getattr(last, "response", "") if last else ""),
            scores_by_id=scores_by_id,
        )
