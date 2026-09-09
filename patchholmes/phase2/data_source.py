"""Per-CVE data snapshot used by the Phase 2 agent.

Encapsulates everything a single agentic run needs:
    - The Phase 1 Top-K candidate list (commit_id + Phase 1 rank).
    - Eagerly loaded commit content (commit_msg + diff) for those candidates.
    - Cached FileChange parses so we don't re-parse the same diff per turn.
    - A slot to record the agent's submitted answer.

The 4 Phase 2 tool executors all receive a `Phase2DataSource` instance and
call methods on it. The runner reads `get_answer()` after the conversation ends.
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from patchholmes.data_models import CommitDoc, CVEQuery, RankedCandidate
from patchholmes.phase2.diff_render import (
    FileChange,
    classify_files,
    parse_diff,
    render_for_agent,
    render_single_file,
)


# ---------------------------------------------------------------------------
# Submitted answer holder
# ---------------------------------------------------------------------------

@dataclass
class SubmittedAnswer:
    commit_id: str
    reasoning: str


# ---------------------------------------------------------------------------
# Commit-content loader (lightweight; no BM25Retriever dependency)
# ---------------------------------------------------------------------------

def _load_commits_for_repo(
    repo2commits_root: Path,
    owner: str,
    repo: str,
    commit_ids: set[str],
) -> dict[str, dict[str, str]]:
    """Scan split_<owner>@@<repo>/*.json and return {commit_id: {msg, diff, datetime, author}}.

    Only commits in `commit_ids` are kept. Stops early once all requested commits
    are found.
    """
    if not commit_ids:
        return {}

    split_dir = repo2commits_root / f"split_{owner}@@{repo}"
    if not split_dir.exists():
        return {}

    target = set(commit_ids)
    found: dict[str, dict[str, str]] = {}
    fields = ("commit_msg", "diff", "datetime", "author")

    for fp in sorted(glob.glob(str(split_dir / "*.json"))):
        if not target:
            break
        try:
            arr = json.loads(Path(fp).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(arr, list):
            continue
        for item in arr:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("commit_id", "")).strip()
            if cid not in target:
                continue
            found[cid] = {f: str(item.get(f, "") or "") for f in fields}
            target.discard(cid)
    return found


# ---------------------------------------------------------------------------
# Phase2DataSource
# ---------------------------------------------------------------------------

class Phase2DataSource:
    """One CVE's view of the world: Phase 1 candidates + commit content + answer slot.

    Parameters
    ----------
    query
        The CVE query (description, owner, repo, fix_commit_ids).
    phase1_candidates
        Ranked candidates from Phase 1 (already sorted by rank). Only the first
        `top_k` are exposed to the agent.
    repo2commits_root
        Filesystem root containing `split_<owner>@@<repo>/*.json`.
    top_k
        Number of top candidates the agent can see (default 100).
    """

    def __init__(
        self,
        query: CVEQuery,
        phase1_candidates: list[RankedCandidate],
        repo2commits_root: str | Path,
        top_k: int = 100,
    ) -> None:
        self.query = query
        self.top_k = top_k
        self.repo2commits_root = Path(repo2commits_root)

        # Trim and re-index ranks so Phase 2 sees ranks 1..top_k.
        self.candidates: list[RankedCandidate] = phase1_candidates[:top_k]

        # Map for fast commit_id → RankedCandidate lookup.
        self._cand_by_id: dict[str, RankedCandidate] = {
            c.commit.commit_id: c for c in self.candidates
        }

        # Eager load commit content (msg + diff) for the candidate set.
        self._commit_content: dict[str, dict[str, str]] = _load_commits_for_repo(
            self.repo2commits_root,
            query.owner,
            query.repo,
            {c.commit.commit_id for c in self.candidates},
        )

        # Caches.
        self._parsed_cache: dict[str, list[FileChange]] = {}
        self._inspected: list[str] = []   # ordered, dedup-by-set on append
        self._inspected_set: set[str] = set()
        self._answer: SubmittedAnswer | None = None

    # ──────────────────────────────────────────────────────────────────────
    # Used by tool executors

    def _stop_signal(self) -> str:
        """Standard 'stop now' error message used by post-submit calls."""
        ans = self._answer
        if ans is None:
            return ""
        return (
            f"[Error] You have already submitted your answer "
            f"({ans.commit_id[:12]}). The task is COMPLETE. "
            f"STOP IMMEDIATELY — do not call any more tools, including "
            f"submit_answer. Your conversation should now end."
        )

    def list_candidates(self) -> str:
        """One-line manifest of every candidate, with quick metadata.

        Format:
            #<rank>  <commit_id_short>  "<msg first line>"
                files: N (Ns src/Nt test/Nd doc/Nb fixture/Nc config)
        """
        if self._answer is not None:
            return self._stop_signal()
        lines = [f"Top-{len(self.candidates)} candidates from Phase 1:"]
        lines.append("")
        for cand in self.candidates:
            cid = cand.commit.commit_id
            short = cid[:12]
            content = self._commit_content.get(cid)
            if content is None:
                lines.append(f"  #{cand.rank:>4}  {short}  [commit not found in repo2commits_diff]")
                continue

            msg = (content.get("commit_msg") or "").strip().splitlines()
            msg_one = msg[0] if msg else "(empty msg)"
            if len(msg_one) > 80:
                msg_one = msg_one[:77] + "..."

            # parse on demand for the file counts
            files = self._parse_cached(cid, content.get("diff") or "")
            counts: dict[str, int] = {}
            for fc in files:
                counts[fc.tag] = counts.get(fc.tag, 0) + 1
            n = len(files)
            tag_summary = "/".join(
                f"{counts.get(t, 0)}{abbr}"
                for t, abbr in [("source", "src"), ("test", "test"),
                                ("doc", "doc"), ("fixture", "bin"), ("config", "cfg")]
                if counts.get(t, 0) > 0
            ) or "0 files"

            lines.append(
                f"  #{cand.rank:>4}  {short}  \"{msg_one}\"\n"
                f"            files: {n} ({tag_summary})"
            )
        return "\n".join(lines)

    def render_commit(self, commit_id: str, char_budget: int = 8000) -> str:
        """Render a commit for the read_commit tool. Returns an error string if not found."""
        if self._answer is not None:
            return self._stop_signal()
        resolved, note = self._normalise_cid(commit_id)
        if resolved is None:
            return self._err_commit_not_found(commit_id)

        content = self._commit_content.get(resolved)
        if content is None:
            return f"[Error] Commit {resolved[:12]} not found in repo2commits_diff."

        self._mark_inspected(resolved)
        files = self._parse_cached(resolved, content.get("diff") or "")
        meta = self._meta_header(resolved, content)
        body = render_for_agent(
            commit_msg=content.get("commit_msg") or "",
            files=files,
            char_budget=char_budget,
        )
        prefix = (note + "\n\n") if note else ""
        return prefix + meta + "\n\n" + body

    def render_file_diff(self, commit_id: str, file_path: str, char_budget: int = 16000) -> str:
        if self._answer is not None:
            return self._stop_signal()
        resolved, note = self._normalise_cid(commit_id)
        if resolved is None:
            return self._err_commit_not_found(commit_id)

        content = self._commit_content.get(resolved)
        if content is None:
            return f"[Error] Commit {resolved[:12]} not found in repo2commits_diff."

        self._mark_inspected(resolved)
        files = self._parse_cached(resolved, content.get("diff") or "")

        # exact match first, then case-insensitive/basename fallback
        match = next((fc for fc in files if fc.path == file_path), None)
        if match is None:
            match = next((fc for fc in files if fc.path.lower() == file_path.lower()), None)
        if match is None:
            base = file_path.rsplit("/", 1)[-1]
            match = next((fc for fc in files if fc.path.endswith(base)), None)
        if match is None:
            available = ", ".join(fc.path for fc in files[:10])
            return (
                f"[Error] File `{file_path}` not found in commit {resolved[:12]}. "
                f"Available files: {available}"
            )

        rendered = render_single_file(match, char_budget=char_budget)
        prefix = (note + "\n\n") if note else ""
        return prefix + rendered

    def submit_answer(self, commit_id: str, reasoning: str) -> str:
        if self._answer is not None:
            return self._stop_signal()
        resolved, note = self._normalise_cid(commit_id)
        # If we couldn't resolve, still record the raw input — that's what the
        # agent gave us; downstream eval will see hit=False if it's not a real
        # candidate, which is the correct outcome.
        normalised = resolved if resolved else commit_id.strip()
        self._answer = SubmittedAnswer(commit_id=normalised, reasoning=reasoning)
        # always treat submitted commit as "inspected" so the trace records it
        if normalised in self._cand_by_id:
            self._mark_inspected(normalised)
        prefix = (note + " ") if note else ""
        return (
            f"{prefix}Answer RECORDED: {normalised[:12]}. The task is COMPLETE. "
            f"STOP NOW — do not call any more tools."
        )

    # ──────────────────────────────────────────────────────────────────────
    # Used by runner

    def get_answer(self) -> SubmittedAnswer | None:
        return self._answer

    def get_inspected_commits(self) -> list[str]:
        return list(self._inspected)

    def phase1_rank_of(self, commit_id: str) -> int | None:
        cand = self._cand_by_id.get(commit_id)
        return cand.rank if cand else None

    def best_rank_in_truth_set(self) -> int | None:
        """Best Phase 1 rank among ground-truth fix commits (within Top-K)."""
        truth = set(self.query.fix_commit_ids)
        ranks = [
            cand.rank for cand in self.candidates
            if cand.commit.commit_id in truth
        ]
        return min(ranks) if ranks else None

    # ──────────────────────────────────────────────────────────────────────
    # Internals

    def _parse_cached(self, commit_id: str, raw_diff: str) -> list[FileChange]:
        if commit_id in self._parsed_cache:
            return self._parsed_cache[commit_id]
        files = parse_diff(raw_diff)
        files = classify_files(files, self.query.description)
        self._parsed_cache[commit_id] = files
        return files

    def _normalise_cid(self, commit_id: str) -> tuple[str | None, str | None]:
        """Match a (possibly truncated) commit_id against the candidate set.

        Returns (resolved_sha, note). The note is non-None when we had to
        auto-correct an input format (e.g. agent passed a rank number like
        '50' instead of a SHA); it should be surfaced to the agent so it
        learns the right format. Returns (None, None) for un-resolvable input.

        Rules
        -----
        - Pure-digit input ≤ 3 chars, or any input starting with '#':
          treat as a rank, look up `candidates[rank-1]`.
        - Otherwise: exact match, then unique prefix match (case-insensitive).
        """
        if not commit_id:
            return None, None
        raw = commit_id.strip()

        # Rank-like detection
        had_hash = raw.startswith("#")
        digits = raw.lstrip("#").strip()
        is_rank_like = digits.isdigit() and (had_hash or len(digits) <= 3)

        if is_rank_like:
            rank = int(digits)
            if 1 <= rank <= len(self.candidates):
                resolved = self.candidates[rank - 1].commit.commit_id
                note = (
                    f"[Note: you passed '{commit_id}' as commit_id; "
                    f"interpreted as rank {rank} → commit {resolved[:12]}. "
                    f"Next time pass the 12-char hex SHA from list_candidates, "
                    f"e.g. '{resolved[:12]}'.]"
                )
                return resolved, note
            # Rank out of range — fall through to "not found"
            return None, None

        # Normal SHA / prefix match (case-insensitive)
        cid_lower = raw.lower()
        if cid_lower in self._cand_by_id:
            return cid_lower, None
        matches = [k for k in self._cand_by_id if k.lower().startswith(cid_lower)]
        if len(matches) == 1:
            return matches[0], None
        return None, None

    def _mark_inspected(self, commit_id: str) -> None:
        if commit_id not in self._inspected_set:
            self._inspected_set.add(commit_id)
            self._inspected.append(commit_id)

    def _err_commit_not_found(self, raw_input: str) -> str:
        """Helpful 'not found' error that tailors the message to the input shape."""
        digits = raw_input.strip().lstrip("#").strip()
        n = len(self.candidates)
        if digits.isdigit():
            rank = int(digits)
            if 1 <= rank <= n:
                # This shouldn't happen given _normalise_cid handles it, but
                # belt-and-suspenders.
                return (
                    f"[Error] Could not interpret '{raw_input}'. "
                    f"To read rank {rank}, pass the SHA from list_candidates."
                )
            return (
                f"[Error] '{raw_input}' looks like a rank number, but {n} candidates "
                f"available and {rank} is out of range. commit_id should be a 12-char "
                f"hex SHA from list_candidates, NOT a rank number."
            )
        return (
            f"[Error] commit_id '{raw_input}' not found in the Top-{n} "
            f"candidate list. Pass the 12-char hex SHA from list_candidates "
            f"(e.g. '3bf5eddb89af'), not a rank or partial guess."
        )

    def _meta_header(self, commit_id: str, content: dict[str, str]) -> str:
        cand = self._cand_by_id.get(commit_id)
        rank = f"#{cand.rank} of {len(self.candidates)}" if cand else "?"
        author = content.get("author") or "?"
        date = content.get("datetime") or "?"
        return (
            f"COMMIT {commit_id[:12]}  (Phase 1 rank {rank})\n"
            f"Author: {author}   Date: {date}"
        )
