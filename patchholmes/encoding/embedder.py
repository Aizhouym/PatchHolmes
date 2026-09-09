"""PatchHolmes — embedding layer for Phase 1 dense retrieval.

Commit diffs can run to 10 000–20 000 tokens, far exceeding any embedding
model's context window.  The solution is static priority truncation (方案一):
diff lines are classified into 5 priority levels and filled into a fixed char
budget from highest to lowest priority.  Budget exhausted → stop.  Lines are
never cut in the middle.

Priority levels
---------------
P1  Commit message   — developer-written summary, always kept in full
                       (handled by build_commit_text, outside this function)
P2  File path lines  — "--- a/src/crypto/rsa.c" / "+++ b/…"
                       Short, high semantic density, always kept in full.
P3  Hunk headers     — "@@ -100,10 +102,15 @@ rsa_encrypt"
                       Contains function name + line numbers, always kept.
P4  Changed lines    — actual "+" / "-" lines, the real fix content.
                       Kept in order until budget runs out.
P5  Context lines    — unchanged surrounding code, lowest value.
                       Appended only if budget still remains after P4.

Classes
-------
BaseEmbedder   Abstract interface with encode_query / encode_corpus / encode
MockEmbedder   Deterministic unit-vector embedder — for testing / CI (no GPU)
QwenEmbedder   Real Qwen2 embedding model (Alibaba-NLP/gte-Qwen2-7B-instruct)
"""
from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Diff truncation — static priority (方案一)
# ---------------------------------------------------------------------------

@dataclass
class _TruncStats:
    """Counts of lines kept / dropped per priority bucket (for debugging)."""
    p2_paths_kept:   int = 0
    p3_hunks_kept:   int = 0
    p4_changed_kept: int = 0;  p4_changed_drop: int = 0
    p5_context_kept: int = 0;  p5_context_drop: int = 0

    def __str__(self) -> str:
        return (
            f"P2(paths)={self.p2_paths_kept} "
            f"P3(hunks)={self.p3_hunks_kept} "
            f"P4(changed) kept={self.p4_changed_kept} drop={self.p4_changed_drop} "
            f"P5(context) kept={self.p5_context_kept} drop={self.p5_context_drop}"
        )


def _classify_line(line: str) -> int:
    """Return the priority bucket (2-5) for a single diff line."""
    if line.startswith("--- ") or line.startswith("+++ "):
        return 2
    if line.startswith("@@"):
        return 3
    if line and line[0] in ("+", "-"):
        return 4
    return 5


def truncate_diff(
    diff: str,
    max_chars: int,
    *,
    stats: _TruncStats | None = None,
) -> str:
    """Apply static priority truncation to a diff string.

    Algorithm
    ---------
    1. Classify every line into P2 / P3 / P4 / P5.
    2. Fill the char budget with P2 lines (file paths).
    3. Fill remaining budget with P3 lines (hunk headers).
    4. Fill remaining budget with P4 lines (changed lines) in original order.
    5. Fill remaining budget with P5 lines (context lines) in original order.

    Lines are NEVER split mid-way — a line that does not fit in the remaining
    budget is skipped entirely.  Output length is always ≤ max_chars.

    Parameters
    ----------
    diff       : raw git diff text
    max_chars  : character budget (characters, not tokens; 1 token ≈ 4 chars)
    stats      : optional _TruncStats filled in-place for debugging / logging

    Returns
    -------
    Truncated diff string.  len(result) <= max_chars is guaranteed.
    """
    if max_chars <= 0 or len(diff) <= max_chars:
        return diff

    # --- Pass 1: classify every line ---
    p2: list[str] = []  # P2 — file path lines   (--- a/..., +++ b/...)
    p3: list[str] = []  # P3 — hunk headers       (@@ ... @@)
    p4: list[str] = []  # P4 — changed lines      (+ / - prefix)
    p5: list[str] = []  # P5 — context lines      (space / empty)

    for line in diff.splitlines():
        pri = _classify_line(line)
        if pri == 2:
            p2.append(line)
        elif pri == 3:
            p3.append(line)
        elif pri == 4:
            p4.append(line)
        else:
            p5.append(line)

    # --- Pass 2: fill budget in priority order ---
    out: list[str] = []
    remaining = max_chars

    def _fill(bucket: list[str]) -> tuple[int, int]:
        """Add lines from bucket while budget allows.  Returns (kept, dropped)."""
        nonlocal remaining
        kept = dropped = 0
        for line in bucket:
            cost = len(line) + 1  # +1 for the '\n' added by join
            if remaining >= cost:
                out.append(line)
                remaining -= cost
                kept += 1
            else:
                dropped += 1
        return kept, dropped

    p2k, p2d = _fill(p2)
    p3k, p3d = _fill(p3)
    p4k, p4d = _fill(p4)
    p5k, p5d = _fill(p5)

    if stats is not None:
        stats.p2_paths_kept   = p2k
        stats.p3_hunks_kept   = p3k
        stats.p4_changed_kept = p4k;  stats.p4_changed_drop = p4d
        stats.p5_context_kept = p5k;  stats.p5_context_drop = p5d

    return "\n".join(out)


# Public alias kept for back-compat with any callers using the old name.
_truncate_diff_layered = truncate_diff


def build_commit_text(
    commit_msg: str,
    diff: str,
    max_diff_chars: int = 6000,
) -> str:
    """Build the text fed to the embedding model for a single commit.

    The commit message (P1) is always included in full.
    The diff is truncated to max_diff_chars via static priority truncation.
    """
    truncated = truncate_diff(diff, max_diff_chars)
    return f"Commit message:\n{commit_msg}\n\nDiff:\n{truncated}".strip()


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    """Abstract embedder interface.

    Subclasses must implement encode_corpus().  encode_query() defaults to
    encode_corpus() — override it when the model uses instruction prefixes
    for asymmetric retrieval (e.g. Qwen2 instruct models).
    """

    def encode_query(self, texts: list[str]) -> np.ndarray:
        """Encode CVE descriptions (query side).  Returns L2-normalised float32 (N, dim)."""
        return self.encode_corpus(texts)

    @abstractmethod
    def encode_corpus(self, texts: list[str]) -> np.ndarray:
        """Encode commit texts (corpus side).  Returns L2-normalised float32 (N, dim)."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Alias for encode_corpus — kept for backward compatibility."""
        return self.encode_corpus(texts)


# ---------------------------------------------------------------------------
# Mock embedder (deterministic, no GPU needed)
# ---------------------------------------------------------------------------

class MockEmbedder(BaseEmbedder):
    """Deterministic unit-vector embedder based on text hash — for testing."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def encode_corpus(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dim).astype(np.float32)
            norm = np.linalg.norm(v)
            vecs[i] = v / norm if norm > 0 else v
        return vecs


# ---------------------------------------------------------------------------
# Qwen2 embedder — backed by tevatron DenseModel
# ---------------------------------------------------------------------------

class QwenEmbedder(BaseEmbedder):
    """Qwen2 text embedding model, powered by tevatron's DenseModel.

    Uses Alibaba-NLP/gte-Qwen2-7B-instruct (or any gte-Qwen2 variant).

    How it works
    ------------
    Internally this wraps tevatron's DenseModel, which handles:
      - Model loading via AutoModel
      - Pooling (cls / mean / last / eos) — we use 'last' (EOS token)
      - L2 normalisation

    Asymmetric encoding (query vs corpus)
    --------------------------------------
    gte-Qwen2-instruct is an instruction-tuned model designed for asymmetric
    retrieval.  Queries receive a task-instruction prefix; corpus texts do not.

        encode_query  →  "Instruct: ...\nQuery: <CVE description>"
        encode_corpus →  "<commit message + diff>"   (no prefix)

    Pooling: EOS token ('last') on right-padded sequences.
    Output:  L2-normalised float32 vectors, shape (N, dim).

    Lazy loading: model is loaded on first encode_query / encode_corpus call.

    Alternatives (same interface, just change model_name):
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct"  (~3 GB VRAM fp16, faster)
        "Alibaba-NLP/gte-Qwen2-7B-instruct"     (~14 GB VRAM fp16, default)
    """

    QUERY_INSTRUCTION = (
        "Instruct: Given a security vulnerability description, "
        "retrieve the commit that fixes it\nQuery: "
    )

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct",
        pooling: str = "last",       # EOS-token pooling for decoder-only Qwen2
        normalize: bool = True,
        device: str | None = None,   # None → auto-detect (cuda if available)
        batch_size: int = 16,
        max_query_length: int = 512,
        max_passage_length: int = 8192,
        dtype: str = "float16",
    ) -> None:
        self.model_name = model_name
        self.pooling = pooling
        self.normalize = normalize
        self.batch_size = batch_size
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length
        self.dtype = dtype
        self._device = device
        self._model = None      # tevatron DenseModel, loaded lazily
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Tevatron path setup
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_tevatron_importable() -> None:
        """Add tevatron/src to sys.path so it can be imported without install."""
        import sys
        from pathlib import Path
        tevatron_src = Path(__file__).resolve().parent.parent.parent / "tevatron" / "src"
        if tevatron_src.exists() and str(tevatron_src) not in sys.path:
            sys.path.insert(0, str(tevatron_src))

    # ------------------------------------------------------------------
    # Lazy loader
    # ------------------------------------------------------------------

    def _load(self) -> None:
        import torch
        from transformers import AutoTokenizer

        self._ensure_tevatron_importable()
        from tevatron.retriever.modeling import DenseModel  # noqa: PLC0415

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Explicitly set the default CUDA device so all allocations go to the
        # right GPU even when CUDA_VISIBLE_DEVICES is not restricted.
        if str(self._device).startswith("cuda"):
            device_idx = 0 if self._device == "cuda" else int(str(self._device).split(":")[1])
            torch.cuda.set_device(device_idx)
            print(f"  device: {self._device}  ({torch.cuda.get_device_name(device_idx)})")

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.float16)

        # DenseModel.load() wraps AutoModel + adds pooling + normalisation.
        self._model = DenseModel.load(
            model_name_or_path=self.model_name,
            pooling=self.pooling,
            normalize=self.normalize,
            dtype=torch_dtype,
            attn_implementation="sdpa",
        ).to(self._device).eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token_id is None:
            # Qwen2 tokenizer has no explicit pad token — use EOS.
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        # Right-padding: tevatron's _pooling handles right-padded EOS correctly.
        self._tokenizer.padding_side = "right"

    # ------------------------------------------------------------------
    # Internal batched encoder
    # ------------------------------------------------------------------

    def _encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        """Encode a list of texts via tevatron DenseModel.

        Parameters
        ----------
        texts     : list of strings to encode
        is_query  : True  → model(query=batch)  → output.q_reps
                    False → model(passage=batch) → output.p_reps
        """
        import torch

        if self._model is None:
            self._load()

        max_len = self.max_query_length if is_query else self.max_passage_length
        all_vecs: list[np.ndarray] = []
        total = len(texts)
        total_batches = (total + self.batch_size - 1) // self.batch_size
        label = "queries" if is_query else "passages"
        start_time = time.monotonic()
        print(
            f"  encoding {total:,} {label} in {total_batches:,} batches "
            f"(batch_size={self.batch_size}, max_len={max_len})",
            flush=True,
        )

        for i in range(0, len(texts), self.batch_size):
            batch_idx = i // self.batch_size + 1
            batch_texts = texts[i : i + self.batch_size]
            if batch_idx == 1 or batch_idx == total_batches or batch_idx % 10 == 0:
                print(f"  batch {batch_idx}/{total_batches} start ({len(batch_texts)} items)", flush=True)

            # Tokenise batch
            batch_enc = self._tokenizer(
                batch_texts,
                max_length=max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch_enc = {k: v.to(self._device) for k, v in batch_enc.items()}

            with torch.no_grad():
                if is_query:
                    # model(query=...) calls DenseModel.encode_query()
                    output = self._model(query=batch_enc)
                    vecs = output.q_reps          # (B, dim), already normalised
                else:
                    # model(passage=...) calls DenseModel.encode_passage()
                    output = self._model(passage=batch_enc)
                    vecs = output.p_reps          # (B, dim), already normalised

            all_vecs.append(vecs.float().cpu().numpy())
            if batch_idx == 1 or batch_idx == total_batches or batch_idx % 10 == 0:
                elapsed = time.monotonic() - start_time
                print(f"  batch {batch_idx}/{total_batches} done ({elapsed:.1f}s elapsed)", flush=True)

        print(f"  encoded {total:,} {label} in {time.monotonic() - start_time:.1f}s", flush=True)
        return np.vstack(all_vecs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_query(self, texts: list[str]) -> np.ndarray:
        prefixed = [self.QUERY_INSTRUCTION + t for t in texts]
        return self._encode(prefixed, is_query=True)

    def encode_corpus(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, is_query=False)


# Backward-compatible alias
QwenEmbeddingClient = QwenEmbedder
