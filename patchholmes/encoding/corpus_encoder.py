"""PatchHolmes — offline Qwen embedding pre-computation.

Pre-computes query and corpus embeddings for a set of repos and saves them as
pickle files compatible with DenseRetriever.retrieve_from_pkl().

Output format (pickle)
----------------------
Each .pkl file stores a (embeddings, ids) tuple:
  embeddings : np.ndarray, shape (N, dim), float32, L2-normalised
  ids        : list[str]  — commit SHAs (corpus) or CVE IDs (queries)

Usage (library)
---------------
    from patchholmes.encoding.corpus_encoder import CorpusEncoder
    from patchholmes.encoding.embedder import QwenEmbedder

    encoder = CorpusEncoder(
        embedder=QwenEmbedder(model_name="Alibaba-NLP/gte-Qwen2-7B-instruct"),
        output_root=Path("./embeddings"),
    )
    encoder.encode_repo(owner="openssl", repo="openssl", ...)
"""
from __future__ import annotations

import csv
import glob
import json
import pickle
from pathlib import Path

import numpy as np

from patchholmes.encoding.embedder import BaseEmbedder, build_commit_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_pkl(path: Path, embeddings: np.ndarray, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((embeddings, ids), f, protocol=4)


def load_repo_list(combined_csv: Path) -> list[tuple[str, str]]:
    """Return ordered list of unique (owner, repo) pairs from combined.csv."""
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    with combined_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            owner = (row.get("owner") or "").strip()
            repo  = (row.get("repo")  or "").strip()
            if owner and repo and (owner, repo) not in seen:
                seen.add((owner, repo))
                pairs.append((owner, repo))
    return pairs


def load_cve_queries_for_repo(
    combined_csv: Path,
    cve2desc: dict[str, str],
    owner: str,
    repo: str,
) -> list[tuple[str, str]]:
    """Return [(cve_id, description)] for all CVEs in a specific (owner, repo)."""
    seen: set[str] = set()
    results: list[tuple[str, str]] = []
    with combined_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("owner") or "").strip() != owner:
                continue
            if (row.get("repo") or "").strip() != repo:
                continue
            cve = (row.get("cve") or "").strip()
            if cve and cve not in seen:
                seen.add(cve)
                results.append((cve, cve2desc.get(cve, "")))
    return results


def load_commits_for_repo(
    repo2commits_root: Path,
    owner: str,
    repo: str,
) -> list[tuple[str, str, str]]:
    """Return [(commit_id, commit_msg, diff)] for all commits in a repo."""
    split_dir = repo2commits_root / f"split_{owner}@@{repo}"
    if not split_dir.exists():
        return []
    commits: list[tuple[str, str, str]] = []
    for fp in glob.glob(str(split_dir / "*.json")):
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
            if not cid:
                continue
            commits.append((
                cid,
                str(item.get("commit_msg", "")),
                str(item.get("diff", "")),
            ))
    return commits


# ---------------------------------------------------------------------------
# CorpusEncoder
# ---------------------------------------------------------------------------

class CorpusEncoder:
    """Encodes commit corpus and CVE queries into per-repo pkl files.

    Parameters
    ----------
    embedder        : any BaseEmbedder (QwenEmbedder or MockEmbedder)
    output_root     : root directory for output pkls
                      layout: <output_root>/<owner>@@<repo>/qwen_embedding/{corpus,queries}.pkl
    max_diff_chars  : character budget for diff truncation before embedding
    skip_existing   : skip repos whose output pkl already exists
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        output_root: Path | str,
        max_diff_chars: int = 6000,
        skip_existing: bool = True,
        embedding_subdir: str = "qwen_embedding",
    ) -> None:
        self.embedder = embedder
        self.output_root = Path(output_root)
        self.max_diff_chars = max_diff_chars
        self.skip_existing = skip_existing
        # Per-repo sub-directory name for the embedding pkls. Default keeps
        # the original "qwen_embedding" layout for backward compatibility;
        # override to e.g. "octen_embedding" when running an alternate-model
        # ablation that should live next to the main Qwen pkls.
        self.embedding_subdir = embedding_subdir

    def _out_dir(self, owner: str, repo: str) -> Path:
        return self.output_root / f"{owner}@@{repo}" / self.embedding_subdir

    # ------------------------------------------------------------------

    def encode_repo_corpus(
        self,
        owner: str,
        repo: str,
        repo2commits_root: Path,
    ) -> Path | None:
        """Encode all commits for one repo → corpus.pkl.

        Returns the output path, or None if skipped / no commits found.
        """
        out_pkl = self._out_dir(owner, repo) / "corpus.pkl"
        if self.skip_existing and out_pkl.exists():
            return out_pkl

        commits = load_commits_for_repo(repo2commits_root, owner, repo)
        if not commits:
            return None

        texts = [build_commit_text(msg, diff, self.max_diff_chars)
                 for _, msg, diff in commits]
        ids   = [cid for cid, _, _ in commits]
        vecs  = self.embedder.encode_corpus(texts)
        save_pkl(out_pkl, vecs, ids)
        return out_pkl

    def encode_repo_queries(
        self,
        owner: str,
        repo: str,
        combined_csv: Path,
        cve2desc: dict[str, str],
    ) -> Path | None:
        """Encode all CVE queries for one repo → queries.pkl.

        Returns the output path, or None if skipped / no queries found.
        """
        out_pkl = self._out_dir(owner, repo) / "queries.pkl"
        if self.skip_existing and out_pkl.exists():
            return out_pkl

        cve_pairs = load_cve_queries_for_repo(combined_csv, cve2desc, owner, repo)
        if not cve_pairs:
            return None

        cve_ids = [c for c, _ in cve_pairs]
        texts   = [desc for _, desc in cve_pairs]
        vecs    = self.embedder.encode_query(texts)
        save_pkl(out_pkl, vecs, cve_ids)
        return out_pkl

