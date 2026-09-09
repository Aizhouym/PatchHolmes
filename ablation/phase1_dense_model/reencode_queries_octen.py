#!/usr/bin/env python3
from __future__ import annotations

import csv
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from patchholmes.encoding.embedder import QwenEmbedder  # noqa: E402


# Octen's official query prompt from config_sentence_transformers.json
OCTEN_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\n"
    "Query: "
)


class OctenPromptEmbedder(QwenEmbedder):
    QUERY_INSTRUCTION = OCTEN_PROMPT


def group_by_repo(csv_path):
    out = {}
    try:
        f = csv_path.open()
    except OSError as e:
        print(f"cannot read {csv_path}: {e}")
        return out
    with f:
        for r in csv.DictReader(f):
            key = f"{r['owner'].strip()}@@{r['repo'].strip()}"
            cve = r["cve"].strip()
            desc = (r.get("cve_description") or "").strip()
            out.setdefault(key, []).append((cve, desc))
    return out


def write_pkl(path, vecs, ids):
    try:
        with open(path, "wb") as f:
            pickle.dump((vecs, ids), f, protocol=4)
        return True
    except OSError as e:
        print(f"  failed to write {path}: {e}")
        return False


def load_model():
    try:
        return OctenPromptEmbedder(
            model_name="Octen/Octen-Embedding-8B",
            batch_size=32,
            max_query_length=512,
            dtype="bfloat16",
        )
    except Exception as e:
        print(f"model load failed: {e}")
        return None


def encode_one_repo(emb, qs):
    ids = [c for c, _ in qs]
    texts = [d for _, d in qs]
    try:
        vecs = emb.encode_query(texts)
        return ids, vecs
    except Exception as e:
        print(f"  encode failed: {e}")
        return None, None


def main():
    gt = Path("./data/sample_ground_truth_810.csv")
    feat_root = Path("./embeddings")
    out_name = "queries_octen_official.pkl"

    repo2qs = group_by_repo(gt)
    n_repos = len(repo2qs)
    n_q = sum(len(v) for v in repo2qs.values())
    print(f"Loaded {n_q} queries across {n_repos} repos")
    print(f"Prompt: {OCTEN_PROMPT!r}\n")

    print("Loading Octen-Embedding-8B ...")
    emb = load_model()
    if emb is None:
        return

    t0 = time.monotonic()
    n_ok = 0
    for i, (repo_key, qs) in enumerate(repo2qs.items(), 1):
        out_pkl = feat_root / repo_key / "octen_embedding" / out_name
        if not out_pkl.parent.exists():
            print(f"  [{i}/{n_repos}] {repo_key}: skip (no octen_embedding/)")
            continue
        ids, vecs = encode_one_repo(emb, qs)
        if ids is None:
            continue
        if write_pkl(out_pkl, vecs, ids):
            n_ok += 1
            print(f"  [{i}/{n_repos}] {repo_key}: {len(qs)} queries -> {out_pkl}  ({time.monotonic()-t0:.1f}s)", flush=True)

    print(f"\nWrote {n_ok} {out_name} (Octen official prompt)")


if __name__ == "__main__":
    main()
