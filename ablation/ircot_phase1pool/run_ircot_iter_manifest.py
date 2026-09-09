"""IRCoT — iterative CoT over the Phase-1 top-100 coarse manifest.

Text-reasoning baseline on PatchHolmes's OWN retrieval pool:
  - Same Phase-1 fused top-100 pool as PatchHolmes (top_k=100).
  - IRCoT-style ITERATIVE chain-of-thought: generate one Thought at a time,
    up to max_iter=15 (same cap as the PatchHolmes agent), terminate on
    "So the answer is: [N]".
  - Input is the COARSE manifest only (commit msg first line + per-tag file
    counts) — IRCoT has no tools, so it cannot read diffs. This isolates
    PatchHolmes's advantage = agentic diff reading, which IRCoT lacks.

Reuses the IRCoT instruction/few-shot and the manifest renderer
`Phase2DataSource.list_candidates()` (the exact list the PatchHolmes agent
sees).

Point --endpoint at any OpenAI-compatible /v1 endpoint (e.g. a local vLLM
server) and set --model to the served model name.
"""
from __future__ import annotations
import argparse, json, re, time, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_pool import (
    load_descriptions, load_sample_ids, build_data_source,
    PATCH_TRACING_INSTRUCTION, PATCH_TRACING_EXAMPLE, PHASE1_JSONL,
)

MAX_ITER = 15
STOP = ["\nThought:", "\nCVE:"]
_NUM = re.compile(r"\[?#?\s*(\d+)\s*\]?")
_SHA = re.compile(r"\b([0-9a-f]{7,40})\b")


def base_prompt(desc: str, ds) -> str:
    """IRCoT instruction + few-shot + COARSE manifest + CVE, ending at 'Thought:'."""
    manifest = ds.list_candidates()           # #rank  <sha>  "msg"  files: N (src/test/doc..)
    return (f"{PATCH_TRACING_INSTRUCTION}\n\n{PATCH_TRACING_EXAMPLE}\n{manifest}\n\n"
            f"Answer with the candidate number, e.g. 'So the answer is: [7]'.\n\n"
            f"CVE: {desc}\nThought:")


def parse_answer(text: str, rank2cid: dict[int, str]) -> str:
    """Map 'So the answer is: [N]' → candidate with rank N; fall back to a SHA prefix."""
    tail = text.split("So the answer is:")[-1]
    m = _NUM.search(tail)
    if m:
        n = int(m.group(1))
        if n in rank2cid:
            return rank2cid[n]
    ms = _SHA.search(tail)
    if ms:
        pref = ms.group(1)
        for cid in rank2cid.values():
            if cid.startswith(pref):
                return cid
    return ""


def run_one(rec, descs, endpoint, model):
    desc = descs.get(rec["cve_id"], "")
    ds = build_data_source(rec, descs, top_k=100)
    rank2cid = {c.rank: c.commit.commit_id for c in ds.candidates}
    base = base_prompt(desc, ds)
    t0 = time.time()
    thoughts, tin, tout, ncall = "", 0, 0, 0
    best, iters = "", 0
    for it in range(MAX_ITER):
        iters = it + 1
        prompt = base + thoughts
        try:
            resp = requests.post(
                f"{endpoint}/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 512, "temperature": 0.0, "stop": STOP},
                timeout=300,
            )
            resp.raise_for_status()
            j = resp.json()
        except Exception as e:
            return {"cve_id": rec["cve_id"], "owner": rec["owner"], "repo": rec["repo"],
                    "fix_commit_ids": rec.get("fix_commit_ids") or [], "best_commit_id": "",
                    "ranking": [], "error": f"{type(e).__name__}: {e}",
                    "iterations_used": iters, "n_llm_calls": ncall,
                    "llm_input_tokens": tin, "llm_output_tokens": tout}
        thought = (j["choices"][0]["message"]["content"] or "").strip()
        u = j.get("usage") or {}
        tin += u.get("prompt_tokens", 0); tout += u.get("completion_tokens", 0); ncall += 1
        if "So the answer is:" in thought:
            best = parse_answer(thought, rank2cid)
            break
        thoughts += " " + thought + "\nThought:"
        if not thought:            # empty generation → stop early
            break
    return {
        "cve_id": rec["cve_id"], "owner": rec["owner"], "repo": rec["repo"],
        "fix_commit_ids": rec.get("fix_commit_ids") or [],
        "best_commit_id": best, "ranking": [best] if best else [],
        "iterations_used": iters, "n_llm_calls": ncall,
        "llm_input_tokens": tin, "llm_output_tokens": tout,
        "wall_time_sec": round(time.time() - t0, 2),
        "stopped_reason": "answered" if best else ("no_answer" if iters < MAX_ITER else "max_iterations"),
        "method": "ircot_iter_manifest_top100",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000/v1",
                    help="OpenAI-compatible /v1 endpoint")
    ap.add_argument("--model", default="qwen3-235b")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--phase1", default=str(PHASE1_JSONL),
                    help="Phase-1 RRF JSONL pool")
    ap.add_argument("--output", default="./logs/ablation/ircot_iter_manifest_top100.jsonl")
    args = ap.parse_args()

    descs = load_descriptions()
    sample = load_sample_ids()
    recs = [json.loads(l) for l in open(args.phase1) if json.loads(l)["cve_id"] in sample]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["cve_id"] for l in open(out)} if out.exists() else set()
    todo = [r for r in recs if r["cve_id"] not in done]
    print(f"sample_810: {len(recs)} CVE, {len(done)} done, {len(todo)} to run "
          f"(iterative CoT, top-100 coarse manifest, max_iter={MAX_ITER})", flush=True)

    n = 0
    with out.open("a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, r, descs, args.endpoint, args.model): r for r in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            f.write(json.dumps(rec) + "\n"); f.flush()
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(todo)} done", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
