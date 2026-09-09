"""IRCoT selector over PatchHolmes's Phase-1 top-10 pool.

Same-pool baseline: reuses the IRCoT method definition
(PATCH_TRACING_INSTRUCTION, PATCH_TRACING_EXAMPLE, and the
"So the answer is: [N]" parse). The ONLY change vs a standard IRCoT run is
that the candidate list is the fixed PatchHolmes Phase-1 RRF top-10 instead
of a fresh dense retrieval (retrieval held fixed -> isolates the
reasoning/selection step).

Records per-CVE token usage so the efficiency/cost comparison is complete.

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

PER_CAND_CHARS = 2500          # per-candidate budget so 10 fit in 32K with room
STOP = ["\nCVE:", "\nThought:"]
_BRACKET = re.compile(r"\[(\d+)\]")
_LOOSE = re.compile(r"\b(\d+)\b")


def build_prompt(desc: str, ds) -> tuple[str, list[str]]:
    """[1]..[10] rendered candidates + CVE + Thought. Returns (prompt, ordered commit_ids)."""
    cand_ids, lines = [], []
    for i, c in enumerate(ds.candidates):        # already top-10, rank order
        cid = c.commit.commit_id
        cand_ids.append(cid)
        content = ds.render_commit(cid, char_budget=PER_CAND_CHARS)
        lines.append(f"[{i+1}] {content}")
    prompt = (f"{PATCH_TRACING_INSTRUCTION}\n\n{PATCH_TRACING_EXAMPLE}\n"
              + "\n\n".join(lines)
              + f"\n\nCVE: {desc}\nThought:")
    return prompt, cand_ids


def parse_answer(text: str, cand_ids: list[str]) -> tuple[str, int | None]:
    """Extract [N] after 'So the answer is:' and map to a candidate commit id."""
    tail = text.split("So the answer is:")[-1] if "So the answer is:" in text else text
    m = _BRACKET.search(tail) or _LOOSE.search(tail)
    if not m:
        return "", None
    n = int(m.group(1))
    if 1 <= n <= len(cand_ids):
        return cand_ids[n - 1], n
    return "", n


def run_one(rec, descs, endpoint, model):
    desc = descs.get(rec["cve_id"], "")
    ds = build_data_source(rec, descs, top_k=10)
    prompt, cand_ids = build_prompt(desc, ds)
    t0 = time.time()
    resp = requests.post(
        f"{endpoint}/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": 1024, "temperature": 0.0, "stop": STOP},
        timeout=300,
    )
    resp.raise_for_status()
    j = resp.json()
    text = j["choices"][0]["message"]["content"] or ""
    usage = j.get("usage") or {}
    best, n = parse_answer(text, cand_ids)
    return {
        "cve_id": rec["cve_id"], "owner": rec["owner"], "repo": rec["repo"],
        "fix_commit_ids": rec.get("fix_commit_ids") or [],
        "best_commit_id": best, "parsed_index": n,
        "ranking": [best] if best else [],
        "raw_pred": text,
        "llm_input_tokens": usage.get("prompt_tokens", 0),
        "llm_output_tokens": usage.get("completion_tokens", 0),
        "n_llm_calls": 1,
        "wall_time_sec": round(time.time() - t0, 2),
        "method": "ircot_phase1pool_optionA",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8001/v1",
                    help="OpenAI-compatible /v1 endpoint")
    ap.add_argument("--model", default="qwen3-235b")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--phase1", default=str(PHASE1_JSONL),
                    help="Phase-1 RRF JSONL pool")
    ap.add_argument("--output", default="./logs/ablation/ircot_phase1pool.jsonl")
    args = ap.parse_args()

    descs = load_descriptions()
    sample = load_sample_ids()
    recs = [json.loads(l) for l in open(args.phase1) if json.loads(l)["cve_id"] in sample]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["cve_id"] for l in open(out)}
    todo = [r for r in recs if r["cve_id"] not in done]
    print(f"sample_810: {len(recs)} CVE, {len(done)} done, {len(todo)} to run", flush=True)

    n = 0
    with out.open("a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, r, descs, args.endpoint, args.model): r for r in todo}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"cve_id": r["cve_id"], "owner": r["owner"], "repo": r["repo"],
                       "fix_commit_ids": r.get("fix_commit_ids") or [], "best_commit_id": "",
                       "ranking": [], "error": f"{type(e).__name__}: {e}",
                       "llm_input_tokens": 0, "llm_output_tokens": 0, "n_llm_calls": 1}
            f.write(json.dumps(rec) + "\n"); f.flush()
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(todo)} done", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
