# Ablations & supplementary experiments

Each subdirectory isolates one design choice of PatchHolmes (or reproduces a
supplementary comparison) and re-runs the affected stage so its effect can be
measured against the main result.

Most `eval_*` scripts are **reference scripts**: they consume the per-CVE result
JSONLs produced by the main runners (`scripts/run_phase2_full.py`,
`scripts/patchholmes.sh phase1`) and reproduce a table once you have generated
those runs. Large run outputs are not shipped — regenerate them. Run any script
with `-h/--help` for its exact flags. Where a script cites a paper number
(e.g. R@1 = 59.95), it is kept as a constant and **not** recomputed.

## Phase 1

| Dir | Isolates | Entry |
|-----|----------|-------|
| `phase1_retrievers/` | single retrieval leg (BM25 / BM25+time / dense) vs. RRF fusion | `make_phase1_jsonl.py --retriever {bm25,bm25_time,dense}` |
| `phase1_dense_model/` | dense backbone: Octen-Embedding-8B vs. Qwen3-Embedding-8B | `encode_with_octen.sh` → `make_phase1_octen.py` (optional `reencode_queries_octen.py` for Octen's own query prompt) |

## Phase 2

| Dir | Isolates / measures | Entry & inputs |
|-----|---------------------|----------------|
| `phase2_cve_report/` | query content: full NVD markdown report vs. short description | `build_report_csv.py` builds the report CSV fed to `run_phase2_full.py` |
| `tool_interface/` | which of the 4 agent tools are available (`list_candidates` / `read_commit` / `read_file_diff` / `submit_answer`) | `run_tool_ablation.py --tier ...` runs the agent with a tool subset (via the runner's `tool_names`); `eval_tool_ablation.py`, `eval_modular_ablation.py` score the runs (reference-only) |
| `cross_family/` | backbone generalization across open model families (Llama / Gemma / Ministral / gpt-oss …) | generate runs with `run_phase2_full.py --llm-model ...`, then `eval_xfamily.py` (reference-only) |
| `efficiency/` | cost & tokens: input/output tokens, tool calls, commits read, latency, token-vs-cost | `analyze_efficiency.py` over a Phase-2 results JSONL (reference-only) |
| `significance/` | statistical significance vs. IRCoT / Favia (McNemar exact test + bootstrap CIs) | `stats_significance.py` (reference-only; needs the three methods' result JSONLs) |
| `favia_samepool/` | Favia's per-pair classifier over PatchHolmes's own Phase-1 top-10 pool (apples-to-apples) | `eval_favia_full.py` / `eval_favia_samepool.py` / `eval_samepool_selectors.py`; `run_favia_capped.py` launches an external Favia checkout; `build_favia_parquet.py` builds the candidate parquet |
| `ircot_phase1pool/` | an IRCoT-style iterative selector over the identical Phase-1 pool | `run_ircot_phase1pool.py` / `run_ircot_iter_manifest.py` (need an OpenAI-compatible `/v1` endpoint); `shared_pool.py` is a shared loader |

## Common inputs

The eval scripts generally expect:

- the ground-truth CSV (`data/sample_ground_truth_810.csv`),
- a Phase-1 candidate JSONL (from `scripts/patchholmes.sh phase1`),
- the per-CVE Phase-2 result JSONLs you generate with `scripts/run_phase2_full.py`.

External baselines (FlashRAG/IRCoT, Favia) install from upstream — see the
project README's "External baselines" section.
