#!/usr/bin/env bash
# Encode all sample_810 repos with Octen/Octen-Embedding-8B and write the
# resulting per-repo pkls next to the existing qwen_embedding/ subdirs.
#
# Output layout (sibling of qwen_embedding/ under --output-root):
#   ./embeddings/<owner>@@<repo>/octen_embedding/{corpus,queries}.pkl
#
# Pre-reqs:
#   - The patchholmes conda env is active (or PYTHON is set to a python
#     that has tevatron + torch + transformers + faiss installed).
#   - CUDA_VISIBLE_DEVICES is set to the GPUs you want to use.
#   - At least 25 GB of free memory per GPU (set BATCH_SIZE lower if not).
#
# Inputs (all paths are repo-relative; override via env vars if your layout differs):
#   QUERIES_CSV     CSV with cve,owner,repo,cve_description,patch columns
#   REPO2COMMITS    Root containing split_<owner>@@<repo>/*.json
#   OUTPUT_ROOT     Where the new octen_embedding/ subdirs land
#   MODEL           HF model id (default Octen/Octen-Embedding-8B)
#   BATCH_SIZE      Encoding batch size (default 32 — fits ~25 GB)
#   MAX_PASSAGE_LEN Token cap per commit (default 2048 — same as main system)
#
# Usage:
#   bash ablation/phase1_dense_model/encode_with_octen.sh

set -euo pipefail

QUERIES_CSV=${QUERIES_CSV:-./data/sample_ground_truth_810.csv}
REPO2COMMITS=${REPO2COMMITS:-./data/repo2commits_diff}
OUTPUT_ROOT=${OUTPUT_ROOT:-./embeddings}
EMBEDDING_SUBDIR=octen_embedding
MODEL=${MODEL:-Octen/Octen-Embedding-8B}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_PASSAGE_LEN=${MAX_PASSAGE_LEN:-2048}
MAX_QUERY_LEN=${MAX_QUERY_LEN:-512}
MAX_DIFF_CHARS=${MAX_DIFF_CHARS:-6000}
DTYPE=${DTYPE:-bfloat16}

PYTHON=${PYTHON:-python}
LOG_DIR=${LOG_DIR:-./logs/octen}
mkdir -p "$LOG_DIR"

# Pick GPU list from CUDA_VISIBLE_DEVICES, default to GPU 0.
GPUS=${CUDA_VISIBLE_DEVICES:-0}
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "[octen-encode] config"
echo "  QUERIES_CSV       = $QUERIES_CSV"
echo "  REPO2COMMITS      = $REPO2COMMITS"
echo "  OUTPUT_ROOT       = $OUTPUT_ROOT"
echo "  EMBEDDING_SUBDIR  = $EMBEDDING_SUBDIR"
echo "  MODEL             = $MODEL"
echo "  BATCH_SIZE        = $BATCH_SIZE"
echo "  MAX_PASSAGE_LEN   = $MAX_PASSAGE_LEN"
echo "  NUM_GPUS          = $NUM_GPUS ($GPUS)"
echo

if [[ ! -f "$QUERIES_CSV" ]]; then
    echo "ERROR: --gt-csv not found: $QUERIES_CSV" >&2
    exit 1
fi
if [[ ! -d "$REPO2COMMITS" ]]; then
    echo "ERROR: --repo2commits-root not found: $REPO2COMMITS" >&2
    exit 1
fi

run_shard () {
    local GPU="$1"
    local SHARD_ID="$2"
    local LOG="$LOG_DIR/gpu${GPU}.log"

    echo "[launch] GPU $GPU  (shard $SHARD_ID of $NUM_GPUS)  log=$LOG"
    CUDA_VISIBLE_DEVICES="$GPU" \
    nohup "$PYTHON" -m patchholmes.cli.encode_worker \
        --gt-csv             "$QUERIES_CSV" \
        --repo2commits-root  "$REPO2COMMITS" \
        --output-root        "$OUTPUT_ROOT" \
        --embedding-subdir   "$EMBEDDING_SUBDIR" \
        --model              "$MODEL" \
        --embedder           qwen \
        --mode               both \
        --batch-size         "$BATCH_SIZE" \
        --max-passage-length "$MAX_PASSAGE_LEN" \
        --max-query-length   "$MAX_QUERY_LEN" \
        --max-diff-chars     "$MAX_DIFF_CHARS" \
        --dtype              "$DTYPE" \
        --num-shards         "$NUM_GPUS" \
        --shard-id           "$SHARD_ID" \
        > "$LOG" 2>&1 &
    echo "  PID=$!"
}

for i in "${!GPU_ARRAY[@]}"; do
    run_shard "${GPU_ARRAY[$i]}" "$i"
done

disown -a

cat <<EOF

✓ Launched $NUM_GPUS shards. Monitor with:
  tail -f $LOG_DIR/gpu*.log
  find $OUTPUT_ROOT -maxdepth 3 -name corpus.pkl -path "*/$EMBEDDING_SUBDIR/*" -size +0 | wc -l
EOF
