#!/usr/bin/env bash
# Unified shell entry point for PatchHolmes.
#
# Wraps the Phase 1 CLIs as subcommands. For Phase 2 use:
#   python scripts/run_phase2_full.py
# For three-way evaluation use:
#   python scripts/eval_methods.py
#
# Override Python with:  PYTHON=/path/to/python scripts/patchholmes.sh ...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

if [ "$#" -lt 1 ]; then
    cat <<EOF
Usage:
  scripts/patchholmes.sh encode {corpus|queries|all} [args...]
  scripts/patchholmes.sh phase1 [args...]
  scripts/patchholmes.sh phase1-grouped [args...]
  scripts/patchholmes.sh phase2 [args...]
  scripts/patchholmes.sh eval-phase2 [args...]

Examples:
  scripts/patchholmes.sh encode corpus --num-gpus 4
  scripts/patchholmes.sh encode queries --num-gpus 2
  scripts/patchholmes.sh phase1 --embedding-root ./embeddings/
  scripts/patchholmes.sh phase2 --phase1-jsonl logs/phase1/phase1.jsonl --dataset-csv data/ground_truth_queries_clean.csv --output logs/phase2/results.jsonl
  scripts/patchholmes.sh eval-phase2 logs/phase2/results.jsonl --phase1 logs/phase1/phase1.jsonl
EOF
    exit 2
fi

COMMAND="$1"
shift

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

case "${COMMAND}" in
    encode)
        exec "${PYTHON}" -m patchholmes.cli.encode "$@"
        ;;
    phase1)
        exec "${PYTHON}" -u -m patchholmes.cli.phase1 "$@"
        ;;
    phase1-grouped)
        exec "${PYTHON}" -u -m patchholmes.cli.phase1_grouped "$@"
        ;;
    phase2)
        exec "${PYTHON}" -u "${REPO_ROOT}/scripts/run_phase2_full.py" "$@"
        ;;
    eval-phase2)
        exec "${PYTHON}" -u -m patchholmes.cli.eval_phase2 "$@"
        ;;
    *)
        echo "Unknown command: ${COMMAND}" >&2
        echo "Run with no args to see usage." >&2
        exit 2
        ;;
esac
