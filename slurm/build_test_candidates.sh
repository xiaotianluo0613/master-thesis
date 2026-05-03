#!/bin/bash
#SBATCH -A uppmax2026-1-95
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -J build_test_candidates
#SBATCH -o logs/build_test_candidates_%j.out
#SBATCH -e logs/build_test_candidates_%j.err

# Build candidate pools for human and synthetic test sets.
# Two-stage retrieval: dense+sparse union top-100 shortlist → integration re-score → top-20.
# Human queries: Gemini Flash pre-filters ~30 candidates to ~10 for manual annotation.
#
# Prerequisite: global_val_chunks.json must exist (built by layer4_evaluate.sh).
# Run AFTER layer4 fine-tuning and evaluation are complete.
#
# Output:
#   data/test_human_candidates.json  + .csv
#   data/test_synthetic_candidates.json + .csv  (if synthetic queries exist)

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

# Load API keys from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

L4_MODEL=output/models/layer4-bge-m3-lora-dense-b4-merged

# Verify corpus exists
if [ ! -f data/global_val_chunks.json ]; then
    echo "ERROR: data/global_val_chunks.json not found."
    echo "Run layer4_evaluate.sh first to build the combined corpus."
    exit 1
fi

# --- Test Set 1: Human queries ---
echo "=== Building human query test set candidates ==="
python scripts/pipeline/build_test_candidates.py \
    --corpus            data/global_val_chunks.json \
    --queries           data/human_queries.txt \
    --queries-format    txt \
    --baseline-model    BAAI/bge-m3 \
    --finetuned-model   $L4_MODEL \
    --dense-k           100 \
    --final-k           20 \
    --batch-size        64 \
    --gemini-prefilter \
    --output-json       data/test_human_candidates.json \
    --output-csv        data/test_human_candidates.csv
echo "Done: data/test_human_candidates.json + .csv"

# --- Test Set 2: Synthetic queries (only if file exists) ---
if [ -f data/test_synthetic_queries_raw.json ]; then
    echo ""
    echo "=== Building synthetic query test set candidates ==="
    python scripts/pipeline/build_test_candidates.py \
        --corpus            data/global_val_chunks.json \
        --queries           data/test_synthetic_queries_raw.json \
        --queries-format    json \
        --baseline-model    BAAI/bge-m3 \
        --finetuned-model   $L4_MODEL \
        --dense-k           100 \
        --final-k           20 \
        --batch-size        64 \
        --output-json       data/test_synthetic_candidates.json \
        --output-csv        data/test_synthetic_candidates.csv
    echo "Done: data/test_synthetic_candidates.json + .csv"
else
    echo ""
    echo "Skipping synthetic test set: data/test_synthetic_queries_raw.json not found."
    echo "Run sample_test_chunks.py then generate_n_to_n_queries_layered.py first."
fi

echo ""
echo "All done."
