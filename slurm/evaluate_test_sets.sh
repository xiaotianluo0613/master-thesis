#!/bin/bash
#SBATCH -A uppmax2026-1-95
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -J evaluate_test_sets
#SBATCH -o logs/evaluate_test_sets_%j.out
#SBATCH -e logs/evaluate_test_sets_%j.err

# Final test set evaluation: all 5 models on both test sets.
#
# Prerequisites:
#   data/test_human_queries.json       (from import_human_annotations.py)
#   data/test_synthetic_queries.json   (from annotate_synthetic_test.py)
#   data/global_val_chunks.json        (built by layer4_evaluate.sh)
#
# Output:
#   output/test_human_eval_results.json
#   output/test_synthetic_eval_results.json

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs output

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

MODELS=(
    BAAI/bge-m3
    output/models/layer1-bge-m3-lora-dense-b4-merged
    output/models/layer2-bge-m3-lora-dense-b4-merged
    output/models/layer3-bge-m3-lora-dense-b4-merged
    output/models/layer4-bge-m3-lora-dense-b4-merged
)

# --- Human test set ---
if [ -f data/test_human_queries.json ]; then
    echo "=== Evaluating on human test set ==="
    python scripts/pipeline/evaluate_comparison.py \
        --chunks     data/global_val_chunks.json \
        --queries    data/test_human_queries.json \
        --models     "${MODELS[@]}" \
        --k-values   10 100 \
        --batch-size 64 \
        --output     output/test_human_eval_results.json
    echo "Done: output/test_human_eval_results.json"
else
    echo "Skipping human test set: data/test_human_queries.json not found."
    echo "Run import_human_annotations.py first."
fi

echo ""

# --- Synthetic test set ---
if [ -f data/test_synthetic_queries.json ]; then
    echo "=== Evaluating on synthetic test set ==="
    python scripts/pipeline/evaluate_comparison.py \
        --chunks     data/global_val_chunks.json \
        --queries    data/test_synthetic_queries.json \
        --models     "${MODELS[@]}" \
        --k-values   10 100 \
        --batch-size 64 \
        --output     output/test_synthetic_eval_results.json
    echo "Done: output/test_synthetic_eval_results.json"
else
    echo "Skipping synthetic test set: data/test_synthetic_queries.json not found."
    echo "Run annotate_synthetic_test.py first."
fi

echo ""
echo "All done."
