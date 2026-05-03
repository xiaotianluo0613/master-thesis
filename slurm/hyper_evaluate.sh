#!/bin/bash
#SBATCH -A uppmax2026-1-95
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 05:00:00
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -J hyper_eval
#SBATCH -o logs/hyper_eval_%j.out
#SBATCH -e logs/hyper_eval_%j.err

# Hyperparameter tuning evaluation
# Evaluates baseline + hyper_A/B/C/D on global val set (1150 queries, 13800-chunk corpus)
# Metrics: MAP, nDCG@10, Recall@10, Recall@100
# Run AFTER all 4 hyper finetune jobs complete.
# Output: output/hyper_eval_results.json

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

# Verify all merged models exist
for run in A B C D; do
    if [ ! -d output/models/hyper_${run}-merged ]; then
        echo "ERROR: output/models/hyper_${run}-merged not found. Run hyper_${run}_finetune.sh first."
        exit 1
    fi
done

echo "=== Hyperparameter tuning evaluation ==="
python scripts/pipeline/evaluate_comparison.py \
    --chunks     data/global_val_chunks.json \
    --queries    data/global_val_queries.json \
    --models     BAAI/bge-m3 \
                 output/models/hyper_A-merged \
                 output/models/hyper_B-merged \
                 output/models/hyper_C-merged \
                 output/models/hyper_D-merged \
    --k-values   10 100 \
    --batch-size 64 \
    --output     output/hyper_eval_results.json

echo "Done: output/hyper_eval_results.json"
