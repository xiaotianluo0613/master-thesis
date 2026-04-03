#!/bin/bash
#SBATCH -A uppmax2025-2-290
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH -J evaluate
#SBATCH -o logs/evaluate_%j.out
#SBATCH -e logs/evaluate_%j.err

# Evaluate baseline vs bge-m3-unified vs bge-m3-gpl
# Output: output/eval_results.json

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs output

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

python scripts/pipeline/evaluate_comparison.py \
    --chunks data/layer1_pilot_pairs_550_grouped_3_4.json \
    --queries data/val_queries.json \
    --batch-size 64 \
    --k 10 \
    --output output/eval_results.json

echo "Done: output/eval_results.json"
