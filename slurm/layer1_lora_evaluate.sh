#!/bin/bash
#SBATCH -A uppmax2026-1-95
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH -J layer1_lora_eval
#SBATCH -o logs/layer1_lora_eval_%j.out
#SBATCH -e logs/layer1_lora_eval_%j.err

# 3-way evaluation: baseline vs full FT vs LoRA.
# Val set: data/layer1_val_queries.json (551 queries)
# Corpus:  data/layer1_chunks_grouped.json (7300 chunks)
# Output:  output/layer1_lora_eval_results.json

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs output

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

python scripts/pipeline/evaluate_comparison.py \
    --chunks     data/layer1_chunks_grouped.json \
    --queries    data/layer1_val_queries.json \
    --models     BAAI/bge-m3 output/models/layer1-bge-m3-unified output/models/layer1-bge-m3-lora \
    --k-values   10 100 \
    --batch-size 64 \
    --output     output/layer1_lora_eval_results.json

echo "Done: output/layer1_lora_eval_results.json"
