#!/bin/bash
#SBATCH -A uppmax2026-1-95
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH -J layer1_eval_all
#SBATCH -o logs/layer1_eval_all_%j.out
#SBATCH -e logs/layer1_eval_all_%j.err

# 5-way evaluation: baseline, full FT, LoRA-b8, LoRA-b4, LoRA+dense-b4
# Val set: data/layer1_val_queries.json (551 queries)
# Corpus:  data/layer1_chunks_grouped.json (7300 chunks)
# Output:  output/layer1_eval_all_results.json

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs output

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

python scripts/pipeline/evaluate_comparison.py \
    --chunks     data/layer1_chunks_grouped.json \
    --queries    data/layer1_val_queries.json \
    --models     BAAI/bge-m3 \
                 output/models/layer1-bge-m3-unified \
                 output/models/layer1-bge-m3-lora \
                 output/models/layer1-bge-m3-lora-b4 \
                 output/models/layer1-bge-m3-lora-dense-b4 \
    --k-values   10 100 \
    --batch-size 64 \
    --output     output/layer1_eval_all_results.json

echo "Done: output/layer1_eval_all_results.json"
