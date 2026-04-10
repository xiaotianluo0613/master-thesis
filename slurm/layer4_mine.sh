#!/bin/bash
#SBATCH -A uppmax2026-1-95
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH -J layer4_mine
#SBATCH -o logs/layer4_mine_%j.out
#SBATCH -e logs/layer4_mine_%j.err

# Step 2: Hard negative mining — BGE-M3 official approach.
# Encodes Layer 4 chunk corpus with BGE-M3, retrieves top-200 per query,
# selects 7 hard negatives (ranks 2-200) per training query.
# Input:  data/layer4_chunks_grouped.json  (~1500 chunks)
#         data/layer4_train_queries.json   (~978 queries)
# Output: output/layer4_bge_negatives.json

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs output

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

python scripts/pipeline/mine_hard_negatives_bge.py \
    --chunks    data/layer4_chunks_grouped.json \
    --queries   data/layer4_train_queries.json \
    --output    output/layer4_bge_negatives.json \
    --model     BAAI/bge-m3 \
    --batch-size 64 \
    --seed      42

echo "Done: output/layer4_bge_negatives.json"
