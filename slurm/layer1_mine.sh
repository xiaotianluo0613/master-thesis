#!/bin/bash
#SBATCH -A uppmax2026-1-95
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH -J layer1_mine
#SBATCH -o logs/layer1_mine_%j.out
#SBATCH -e logs/layer1_mine_%j.err

# Step 2: Hard negative mining — BGE-M3 official approach.
# Encodes 7300-chunk corpus with BGE-M3, retrieves top-200 per query,
# selects 7 hard negatives (ranks 2-200) per training query.
# Input:  data/layer1_chunks_grouped.json  (7300 chunks)
#         data/layer1_train_queries.json   (~4950 queries)
# Output: output/layer1_bge_negatives.json

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs output

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

python scripts/pipeline/mine_hard_negatives_bge.py \
    --chunks    data/layer1_chunks_grouped.json \
    --queries   data/layer1_train_queries.json \
    --output    output/layer1_bge_negatives.json \
    --model     BAAI/bge-m3 \
    --batch-size 64 \
    --seed      42

echo "Done: output/layer1_bge_negatives.json"
