#!/bin/bash
#SBATCH -A uppmax2025-2-290
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH -J mine_bge
#SBATCH -o logs/mine_bge_%j.out
#SBATCH -e logs/mine_bge_%j.err

# Hard negative mining — BGE-M3 official approach
# Encodes corpus with BGE-M3, retrieves top-200, selects 7 hard negatives per query.
# Output: output/bge_negatives.json

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs output

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

python scripts/pipeline/mine_hard_negatives_bge.py \
    --chunks data/layer1_pilot_pairs_550_grouped_3_4.json \
    --queries data/train_queries.json \
    --output output/bge_negatives.json \
    --model BAAI/bge-m3 \
    --batch-size 64 \
    --seed 42

echo "Done: output/bge_negatives.json"
