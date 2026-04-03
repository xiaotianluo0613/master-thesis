#!/bin/bash
#SBATCH -A uppmax2025-2-290
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH -J score_gpl
#SBATCH -o logs/score_gpl_%j.out
#SBATCH -e logs/score_gpl_%j.err

# GPL scoring — cross-encoder margin scoring for MarginMSE training
# Input:  output/gpl_negatives.json
# Output: output/gpl_training_data.jsonl

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs output

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

python scripts/pipeline/score_margins_gpl.py \
    --input output/gpl_negatives.json \
    --output output/gpl_training_data.jsonl \
    --reranker BAAI/bge-reranker-v2-m3 \
    --batch-size 64 \
    --max-length 512

echo "Done: output/gpl_training_data.jsonl"
