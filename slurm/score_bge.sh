#!/bin/bash
#SBATCH -A uppmax2025-2-290
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH -J score_bge
#SBATCH -o logs/score_bge_%j.out
#SBATCH -e logs/score_bge_%j.err

# BGE-M3 integration scoring — compute teacher scores for m3_kd_loss
# Input:  output/bge_training_data.jsonl
# Output: output/bge_training_data_scored.jsonl

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs output

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

python scripts/score_bge_integration.py \
    --input output/bge_training_data.jsonl \
    --output output/bge_training_data_scored.jsonl \
    --model BAAI/bge-m3 \
    --batch-size 32

echo "Done: output/bge_training_data_scored.jsonl"
