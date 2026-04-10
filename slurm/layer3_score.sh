#!/bin/bash
#SBATCH -A uppmax2026-1-95
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH -J layer3_score
#SBATCH -o logs/layer3_score_%j.out
#SBATCH -e logs/layer3_score_%j.err

# Step 3: Convert negatives to FlagEmbedding format, then teacher-score with
# BGE-M3 integration score (dense 0.4 + sparse 0.2 + ColBERT 0.4).
# Input:  output/layer3_bge_negatives.json
# Output: output/layer3_bge_training_data_scored.jsonl

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs output

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

python scripts/pipeline/convert_to_flagembedding_format.py \
    --input  output/layer3_bge_negatives.json \
    --output output/layer3_bge_training_data.jsonl

python scripts/pipeline/score_bge_integration.py \
    --input      output/layer3_bge_training_data.jsonl \
    --output     output/layer3_bge_training_data_scored.jsonl \
    --model      BAAI/bge-m3 \
    --batch-size 32

echo "Done: output/layer3_bge_training_data_scored.jsonl"
