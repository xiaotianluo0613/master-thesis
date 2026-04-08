#!/bin/bash
#SBATCH -A uppmax2026-1-95
#SBATCH -p core
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH --mem=4G
#SBATCH -J layer4_split
#SBATCH -o logs/layer4_split_%j.out
#SBATCH -e logs/layer4_split_%j.err

# Step 1: Split Layer 4 queries into train/val sets.
# Groups by date so no document group leaks across splits.
# Input:  data/layer4_queries.json  (~1500 queries)
# Output: data/layer4_train_queries.json (~90% train)
#         data/layer4_val_queries.json   (~10% val, all positives kept)

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

python scripts/pipeline/split_train_val.py \
    --queries      data/layer4_queries.json \
    --train-output data/layer4_train_queries.json \
    --val-output   data/layer4_val_queries.json \
    --val-size     150 \
    --seed         42

echo "Done: data/layer4_train_queries.json + data/layer4_val_queries.json"
