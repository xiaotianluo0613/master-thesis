#!/bin/bash
#SBATCH -A uppmax2026-1-95
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH --mem=4G
#SBATCH -J layer1_split
#SBATCH -o logs/layer1_split_%j.out
#SBATCH -e logs/layer1_split_%j.err

# Step 1: Split Layer 1 queries into train/val sets.
# Groups by date so no document group leaks across splits.
# Input:  data/layer1_queries.json  (5514 queries, 1838 groups)
# Output: data/layer1_train_queries.json (~4950 train)
#         data/layer1_val_queries.json   (~87 val, all positives kept)

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

python scripts/pipeline/split_train_val.py \
    --queries      data/layer1_queries.json \
    --train-output data/layer1_train_queries.json \
    --val-output   data/layer1_val_queries.json \
    --val-size     551 \
    --seed         42

echo "Done: data/layer1_train_queries.json + data/layer1_val_queries.json"
