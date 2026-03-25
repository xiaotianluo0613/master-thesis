#!/bin/bash
# Activate thesis conda environment and run baseline test

echo "Activating thesis environment..."
eval "$(conda shell.bash hook)"
conda activate thesis

echo "Running BGE-M3 baseline evaluation..."
python scripts/baseline_bge_m3.py "$@"
