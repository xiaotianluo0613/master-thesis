#!/bin/bash
# Layer 4 query generation — runs on UPPMAX LOGIN NODE (not sbatch).
# Login nodes have internet access; compute nodes do not.
#
# Generates N-to-N queries via Gemini for Layer 4 chunks.
# Uses --resume so it is safe to re-run if interrupted.
#
# Input:  data/layer4_chunks_grouped.json  (scp'd from local Mac)
# Output: data/layer4_queries.json
#
# Usage (run directly on login node, not via sbatch):
#   export GEMINI_API_KEY=<your-key>
#   nohup bash slurm/layer4_query_gen.sh > logs/layer4_query_gen.log 2>&1 &
#   tail -f logs/layer4_query_gen.log

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs data

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY is not set. Export it before running."
    exit 1
fi

echo "=== Layer 4 query generation ==="
echo "Input:  data/layer4_chunks_grouped.json"
echo "Output: data/layer4_queries.json"
echo "Started: $(date)"

python scripts/pipeline/generate_n_to_n_queries_layered.py \
    --chunks                  data/layer4_chunks_grouped.json \
    --fewshot                 data/n_to_n_fewshot_examples.json \
    --pool-dir                output/data_pools \
    --output                  data/layer4_queries.json \
    --provider                gemini \
    --model                   gemini-2.5-flash \
    --queries-per-day         3 \
    --temperature             1.0 \
    --delay                   4.0 \
    --disable-baseline-filter \
    --resume

echo "Done: data/layer4_queries.json"
echo "Finished: $(date)"
echo "Next: sbatch slurm/layer4_split.sh"
