#!/bin/bash
# Layer 1 data preparation — runs LOCALLY (XML transcription files are local)
#
# Step 1: Build chunks   (~7300 chunks, proportional sampling)
#   Input:  output/data_pools/train_layer1_pool.txt
#   Output: data/layer1_chunks.json
#
# Step 2: Group chunks   (groups of 3-4 consecutive pages per volume)
#   Input:  data/layer1_chunks.json
#   Output: data/layer1_chunks_grouped.json
#
# Step 3: Generate queries  (~2000 groups × 3 queries = ~6000 queries)
#   Input:  data/layer1_chunks_grouped.json
#   Output: data/layer1_queries.json
#
# After this script: scp data files to UPPMAX, then run slurm jobs.
#
# Requires: GEMINI_API_KEY set in environment
# Resume:   re-run with same command if interrupted — step 3 uses --resume

set -e

cd "$(dirname "$0")/../.."

source .venv/bin/activate

# ── Step 1: Build chunks ──────────────────────────────────────────────────────
echo "=== Step 1: Building Layer 1 chunks ==="

python scripts/pipeline/build_layer1_chunks.py \
    --pool                output/data_pools/train_layer1_pool.txt \
    --fingerprints        output/comprehensive_volume_fingerprints.csv \
    --transcriptions-root Filedrop-7hXHfBBqt2nJHQuk/home/dgxuser/erik/data/transcriptions \
    --target-chunks       7300 \
    --min-text-chars      220 \
    --seed                42 \
    --output              data/layer1_chunks.json

echo "Done: data/layer1_chunks.json"

# ── Step 2: Group chunks ──────────────────────────────────────────────────────
echo "=== Step 2: Grouping chunks (3-4 per group) ==="

python scripts/pipeline/group_layer1_pairs_chunks_3_4.py \
    --input   data/layer1_chunks.json \
    --output  data/layer1_chunks_grouped.json \
    --summary data/layer1_chunks_grouped_summary.txt

echo "Done: data/layer1_chunks_grouped.json"

# ── Step 3: Generate queries ──────────────────────────────────────────────────
echo "=== Step 3: Generating N-to-N queries ==="

python scripts/pipeline/generate_n_to_n_queries_layered.py \
    --chunks          data/layer1_chunks_grouped.json \
    --fewshot         data/n_to_n_fewshot_examples.json \
    --pool-dir        output/data_pools \
    --output          data/layer1_queries.json \
    --provider        gemini \
    --model           gemini-2.5-flash \
    --queries-per-day 3 \
    --temperature     1.0 \
    --delay           4.0 \
    --resume

echo "Done: data/layer1_queries.json"
echo ""
echo "=== All steps complete ==="
echo "Next: scp data/layer1_queries.json and data/layer1_chunks_grouped.json to UPPMAX"
