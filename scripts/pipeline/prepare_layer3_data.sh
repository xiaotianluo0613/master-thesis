#!/bin/bash
# Layer 3 local data preparation — runs on local Mac (XML transcription files are local)
#
# Step 1: Build chunks   (~2500 chunks, Legal ALL up to target)
#   Input:  output/data_pools/train_layer3_pool.txt
#   Output: data/layer3_chunks.json
#
# Step 2: Group chunks   (groups of 3-4 consecutive pages per volume)
#   Input:  data/layer3_chunks.json
#   Output: data/layer3_chunks_grouped.json
#
# Query generation (Step 3) runs on UPPMAX login node — see slurm/layer3_query_gen.sh
#
# After this script:
#   scp data/layer3_chunks_grouped.json to UPPMAX, then run layer3_query_gen.sh

set -e

cd "$(dirname "$0")/../.."

source .venv/bin/activate

# ── Step 1: Build chunks ──────────────────────────────────────────────────────
echo "=== Step 1: Building Layer 3 chunks ==="

python scripts/pipeline/build_layer3_chunks.py \
    --pool                output/data_pools/train_layer3_pool.txt \
    --fingerprints        output/comprehensive_volume_fingerprints.csv \
    --transcriptions-root Filedrop-7hXHfBBqt2nJHQuk/home/dgxuser/erik/data/transcriptions \
    --target-chunks       2500 \
    --min-text-chars      220 \
    --seed                42 \
    --output              data/layer3_chunks.json

echo "Done: data/layer3_chunks.json"

# ── Step 2: Group chunks ──────────────────────────────────────────────────────
echo "=== Step 2: Grouping chunks (3-4 per group) ==="

python scripts/pipeline/group_layer1_pairs_chunks_3_4.py \
    --input   data/layer3_chunks.json \
    --output  data/layer3_chunks_grouped.json \
    --summary data/layer3_chunks_grouped_summary.txt

echo "Done: data/layer3_chunks_grouped.json"
echo ""
echo "=== Local steps complete ==="
echo "Next: scp data/layer3_chunks_grouped.json to UPPMAX"
echo "Then on UPPMAX login node: bash slurm/layer3_query_gen.sh"
