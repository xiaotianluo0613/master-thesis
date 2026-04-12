#!/bin/bash
#SBATCH -A uppmax2026-1-95
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -J layer3_eval
#SBATCH -o logs/layer3_eval_%j.out
#SBATCH -e logs/layer3_eval_%j.err

# Layer 3 evaluation — two passes:
#   Global: 1150 queries (all 4 layers), 13800-chunk corpus
#   Layer-specific: L3 val queries (250), L3 corpus only (2500 chunks)
# Step 1: merge L3 LoRA adapter into full checkpoint
# Step 2: build combined corpus (L1+L2+L3+L4, 13800 chunks) if not already built
# Step 3: evaluate baseline -> L1 -> L2 -> L3 on global val
# Step 4: evaluate baseline -> L1 -> L2 -> L3 on L3 val only
# Output: output/layer3_eval_results.json, output/layer3_eval_layer_specific_results.json

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs output output/models/layer3-bge-m3-lora-dense-b4-merged

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

# Step 1: merge L3 LoRA checkpoint
echo "=== Merging L3 LoRA checkpoint ==="
python scripts/pipeline/merge_lora_checkpoint.py \
    --adapter-dir output/models/layer3-bge-m3-lora-dense-b4 \
    --output-dir  output/models/layer3-bge-m3-lora-dense-b4-merged \
    --base-model  output/models/layer2-bge-m3-lora-dense-b4-merged

# Step 2: build combined corpus (reuse if already exists)
if [ ! -f data/global_val_chunks.json ]; then
    echo "=== Building combined corpus ==="
    python - <<'EOF'
import json
from pathlib import Path

layers = ["layer1", "layer2", "layer3", "layer4"]
all_chunks = []
for layer in layers:
    path = Path(f"data/{layer}_chunks_grouped.json")
    d = json.load(open(path, encoding="utf-8"))
    all_chunks.extend(d["chunks"])
    print(f"  {layer}: {len(d['chunks'])} chunks")

out = {"chunks": all_chunks}
with open("data/global_val_chunks.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print(f"Combined corpus: {len(all_chunks)} chunks -> data/global_val_chunks.json")
EOF
else
    echo "=== Combined corpus already exists, skipping ==="
fi

# Step 3: global eval
echo "=== Global evaluation ==="
python scripts/pipeline/evaluate_comparison.py \
    --chunks     data/global_val_chunks.json \
    --queries    data/global_val_queries.json \
    --models     BAAI/bge-m3 \
                 output/models/layer1-bge-m3-lora-dense-b4-merged \
                 output/models/layer2-bge-m3-lora-dense-b4-merged \
                 output/models/layer3-bge-m3-lora-dense-b4-merged \
    --k-values   10 100 \
    --batch-size 64 \
    --output     output/layer3_eval_results.json

echo "Done: output/layer3_eval_results.json"

# Step 4: layer-specific eval
echo "=== Layer 3-specific evaluation ==="
python scripts/pipeline/evaluate_comparison.py \
    --chunks     data/layer3_chunks_grouped.json \
    --queries    data/layer3_val_queries.json \
    --models     BAAI/bge-m3 \
                 output/models/layer1-bge-m3-lora-dense-b4-merged \
                 output/models/layer2-bge-m3-lora-dense-b4-merged \
                 output/models/layer3-bge-m3-lora-dense-b4-merged \
    --k-values   10 100 \
    --batch-size 64 \
    --output     output/layer3_eval_layer_specific_results.json

echo "Done: output/layer3_eval_layer_specific_results.json"
