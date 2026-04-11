#!/bin/bash
#SBATCH -A uppmax2026-1-95
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -J layer3_finetune
#SBATCH -o logs/layer3_finetune_%j.out
#SBATCH -e logs/layer3_finetune_%j.err

# Layer 3 cumulative fine-tune: LoRA + dense, trained on L1+L2+L3 combined data.
# Continues from the Layer 2 merged checkpoint.
# LoRA: r=16, alpha=32, target=Q+K+V+dense, lr=1e-4, batch=4
# Input:  output/layer1_bge_training_data_scored.jsonl
#         output/layer2_bge_training_data_scored.jsonl
#         output/layer3_bge_training_data_scored.jsonl
# Combined: output/layer1_layer2_layer3_combined_training_data.jsonl
# Output: output/models/layer3-bge-m3-lora-dense-b4/
# Merged: output/models/layer3-bge-m3-lora-dense-b4-merged/

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs output/models/layer3-bge-m3-lora-dense-b4

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

# Combine L1 + L2 + L3 training data
echo "Combining Layer 1, 2, and 3 training data..."
cat output/layer1_bge_training_data_scored.jsonl \
    output/layer2_bge_training_data_scored.jsonl \
    output/layer3_bge_training_data_scored.jsonl \
    > output/layer1_layer2_layer3_combined_training_data.jsonl
echo "Combined: output/layer1_layer2_layer3_combined_training_data.jsonl"

# Fine-tune starting from the Layer 2 merged checkpoint
torchrun --nproc_per_node 1 \
    -m FlagEmbedding.finetune.embedder.encoder_only.m3 \
    --model_name_or_path        output/models/layer2-bge-m3-lora-dense-b4-merged \
    --train_data                output/layer1_layer2_layer3_combined_training_data.jsonl \
    --output_dir                output/models/layer3-bge-m3-lora-dense-b4 \
    --train_group_size          8 \
    --query_max_len             512 \
    --passage_max_len           512 \
    --knowledge_distillation    True \
    --unified_finetuning        True \
    --use_self_distill          True \
    --kd_loss_type              m3_kd_loss \
    --self_distill_start_step   0 \
    --num_train_epochs          3 \
    --per_device_train_batch_size 4 \
    --learning_rate             1e-4 \
    --warmup_ratio              0.1 \
    --temperature               0.02 \
    --sentence_pooling_method   cls \
    --normalize_embeddings      True \
    --fp16 \
    --gradient_checkpointing \
    --dataloader_drop_last      True \
    --logging_steps             50 \
    --save_steps                500 \
    --overwrite_output_dir \
    --use_lora                  True \
    --lora_rank                 16 \
    --lora_alpha                32 \
    --lora_dropout              0.05 \
    --lora_target_modules       "query,key,value,dense"

echo "Done: output/models/layer3-bge-m3-lora-dense-b4"

# Merge LoRA adapter into full model for next layer
echo "=== Merging L3 LoRA checkpoint ==="
python scripts/pipeline/merge_lora_checkpoint.py \
    --adapter-dir output/models/layer3-bge-m3-lora-dense-b4 \
    --output-dir  output/models/layer3-bge-m3-lora-dense-b4-merged \
    --base-model  output/models/layer2-bge-m3-lora-dense-b4-merged

echo "Done: output/models/layer3-bge-m3-lora-dense-b4-merged"
