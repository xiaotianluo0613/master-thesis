#!/bin/bash
#SBATCH -A uppmax2026-1-95
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -J layer1_finetune
#SBATCH -o logs/layer1_finetune_%j.out
#SBATCH -e logs/layer1_finetune_%j.err

# Step 4: Fine-tune BGE-M3 unified (m3_kd_loss, self-distillation).
# ~4950 training examples, 7 hard negatives each.
# Input:  output/layer1_bge_training_data_scored.jsonl
# Output: output/models/layer1-bge-m3-unified/

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs output/models/layer1-bge-m3-unified

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

torchrun --nproc_per_node 1 \
    -m FlagEmbedding.finetune.embedder.encoder_only.m3 \
    --model_name_or_path        BAAI/bge-m3 \
    --train_data                output/layer1_bge_training_data_scored.jsonl \
    --output_dir                output/models/layer1-bge-m3-unified \
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
    --learning_rate             1e-5 \
    --warmup_ratio              0.1 \
    --temperature               0.02 \
    --sentence_pooling_method   cls \
    --normalize_embeddings      True \
    --fp16 \
    --gradient_checkpointing \
    --dataloader_drop_last      True \
    --logging_steps             50 \
    --save_steps                500 \
    --overwrite_output_dir

echo "Done: output/models/layer1-bge-m3-unified"
