#!/bin/bash
#SBATCH -A uppmax2025-2-290
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH -J finetune_gpl
#SBATCH -o logs/finetune_gpl_%j.out
#SBATCH -e logs/finetune_gpl_%j.err

# GPL fine-tuning — kl_div loss, full fine-tuning
# Input:  output/gpl_training_data.jsonl  (from score_margins_gpl.py)
# Output: output/models/bge-m3-gpl/

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs output/models/bge-m3-gpl

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

torchrun --nproc_per_node 1 \
    -m FlagEmbedding.finetune.embedder.encoder_only.m3 \
    --model_name_or_path BAAI/bge-m3 \
    --train_data output/gpl_training_data.jsonl \
    --output_dir output/models/bge-m3-gpl \
    --train_group_size 2 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --knowledge_distillation True \
    --kd_loss_type kl_div \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --fp16 \
    --gradient_checkpointing \
    --dataloader_drop_last True \
    --logging_steps 10 \
    --save_steps 500 \
    --overwrite_output_dir

echo "Done: output/models/bge-m3-gpl"
