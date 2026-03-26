#!/bin/bash
#SBATCH -A uppmax2025-2-290
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH -J finetune_bge
#SBATCH -o logs/finetune_bge_%j.out
#SBATCH -e logs/finetune_bge_%j.err

# BGE-M3 unified fine-tuning with LoRA + self-distillation (m3_kd_loss)
# Input:  output/bge_training_data.jsonl  (from convert_to_flagembedding_format.py)
# Output: output/models/bge-m3-unified/

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs output/models/bge-m3-unified

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

torchrun --nproc_per_node 1 \
    -m FlagEmbedding.finetune.embedder.encoder_only.m3 \
    --model_name_or_path BAAI/bge-m3 \
    --train_data output/bge_training_data.jsonl \
    --output_dir output/models/bge-m3-unified \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --knowledge_distillation True \
    --unified_finetuning True \
    --use_self_distill True \
    --kd_loss_type m3_kd_loss \
    --self_distill_start_step 0 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --fp16 \
    --gradient_checkpointing \
    --dataloader_drop_last True \
    --logging_steps 10 \
    --save_steps 500 \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --overwrite_output_dir

echo "Done: output/models/bge-m3-unified"
