#!/bin/bash
#SBATCH -A uppmax2026-1-95
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -J hyper_B
#SBATCH -o logs/hyper_B_%j.out
#SBATCH -e logs/hyper_B_%j.err

# Hyperparameter tuning — Run B
# LR: 1e-5 | Scheduler: cosine | Epochs: 2
# Starts from BAAI/bge-m3 base, trains on all 4 layers combined (~45K examples)

set -e

PROJECT_DIR=/proj/uppmax2025-2-505/$USER/master_thesis
cd $PROJECT_DIR

mkdir -p logs output/models/hyper_B

module load Python/3.11.5-GCCcore-13.3.0
source .venv/bin/activate

TRAIN_DATA=output/layer1_layer2_layer3_layer4_combined_training_data.jsonl
if [ ! -f $TRAIN_DATA ]; then
    echo "Combining training data..."
    cat output/layer1_bge_training_data_scored.jsonl \
        output/layer2_bge_training_data_scored.jsonl \
        output/layer3_bge_training_data_scored.jsonl \
        output/layer4_bge_training_data_scored.jsonl \
        > $TRAIN_DATA
fi

torchrun --nproc_per_node 1 \
    -m FlagEmbedding.finetune.embedder.encoder_only.m3 \
    --model_name_or_path        BAAI/bge-m3 \
    --train_data                $TRAIN_DATA \
    --output_dir                output/models/hyper_B \
    --train_group_size          8 \
    --query_max_len             512 \
    --passage_max_len           512 \
    --knowledge_distillation    True \
    --unified_finetuning        True \
    --use_self_distill          True \
    --kd_loss_type              m3_kd_loss \
    --self_distill_start_step   0 \
    --num_train_epochs          2 \
    --per_device_train_batch_size 4 \
    --learning_rate             1e-5 \
    --lr_scheduler_type         cosine \
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

echo "Done: output/models/hyper_B"

echo "=== Merging hyper_B LoRA checkpoint ==="
python scripts/pipeline/merge_lora_checkpoint.py \
    --adapter-dir output/models/hyper_B \
    --output-dir  output/models/hyper_B-merged \
    --base-model  BAAI/bge-m3

echo "Done: output/models/hyper_B-merged"
