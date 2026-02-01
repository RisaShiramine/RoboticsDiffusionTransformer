#!/bin/bash

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rdt

# ==========================================
# RDT-1B 恢复训练脚本
# 从 checkpoint-2000 恢复
# ==========================================

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/rdt-1b-stack-blocks"
export WANDB_PROJECT="rdt-stack-blocks-finetune"
export WANDB_MODE="offline"

# PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

echo "================================================"
echo "🚀 RDT-1B 恢复训练 (Resume Training)"
echo "================================================"
echo "▶️ 从断点恢复: checkpoint-2000"
echo "💾 数据集: stack_blocks_three (50 episodes)"
echo "💻 GPU: RTX 4080 SUPER (16GB)"
echo "================================================"

# 添加了 --resume_from_checkpoint 参数
python main.py \
    --pretrained_model_name_or_path="robotics-diffusion-transformer/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=1 \
    --gradient_accumulation_steps=16 \
    --sample_batch_size=1 \
    --num_sample_batches=2 \
    --max_train_steps=5000 \
    --checkpointing_period=2000 \
    --sample_period=2000 \
    --checkpoints_total_limit=5 \
    --lr_scheduler="constant" \
    --learning_rate=5e-5 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=1 \
    --use_8bit_adam \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --precomp_lang_embed \
    --report_to=wandb \
    --resume_from_checkpoint="checkpoint-2000"

