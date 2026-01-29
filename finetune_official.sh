#!/bin/bash

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rdt

# 按照官方文档配置
export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/rdt-170m-stack-blocks"
export WANDB_PROJECT="rdt-stack-blocks-finetune"
export WANDB_MODE="offline"

# PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "✅ 创建输出目录: '$OUTPUT_DIR'"
else
    echo "📁 输出目录已存在: '$OUTPUT_DIR'"
fi

# 单机训练不需要 hostfile，直接使用 deepspeed
deepspeed --num_gpus=1 main.py \
    --deepspeed="./configs/zero2_offload.json" \
    --pretrained_model_name_or_path="robotics-diffusion-transformer/rdt-170m" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=8 \
    --sample_batch_size=8 \
    --max_train_steps=50000 \
    --checkpointing_period=2000 \
    --sample_period=2000 \
    --checkpoints_total_limit=10 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=2 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --precomp_lang_embed \
    --report_to=wandb

# 针对 16GB 显存的激进优化:
# - 使用 zero2_offload.json (优化器卸载到 CPU)
# - train_batch_size: 1 (最小)
# - sample_batch_size: 1 (最小)
# - sample_period: 2000 (减少验证频率)
# - dataloader_num_workers: 2 (减少内存占用)
# - checkpoints_total_limit: 10 (减少保存数量)
