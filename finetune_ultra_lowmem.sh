#!/bin/bash

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rdt

# ==========================================
# RDT-1B 微调脚本 - 极限省显存版本
# 使用 DeepSpeed ZeRO-3 + CPU Offload
# 适用于: RTX 4090 16GB
# ==========================================

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/rdt-1b-stack-blocks"
export WANDB_PROJECT="rdt-stack-blocks-finetune"

# PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "================================================"
echo "🚀 RDT-1B 微调 - 极限省显存版本"
echo "================================================"
echo "⚡ 使用 DeepSpeed ZeRO-3 + CPU Offload"
echo "   优化器状态 -> CPU"
echo "   模型参数 -> CPU"
echo "   梯度 -> 分片"
echo "================================================"

accelerate launch main.py \
    --deepspeed="./configs/zero2_offload.json" \
    --pretrained_model_name_or_path="robotics-diffusion-transformer/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=1 \
    --gradient_accumulation_steps=16 \
    --sample_batch_size=1 \
    --max_train_steps=50000 \
    --checkpointing_period=2000 \
    --sample_period=2000 \
    --checkpoints_total_limit=5 \
    --lr_scheduler="constant" \
    --learning_rate=5e-5 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=2 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --precomp_lang_embed \
    --report_to=wandb

# 关键优化：
# 1. ZeRO-2 + CPU Offload: 优化器和参数卸载到CPU
# 2. Batch Size = 1: 最小batch size
# 3. Gradient Accumulation = 16: 累积更多步骤
# 4. Gradient Checkpointing: 重计算代替存储
# 5. 减少验证频率: sample_period=2000
# 6. 减少checkpoint数量: checkpoints_total_limit=5
