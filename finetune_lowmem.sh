#!/bin/bash

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rdt

# ==========================================
# RDT-1B 微调脚本 - 低显存优化版本
# 适用于: RTX 4090 16GB
# ==========================================

# 模型路径配置
export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/rdt-1b-stack-blocks"

# Wandb 配置（可选）
export WANDB_PROJECT="rdt-stack-blocks-finetune"

# PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# 创建输出目录
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "✅ 创建输出目录: '$OUTPUT_DIR'"
else
    echo "📁 输出目录已存在: '$OUTPUT_DIR'"
fi

echo "================================================"
echo "🚀 RDT-1B 微调 - 低显存优化版本"
echo "================================================"
echo "💾 数据集: stack_blocks_three (50 episodes)"
echo "💻 GPU: RTX 4090 (16GB)"
echo "🔧 优化策略:"
echo "   - Batch Size: 1 (极小)"
echo "   - Gradient Accumulation: 8 (模拟 batch=8)"
echo "   - DeepSpeed ZeRO-2: 启用"
echo "   - Mixed Precision: BF16"
echo "   - Precomputed Lang Embeds: 启用"
echo "================================================"

# 使用 accelerate 启动（单机单卡）
accelerate launch main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="robotics-diffusion-transformer/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --sample_batch_size=1 \
    --max_train_steps=50000 \
    --checkpointing_period=2000 \
    --sample_period=1000 \
    --checkpoints_total_limit=10 \
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

# 参数说明：
# --train_batch_size=1              # 每个GPU的batch size设为1（最小）
# --gradient_accumulation_steps=8   # 梯度累积8步，相当于batch=8
# --sample_batch_size=1             # 验证时batch size也设为1
# --learning_rate=5e-5              # 降低学习率（因为有效batch size更小）
# --checkpointing_period=2000       # 每2000步保存（减少保存频率）
# --checkpoints_total_limit=10      # 只保留10个checkpoint
# --dataloader_num_workers=2        # 减少数据加载workers
