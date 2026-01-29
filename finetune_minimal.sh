#!/bin/bash

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rdt

# ==========================================
# RDT-1B 微调脚本 - 超低显存版本
# 使用 8bit optimizer + 更小的 batch
# 适用于: RTX 4080 SUPER 16GB
# ==========================================

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/rdt-1b-stack-blocks"
export WANDB_PROJECT="rdt-stack-blocks-finetune"
export WANDB_MODE="offline"

# PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

mkdir -p "$OUTPUT_DIR"

echo "================================================"
echo "🚀 RDT-1B 微调 - 超低显存版本"
echo "================================================"
echo "💾 数据集: stack_blocks_three (50 episodes)"
echo "💻 GPU: RTX 4080 SUPER (16GB)"
echo "🔧 优化策略:"
echo "   - Batch Size: 1"
echo "   - Gradient Accumulation: 16 (有效 batch=16)"
echo "   - 8-bit Adam Optimizer: 启用"
echo "   - Mixed Precision: BF16"
echo "   - 不使用 DeepSpeed (避免配置问题)"
echo "================================================"

# 不使用 DeepSpeed，使用 accelerate 的原生优化
python main.py \
    --pretrained_model_name_or_path="robotics-diffusion-transformer/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=1 \
    --gradient_accumulation_steps=16 \
    --sample_batch_size=1 \
    --num_sample_batches=2 \
    --max_train_steps=50000 \
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
    --report_to=wandb

# 关键优化：
# 1. 移除 DeepSpeed (配置复杂且未正常工作)
# 2. 使用 8-bit Adam (--use_8bit_adam) 大幅降低优化器显存
# 3. Gradient Accumulation = 16 (更大的累积)
# 4. num_sample_batches=2 (减少验证batch数量)
# 5. dataloader_num_workers=1 (减少内存占用)
