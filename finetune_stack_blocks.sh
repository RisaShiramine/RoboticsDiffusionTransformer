#!/bin/bash

# 网络配置（单机单卡可以忽略这些）
# export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
# export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_DEBUG=INFO
# export NCCL_NVLS_ENABLE=0

# 模型路径配置
export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/rdt-1b-stack-blocks"

# CUTLASS 路径（如果没有可以不设置）
# export CUTLASS_PATH="/path/to/cutlass"

# Wandb 配置（如果要使用 wandb 记录训练过程）
export WANDB_PROJECT="rdt-stack-blocks-finetune"

# 创建输出目录
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "✅ 创建输出目录: '$OUTPUT_DIR'"
else
    echo "📁 输出目录已存在: '$OUTPUT_DIR'"
fi

echo "================================================"
echo "🚀 开始微调 RDT-1B 模型 - Stack Blocks 任务"
echo "================================================"
echo "📦 数据集: stack_blocks_three (50 episodes)"
echo "💾 语言嵌入: data/datasets/stack_blocks_three/data/lang_embeds/"
echo "💻 GPU: RTX 4090 (16GB)"
echo "🎯 预训练模型: robotics-diffusion-transformer/rdt-1b"
echo "================================================"

# 单机单卡训练（使用 accelerate）
accelerate launch main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="robotics-diffusion-transformer/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=8 \
    --sample_batch_size=16 \
    --max_train_steps=50000 \
    --checkpointing_period=1000 \
    --sample_period=500 \
    --checkpoints_total_limit=20 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=4 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --precomp_lang_embed \
    --report_to=wandb

# 如果要从某个 checkpoint 恢复训练，取消下面的注释
# --resume_from_checkpoint="checkpoint-5000" \

# 说明：
# 1. 使用 --precomp_lang_embed 标志，因为我们已经预计算了语言嵌入
# 2. batch_size 设为 8（考虑到 16GB 显存）
# 3. max_train_steps 设为 50000（可以根据需要调整）
# 4. 使用 bf16 混合精度训练以节省显存
# 5. 启用图像增强 (--image_aug) 提升泛化能力
