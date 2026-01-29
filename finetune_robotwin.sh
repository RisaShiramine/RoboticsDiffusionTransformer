# export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=1

# Use precomputed embeddings (Required for 16GB VRAM)
export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/rdt-finetune-robotwin-170m"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# Run without DeepSpeed (Single GPU, Standard PyTorch)
# Using python directly avoiding distributed launchers
python main.py \
    --pretrained_model_name_or_path="/mnt/hdd/RoboticsDiffusionTransformer/checkpoints/rdt-170m-stack-blocks" \
    --pretrained_text_encoder_name_or_path="google/t5-v1_1-xxl" \
    --pretrained_vision_encoder_name_or_path="google/siglip-so400m-patch14-384" \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=2 \
    --sample_batch_size=2 \
    --max_train_steps=200000 \
    --checkpointing_period=1000 \
    --sample_period=1000 \
    --checkpoints_total_limit=10 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=4 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --report_to=tensorboard \
    --precomp_lang_embed \
    --use_8bit_adam \
    --gradient_accumulation_steps=16
