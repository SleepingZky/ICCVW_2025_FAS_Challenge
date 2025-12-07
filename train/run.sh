#!/bin/bash
# Single-GPU CLIP-LoRA training with ACCUMU_STEPS=1

# ====== CONFIGURATION ======
# Data paths
REAL_DATA="dataset/MSCOCO_vae_rec/real/"
VAE_DATA="dataset/MSCOCO_vae_rec/sd2.1/"

# Training config
BATCH_SIZE=16
LEARNING_RATE=1e-4
EPOCHS=2
EXP_NAME="Checkpoints"
ACCUMU_STEPS=1

# ====== TRAINING ======
echo "Starting single-GPU CLIP-LoRA training on GPU 0..."


EFFECTIVE_BATCH=$((BATCH_SIZE * ACCUMU_STEPS))
CHECKPOINT_DIR="checkpoints_batch_${EFFECTIVE_BATCH}"
echo "Training with accumulation steps: $ACCUMU_STEPS (effective batch: $EFFECTIVE_BATCH)"
mkdir -p "${EXP_NAME}/${CHECKPOINT_DIR}"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --gpu_ids 0 \
    --seed 42 \
    --name "$EXP_NAME" \
    --checkpoints_dir "$CHECKPOINT_DIR" \
    --contrastive \
    --cropSize 224 \
    --real_data_path "$REAL_DATA" \
    --vae_rec_data_path "$VAE_DATA" \
    --data_mode "mscoco" \
    --arch "CLIP-LoRA:ViT-L/14" \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --accumulation_steps $ACCUMU_STEPS \
    --optim "adam" \
    --niter $EPOCHS \
    --p_jpeg_fake 0.5 \
    --jpeg_quality 100 \
    --down_resize_factors 0.2 \
    --upper_resize_factors 3.5 \
    --quality_json "./MSCOCO_train2017.json" \
    --mix_color_space "RGB" \
    --p_pixelmix 0.2 \
    --r_pixelmix 0.8 \
    --meth_pixelmix "uniform" \
    --p_freqmix 0.2 \
    --r_freqmix 0.8 \
    --meth_freqmix "uniform" \
    2>&1 | tee "${EXP_NAME}/${CHECKPOINT_DIR}/training_batch${EFFECTIVE_BATCH}.log"

echo "CLIP-LoRA training completed!"
echo "Results saved in: ${EXP_NAME}/${CHECKPOINT_DIR}/"
echo "Check log: ${EXP_NAME}/${CHECKPOINT_DIR}/training_batch${EFFECTIVE_BATCH}.log"