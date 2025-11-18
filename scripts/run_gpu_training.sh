#!/bin/bash
# GPU Training Deployment Script for V7 Enhanced Model
# Runs on existing g5.xlarge instance for cost efficiency

set -e

# Configuration
GPU_INSTANCE_IP="35.153.176.224"
TRAINING_DATA_PATH="/home/ubuntu/training_data"
MODEL_OUTPUT_PATH="/home/ubuntu/models/v7_enhanced"
BATCH_SIZE=512
EPOCHS=50
LEARNING_RATE=0.001

echo "🚀 Starting V7 Enhanced Model Training on GPU Instance..."

# Copy training script to GPU instance
echo "📁 Copying training script..."
scp -i ~/.ssh/id_rsa apps/trainer/sagemaker_train.py ubuntu@${GPU_INSTANCE_IP}:/home/ubuntu/

# Copy training data if not already present
echo "📊 Ensuring training data is available..."
ssh -i ~/.ssh/id_rsa ubuntu@${GPU_INSTANCE_IP} "mkdir -p ${TRAINING_DATA_PATH} ${MODEL_OUTPUT_PATH}"

# Check if data exists, if not copy it
ssh -i ~/.ssh/id_rsa ubuntu@${GPU_INSTANCE_IP} "
if [ ! -f ${TRAINING_DATA_PATH}/BTC_features.parquet ]; then
    echo 'Training data not found, please upload data first'
    exit 1
fi
"

# Run training with optimized settings for single GPU
echo "🔥 Starting training on NVIDIA A10G GPU..."
ssh -i ~/.ssh/id_rsa ubuntu@${GPU_INSTANCE_IP} "
cd /home/ubuntu
export CUDA_VISIBLE_DEVICES=0

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env

# Run training with mixed precision for faster training
python sagemaker_train.py \
    --data-path ${TRAINING_DATA_PATH} \
    --model-dir ${MODEL_OUTPUT_PATH} \
    --batch-size ${BATCH_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    --epochs ${EPOCHS} \
    --symbols BTC,ETH,SOL
"

# Copy trained model back
echo "📥 Copying trained model back to local..."
mkdir -p models/v7_enhanced
scp -i ~/.ssh/id_rsa ubuntu@${GPU_INSTANCE_IP}:${MODEL_OUTPUT_PATH}/v7_enhanced_model.pt models/v7_enhanced/

echo "✅ V7 Enhanced Model Training Complete!"
echo "📊 Model saved to: models/v7_enhanced/v7_enhanced_model.pt"
echo "💰 Estimated cost: ~$3-5 for 2-3 hours on g5.xlarge"
