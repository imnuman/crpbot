#!/bin/bash
# Multi-GPU training script for p3.8xlarge (4x V100)
# Trains all 3 models in parallel across 4 GPUs

set -e

echo "=== Multi-GPU Training ==="
echo "GPUs available: $(nvidia-smi --list-gpus | wc -l)"
echo "Models: BTC-USD, ETH-USD, SOL-USD"
echo ""

# Check CUDA availability
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
EOF

echo ""
read -p "Start training? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 0
fi

# Get bucket name
if [ -f .s3_bucket_name ]; then
  BUCKET_NAME=$(cat .s3_bucket_name)
else
  echo "Warning: No bucket name found, using local data"
  BUCKET_NAME=""
fi

# Train all models in parallel using different GPUs
echo "[1/3] Starting BTC training on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup python apps/trainer/main.py \
  --task lstm \
  --coin BTC \
  --epochs 15 \
  --device cuda \
  > /tmp/train_btc_gpu.log 2>&1 &
BTC_PID=$!

echo "[2/3] Starting ETH training on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup python apps/trainer/main.py \
  --task lstm \
  --coin ETH \
  --epochs 15 \
  --device cuda \
  > /tmp/train_eth_gpu.log 2>&1 &
ETH_PID=$!

echo "[3/3] Starting SOL training on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup python apps/trainer/main.py \
  --task lstm \
  --coin SOL \
  --epochs 15 \
  --device cuda \
  > /tmp/train_sol_gpu.log 2>&1 &
SOL_PID=$!

echo ""
echo "✅ All training jobs started!"
echo ""
echo "Process IDs:"
echo "  BTC: $BTC_PID (GPU 0)"
echo "  ETH: $ETH_PID (GPU 1)"
echo "  SOL: $SOL_PID (GPU 2)"
echo ""
echo "Monitor training:"
echo "  nvidia-smi      # GPU usage"
echo "  tail -f /tmp/train_btc_gpu.log"
echo "  tail -f /tmp/train_eth_gpu.log"
echo "  tail -f /tmp/train_sol_gpu.log"
echo ""
echo "Estimated completion: 3 minutes"
echo ""

# Monitor training
echo "Monitoring training progress..."
while kill -0 $BTC_PID 2>/dev/null || kill -0 $ETH_PID 2>/dev/null || kill -0 $SOL_PID 2>/dev/null; do
  clear
  echo "=== Training Progress ==="
  date
  echo ""

  # Check BTC
  if kill -0 $BTC_PID 2>/dev/null; then
    echo "🔄 BTC: Running on GPU 0"
    tail -3 /tmp/train_btc_gpu.log | grep -E "Epoch|val_" || echo "  Starting..."
  else
    echo "✅ BTC: Complete"
  fi
  echo ""

  # Check ETH
  if kill -0 $ETH_PID 2>/dev/null; then
    echo "🔄 ETH: Running on GPU 1"
    tail -3 /tmp/train_eth_gpu.log | grep -E "Epoch|val_" || echo "  Starting..."
  else
    echo "✅ ETH: Complete"
  fi
  echo ""

  # Check SOL
  if kill -0 $SOL_PID 2>/dev/null; then
    echo "🔄 SOL: Running on GPU 2"
    tail -3 /tmp/train_sol_gpu.log | grep -E "Epoch|val_" || echo "  Starting..."
  else
    echo "✅ SOL: Complete"
  fi
  echo ""

  # Show GPU usage
  echo "GPU Usage:"
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F, '{printf "  GPU %s: %s%% utilization, %sMB / %sMB memory\n", $1, $3, $4, $5}'
  echo ""

  sleep 10
done

echo ""
echo "✅ All training complete!"
echo ""

# Show final results
echo "Final results:"
echo ""
echo "BTC Model:"
grep -E "Best model|Test accuracy" /tmp/train_btc_gpu.log | tail -3
echo ""
echo "ETH Model:"
grep -E "Best model|Test accuracy" /tmp/train_eth_gpu.log | tail -3
echo ""
echo "SOL Model:"
grep -E "Best model|Test accuracy" /tmp/train_sol_gpu.log | tail -3
echo ""

# Upload models to S3 if bucket is configured
if [ -n "$BUCKET_NAME" ]; then
  echo "Uploading trained models to S3..."
  aws s3 sync models/ "s3://${BUCKET_NAME}/models/production/" \
    --exclude "*.log" \
    --exclude "*.tmp"
  echo "✅ Models uploaded to s3://${BUCKET_NAME}/models/production/"
fi

echo ""
echo "Training complete! Don't forget to terminate the instance to stop charges."
echo "  aws ec2 terminate-instances --instance-ids \$(cat .gpu_instance_info | grep INSTANCE_ID | cut -d= -f2)"
