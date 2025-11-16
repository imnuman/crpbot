#!/bin/bash
# Download V6 models from GPU instance after training completes

set -e

GPU_IP="98.91.192.206"
KEY="~/.ssh/crpbot-training.pem"

echo "🔽 Downloading V6 Models from GPU Instance"
echo "==========================================="
echo ""

# Create v6 directory
mkdir -p models/v6_gpu

# Download models
echo "Downloading trained models..."
rsync -avz --progress -e "ssh -i $KEY" \
    ec2-user@${GPU_IP}:~/models/ \
    models/v6_gpu/

echo ""
echo "✅ Models downloaded to models/v6_gpu/"
echo ""

# List downloaded models
echo "Downloaded models:"
ls -lh models/v6_gpu/*.pt 2>/dev/null || echo "No .pt files found"
echo ""

# Promote to runtime
echo "Promoting V6 models to production..."
cp models/v6_gpu/*.pt models/promoted/ 2>/dev/null || echo "No models to promote yet"

echo ""
echo "✅ V6 models ready for testing!"
echo ""
echo "Next steps:"
echo "  1. Test predictions: ./run_runtime_with_env.sh --mode dryrun --iterations 5"
echo "  2. Verify >50% confidence in predictions"
echo "  3. If good, restart live bot"
