#!/bin/bash
# V8 GPU Training Setup Script
# Run this on AWS g5.xlarge instance

set -e

echo "🚀 V8 GPU Training Setup"
echo "========================"

# Check if running on GPU instance
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ No GPU detected. This script requires a GPU instance."
    exit 1
fi

echo "✅ GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y git wget curl htop

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other ML libraries
echo "📊 Installing ML libraries..."
pip install pandas numpy scikit-learn matplotlib seaborn
pip install jupyter notebook

# Verify CUDA installation
echo "🔍 Verifying CUDA installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Create directories
echo "📁 Creating directories..."
mkdir -p models/v8_enhanced
mkdir -p data
mkdir -p logs
mkdir -p reports

# Download training data (if not already present)
if [ ! -f "btc_data.csv" ]; then
    echo "📈 Downloading training data..."
    # Add your data download commands here
    echo "⚠️  Please upload your training data files:"
    echo "   - btc_data.csv"
    echo "   - eth_data.csv" 
    echo "   - sol_data.csv"
fi

# Set up monitoring
echo "📊 Setting up monitoring..."
cat > monitor_training.sh << 'EOF'
#!/bin/bash
# Monitor GPU usage during training
while true; do
    clear
    echo "=== GPU Monitoring ==="
    nvidia-smi
    echo ""
    echo "=== Disk Usage ==="
    df -h
    echo ""
    echo "=== Memory Usage ==="
    free -h
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 5
done
EOF
chmod +x monitor_training.sh

# Create training launcher
echo "🎯 Creating training launcher..."
cat > train_v8_all.sh << 'EOF'
#!/bin/bash
# Train all V8 models

set -e

echo "🚀 Starting V8 Training for All Models"
echo "======================================"

# Log file
LOG_FILE="logs/v8_training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting V8 enhanced training"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Train each model
for symbol in BTC-USD ETH-USD SOL-USD; do
    log "Training $symbol model..."
    
    if python v8_enhanced_training.py --symbol "$symbol" --epochs 100 --batch-size 256 2>&1 | tee -a "$LOG_FILE"; then
        log "✅ $symbol training completed successfully"
    else
        log "❌ $symbol training failed"
        exit 1
    fi
    
    # Brief pause between models
    sleep 10
done

log "🎉 All V8 models trained successfully!"

# Run diagnostics
log "Running model diagnostics..."
if python diagnose_v8_models.py --all-models --output "reports/v8_diagnostic_$(date +%Y%m%d_%H%M%S).json" 2>&1 | tee -a "$LOG_FILE"; then
    log "✅ Diagnostics completed"
else
    log "⚠️ Diagnostics had issues"
fi

log "Training session complete. Check $LOG_FILE for details."
EOF
chmod +x train_v8_all.sh

# Create quick diagnostic script
echo "🔍 Creating diagnostic script..."
cat > quick_diagnostic.sh << 'EOF'
#!/bin/bash
# Quick diagnostic of V8 models

echo "🔍 Quick V8 Model Diagnostic"
echo "============================"

for symbol in BTC-USD ETH-USD SOL-USD; do
    echo ""
    echo "Checking $symbol model..."
    
    model_file="models/v8_enhanced/lstm_${symbol}_v8_enhanced.pt"
    processor_file="models/v8_enhanced/processor_${symbol}_v8.pkl"
    
    if [ -f "$model_file" ] && [ -f "$processor_file" ]; then
        echo "✅ Files exist"
        echo "   Model: $(ls -lh $model_file | awk '{print $5}')"
        echo "   Processor: $(ls -lh $processor_file | awk '{print $5}')"
    else
        echo "❌ Missing files"
        [ ! -f "$model_file" ] && echo "   Missing: $model_file"
        [ ! -f "$processor_file" ] && echo "   Missing: $processor_file"
    fi
done

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
EOF
chmod +x quick_diagnostic.sh

# Create cost monitoring script
echo "💰 Creating cost monitoring..."
cat > cost_monitor.sh << 'EOF'
#!/bin/bash
# Monitor training costs

INSTANCE_TYPE="g5.xlarge"
HOURLY_RATE="1.006"  # USD per hour

start_time=$(date +%s)

while true; do
    current_time=$(date +%s)
    elapsed_hours=$(echo "scale=2; ($current_time - $start_time) / 3600" | bc)
    estimated_cost=$(echo "scale=2; $elapsed_hours * $HOURLY_RATE" | bc)
    
    clear
    echo "💰 Training Cost Monitor"
    echo "======================="
    echo "Instance: $INSTANCE_TYPE"
    echo "Rate: \$$HOURLY_RATE/hour"
    echo "Elapsed: ${elapsed_hours}h"
    echo "Estimated Cost: \$$estimated_cost"
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    
    sleep 60
done
EOF
chmod +x cost_monitor.sh

# Final setup verification
echo ""
echo "🎉 V8 GPU Training Setup Complete!"
echo "=================================="
echo ""
echo "Available commands:"
echo "  ./train_v8_all.sh          - Train all V8 models"
echo "  ./monitor_training.sh      - Monitor GPU usage"
echo "  ./quick_diagnostic.sh      - Quick model check"
echo "  ./cost_monitor.sh          - Monitor training costs"
echo ""
echo "Manual training:"
echo "  python v8_enhanced_training.py --symbol BTC-USD --epochs 100"
echo "  python v8_enhanced_training.py --all --epochs 100"
echo ""
echo "Diagnostics:"
echo "  python diagnose_v8_models.py --all-models"
echo ""
echo "Next steps:"
echo "1. Upload your training data (btc_data.csv, eth_data.csv, sol_data.csv)"
echo "2. Run: ./train_v8_all.sh"
echo "3. Monitor progress with: ./monitor_training.sh"
echo "4. Check results with: ./quick_diagnostic.sh"
echo ""
echo "⚠️  Remember to terminate the instance when training is complete!"
echo "   Estimated cost for full training: \$6-8"
