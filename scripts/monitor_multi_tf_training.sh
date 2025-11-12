#!/bin/bash
# Monitor multi-TF model retraining progress

echo "=== Multi-TF LSTM Retraining Progress ==="
echo "Started: $(date)"
echo ""

# Function to extract training progress
extract_progress() {
    local log_file=$1
    local coin=$2

    if [ ! -f "$log_file" ]; then
        echo "$coin: Log file not found"
        return
    fi

    # Get current epoch
    local epoch=$(grep -oP "Epoch \K\d+/\d+" "$log_file" | tail -1)

    # Get latest validation metrics
    local val_loss=$(grep "val_loss" "$log_file" | tail -1 | grep -oP "val_loss=\K[0-9.]+")
    local val_acc=$(grep "val_acc" "$log_file" | tail -1 | grep -oP "val_acc=\K[0-9.]+")

    # Get best metrics
    local best_val_loss=$(grep "New best model" "$log_file" | tail -1 | grep -oP "val_loss=\K[0-9.]+")
    local best_epoch=$(grep "New best model" "$log_file" | tail -1 | grep -oP "epoch \K\d+")

    echo "[$coin]"
    echo "  Current Epoch: ${epoch:-N/A}"
    echo "  Latest Val Loss: ${val_loss:-N/A}"
    echo "  Latest Val Acc: ${val_acc:-N/A}"
    if [ -n "$best_val_loss" ]; then
        echo "  Best Val Loss: $best_val_loss (epoch $best_epoch)"
    fi
    echo ""
}

# Extract progress for each coin
extract_progress "/tmp/train_btc_multi_tf_v2.log" "BTC-USD"
extract_progress "/tmp/train_eth_multi_tf_v2.log" "ETH-USD"
extract_progress "/tmp/train_sol_multi_tf_v2.log" "SOL-USD"

# Check if any training is still running
running_count=$(ps aux | grep "train.*lstm" | grep -v grep | wc -l)
echo "Active training processes: $((running_count / 2))"  # Divide by 2 (uv + python)

# Estimate completion time (assuming 4 minutes per epoch, 15 epochs)
if [ "$running_count" -gt 0 ]; then
    echo ""
    echo "Estimated completion time: ~60 minutes per model"
    echo "All 3 models (parallel): ~60 minutes total"
    echo ""
    echo "Next check: $(date -d '+10 minutes' '+%Y-%m-%d %H:%M:%S')"
fi
