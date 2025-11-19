#!/bin/bash
#
# Auto-Terminate AWS GPU Instance After Training
#
# This script should be run on the AWS GPU instance itself.
# It monitors the training process and automatically terminates the instance when done.
#
# Usage:
#   1. Copy this script to the GPU instance
#   2. Run training with: nohup ./auto_terminate_after_training.sh "training_command" &
#   3. The instance will self-terminate when training completes
#

set -e

TRAINING_COMMAND="$1"
MAX_TRAINING_HOURS="${2:-4}"  # Default: 4 hours max
LOG_FILE="/tmp/auto_terminate.log"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Validate we're on EC2
if ! command -v ec2-metadata &> /dev/null; then
    log "ERROR: This script must run on an AWS EC2 instance"
    exit 1
fi

# Get instance ID
INSTANCE_ID=$(ec2-metadata --instance-id | cut -d' ' -f2)
log "Running on instance: $INSTANCE_ID"

if [ -z "$TRAINING_COMMAND" ]; then
    log "ERROR: No training command provided"
    log "Usage: $0 'training_command' [max_hours]"
    exit 1
fi

log "=========================================="
log "AUTO-TERMINATE TRAINING MONITOR"
log "=========================================="
log "Training Command: $TRAINING_COMMAND"
log "Max Training Time: $MAX_TRAINING_HOURS hours"
log "Instance ID: $INSTANCE_ID"
log "=========================================="

# Start training in background
log "Starting training process..."
eval "$TRAINING_COMMAND" &
TRAINING_PID=$!
log "Training PID: $TRAINING_PID"

# Calculate timeout timestamp
TIMEOUT_SECONDS=$((MAX_TRAINING_HOURS * 3600))
START_TIME=$(date +%s)
TIMEOUT_TIME=$((START_TIME + TIMEOUT_SECONDS))

# Monitor training process
while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    ELAPSED_HOURS=$((ELAPSED / 3600))
    ELAPSED_MINS=$(((ELAPSED % 3600) / 60))

    # Check if process is still running
    if ! kill -0 $TRAINING_PID 2>/dev/null; then
        wait $TRAINING_PID
        EXIT_CODE=$?
        log "Training process completed with exit code: $EXIT_CODE"
        log "Total training time: ${ELAPSED_HOURS}h ${ELAPSED_MINS}m"

        if [ $EXIT_CODE -eq 0 ]; then
            log "✅ Training completed successfully!"
        else
            log "❌ Training failed with exit code $EXIT_CODE"
        fi

        break
    fi

    # Check timeout
    if [ $CURRENT_TIME -ge $TIMEOUT_TIME ]; then
        log "⏰ Maximum training time ($MAX_TRAINING_HOURS hours) reached!"
        log "Killing training process..."
        kill -TERM $TRAINING_PID 2>/dev/null || true
        sleep 10
        kill -KILL $TRAINING_PID 2>/dev/null || true
        log "Training process terminated due to timeout"
        break
    fi

    # Log progress every 5 minutes
    if [ $((ELAPSED % 300)) -eq 0 ]; then
        log "Training in progress... (${ELAPSED_HOURS}h ${ELAPSED_MINS}m elapsed)"
    fi

    sleep 10
done

log "=========================================="
log "INITIATING INSTANCE TERMINATION"
log "=========================================="

# Wait a bit to ensure logs are flushed
sleep 5

# Self-terminate the instance
log "Terminating instance $INSTANCE_ID in 10 seconds..."
sleep 10

log "🔴 TERMINATING NOW..."
sudo shutdown -h now

# Fallback: Use AWS CLI if available and shutdown fails
if command -v aws &> /dev/null; then
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region us-east-1
fi
