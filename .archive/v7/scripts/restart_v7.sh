#!/bin/bash
#
# V7 Runtime Auto-Restart Watchdog
# Monitors v7_runtime.py and restarts it if it crashes
#
# Usage: nohup ./restart_v7.sh > /tmp/v7_watchdog.log 2>&1 &
#

LOG_DIR="/tmp"
RUNTIME_SCRIPT="apps/runtime/v7_runtime.py"
PROJECT_DIR="/root/crpbot"

echo "========================================="
echo "V7 Runtime Watchdog Started"
echo "Time: $(date)"
echo "PID: $$"
echo "========================================="

while true; do
    # Check if v7_runtime.py is running
    if ! pgrep -f "$RUNTIME_SCRIPT" > /dev/null; then
        echo ""
        echo "========================================="
        echo "$(date): V7 CRASHED - Restarting..."
        echo "========================================="

        # Change to project directory
        cd "$PROJECT_DIR" || exit 1

        # Generate timestamped log file
        LOGFILE="$LOG_DIR/v7_runtime_$(date +%Y%m%d_%H%M).log"

        # Start V7 runtime with all parameters
        nohup .venv/bin/python3 -u "$RUNTIME_SCRIPT" \
          --iterations -1 \
          --sleep-seconds 60 \
          --max-signals-per-hour 12 \
          > "$LOGFILE" 2>&1 &

        NEW_PID=$!

        echo "$(date): V7 restarted"
        echo "  PID: $NEW_PID"
        echo "  Log: $LOGFILE"
        echo "========================================="

        # Give it a moment to start
        sleep 5

        # Verify it started
        if pgrep -f "$RUNTIME_SCRIPT" > /dev/null; then
            echo "$(date): ✓ V7 running successfully (PID: $NEW_PID)"
        else
            echo "$(date): ✗ WARNING - V7 failed to start!"
        fi
    fi

    # Check every 30 seconds
    sleep 30
done
