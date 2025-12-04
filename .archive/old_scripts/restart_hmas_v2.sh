#!/bin/bash
#
# HMAS V2 Runtime Auto-Restart Watchdog
# Monitors hmas_v2_runtime.py and restarts it if it crashes
#

LOG_DIR="/tmp"
RUNTIME_SCRIPT="apps/runtime/hmas_v2_runtime.py"
PROJECT_DIR="/root/crpbot"

echo "========================================="
echo "HMAS V2 Runtime Watchdog Started"
echo "Time: $(date)"
echo "PID: $$"
echo "========================================="

while true; do
    # Check if hmas_v2_runtime.py is running
    if ! pgrep -f "$RUNTIME_SCRIPT" > /dev/null; then
        echo ""
        echo "========================================="
        echo "$(date): HMAS V2 CRASHED - Restarting..."
        echo "========================================="

        cd "$PROJECT_DIR" || exit 1

        # Generate timestamped log file
        LOGFILE="$LOG_DIR/hmas_v2_production_$(date +%Y%m%d_%H%M).log"

        # Start HMAS V2 runtime
        nohup .venv/bin/python3 -u "$RUNTIME_SCRIPT" \
          --symbols BTC-USD ETH-USD SOL-USD DOGE-USD \
          --max-signals-per-day 20 \
          --iterations -1 \
          --sleep-seconds 3600 \
          > "$LOGFILE" 2>&1 &

        NEW_PID=$!
        echo "$(date): HMAS V2 restarted (PID: $NEW_PID, Log: $LOGFILE)"
    fi

    sleep 60
done
