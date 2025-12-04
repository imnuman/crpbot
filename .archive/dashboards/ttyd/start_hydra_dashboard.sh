#!/usr/bin/env bash
#
# Start HYDRA 3.0 Terminal Dashboard
#
# This script serves the terminal monitor via ttyd for web viewing.
# The dashboard is accessible at http://178.156.136.185:9090
#

set -e

# Configuration
PORT=9090
MONITOR_SCRIPT="/root/crpbot/apps/dashboard_terminal/hydra_monitor.py"
LOG_FILE="/tmp/hydra_dashboard_ttyd.log"

# Check if already running
if pgrep -f "ttyd.*hydra_monitor" > /dev/null 2>&1; then
    echo "HYDRA Dashboard is already running on port $PORT"
    echo "PID: $(pgrep -f 'ttyd.*hydra_monitor')"
    echo "Access at: http://178.156.136.185:$PORT"
    exit 0
fi

# Start ttyd with the monitor script
echo "Starting HYDRA 3.0 Terminal Dashboard..."
echo "Port: $PORT"
echo "Monitor: $MONITOR_SCRIPT"
echo "Log: $LOG_FILE"
echo ""

# Start ttyd in background
nohup ttyd \
    --port $PORT \
    --interface 0.0.0.0 \
    --writable \
    --once \
    --title "HYDRA 3.0 Dashboard" \
    python3 "$MONITOR_SCRIPT" \
    > "$LOG_FILE" 2>&1 &

TTYD_PID=$!

# Wait a moment for startup
sleep 2

# Check if running
if ps -p $TTYD_PID > /dev/null 2>&1; then
    echo "✅ HYDRA Dashboard started successfully!"
    echo "PID: $TTYD_PID"
    echo "URL: http://178.156.136.185:$PORT"
    echo ""
    echo "To stop: kill $TTYD_PID"
else
    echo "❌ Failed to start dashboard"
    echo "Check log: tail -50 $LOG_FILE"
    exit 1
fi
