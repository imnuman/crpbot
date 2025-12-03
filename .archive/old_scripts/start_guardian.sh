#!/bin/bash
# HYDRA Guardian Startup Script

cd /root/crpbot

# Kill any existing Guardian
pkill -f hydra_guardian.py

# Wait for clean shutdown
sleep 2

# Start fresh Guardian
LOGFILE="/tmp/hydra_guardian_$(date +%Y%m%d_%H%M%S).log"
nohup .venv/bin/python3 apps/runtime/hydra_guardian.py --check-interval 300 > "$LOGFILE" 2>&1 &

# Wait for startup
sleep 3

# Verify running
if ps aux | grep -q "[h]ydra_guardian.py"; then
    echo "✅ HYDRA Guardian started successfully"
    echo "Log: $LOGFILE"
    echo ""
    echo "=== First Log Lines ==="
    tail -10 "$LOGFILE"
else
    echo "❌ Guardian failed to start"
    echo "Check log: $LOGFILE"
fi
