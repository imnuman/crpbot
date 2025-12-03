#!/bin/bash
# Deploy V7 Enhancements (Multi-Timeframe + Sharpe + CVaR)
# Date: 2025-11-24
# Status: Ready to execute when desired

echo "======================================================================"
echo "V7 ULTIMATE - DEPLOY ENHANCEMENTS"
echo "======================================================================"
echo ""
echo "This will restart V7 with the following NEW features:"
echo "  ✅ Multi-Timeframe Confirmation (1m + 5m alignment)"
echo "  ✅ Sharpe Ratio Tracker (7/14/30/90-day performance)"
echo "  ✅ CVaR Calculator (tail risk analysis)"
echo ""
echo "Already running features (will continue):"
echo "  ✅ Safety Guards (4 modules: Regime/Drawdown/Correlation/Rejection)"
echo "  ✅ Volatility Regime Detection (adaptive stops/targets)"
echo ""
echo "======================================================================"
echo ""

# Check if V7 is currently running
if pgrep -f "v7_runtime.py" > /dev/null; then
    echo "🔴 V7 is currently RUNNING (PID: $(pgrep -f v7_runtime.py | tail -1))"
    echo ""
    read -p "Do you want to restart V7 with new enhancements? (yes/no): " CONFIRM

    if [ "$CONFIRM" != "yes" ]; then
        echo "❌ Deployment cancelled."
        exit 0
    fi

    echo ""
    echo "⏹️  Stopping current V7 runtime..."
    pkill -f v7_runtime.py
    sleep 3

    # Verify stopped
    if pgrep -f "v7_runtime.py" > /dev/null; then
        echo "❌ Failed to stop V7. Manual intervention required."
        exit 1
    fi

    echo "✅ V7 stopped successfully."
else
    echo "🟢 V7 is not currently running."
fi

echo ""
echo "🚀 Starting V7 with ALL enhancements..."
echo ""

# Generate log filename with timestamp
LOG_FILE="/tmp/v7_runtime_$(date +%Y%m%d_%H%M).log"

# Start V7 in background
nohup /root/crpbot/.venv/bin/python3 /root/crpbot/apps/runtime/v7_runtime.py \
  --iterations -1 \
  --sleep-seconds 300 \
  --max-signals-per-hour 3 \
  > "$LOG_FILE" 2>&1 &

# Wait for startup
sleep 5

# Verify running
if pgrep -f "v7_runtime.py" > /dev/null; then
    NEW_PID=$(pgrep -f v7_runtime.py | tail -1)
    echo "✅ V7 started successfully!"
    echo "   PID: $NEW_PID"
    echo "   Log: $LOG_FILE"
    echo ""
    echo "======================================================================"
    echo "INITIALIZATION CHECK"
    echo "======================================================================"
    echo ""

    # Show initialization messages
    echo "Last 50 lines of initialization log:"
    echo ""
    tail -50 "$LOG_FILE" | grep -E "(initialized|Loaded|Safety|Sharpe|CVaR|Multi-TF)" || echo "(Waiting for initialization logs...)"

    echo ""
    echo "======================================================================"
    echo "MONITORING COMMANDS"
    echo "======================================================================"
    echo ""
    echo "Live logs:        tail -f $LOG_FILE"
    echo "Sharpe metrics:   grep 'Sharpe Ratio' $LOG_FILE"
    echo "CVaR metrics:     grep 'CVaR' $LOG_FILE"
    echo "Safety Guards:    grep 'SAFETY GUARDS' $LOG_FILE"
    echo "Multi-TF:         grep 'Multi-TF' $LOG_FILE"
    echo "Rejections:       sqlite3 tradingai.db 'SELECT COUNT(*) FROM signal_rejections;'"
    echo ""
    echo "✅ DEPLOYMENT COMPLETE!"
    echo ""
else
    echo "❌ V7 failed to start. Check logs:"
    echo "   cat $LOG_FILE"
    exit 1
fi
