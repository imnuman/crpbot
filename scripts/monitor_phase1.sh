#!/bin/bash
# Daily Phase 1 monitoring script
# Usage: ./scripts/monitor_phase1.sh

echo "===================================================================="
echo "V7 PHASE 1 MONITORING REPORT"
echo "===================================================================="
echo "Date: $(date)"
echo ""

# Runtime Status
echo "=== Runtime Status ==="
if ps aux | grep v7_runtime_phase1 | grep -v grep > /dev/null; then
    echo "✅ Phase 1 Runtime: RUNNING"
    ps aux | grep v7_runtime_phase1 | grep -v grep | awk '{print "   PID: "$2", CPU: "$3"%, MEM: "$4"%, Uptime: "$10}'
else
    echo "❌ Phase 1 Runtime: NOT RUNNING"
fi

if ps aux | grep "v7_runtime.py" | grep -v "phase1" | grep -v grep > /dev/null; then
    echo "✅ Current V7 Runtime: RUNNING"
    ps aux | grep "v7_runtime.py" | grep -v "phase1" | grep -v grep | awk '{print "   PID: "$2", CPU: "$3"%, MEM: "$4"%, Uptime: "$10}'
else
    echo "⚠️  Current V7 Runtime: NOT RUNNING"
fi

echo ""

# Performance Comparison
echo "=== Performance Comparison (Last 7 Days) ==="
sqlite3 /root/crpbot/tradingai.db <<EOF
.mode column
.headers on
.width 15 8 6 10 10 11
SELECT
    COALESCE(s.strategy, 'v7_current') as variant,
    COUNT(*) as trades,
    SUM(CASE WHEN sr.outcome = 'win' THEN 1 ELSE 0 END) as wins,
    ROUND(100.0 * SUM(CASE WHEN sr.outcome = 'win' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate,
    ROUND(AVG(sr.pnl_percent), 2) as avg_pnl,
    ROUND(SUM(sr.pnl_percent), 2) as total_pnl
FROM signal_results sr
JOIN signals s ON sr.signal_id = s.id
WHERE sr.outcome IN ('win', 'loss')
AND sr.created_at > datetime('now', '-7 days')
GROUP BY s.strategy
ORDER BY win_rate DESC;
EOF

echo ""

# Signal Counts
echo "=== Signal Generation (Last 24 Hours) ==="
sqlite3 /root/crpbot/tradingai.db <<EOF
.mode column
.headers on
SELECT
    COALESCE(strategy, 'v7_current') as variant,
    COUNT(*) as total_signals,
    SUM(CASE WHEN direction = 'long' THEN 1 ELSE 0 END) as long_signals,
    SUM(CASE WHEN direction = 'short' THEN 1 ELSE 0 END) as short_signals,
    SUM(CASE WHEN direction = 'hold' THEN 1 ELSE 0 END) as hold_signals
FROM signals
WHERE timestamp > datetime('now', '-24 hours')
GROUP BY strategy
ORDER BY total_signals DESC;
EOF

echo ""

# Recent Phase 1 Signals
echo "=== Recent Phase 1 Signals (Last 10) ==="
sqlite3 /root/crpbot/tradingai.db <<EOF
.mode column
.headers on
.width 20 10 10 6
SELECT
    datetime(timestamp, 'localtime') as time,
    symbol,
    direction,
    ROUND(confidence, 2) as conf
FROM signals
WHERE strategy LIKE '%phase1%'
ORDER BY timestamp DESC
LIMIT 10;
EOF

echo ""

# Open Positions
echo "=== Open Paper Trading Positions ==="
sqlite3 /root/crpbot/tradingai.db <<EOF
.mode column
.headers on
.width 20 10 10 12 10
SELECT
    datetime(sr.entry_timestamp, 'localtime') as entry_time,
    s.symbol,
    s.direction,
    ROUND(sr.entry_price, 2) as entry_price,
    COALESCE(s.strategy, 'v7_current') as variant
FROM signal_results sr
JOIN signals s ON sr.signal_id = s.id
WHERE sr.outcome = 'open'
ORDER BY sr.entry_timestamp DESC
LIMIT 10;
EOF

echo ""

# Phase 1 Specific Metrics
echo "=== Phase 1 Component Status ==="

# Kelly Fraction (from recent logs)
if [ -f /tmp/v7_phase1_*.log ]; then
    echo "Kelly Fraction:"
    grep "Kelly Updated" /tmp/v7_phase1_*.log 2>/dev/null | tail -1 | sed 's/.*Kelly Updated:/  /'
fi

# Regime Filtering
if [ -f /tmp/v7_phase1_*.log ]; then
    FILTERED=$(grep -c "Signal filtered by regime" /tmp/v7_phase1_*.log 2>/dev/null || echo 0)
    PASSED=$(grep -c "Regime filter passed" /tmp/v7_phase1_*.log 2>/dev/null || echo 0)
    echo "Regime Filtering:"
    echo "   Passed: $PASSED signals"
    echo "   Blocked: $FILTERED signals"
fi

# Correlation Blocking
if [ -f /tmp/v7_phase1_*.log ]; then
    CORR_BLOCKED=$(grep -c "blocked by correlation" /tmp/v7_phase1_*.log 2>/dev/null || echo 0)
    CORR_PASSED=$(grep -c "Correlation check passed" /tmp/v7_phase1_*.log 2>/dev/null || echo 0)
    echo "Correlation Analysis:"
    echo "   Diversified: $CORR_PASSED signals"
    echo "   Blocked: $CORR_BLOCKED signals"
fi

echo ""

# Error Check
echo "=== Recent Errors (Last 10) ==="
if [ -f /tmp/v7_phase1_*.log ]; then
    ERROR_COUNT=$(grep -c -i "error\|exception\|failed" /tmp/v7_phase1_*.log 2>/dev/null || echo 0)
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "⚠️  Found $ERROR_COUNT errors in logs:"
        grep -i "error\|exception\|failed" /tmp/v7_phase1_*.log 2>/dev/null | tail -10
    else
        echo "✅ No errors found in Phase 1 logs"
    fi
else
    echo "⚠️  Phase 1 log file not found"
fi

echo ""

# Log Tail
echo "=== Recent Phase 1 Activity (Last 15 Lines) ==="
if [ -f /tmp/v7_phase1_*.log ]; then
    tail -15 /tmp/v7_phase1_*.log
else
    echo "⚠️  Phase 1 log file not found"
fi

echo ""
echo "===================================================================="
echo "END OF REPORT"
echo "===================================================================="
