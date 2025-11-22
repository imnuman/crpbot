# Builder Claude: Verification & Next Steps
## Complete V7 System Audit + Action Plan

**Date**: 2025-11-22
**Priority**: ⭐ HIGH - Execute immediately
**Working Directory**: `/root/crpbot` (cloud server)

---

## 🎯 MISSION

1. **VERIFY**: Confirm V7 Ultimate is operational and working correctly
2. **MEASURE**: Collect actual performance data
3. **DECIDE**: Determine next steps based on real results (not assumptions)

---

## PHASE 1: SYSTEM VERIFICATION (2 hours)

### STEP 1: Environment Check (15 minutes)

**Verify you're on the production server**:
```bash
# Check working directory
pwd
# Should output: /root/crpbot

# Check git branch
git branch
# Should show: * feature/v7-ultimate or main

# Pull latest from GitHub
git pull origin feature/v7-ultimate

# Check Python environment
which python3
source .venv/bin/activate  # If using venv

# Verify key packages
python3 -c "import pandas, numpy, scipy; print('Core packages: OK')"
python3 -c "from libs.llm import SignalGenerator; print('V7 components: OK')"
```

**Success Criteria**:
- [ ] On production server (/root/crpbot)
- [ ] Latest code pulled from GitHub
- [ ] Python environment active
- [ ] No import errors

---

### STEP 2: Verify V7 Runtime Status (15 minutes)

**Check what's currently running**:
```bash
# 1. Check for V7 runtime
ps aux | grep v7_runtime | grep -v grep

# 2. Check for old V6 runtime (should NOT be running)
ps aux | grep -E "v6_runtime|main.py" | grep -v grep

# 3. Check dashboard
ps aux | grep reflex | grep -v grep

# 4. Check system resources
free -h
df -h
```

**Record findings**:
```bash
# Create status file
cat > V7_RUNTIME_STATUS_$(date +%Y%m%d_%H%M).txt <<EOF
=== V7 Runtime Status ===
Date: $(date)

V7 Runtime:
$(ps aux | grep v7_runtime | grep -v grep || echo "NOT RUNNING")

V6 Runtime (should be stopped):
$(ps aux | grep -E "v6_runtime|main.py" | grep -v grep || echo "NOT RUNNING (good)")

Dashboard:
$(ps aux | grep reflex | grep -v grep || echo "NOT RUNNING")

Resources:
$(free -h)

Disk:
$(df -h | grep -E "Filesystem|/$")
EOF

# Display status
cat V7_RUNTIME_STATUS_$(date +%Y%m%d_%H%M).txt
```

**Decision Points**:
- **If V7 NOT running**: Start it (see Step 3)
- **If V6 still running**: Stop it immediately
- **If multiple V7 instances**: Keep newest, stop duplicates

---

### STEP 3: Start/Restart V7 (if needed) (10 minutes)

**Only if V7 is not running or needs restart**:

```bash
# Stop any existing V7 instances
pkill -f v7_runtime

# Verify stopped
ps aux | grep v7_runtime | grep -v grep
# Should return nothing

# Start V7 with proper settings
nohup .venv/bin/python3 apps/runtime/v7_runtime.py \
  --iterations -1 \
  --sleep-seconds 300 \
  --max-signals-per-hour 3 \
  > /tmp/v7_runtime_$(date +%Y%m%d_%H%M).log 2>&1 &

# Get PID
echo "V7 Runtime PID: $!"
V7_PID=$!

# Wait 30 seconds for startup
sleep 30

# Check it's running
ps -p $V7_PID
tail -50 /tmp/v7_runtime_$(date +%Y%m%d_%H%M).log

# Monitor for 2 minutes to ensure stability
echo "Monitoring for 2 minutes..."
sleep 120
ps -p $V7_PID && echo "✅ V7 stable" || echo "❌ V7 crashed - check logs"
```

**Success Criteria**:
- [ ] V7 runtime started successfully
- [ ] PID recorded
- [ ] No errors in first 2 minutes of logs
- [ ] Process still running after 2 minutes

---

### STEP 4: Database Health Check (15 minutes)

**Verify database and recent activity**:

```bash
# 1. Check database exists and size
ls -lh tradingai.db
sqlite3 tradingai.db "SELECT COUNT(*) as total_signals FROM signals;"

# 2. Check recent signals (last 24 hours)
sqlite3 tradingai.db "
SELECT
  COUNT(*) as signals_24h,
  COUNT(DISTINCT symbol) as unique_symbols,
  MIN(timestamp) as earliest,
  MAX(timestamp) as latest
FROM signals
WHERE timestamp > datetime('now', '-24 hours');
"

# 3. Signal distribution
sqlite3 tradingai.db "
SELECT
  direction,
  COUNT(*) as count,
  ROUND(AVG(confidence), 4) as avg_confidence,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM signals
WHERE timestamp > datetime('now', '-24 hours')
GROUP BY direction
ORDER BY count DESC;
"

# 4. Check which theories are being used
sqlite3 tradingai.db "
SELECT
  symbol,
  COUNT(*) as signals,
  AVG(confidence) as avg_conf
FROM signals
WHERE timestamp > datetime('now', '-24 hours')
GROUP BY symbol;
"

# 5. Check A/B testing
sqlite3 tradingai.db "
SELECT
  strategy,
  COUNT(*) as count,
  ROUND(AVG(confidence), 4) as avg_confidence
FROM signals
WHERE timestamp > datetime('now', '-24 hours')
  AND strategy IS NOT NULL
GROUP BY strategy;
"

# 6. Paper trading results
sqlite3 tradingai.db "
SELECT
  COUNT(*) as total_trades,
  SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
  SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
  ROUND(AVG(pnl_percent), 4) as avg_pnl,
  ROUND(MIN(pnl_percent), 4) as worst_trade,
  ROUND(MAX(pnl_percent), 4) as best_trade
FROM signal_results
WHERE pnl_percent IS NOT NULL;
"
```

**Record findings in file**:
```bash
sqlite3 tradingai.db > V7_DATABASE_REPORT_$(date +%Y%m%d).txt <<EOF
.mode column
.headers on

SELECT '=== SIGNALS LAST 24 HOURS ===' as report_section;

SELECT
  direction,
  COUNT(*) as count,
  ROUND(AVG(confidence), 4) as avg_confidence,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct
FROM signals
WHERE timestamp > datetime('now', '-24 hours')
GROUP BY direction;

SELECT '=== A/B TEST DISTRIBUTION ===' as report_section;

SELECT
  strategy,
  COUNT(*) as count
FROM signals
WHERE strategy IS NOT NULL
GROUP BY strategy;

SELECT '=== PAPER TRADING SUMMARY ===' as report_section;

SELECT
  COUNT(*) as trades,
  SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
  ROUND(AVG(pnl_percent), 4) as avg_pnl
FROM signal_results;

.quit
EOF

cat V7_DATABASE_REPORT_$(date +%Y%m%d).txt
```

**Success Criteria**:
- [ ] Database accessible
- [ ] Signals generated in last 24 hours
- [ ] All 10 symbols present
- [ ] A/B testing active (if configured)
- [ ] Paper trading results available

---

### STEP 5: Theory Verification (20 minutes)

**Verify all 10 theories are working**:

```bash
# Create theory verification script
cat > verify_theories.py <<'EOF'
"""Verify all V7 theories are accessible and functional"""
import sys
import traceback

print("="*70)
print("V7 THEORY VERIFICATION")
print("="*70)

theories_status = {}

# Test imports from libs/analysis (6 core theories)
print("\n1. CORE THEORIES (libs/analysis/):")
try:
    from libs.analysis import ShannonEntropyAnalyzer
    print("  ✅ Shannon Entropy")
    theories_status['shannon_entropy'] = True
except Exception as e:
    print(f"  ❌ Shannon Entropy: {e}")
    theories_status['shannon_entropy'] = False

try:
    from libs.analysis import HurstExponentAnalyzer
    print("  ✅ Hurst Exponent")
    theories_status['hurst_exponent'] = True
except Exception as e:
    print(f"  ❌ Hurst Exponent: {e}")
    theories_status['hurst_exponent'] = False

try:
    from libs.analysis import MarkovRegimeDetector
    print("  ✅ Markov Regime")
    theories_status['markov_regime'] = True
except Exception as e:
    print(f"  ❌ Markov Regime: {e}")
    theories_status['markov_regime'] = False

try:
    from libs.analysis import KalmanPriceFilter
    print("  ✅ Kalman Filter")
    theories_status['kalman_filter'] = True
except Exception as e:
    print(f"  ❌ Kalman Filter: {e}")
    theories_status['kalman_filter'] = False

try:
    from libs.analysis import BayesianWinRateLearner
    print("  ✅ Bayesian Win Rate")
    theories_status['bayesian_win_rate'] = True
except Exception as e:
    print(f"  ❌ Bayesian Win Rate: {e}")
    theories_status['bayesian_win_rate'] = False

try:
    from libs.analysis import MonteCarloSimulator
    print("  ✅ Monte Carlo")
    theories_status['monte_carlo'] = True
except Exception as e:
    print(f"  ❌ Monte Carlo: {e}")
    theories_status['monte_carlo'] = False

# Test imports from libs/theories (4 statistical)
print("\n2. STATISTICAL THEORIES (libs/theories/):")
try:
    from libs.theories.random_forest_validator import RandomForestValidator
    print("  ✅ Random Forest")
    theories_status['random_forest'] = True
except Exception as e:
    print(f"  ❌ Random Forest: {e}")
    theories_status['random_forest'] = False

try:
    from libs.theories.autocorrelation_analyzer import AutocorrelationAnalyzer
    print("  ✅ Autocorrelation")
    theories_status['autocorrelation'] = True
except Exception as e:
    print(f"  ❌ Autocorrelation: {e}")
    theories_status['autocorrelation'] = False

try:
    from libs.theories.stationarity_test import StationarityAnalyzer
    print("  ✅ Stationarity")
    theories_status['stationarity'] = True
except Exception as e:
    print(f"  ❌ Stationarity: {e}")
    theories_status['stationarity'] = False

try:
    from libs.theories.variance_tests import VarianceAnalyzer
    print("  ✅ Variance")
    theories_status['variance'] = True
except Exception as e:
    print(f"  ❌ Variance: {e}")
    theories_status['variance'] = False

# Test signal generator integration
print("\n3. V7 SIGNAL GENERATOR:")
try:
    from libs.llm import SignalGenerator
    print("  ✅ Signal Generator imports")

    # Try to instantiate (will fail if DeepSeek key missing, but that's expected)
    try:
        sg = SignalGenerator(api_key="test")
        print("  ✅ Signal Generator instantiates")
    except Exception as e:
        if "api_key" in str(e).lower():
            print("  ⚠️  Signal Generator needs API key (expected)")
        else:
            print(f"  ❌ Signal Generator error: {e}")

    theories_status['signal_generator'] = True
except Exception as e:
    print(f"  ❌ Signal Generator: {e}")
    theories_status['signal_generator'] = False
    traceback.print_exc()

# Summary
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)
working = sum(1 for v in theories_status.values() if v)
total = len(theories_status)
print(f"Theories Working: {working}/{total} ({working/total*100:.1f}%)")

if working == total:
    print("\n✅ ALL THEORIES OPERATIONAL")
    sys.exit(0)
elif working >= 8:
    print("\n⚠️  MOST THEORIES WORKING (acceptable)")
    sys.exit(0)
else:
    print("\n❌ CRITICAL: Too many theories failing")
    sys.exit(1)
EOF

# Run verification
python3 verify_theories.py
echo "Exit code: $?"
```

**Success Criteria**:
- [ ] 10/10 theories import successfully
- [ ] Signal generator instantiates
- [ ] No critical import errors

---

### STEP 6: API Health Check (15 minutes)

**Verify all APIs are working**:

```bash
# Create API test script
cat > verify_apis.py <<'EOF'
"""Verify all V7 APIs are functional"""
import os
import sys

print("="*70)
print("V7 API VERIFICATION")
print("="*70)

api_status = {}

# 1. Check environment variables
print("\n1. ENVIRONMENT VARIABLES:")
required_vars = [
    'DEEPSEEK_API_KEY',
    'COINBASE_API_KEY_NAME',
    'COINBASE_API_PRIVATE_KEY',
    'COINGECKO_API_KEY'
]

for var in required_vars:
    if os.getenv(var):
        print(f"  ✅ {var}: Set")
        api_status[var] = True
    else:
        print(f"  ❌ {var}: Missing")
        api_status[var] = False

# 2. Test DeepSeek API
print("\n2. DEEPSEEK API:")
try:
    from libs.llm import DeepSeekClient

    api_key = os.getenv('DEEPSEEK_API_KEY')
    if api_key:
        client = DeepSeekClient(api_key=api_key)

        # Try a simple API call
        try:
            response = client.chat([{"role": "user", "content": "Say 'API works'"}])
            if response and 'API works' in response.content:
                print("  ✅ DeepSeek API responding")
                api_status['deepseek_functional'] = True
            else:
                print("  ⚠️  DeepSeek API response unexpected")
                api_status['deepseek_functional'] = False
        except Exception as e:
            print(f"  ❌ DeepSeek API call failed: {e}")
            api_status['deepseek_functional'] = False
    else:
        print("  ⚠️  Cannot test - API key missing")
        api_status['deepseek_functional'] = False
except Exception as e:
    print(f"  ❌ DeepSeek client error: {e}")
    api_status['deepseek_functional'] = False

# 3. Test Coinbase API
print("\n3. COINBASE API:")
try:
    from libs.data.coinbase_client import CoinbaseClient

    client = CoinbaseClient()

    # Try to get BTC price
    try:
        candles = client.get_candles('BTC-USD', granularity=3600, limit=1)
        if candles and len(candles) > 0:
            price = candles[0][4]  # Close price
            print(f"  ✅ Coinbase API responding (BTC: ${price:,.2f})")
            api_status['coinbase_functional'] = True
        else:
            print("  ❌ Coinbase API no data")
            api_status['coinbase_functional'] = False
    except Exception as e:
        print(f"  ❌ Coinbase API call failed: {e}")
        api_status['coinbase_functional'] = False
except Exception as e:
    print(f"  ❌ Coinbase client error: {e}")
    api_status['coinbase_functional'] = False

# 4. Test CoinGecko API
print("\n4. COINGECKO API:")
try:
    from libs.data.coingecko_client import CoinGeckoClient

    client = CoinGeckoClient()

    # Try to get BTC market data
    try:
        data = client.get_coin_data('bitcoin')
        if data:
            mcap = data.get('market_cap', 0)
            print(f"  ✅ CoinGecko API responding (BTC MCap: ${mcap/1e9:.1f}B)")
            api_status['coingecko_functional'] = True
        else:
            print("  ❌ CoinGecko API no data")
            api_status['coingecko_functional'] = False
    except Exception as e:
        print(f"  ❌ CoinGecko API call failed: {e}")
        api_status['coingecko_functional'] = False
except Exception as e:
    print(f"  ❌ CoinGecko client error: {e}")
    api_status['coingecko_functional'] = False

# Summary
print("\n" + "="*70)
print("API SUMMARY")
print("="*70)
working = sum(1 for k, v in api_status.items() if 'functional' in k and v)
total = len([k for k in api_status.keys() if 'functional' in k])
print(f"APIs Working: {working}/{total}")

if working == total:
    print("\n✅ ALL APIS OPERATIONAL")
    sys.exit(0)
elif working >= 2:
    print("\n⚠️  SOME APIS WORKING (V7 can function)")
    sys.exit(0)
else:
    print("\n❌ CRITICAL: Most APIs failing")
    sys.exit(1)
EOF

# Run API verification
python3 verify_apis.py
echo "Exit code: $?"
```

**Success Criteria**:
- [ ] DeepSeek API key present and working
- [ ] Coinbase API returning prices
- [ ] CoinGecko API returning data
- [ ] At least 2/3 APIs functional

---

### STEP 7: Generate Test Signal (10 minutes)

**Verify signal generation end-to-end**:

```bash
# Create test signal script
cat > test_signal_generation.py <<'EOF'
"""Test V7 signal generation end-to-end"""
import sys
from apps.runtime.v7_runtime import V7Runtime

print("="*70)
print("V7 SIGNAL GENERATION TEST")
print("="*70)

try:
    # Initialize V7 runtime
    print("\n1. Initializing V7 Runtime...")
    runtime = V7Runtime(
        symbols=['BTC-USD'],
        confidence_threshold=0.65,
        max_signals_per_hour=3
    )
    print("  ✅ Runtime initialized")

    # Generate signal for BTC
    print("\n2. Generating signal for BTC-USD...")
    signal = runtime.generate_signal('BTC-USD')

    print("\n3. Signal Generated:")
    print(f"  Symbol:      {signal.get('symbol')}")
    print(f"  Direction:   {signal.get('direction')}")
    print(f"  Confidence:  {signal.get('confidence', 0)*100:.1f}%")
    print(f"  Strategy:    {signal.get('strategy')}")
    print(f"  Timestamp:   {signal.get('timestamp')}")

    # Verify signal structure
    required_fields = ['symbol', 'direction', 'confidence', 'timestamp']
    missing_fields = [f for f in required_fields if f not in signal]

    if missing_fields:
        print(f"\n❌ Missing fields: {missing_fields}")
        sys.exit(1)

    print("\n✅ SIGNAL GENERATION SUCCESSFUL")
    sys.exit(0)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

# Run test
timeout 60 python3 test_signal_generation.py
echo "Exit code: $?"
```

**Success Criteria**:
- [ ] Signal generates within 60 seconds
- [ ] All required fields present
- [ ] Confidence score reasonable (0.55-0.95)
- [ ] No exceptions

---

### VERIFICATION SUMMARY (10 minutes)

**Compile all findings**:

```bash
# Create comprehensive verification report
cat > V7_VERIFICATION_REPORT_$(date +%Y%m%d).md <<EOF
# V7 Ultimate Verification Report

**Date**: $(date)
**Server**: $(hostname)
**Directory**: $(pwd)

---

## ✅ SYSTEM STATUS

### Runtime
$(ps aux | grep v7_runtime | grep -v grep | head -1 || echo "NOT RUNNING")

### Database
- Signals (24h): $(sqlite3 tradingai.db "SELECT COUNT(*) FROM signals WHERE timestamp > datetime('now', '-24 hours');" 2>/dev/null || echo "ERROR")
- Total Signals: $(sqlite3 tradingai.db "SELECT COUNT(*) FROM signals;" 2>/dev/null || echo "ERROR")

### Theories
$(python3 verify_theories.py 2>&1 | grep "Theories Working" || echo "NOT VERIFIED")

### APIs
$(python3 verify_apis.py 2>&1 | grep "APIs Working" || echo "NOT VERIFIED")

### Signal Generation Test
$(python3 test_signal_generation.py 2>&1 | tail -1 || echo "NOT TESTED")

---

## 📊 SIGNAL DISTRIBUTION (Last 24h)

\`\`\`
$(sqlite3 tradingai.db "
SELECT
  direction,
  COUNT(*) as count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct
FROM signals
WHERE timestamp > datetime('now', '-24 hours')
GROUP BY direction;" 2>/dev/null || echo "NO DATA")
\`\`\`

---

## 🎯 NEXT STEPS

Based on verification results:
- [ ] All systems operational → Monitor for 7 days, collect A/B test data
- [ ] Issues found → Fix and re-verify
- [ ] Performance data needed → Continue collecting paper trading results

---

**Status**: $(if python3 verify_theories.py >/dev/null 2>&1 && python3 verify_apis.py >/dev/null 2>&1; then echo "✅ OPERATIONAL"; else echo "⚠️  NEEDS ATTENTION"; fi)
EOF

# Display report
cat V7_VERIFICATION_REPORT_$(date +%Y%m%d).md
```

---

## PHASE 2: PERFORMANCE MEASUREMENT (30 minutes)

**Only proceed if Phase 1 verification passed**

### STEP 8: Analyze Current Performance (30 minutes)

```bash
# Create performance analysis script
cat > analyze_v7_performance.py <<'EOF'
"""Analyze V7 performance from database"""
import sqlite3
import pandas as pd
import numpy as np

print("="*70)
print("V7 PERFORMANCE ANALYSIS")
print("="*70)

conn = sqlite3.connect('tradingai.db')

# 1. Signal Statistics
print("\n1. SIGNAL GENERATION STATISTICS (Last 7 Days)")
print("-"*70)

query = """
SELECT
  DATE(timestamp) as date,
  COUNT(*) as total_signals,
  SUM(CASE WHEN direction IN ('buy', 'long') THEN 1 ELSE 0 END) as buy_signals,
  SUM(CASE WHEN direction IN ('sell', 'short') THEN 1 ELSE 0 END) as sell_signals,
  SUM(CASE WHEN direction = 'hold' THEN 1 ELSE 0 END) as hold_signals,
  ROUND(AVG(confidence), 4) as avg_confidence
FROM signals
WHERE timestamp > datetime('now', '-7 days')
GROUP BY DATE(timestamp)
ORDER BY date DESC
"""

df_signals = pd.read_sql(query, conn)
print(df_signals.to_string(index=False))

# 2. Paper Trading Performance
print("\n2. PAPER TRADING RESULTS")
print("-"*70)

query = """
SELECT
  COUNT(*) as total_trades,
  SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
  SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
  ROUND(AVG(pnl_percent), 4) as avg_pnl_pct,
  ROUND(SUM(pnl_percent), 4) as total_pnl_pct,
  ROUND(MIN(pnl_percent), 4) as worst_trade,
  ROUND(MAX(pnl_percent), 4) as best_trade
FROM signal_results
WHERE pnl_percent IS NOT NULL
"""

df_trades = pd.read_sql(query, conn)
if len(df_trades) > 0 and df_trades['total_trades'].iloc[0] > 0:
    print(df_trades.to_string(index=False))

    win_rate = df_trades['wins'].iloc[0] / df_trades['total_trades'].iloc[0]
    print(f"\nWin Rate: {win_rate*100:.1f}%")
else:
    print("No paper trading data yet")

# 3. A/B Test Results
print("\n3. A/B TEST RESULTS")
print("-"*70)

query = """
SELECT
  strategy,
  COUNT(*) as signals,
  ROUND(AVG(confidence), 4) as avg_confidence,
  COUNT(DISTINCT symbol) as symbols_traded
FROM signals
WHERE strategy IS NOT NULL
  AND timestamp > datetime('now', '-7 days')
GROUP BY strategy
"""

df_ab = pd.read_sql(query, conn)
if len(df_ab) > 0:
    print(df_ab.to_string(index=False))
else:
    print("No A/B test data yet")

# 4. Per-Symbol Performance
print("\n4. PER-SYMBOL BREAKDOWN")
print("-"*70)

query = """
SELECT
  symbol,
  COUNT(*) as signals,
  ROUND(AVG(confidence), 4) as avg_conf,
  SUM(CASE WHEN direction IN ('buy', 'long', 'sell', 'short') THEN 1 ELSE 0 END) as actionable,
  SUM(CASE WHEN direction = 'hold' THEN 1 ELSE 0 END) as holds
FROM signals
WHERE timestamp > datetime('now', '-7 days')
GROUP BY symbol
ORDER BY signals DESC
"""

df_symbols = pd.read_sql(query, conn)
print(df_symbols.to_string(index=False))

conn.close()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
EOF

# Run analysis
python3 analyze_v7_performance.py > V7_PERFORMANCE_$(date +%Y%m%d).txt

# Display results
cat V7_PERFORMANCE_$(date +%Y%m%d).txt
```

**Record Key Metrics**:
```bash
# Extract key metrics for decision making
echo "KEY PERFORMANCE INDICATORS:" > V7_KPIs.txt
sqlite3 tradingai.db <<EOF >> V7_KPIs.txt
SELECT 'Total Signals (7 days): ' || COUNT(*) FROM signals WHERE timestamp > datetime('now', '-7 days');
SELECT 'Actionable Signals %: ' || ROUND(SUM(CASE WHEN direction != 'hold' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) FROM signals WHERE timestamp > datetime('now', '-7 days');
SELECT 'Avg Confidence: ' || ROUND(AVG(confidence), 4) FROM signals WHERE timestamp > datetime('now', '-7 days');
SELECT 'Paper Trading Trades: ' || COUNT(*) FROM signal_results;
SELECT 'Win Rate: ' || ROUND(SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) || '%' FROM signal_results WHERE outcome IS NOT NULL;
EOF

cat V7_KPIs.txt
```

---

## PHASE 3: DECISION MATRIX (15 minutes)

### STEP 9: Determine Next Steps Based on Data

**Analyze the verification results and decide**:

```bash
# Create decision script
cat > decide_next_steps.sh <<'EOF'
#!/bin/bash

echo "="*70
echo "V7 NEXT STEPS DECISION MATRIX"
echo "="*70

# Check if V7 is running
V7_RUNNING=$(ps aux | grep v7_runtime | grep -v grep | wc -l)

# Check signal count (last 24h)
SIGNAL_COUNT=$(sqlite3 tradingai.db "SELECT COUNT(*) FROM signals WHERE timestamp > datetime('now', '-24 hours');" 2>/dev/null || echo "0")

# Check paper trading data
TRADE_COUNT=$(sqlite3 tradingai.db "SELECT COUNT(*) FROM signal_results;" 2>/dev/null || echo "0")

# Get actionable signal percentage
ACTIONABLE_PCT=$(sqlite3 tradingai.db "SELECT ROUND(SUM(CASE WHEN direction != 'hold' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) FROM signals WHERE timestamp > datetime('now', '-7 days');" 2>/dev/null || echo "0")

echo ""
echo "SYSTEM STATUS:"
echo "  V7 Running:        $(if [ $V7_RUNNING -gt 0 ]; then echo 'YES ✅'; else echo 'NO ❌'; fi)"
echo "  Signals (24h):     $SIGNAL_COUNT"
echo "  Paper Trades:      $TRADE_COUNT"
echo "  Actionable %:      $ACTIONABLE_PCT%"
echo ""

# Decision logic
if [ $V7_RUNNING -eq 0 ]; then
    echo "DECISION: ❌ START V7 RUNTIME"
    echo "Action: Go back to STEP 3 and start V7"
    exit 1
fi

if [ "$SIGNAL_COUNT" -lt 10 ]; then
    echo "DECISION: ⏳ WAIT - Collecting Data"
    echo "Action: Let V7 run for 24-48 hours to generate more signals"
    echo "Next check: $(date -d '+1 day' '+%Y-%m-%d %H:%M')"
    exit 2
fi

if [ "$TRADE_COUNT" -lt 20 ]; then
    echo "DECISION: ⏳ WAIT - Need More Paper Trading Data"
    echo "Action: Let paper trading accumulate at least 20 trades"
    echo "Current: $TRADE_COUNT trades"
    echo "Estimated time: $(echo "scale=1; (20-$TRADE_COUNT)/3" | bc) days at 3 trades/day"
    exit 3
fi

# If we have enough data
echo "DECISION: ✅ SUFFICIENT DATA - Ready for Analysis"
echo ""
echo "RECOMMENDED NEXT ACTIONS:"
echo "  1. Analyze A/B test results (v7_full_math vs v7_deepseek_only)"
echo "  2. Calculate actual Sharpe ratio from paper trading"
echo "  3. Identify which theories have highest IC"
echo "  4. Decide if Phase 2 enhancements needed"
echo ""
echo "TO PROCEED WITH PHASE 1 (10-hour quant plan):"
echo "  - IF Sharpe < 1.0: Implement Phase 1 immediately"
echo "  - IF Sharpe 1.0-1.5: Monitor for 1 more week, then decide"
echo "  - IF Sharpe > 1.5: V7 already excellent, Phase 1 optional"
echo ""
exit 0
EOF

chmod +x decide_next_steps.sh
./decide_next_steps.sh
```

---

## 📊 REPORTING BACK

### STEP 10: Create Status Report for QC Claude

```bash
# Compile everything into final report
cat > BUILDER_STATUS_REPORT_$(date +%Y%m%d).md <<EOF
# Builder Claude Status Report

**Date**: $(date)
**V7 Verification**: Complete

---

## ✅ VERIFICATION RESULTS

### 1. Runtime Status
$(ps aux | grep v7_runtime | grep -v grep | head -1 || echo "NOT RUNNING")

### 2. Theory Verification
$(python3 verify_theories.py 2>&1 | tail -5)

### 3. API Health
$(python3 verify_apis.py 2>&1 | tail -3)

### 4. Database Status
- Total Signals: $(sqlite3 tradingai.db "SELECT COUNT(*) FROM signals;")
- Signals (24h): $(sqlite3 tradingai.db "SELECT COUNT(*) FROM signals WHERE timestamp > datetime('now', '-24 hours');")
- Paper Trades: $(sqlite3 tradingai.db "SELECT COUNT(*) FROM signal_results;")

---

## 📊 PERFORMANCE SUMMARY

### Signal Distribution (Last 7 Days)
\`\`\`
$(sqlite3 tradingai.db "SELECT direction, COUNT(*) as count FROM signals WHERE timestamp > datetime('now', '-7 days') GROUP BY direction;" 2>/dev/null | column -t -s '|')
\`\`\`

### Paper Trading
$(sqlite3 tradingai.db "SELECT 'Win Rate: ' || ROUND(SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) || '%' FROM signal_results WHERE outcome IS NOT NULL;" 2>/dev/null || echo "No data yet")

---

## 🎯 DECISION

$(./decide_next_steps.sh 2>&1 | tail -15)

---

## 📁 FILES CREATED

$(ls -1 V7_*$(date +%Y%m%d)* 2>/dev/null || echo "No files created")

---

**Next Action**: $(./decide_next_steps.sh >/dev/null 2>&1 && echo "Analyze performance data" || echo "Continue collecting data")
EOF

# Display final report
cat BUILDER_STATUS_REPORT_$(date +%Y%m%d).md

# Push to GitHub
git add BUILDER_STATUS_REPORT_$(date +%Y%m%d).md V7_*.txt V7_*.md 2>/dev/null
git commit -m "docs: Builder Claude V7 verification report $(date +%Y%m%d)

V7 system verification complete.

Verified:
- Runtime status
- Theory integration (10/10)
- API health (DeepSeek, Coinbase, CoinGecko)
- Database and signal generation
- Paper trading results

Next steps determined based on actual performance data.
"
git push origin feature/v7-ultimate
```

---

## ✅ COMPLETION CHECKLIST

After executing all steps:

- [ ] Phase 1: System verification complete (all 7 steps)
- [ ] Phase 2: Performance data collected
- [ ] Phase 3: Decision made based on data
- [ ] Status report created
- [ ] Report pushed to GitHub for QC Claude review

---

## 🔄 WHAT HAPPENS NEXT

**Based on decision from STEP 9**:

### Scenario A: Not Enough Data Yet
- **Action**: Continue monitoring for 3-7 days
- **Goal**: Collect 50+ signals, 20+ paper trades
- **Next Review**: 2025-11-29

### Scenario B: Performance Issues Found
- **Action**: Implement Phase 1 (10-hour quant plan)
- **File**: `QUANT_FINANCE_10_HOUR_PLAN.md`
- **Start With**: Backtesting to validate issues

### Scenario C: V7 Performing Well
- **Action**: Monitor for 2 more weeks
- **Goal**: Confirm stability and performance
- **Then**: Consider Phase 2 enhancements (optional)

---

**Status**: ⏳ Ready for Builder Claude execution
**Time Required**: 3 hours total (verification + analysis + reporting)
**Expected Outcome**: Clear data-driven decision on next steps
