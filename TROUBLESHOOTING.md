# Troubleshooting Guide

**Last Updated**: 2025-11-21

This guide covers common issues with the V7 trading bot and their solutions.

---

## Table of Contents

1. [Dashboard Issues](#dashboard-issues)
2. [Database Issues](#database-issues)
3. [Runtime Issues](#runtime-issues)
4. [Performance Tracking Issues](#performance-tracking-issues)
5. [Model Loading Issues](#model-loading-issues)
6. [API Connection Issues](#api-connection-issues)
7. [Feature Engineering Issues](#feature-engineering-issues)

---

## Dashboard Issues

### Issue: Performance Tracking page shows blank/empty

**Symptoms**:
- Dashboard loads but Performance page shows no data
- "No performance data available" message
- Python error: `sqlite3.OperationalError: no such table: signal_results`

**Root Cause**:
The `signal_results` table is missing from the database. This table is required for tracking paper trading performance.

**Solution**:

```bash
# 1. Verify tables exist
.venv/bin/python3 -c "
import sqlite3
conn = sqlite3.connect('tradingai.db')
cursor = conn.cursor()
cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")
tables = [t[0] for t in cursor.fetchall()]
print('Tables:', tables)
print('signal_results exists:', 'signal_results' in tables)
print('theory_performance exists:', 'theory_performance' in tables)
conn.close()
"

# 2. Create missing tables
.venv/bin/python3 scripts/create_missing_tables.py

# 3. Restart dashboard
# (Kill existing dashboard process and restart)
```

**Prevention**:
Always run database migrations after pulling code changes:
```bash
.venv/bin/python3 scripts/create_missing_tables.py
```

---

### Issue: A/B Test Comparison page shows blank

**Symptoms**:
- A/B Test page loads but shows no comparison data
- Both strategies show 0 trades
- Error: `sqlite3.OperationalError: no such table: signal_results`

**Root Cause**:
Same as Performance page - missing `signal_results` table.

**Solution**:
Follow the same steps as "Performance Tracking page shows blank" above.

---

### Issue: Dashboard won't start (port already in use)

**Symptoms**:
```
Error: Address already in use
OSError: [Errno 98] Address already in use
```

**Root Cause**:
Another dashboard instance is already running on the same port.

**Solution**:

```bash
# 1. Find the process using port 3000 (Reflex frontend)
lsof -i :3000
# or for port 8000 (Reflex backend)
lsof -i :8000

# 2. Kill the process
kill -9 <PID>

# 3. Restart dashboard
cd apps/dashboard_reflex
../../.venv/bin/python3 -m reflex run
```

---

## Database Issues

### Issue: Database locked error

**Symptoms**:
```
sqlite3.OperationalError: database is locked
```

**Root Cause**:
Multiple processes are trying to write to the database simultaneously.

**Solution**:

```bash
# 1. Check for multiple V7 runtime instances
ps aux | grep v7_runtime

# 2. Kill duplicate processes (keep only one)
ps aux | grep v7_runtime | grep -v grep | awk '{print $2}' | xargs -r kill -9

# 3. Restart V7 runtime
.venv/bin/python3 apps/runtime/v7_runtime.py --iterations -1 --sleep-seconds 900
```

**Prevention**:
- Always check for existing processes before starting new ones
- Use process management tools like `systemd` for production

---

### Issue: Missing columns in signals table

**Symptoms**:
```
sqlite3.OperationalError: no such column: strategy
```

**Root Cause**:
Database schema is outdated (missing new columns added in recent updates).

**Solution**:

```bash
# Option 1: Add column manually (SQLite)
.venv/bin/python3 -c "
import sqlite3
conn = sqlite3.connect('tradingai.db')
try:
    conn.execute('ALTER TABLE signals ADD COLUMN strategy VARCHAR(50) DEFAULT \"v7_full_math\"')
    conn.commit()
    print('✅ Added strategy column')
except sqlite3.OperationalError as e:
    print(f'Column might already exist: {e}')
conn.close()
"

# Option 2: Recreate database (WARNING: loses all data)
mv tradingai.db tradingai_backup_$(date +%Y%m%d_%H%M%S).db
.venv/bin/python3 -c "from libs.db.models import create_tables; create_tables()"
```

---

## Runtime Issues

### Issue: V7 runtime not generating signals

**Symptoms**:
- Runtime runs but no signals appear in logs
- Log shows: "No signals generated this cycle"
- Dashboard shows no recent signals

**Possible Causes & Solutions**:

**1. Confidence threshold too high**
```bash
# Check current threshold
grep "CONFIDENCE_THRESHOLD" .env

# Lower threshold temporarily for testing
export CONFIDENCE_THRESHOLD=0.50  # Default is 0.65
```

**2. Rate limiter blocking signals**
```bash
# Check recent signals
tail -100 /tmp/v7_*.log | grep "Rate limit"

# Temporarily increase rate limit
.venv/bin/python3 apps/runtime/v7_runtime.py --max-signals-per-hour 10
```

**3. Market conditions (choppy/sideways)**
- V7 is designed to be conservative and avoid low-quality setups
- Check if market is ranging or has low volatility
- This is normal behavior - wait for better conditions

---

### Issue: V7 runtime crashes with import error

**Symptoms**:
```
ModuleNotFoundError: No module named 'libs.theories.random_forest_validator'
ImportError: cannot import name 'RandomForestValidator'
```

**Root Cause**:
New theory modules were added but Python cache wasn't cleared.

**Solution**:

```bash
# 1. Clear all Python cache
find /root/crpbot -path "*/__pycache__/*" -name "*.pyc" -delete
rm -rf /root/crpbot/**/__pycache__

# 2. Restart V7 runtime
ps aux | grep v7_runtime | grep -v grep | awk '{print $2}' | xargs -r kill -9
.venv/bin/python3 apps/runtime/v7_runtime.py --iterations -1 --sleep-seconds 900
```

---

### Issue: High API costs (DeepSeek)

**Symptoms**:
- DeepSeek costs exceed $5/day budget
- Log shows many LLM calls per hour

**Root Cause**:
Too many signals being generated or rate limiter not enforced.

**Solution**:

```bash
# 1. Check current signal rate
tail -200 /tmp/v7_*.log | grep "Signal generated" | wc -l

# 2. Reduce signal rate
.venv/bin/python3 apps/runtime/v7_runtime.py \
  --iterations -1 \
  --sleep-seconds 1800 \  # 30 minutes between cycles
  --max-signals-per-hour 3  # Max 3 signals/hour

# 3. Monitor costs in logs
tail -f /tmp/v7_*.log | grep "DeepSeek cost"
```

---

## Performance Tracking Issues

### Issue: Paper trading not tracking results

**Symptoms**:
- Signals are generated but no entries in `signal_results` table
- Performance page shows 0 trades
- No paper trading logs in runtime output

**Root Cause**:
Performance tracker not initialized or not being called.

**Solution**:

```bash
# 1. Check if paper trading code exists in v7_runtime.py
grep -n "PerformanceTracker" apps/runtime/v7_runtime.py

# 2. If missing, check git for latest version
git status
git pull origin main

# 3. Verify signal_results table exists
.venv/bin/python3 -c "
import sqlite3
conn = sqlite3.connect('tradingai.db')
cursor = conn.cursor()
cursor.execute(\"SELECT COUNT(*) FROM signal_results\")
print(f'Signal results count: {cursor.fetchone()[0]}')
conn.close()
"

# 4. Restart V7 with paper trading enabled
ps aux | grep v7_runtime | grep -v grep | awk '{print $2}' | xargs -r kill -9
.venv/bin/python3 apps/runtime/v7_runtime.py --iterations -1
```

---

### Issue: Win rate calculation incorrect

**Symptoms**:
- Win rate shows 100% or 0% when trades exist
- Average P&L doesn't match individual trade results

**Root Cause**:
- P&L calculation bug (direction not considered)
- Outcome classification wrong (win/loss threshold)

**Solution**:

```bash
# 1. Check performance_tracker.py for correct P&L logic
grep -A 20 "Calculate P&L" libs/tracking/performance_tracker.py

# 2. Verify direction is being used correctly
.venv/bin/python3 -c "
import sqlite3
conn = sqlite3.connect('tradingai.db')
cursor = conn.cursor()
cursor.execute('''
    SELECT sr.signal_id, s.direction, sr.entry_price, sr.exit_price, sr.pnl_percent, sr.outcome
    FROM signal_results sr
    JOIN signals s ON sr.signal_id = s.id
    WHERE sr.outcome != 'open'
    LIMIT 10
''')
for row in cursor.fetchall():
    print(row)
conn.close()
"

# 3. If logic is wrong, update libs/tracking/performance_tracker.py
# See line 84-92 in performance_tracker.py for correct calculation
```

---

## Model Loading Issues

### Issue: Model file not found

**Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/promoted/lstm_BTC-USD_v7.pt'
```

**Root Cause**:
Model files not present in `models/promoted/` directory.

**Solution**:

```bash
# 1. Check what model files exist
ls -lh models/promoted/

# 2. Download from S3 (if using AWS)
aws s3 sync s3://crpbot-ml-data/models/v7/ models/promoted/

# 3. Or use existing V6 models temporarily
cp models/v6/* models/promoted/

# 4. Verify model loads
.venv/bin/python3 -c "
import torch
model = torch.load('models/promoted/lstm_BTC-USD_v6_enhanced.pt', map_location='cpu')
print(f'Model input size: {model[\"input_size\"]}')
print(f'Model version: {model.get(\"version\", \"unknown\")}')
"
```

---

### Issue: Model input size mismatch

**Symptoms**:
```
RuntimeError: size mismatch, expected 72 features but got 54
```

**Root Cause**:
Feature engineering produces different number of features than model expects.

**Solution**:

```bash
# 1. Check feature count
.venv/bin/python3 -c "
import pandas as pd
df = pd.read_parquet('data/features/features_BTC-USD_1m_latest.parquet')
numeric_cols = [c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'session', 'volatility_regime'] and df[c].dtype in ['float64', 'int64']]
print(f'Feature count: {len(numeric_cols)}')
"

# 2. Check model expected size
.venv/bin/python3 -c "
import torch
model = torch.load('models/promoted/lstm_BTC-USD_v7.pt', map_location='cpu')
print(f'Model expects: {model[\"input_size\"]} features')
"

# 3. Re-engineer features or retrain model to match
# See MASTER_TRAINING_WORKFLOW.md for training instructions
```

---

## API Connection Issues

### Issue: Coinbase API authentication failed

**Symptoms**:
```
coinbaseerror: Unauthorized
401 Client Error: Unauthorized for url
```

**Root Cause**:
Invalid or expired API credentials.

**Solution**:

```bash
# 1. Check .env file has correct credentials
grep "COINBASE_API_KEY_NAME" .env
grep "COINBASE_API_PRIVATE_KEY" .env

# 2. Test API connection
.venv/bin/python3 -c "
from libs.data.coinbase_client import CoinbaseClient
client = CoinbaseClient()
try:
    candles = client.get_candles('BTC-USD', granularity='ONE_MINUTE', limit=10)
    print(f'✅ API works! Got {len(candles)} candles')
except Exception as e:
    print(f'❌ API failed: {e}')
"

# 3. If still failing, regenerate API keys at:
# https://portal.cdp.coinbase.com/access/api
```

---

### Issue: CoinGecko rate limit exceeded

**Symptoms**:
```
coingecko.exceptions.RateLimitException: Rate limit exceeded
429 Too Many Requests
```

**Root Cause**:
Making too many CoinGecko API calls.

**Solution**:

```bash
# 1. Check if using premium API key
grep "COINGECKO_API_KEY" .env

# 2. Add rate limiting to CoinGecko calls
# (Should already be implemented in libs/data/coingecko_client.py)

# 3. Reduce call frequency
# V7 should only call CoinGecko once per signal generation cycle
```

---

## Feature Engineering Issues

### Issue: Feature engineering takes too long

**Symptoms**:
- Each signal takes 30+ seconds to generate
- High CPU usage during feature calculation

**Root Cause**:
- Inefficient feature calculations
- Too many historical candles being fetched

**Solution**:

```bash
# 1. Profile feature engineering
.venv/bin/python3 -c "
import time
from libs.features.feature_pipeline import engineer_features
# ... measure time ...
"

# 2. Reduce lookback window
# Edit libs/features/feature_pipeline.py
# Reduce number of candles fetched (e.g., 500 instead of 1000)

# 3. Cache calculations where possible
# Implement caching for expensive operations
```

---

### Issue: NaN values in features

**Symptoms**:
```
ValueError: Input contains NaN
RuntimeError: Model input contains NaN values
```

**Root Cause**:
Insufficient data for indicator calculations or missing data handling.

**Solution**:

```bash
# 1. Check for NaN values
.venv/bin/python3 -c "
import pandas as pd
df = pd.read_parquet('data/features/features_BTC-USD_1m_latest.parquet')
nan_cols = df.columns[df.isna().any()].tolist()
print(f'Columns with NaN: {nan_cols}')
print(df[nan_cols].isna().sum())
"

# 2. Fix by filling NaN values
# In libs/features/feature_pipeline.py:
# df = df.fillna(method='ffill').fillna(0)

# 3. Ensure sufficient lookback period
# Some indicators need minimum data points (e.g., 200-period SMA needs 200 candles)
```

---

## General Debugging Tips

### Enable verbose logging

```bash
# Set logging level to DEBUG
export LOG_LEVEL=DEBUG
.venv/bin/python3 apps/runtime/v7_runtime.py --iterations 1
```

### Check recent logs

```bash
# V7 runtime logs
tail -100 /tmp/v7_*.log

# Dashboard logs
tail -100 /tmp/reflex_*.log

# Python errors
tail -100 /tmp/v7_*.log | grep -A 5 "Error\|Exception\|Traceback"
```

### Verify system resources

```bash
# Check disk space
df -h

# Check memory
free -h

# Check CPU
top -bn1 | head -20
```

### Test individual components

```bash
# Test database connection
.venv/bin/python3 -c "from libs.db.models import get_session; session = get_session(); print('✅ DB connection works')"

# Test Coinbase API
.venv/bin/python3 -c "from libs.data.coinbase_client import CoinbaseClient; client = CoinbaseClient(); print('✅ Coinbase API works')"

# Test CoinGecko API
.venv/bin/python3 -c "from libs.data.coingecko_client import CoinGeckoClient; client = CoinGeckoClient(); print('✅ CoinGecko API works')"

# Test model loading
.venv/bin/python3 -c "import torch; model = torch.load('models/promoted/lstm_BTC-USD_v7.pt', map_location='cpu'); print('✅ Model loads')"
```

---

## Getting Help

If you've tried the solutions above and still have issues:

1. **Check git history** for recent changes:
   ```bash
   git log --oneline -20
   git diff HEAD~5
   ```

2. **Search for similar issues**:
   ```bash
   grep -r "your error message" /root/crpbot/
   ```

3. **Check documentation**:
   - `CLAUDE.md` - Project architecture and setup
   - `DATABASE_SCHEMA.md` - Database structure and queries
   - `MASTER_TRAINING_WORKFLOW.md` - Model training process
   - `PROJECT_MEMORY.md` - Session history and context

4. **Create a diagnostic report**:
   ```bash
   echo "=== System Info ===" > diagnostic_report.txt
   uname -a >> diagnostic_report.txt
   python3 --version >> diagnostic_report.txt
   echo "\n=== Database Tables ===" >> diagnostic_report.txt
   .venv/bin/python3 -c "import sqlite3; conn = sqlite3.connect('tradingai.db'); cursor = conn.cursor(); cursor.execute('SELECT name FROM sqlite_master WHERE type=\"table\"'); print('\n'.join([t[0] for t in cursor.fetchall()]))" >> diagnostic_report.txt
   echo "\n=== Recent Signals ===" >> diagnostic_report.txt
   .venv/bin/python3 -c "import sqlite3; conn = sqlite3.connect('tradingai.db'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM signals'); print(f'Total signals: {cursor.fetchone()[0]}')" >> diagnostic_report.txt
   echo "\n=== Recent Logs ===" >> diagnostic_report.txt
   tail -50 /tmp/v7_*.log >> diagnostic_report.txt
   cat diagnostic_report.txt
   ```

---

**Version**: 1.0
**Last Updated**: 2025-11-21
