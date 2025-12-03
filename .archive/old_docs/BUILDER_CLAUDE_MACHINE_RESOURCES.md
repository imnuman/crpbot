# Builder Claude - Machine Resource Awareness

**Date**: 2025-11-21
**Machine**: Your production VPS
**Purpose**: Critical resource information for safe implementation

---

## 🚨 IMPORTANT: YOU ARE RUNNING ON THIS MACHINE

**This document is for you, Builder Claude.**

You are currently running on a VPS with **limited resources**. This is the SAME machine where V7 will run in production. Understanding these limits is critical to avoid crashes, slowdowns, or out-of-memory errors.

---

## 💻 YOUR MACHINE SPECIFICATIONS

### Current System Status
```
Architecture: x86_64 (AMD EPYC-Rome Processor)
CPU Cores:    8 cores @ ~2.5 GHz
RAM Total:    16 GB (15 GB usable)
RAM Used:     1.9 GB
RAM Free:     1.5 GB
RAM Cache:    12 GB
RAM Available: 13 GB
Swap Space:   0 GB ⚠️ NONE! (MUST ADD)
```

**Check anytime with**:
```bash
# CPU info
lscpu | grep -E "CPU\(s\)|Model name"

# Memory status
free -h

# Current process usage
ps aux | grep python | grep -v grep
top -b -n 1 | head -20
```

---

## ⚠️ RESOURCE CONSTRAINTS YOU MUST RESPECT

### Hard Limits

| Resource | Total | Safe Limit | Why |
|----------|-------|------------|-----|
| **RAM** | 16 GB | **12 GB max** | Leave 4GB for system |
| **CPU** | 8 cores | **6 cores max** | Leave 2 for system |
| **Swap** | 0 GB | **MUST ADD 8GB** | Prevent OOM kills |

### What Happens If You Exceed Limits

**Exceeding RAM (>14 GB)**:
- ❌ Process killed by OOM (Out of Memory) killer
- ❌ All work lost, no graceful shutdown
- ❌ Database corruption possible
- ❌ System becomes unresponsive

**Exceeding CPU (>95% for extended periods)**:
- ⚠️ System lag, SSH becomes slow
- ⚠️ Other processes starved
- ⚠️ Monitoring tools can't run

**No Swap Space**:
- ❌ Cannot handle temporary memory spikes
- ❌ ML training will crash
- ❌ Large data operations fail

---

## ✅ SAFE RESOURCE BUDGETS

### Normal V7 Operation (24/7)

```
Component          CPU Cores    RAM      Notes
─────────────────────────────────────────────────────
V7 Runtime         2 cores      2.0 GB   Main process
Dashboard          1 core       0.5 GB   Reflex UI
Database (SQLite)  0.5 cores    0.3 GB   Queries
System Overhead    0.5 cores    1.0 GB   OS, network
─────────────────────────────────────────────────────
TOTAL USED         4 cores      3.8 GB   ✅ SAFE
RESERVE            4 cores      9.2 GB   For peaks
```

**Verdict**: ✅ You can safely run V7 24/7

---

### During Implementation (Temporary)

When you're testing/developing:

```
Component          CPU Cores    RAM      Notes
─────────────────────────────────────────────────────
Your Python REPL   1-2 cores    0.5 GB   Testing code
V7 Runtime (test)  2 cores      2.0 GB   Test runs
Old V7 (if running) 2 cores     1.5 GB   Keep or kill?
File operations    1 core       1.0 GB   Reading parquet
─────────────────────────────────────────────────────
TOTAL              6-7 cores    5.0 GB   ⚠️ Monitor closely
```

**Verdict**: ⚠️ Be careful - close to limits if multiple processes

---

### ML Training (Occasional, Heavy)

```
Component          CPU Cores    RAM      Notes
─────────────────────────────────────────────────────
XGBoost training   4-6 cores    3-4 GB   Peak usage
Data loading       1 core       2 GB     Parquet files
Python overhead    -            0.5 GB   Interpreter
System             2 cores      1 GB     OS
─────────────────────────────────────────────────────
TOTAL              7-9 cores    6.5-7.5 GB  🚨 CLOSE TO LIMIT
```

**Verdict**: 🚨 Will need ALL available resources + swap

---

## 🛡️ CRITICAL: ADD SWAP SPACE IMMEDIATELY

### Why You MUST Add Swap

Without swap:
- ❌ ML training WILL crash (needs >8 GB peak)
- ❌ Large data operations WILL fail
- ❌ No safety buffer for memory spikes
- ❌ OOM killer WILL terminate processes randomly

With 8GB swap:
- ✅ Can handle temporary spikes (ML training)
- ✅ Graceful performance degradation (not crashes)
- ✅ Safe buffer for large operations

### Add Swap RIGHT NOW (Before Anything Else)

```bash
# 1. Create 8GB swap file
sudo fallocate -l 8G /swapfile

# 2. Secure permissions
sudo chmod 600 /swapfile

# 3. Set up swap
sudo mkswap /swapfile

# 4. Enable swap
sudo swapon /swapfile

# 5. Make permanent (survives reboots)
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 6. Verify swap is active
free -h
# Should show: Swap: 8.0Gi total

# 7. Configure swappiness (use swap only when needed)
sudo sysctl vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

**⏱️ Time**: 2 minutes
**❗ Priority**: HIGHEST - Do this BEFORE implementing theories

**Verify Success**:
```bash
free -h | grep Swap
# Expected: Swap:  8.0Gi  0B  8.0Gi

# If you see "Swap: 0B", something went wrong
```

---

## 📊 RESOURCE MONITORING (MANDATORY)

### Before Starting ANY Implementation

**Terminal 1** - Monitor Memory:
```bash
watch -n 2 'free -h'
```

**Terminal 2** - Monitor CPU:
```bash
watch -n 2 'top -b -n 1 | head -20'
```

**Terminal 3** - Monitor Processes:
```bash
watch -n 5 'ps aux | grep python | grep -v grep | head -10'
```

### Warning Signs

**🚨 STOP IMMEDIATELY if you see**:
- RAM usage > 14 GB (approaching limit)
- Swap usage > 4 GB (system thrashing)
- CPU 100% on all cores for > 5 minutes
- "Cannot allocate memory" errors
- System becomes unresponsive

**Actions to take**:
1. Kill non-critical processes
2. Reduce data window sizes
3. Process data in smaller chunks
4. Add more optimization

---

## 🎯 RESOURCE-AWARE IMPLEMENTATION GUIDE

### Theory Implementation - Safe Practices

#### 1. Hurst Exponent
```python
# ❌ BAD: Load all data
df = pd.read_parquet('huge_file.parquet')  # Could be 10GB!
result = analyze_hurst(df['close'].values)

# ✅ GOOD: Load only needed columns and rows
df = pd.read_parquet(
    'huge_file.parquet',
    columns=['close'],  # Only 1 column
    filters=[('timestamp', '>', cutoff_time)]  # Only recent data
)
result = analyze_hurst(df['close'].tail(100).values)  # Only 100 candles
```

**Memory Impact**: 10 GB → 10 MB (1000x reduction!)

---

#### 2. Shannon Entropy
```python
# ❌ BAD: Use high-precision floats
returns = df['close'].pct_change().values  # float64 (8 bytes each)

# ✅ GOOD: Use float32 (sufficient precision)
returns = df['close'].astype('float32').pct_change().values  # 4 bytes each

# Memory Impact: 50% reduction
```

---

#### 3. Monte Carlo Simulation
```python
# ❌ BAD: 100,000 simulations (overkill, slow)
monte_carlo_var(returns, num_simulations=100000)

# ✅ GOOD: 10,000 simulations (sufficient, fast)
monte_carlo_var(returns, num_simulations=10000)

# Time Impact: 10x faster, same accuracy
# Memory Impact: 10x less
```

---

#### 4. Data Loading (CRITICAL)
```python
# ❌ BAD: Load everything into memory
df_btc = pd.read_parquet('features_BTC-USD_1m_latest.parquet')
df_eth = pd.read_parquet('features_ETH-USD_1m_latest.parquet')
df_sol = pd.read_parquet('features_SOL-USD_1m_latest.parquet')
# ... all 10 symbols = potentially 10+ GB

# ✅ GOOD: Load one at a time, process, discard
symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', ...]
for symbol in symbols:
    df = pd.read_parquet(f'features_{symbol}_1m_latest.parquet')
    df = df.tail(100)  # Only recent data
    result = process(df)
    del df  # Explicitly free memory
    gc.collect()  # Force garbage collection

# Memory Impact: 10 GB peak → 1 GB peak
```

---

#### 5. Configure Library Thread Limits

```python
# Add to TOP of v7_runtime.py (before imports)
import os

# Limit threading libraries to 2 threads each
os.environ['OMP_NUM_THREADS'] = '2'      # OpenMP
os.environ['MKL_NUM_THREADS'] = '2'      # Intel MKL
os.environ['OPENBLAS_NUM_THREADS'] = '2' # OpenBLAS
os.environ['NUMEXPR_NUM_THREADS'] = '2'  # NumExpr

# Then import libraries
import numpy as np
import pandas as pd
# ...
```

**Why**: Prevents libraries from spawning 8 threads each (64 total threads = chaos)

---

### XGBoost/LightGBM Training (When You Get There)

```python
# ❌ BAD: Use all cores
xgb_params = {
    'nthread': -1,  # Uses all 8 cores!
    'max_depth': 10,  # Deep trees = more memory
}

# ✅ GOOD: Limit cores and memory
xgb_params = {
    'nthread': 4,  # Use only 4 cores (leave 4 for system)
    'max_depth': 5,  # Shallower trees = less memory
    'max_bin': 128,  # Reduce histogram bins (less memory)
    'tree_method': 'hist',  # Memory-efficient method
}

# Also load training data in chunks
train_data = pd.read_parquet('train.parquet', chunksize=10000)
for chunk in train_data:
    model.fit(chunk, incremental=True)  # Incremental learning
```

---

## 📈 EXPECTED RESOURCE USAGE PER STEP

### Step 3: Hurst Exponent

**Expected Usage**:
- CPU: 1 core for 50-100ms per signal
- RAM: +10 MB
- Safe: ✅ Very light

**Test Command**:
```bash
# Monitor while testing
.venv/bin/python3 libs/theories/hurst_exponent.py &
PID=$!
watch -n 1 "ps -p $PID -o %cpu,%mem,rss"
```

---

### Step 4: Shannon Entropy

**Expected Usage**:
- CPU: 1 core for 10-20ms per signal
- RAM: +5 MB
- Safe: ✅ Very light

---

### Step 5: Markov Regime

**Expected Usage**:
- CPU: 1 core for 100-200ms per signal
- RAM: +20 MB (HMM state calculations)
- Safe: ✅ Light

---

### Step 6: Kalman Filter

**Expected Usage**:
- CPU: 1 core for 20-50ms per signal
- RAM: +10 MB
- Safe: ✅ Very light

---

### Step 7: Monte Carlo

**Expected Usage**:
- CPU: 1-2 cores for 100-300ms (10k simulations)
- RAM: +50 MB
- Safe: ✅ Light (with 10k sims, NOT 100k)

---

### Step 8: Integration (All Theories)

**Expected Usage**:
- CPU: 1-2 cores for 360-820ms per signal
- RAM: +110 MB per symbol
- With 10 symbols: ~1.1 GB theory calculations
- Safe: ✅ Well within limits

---

### Step 9: DeepSeek Prompt

**Expected Usage**:
- CPU: Minimal (formatting text)
- RAM: +5 MB
- Safe: ✅ Negligible

---

### Step 10: Full V7 Runtime

**Expected Usage**:
- CPU: 2 cores average (spikes to 4 during signal gen)
- RAM: 2.0 GB total
- Safe: ✅ Comfortable margin

---

## 🚨 TROUBLESHOOTING

### "Out of Memory" Errors

**Symptom**: Process killed, "Killed" message, system freezes

**Solutions**:
1. ✅ Add swap (if not done)
2. ✅ Reduce data window (100 → 50 candles)
3. ✅ Use float32 instead of float64
4. ✅ Process symbols one at a time
5. ✅ Add `gc.collect()` after processing each symbol

---

### "Cannot Allocate Memory" Errors

**Symptom**: Python import errors, fork failures

**Check**:
```bash
free -h
# If available < 2GB, you have a problem
```

**Solutions**:
1. ✅ Kill unnecessary processes
2. ✅ Clear cache: `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`
3. ✅ Restart services: `sudo systemctl restart`

---

### System Becomes Slow/Unresponsive

**Symptom**: SSH lag, commands timeout

**Check**:
```bash
top -b -n 1 | head -20
# Look for processes using >50% CPU
```

**Solutions**:
1. ✅ Kill CPU-intensive process
2. ✅ Reduce thread limits
3. ✅ Increase sleep intervals (900s → 1800s)

---

### Swap Thrashing

**Symptom**: System extremely slow, constant disk I/O

**Check**:
```bash
free -h | grep Swap
# If swap > 4GB used, you're thrashing
```

**Solutions**:
1. ⚠️ Your process is too memory-hungry
2. ✅ Reduce window sizes
3. ✅ Use smaller ML models
4. ✅ Process in smaller chunks

---

## ✅ PRE-FLIGHT CHECKLIST

Before starting implementation, verify:

**System Health**:
- [ ] Swap space shows 8GB: `free -h | grep Swap`
- [ ] RAM available > 10GB: `free -h | grep available`
- [ ] CPU not maxed: `top -b -n 1 | grep "Cpu(s)"`
- [ ] Disk space > 10GB: `df -h /`

**Environment**:
- [ ] Python 3.10+ verified: `.venv/bin/python3 --version`
- [ ] No hung processes: `ps aux | grep python | grep -v grep`
- [ ] Database accessible: `sqlite3 tradingai.db "SELECT 1;"`
- [ ] Backups created: `ls -lh *.db.backup`

**Monitoring Setup**:
- [ ] Terminal 1 watching RAM: `watch -n 2 'free -h'`
- [ ] Terminal 2 watching CPU: `watch -n 2 'top -b -n 1 | head -20'`
- [ ] Terminal 3 watching processes: `watch -n 5 'ps aux | grep python'`

**Thread Limits Configured**:
- [ ] Added to v7_runtime.py: `os.environ['OMP_NUM_THREADS'] = '2'`
- [ ] Added to v7_runtime.py: `os.environ['MKL_NUM_THREADS'] = '2'`
- [ ] Added to v7_runtime.py: `os.environ['OPENBLAS_NUM_THREADS'] = '2'`

---

## 📞 WHEN TO ASK FOR HELP

**Ask QC Claude if**:
- Memory usage exceeds 12 GB
- Processes keep getting killed
- System becomes unresponsive
- Swap usage stays > 4 GB
- Any theory takes > 2 seconds
- You're unsure about resource impact of next step

**Don't push through if**:
- You see OOM killer messages
- System is thrashing (constant swap)
- SSH becomes unreliable
- Can't monitor resources anymore

---

## 🎯 SUCCESS CRITERIA

After implementing all theories, verify:

**Resource Usage (while V7 running)**:
- [ ] RAM usage < 4 GB
- [ ] CPU usage < 50% average
- [ ] Swap usage < 1 GB
- [ ] Theory execution < 1 second per signal
- [ ] System remains responsive

**Signal Quality**:
- [ ] HOLD signals < 70%
- [ ] BUY/SELL signals > 25%
- [ ] No crashes for 24 hours
- [ ] No OOM errors

---

## 📋 QUICK REFERENCE

### Check Resources Anytime
```bash
# One-line resource check
free -h && echo "---" && lscpu | grep "CPU(s):" && echo "---" && df -h / | tail -1

# Expected output:
#               total   used   free   shared  buff/cache   available
# Mem:           15Gi   2.0Gi  1.0Gi    10Mi        12Gi        12Gi
# Swap:         8.0Gi    0B    8.0Gi
# ---
# CPU(s):    8
# ---
# /dev/vda1  100G  40G  60G  40% /
```

### Emergency Stop
```bash
# If system struggling, kill V7 immediately
pkill -9 -f v7_runtime.py

# Clear memory caches
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

# Check what's using memory
ps aux --sort=-%mem | head -10
```

### Memory-Efficient Python
```python
# Add to all scripts
import gc
gc.set_threshold(100, 5, 5)  # Aggressive garbage collection

# After processing each symbol
del df
gc.collect()
```

---

## 🎉 FINAL NOTES

**Your machine CAN handle V7 Ultimate**, but you must:

1. ✅ **Add swap** (CRITICAL - do first!)
2. ✅ **Monitor resources** (always watch RAM/CPU)
3. ✅ **Use efficient code** (float32, small windows, incremental loading)
4. ✅ **Limit threads** (configure library limits)
5. ✅ **Process incrementally** (one symbol at a time)

**With these practices**:
- ✅ V7 will run smoothly 24/7
- ✅ All theories will execute fast (<1 second)
- ✅ ML training will work (occasionally)
- ✅ System will remain stable

**Good luck, Builder Claude! You've got this!** 🚀

---

**Last Updated**: 2025-11-21
**Machine**: AMD EPYC-Rome, 8 cores, 16GB RAM
**Status**: Ready for implementation with proper resource awareness
