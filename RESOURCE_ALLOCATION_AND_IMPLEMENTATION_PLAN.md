# V7 Resource Allocation & Step-Based Implementation Plan

**Date**: 2025-11-21
**System**: AMD EPYC-Rome, 8 cores, 16GB RAM
**Purpose**: Resource planning and step-by-step implementation guide

---

## 💻 CURRENT SYSTEM ANALYSIS

### Your Hardware
```
CPU:  8 cores @ ~2.5 GHz (AMD EPYC-Rome)
RAM:  16 GB total, 13 GB available
Disk: Not specified (likely SSD based on VPS)
Swap: 0 GB (⚠️ RECOMMEND ADDING)
```

### Current Usage
```
RAM Used:     1.9 GB
RAM Free:     1.5 GB
RAM Cached:   12 GB
RAM Available: 13 GB
```

**Assessment**: ✅ **SUFFICIENT** for V7 Ultimate with all enhancements

---

## 📊 RESOURCE REQUIREMENTS

### V7 Runtime Components

| Component | CPU Cores | RAM | Notes |
|-----------|-----------|-----|-------|
| V7 Runtime Process | 1-2 cores | 500-800 MB | Mathematical theories |
| DeepSeek API Calls | 0 (external) | Minimal | Network only |
| CoinGecko API | 0 (external) | Minimal | Network only |
| Coinbase WebSocket | 0.5 core | 100 MB | Real-time data |
| Database (SQLite) | 0.5 core | 200-300 MB | Query processing |
| Reflex Dashboard | 1 core | 300-500 MB | Frontend + backend |
| Python Base | - | 200-300 MB | Interpreter overhead |
| **Total (Runtime)** | **3-4 cores** | **1.5-2.0 GB** | **Normal operation** |

### ML Training (Optional)

| Component | CPU Cores | RAM | Notes |
|-----------|-----------|-----|-------|
| XGBoost Training | 4-6 cores | 2-4 GB | Parallel tree building |
| LightGBM Training | 4-6 cores | 1-3 GB | More memory efficient |
| Prophet Training | 2-3 cores | 1-2 GB | Time series modeling |
| Feature Engineering | 2-4 cores | 1-2 GB | Pandas operations |
| **Total (Training)** | **4-6 cores** | **3-6 GB** | **Peak during training** |

### Theory Calculations (Per Signal)

| Theory | CPU Time | RAM | Complexity |
|--------|----------|-----|------------|
| Hurst Exponent | 50-100 ms | 10 MB | O(n log n) |
| Shannon Entropy | 10-20 ms | 5 MB | O(n) |
| Markov Regime (HMM) | 100-200 ms | 20 MB | O(n²) |
| Kalman Filter | 20-50 ms | 10 MB | O(n) |
| Monte Carlo (10k sims) | 100-300 ms | 50 MB | O(n*m) |
| CoinGecko Context | 50-100 ms | 5 MB | API call |
| Market Microstructure | 30-50 ms | 10 MB | O(n) |
| **Total per Signal** | **360-820 ms** | **110 MB** | **< 1 second** |

---

## ✅ RESOURCE ALLOCATION PLAN

### Recommended Allocation (Your System)

```
Total Available: 8 cores, 13 GB RAM

ALLOCATION:
├── V7 Runtime:          2 cores,  2 GB  (25% CPU, 15% RAM)
├── Dashboard:           1 core,   0.5 GB (12% CPU, 4% RAM)
├── Database:            0.5 cores, 0.3 GB (6% CPU, 2% RAM)
├── System Overhead:     0.5 cores, 1 GB   (6% CPU, 8% RAM)
├── ML Training (peak):  4 cores,   4 GB   (50% CPU, 30% RAM)
└── Reserve:             4 cores,   5.2 GB (50% CPU, 40% RAM)

TYPICAL USAGE (non-training):
├── Active: 4 cores, 3.8 GB (50% CPU, 29% RAM)
└── Reserve: 4 cores, 9.2 GB (50% CPU, 71% RAM)
```

**Verdict**: ✅ **Your system can comfortably run V7 Ultimate**

---

## ⚙️ OPTIMIZATION RECOMMENDATIONS

### 1. Add Swap Space (CRITICAL)

**Why**: Prevent OOM (Out of Memory) kills during ML training

```bash
# Create 8GB swap file
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify
free -h
# Should show: Swap: 8.0Gi
```

**Impact**: Prevents crashes during memory-intensive operations

---

### 2. Configure Process Limits

**File**: `apps/runtime/v7_runtime.py`

```python
# Limit number of parallel processes
import os
os.environ['OMP_NUM_THREADS'] = '2'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '2'  # Limit Intel MKL threads
os.environ['OPENBLAS_NUM_THREADS'] = '2'  # Limit OpenBLAS threads

# For XGBoost/LightGBM
xgb_params = {
    'nthread': 4,  # Use 4 cores max for training
    'max_depth': 5,  # Limit tree depth (less memory)
}
```

---

### 3. Use Incremental Learning

**Instead of**: Loading all historical data into memory
**Do**: Process data in chunks

```python
# Bad: Load all data
df = pd.read_parquet('huge_file.parquet')  # 10 GB!

# Good: Load in chunks
chunks = pd.read_parquet('huge_file.parquet', chunksize=10000)
for chunk in chunks:
    process(chunk)
```

---

### 4. Optimize Database Queries

```python
# Add index on timestamp (faster queries)
sqlite3 tradingai.db "CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);"

# Add index on symbol
sqlite3 tradingai.db "CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);"

# Add index on direction
sqlite3 tradingai.db "CREATE INDEX IF NOT EXISTS idx_signals_direction ON signals(direction);"
```

---

### 5. Use Efficient Data Types

```python
# Bad: Default float64 (8 bytes per value)
df['price'] = df['price'].astype('float64')

# Good: float32 (4 bytes per value) - sufficient precision
df['price'] = df['price'].astype('float32')

# Good: Categorical for repeating values
df['symbol'] = df['symbol'].astype('category')
```

**Impact**: 50% memory reduction for numerical data

---

## 📋 STEP-BY-STEP IMPLEMENTATION PLAN

**NOTE**: Steps are independent of time - complete each fully before moving to next.

---

### STEP 1: Environment Preparation

**Objective**: Prepare system for V7 enhancements

**Tasks**:
1. ✅ Add 8GB swap space
2. ✅ Install required system packages
3. ✅ Verify Python environment
4. ✅ Create backup of current V7

**Commands**:
```bash
# 1.1 Add swap
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstag
free -h  # Verify

# 1.2 Install system packages
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    python3-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev

# 1.3 Verify Python
.venv/bin/python3 --version  # Should be 3.10+
.venv/bin/pip list | grep -E "numpy|pandas|scipy"

# 1.4 Backup current V7
cp apps/runtime/v7_runtime.py apps/runtime/v7_runtime_backup.py
sqlite3 tradingai.db ".backup tradingai_backup_$(date +%Y%m%d).db"
```

**Success Criteria**:
- [ ] Swap space showing 8GB
- [ ] All system packages installed
- [ ] Python 3.10+ verified
- [ ] Backups created

**Estimated Time**: 30 minutes

---

### STEP 2: Install Core Theory Libraries

**Objective**: Install mathematical theory libraries (Phase 1)

**Libraries**:
- `hurst` - Hurst exponent
- `hmmlearn` - Markov regime detection
- `EntropyHub` - Shannon entropy
- `pykalman` - Kalman filtering
- `numpy` - Monte Carlo (already have)

**Commands**:
```bash
# 2.1 Install libraries
.venv/bin/pip install hurst
.venv/bin/pip install hmmlearn
.venv/bin/pip install EntropyHub
.venv/bin/pip install pykalman
.venv/bin/pip install filterpy  # Alternative Kalman

# 2.2 Verify installations
.venv/bin/python3 -c "from hurst import compute_Hc; print('✅ hurst')"
.venv/bin/python3 -c "from hmmlearn import hmm; print('✅ hmmlearn')"
.venv/bin/python3 -c "import EntropyHub; print('✅ EntropyHub')"
.venv/bin/python3 -c "from pykalman import KalmanFilter; print('✅ pykalman')"

# 2.3 Check memory usage after imports
.venv/bin/python3 -c "
from hurst import compute_Hc
from hmmlearn import hmm
import EntropyHub
from pykalman import KalmanFilter
import psutil
print(f'Memory: {psutil.Process().memory_info().rss / 1024**2:.1f} MB')
"
```

**Success Criteria**:
- [ ] All 4 libraries import without errors
- [ ] Memory usage < 200 MB for imports
- [ ] No version conflicts

**Estimated Time**: 15 minutes

---

### STEP 3: Implement Hurst Exponent Theory

**Objective**: Create working Hurst exponent module

**File**: `libs/theories/hurst_exponent.py`

**Implementation**:
```python
"""
Hurst Exponent Theory
Measures trend persistence vs mean reversion
"""
from hurst import compute_Hc
import numpy as np
from typing import Dict, Any

def analyze_hurst(prices: np.ndarray, window: int = 100) -> Dict[str, Any]:
    """
    Calculate Hurst exponent for trend analysis

    Args:
        prices: Array of prices
        window: Lookback window (default 100)

    Returns:
        dict: Hurst analysis results
    """
    try:
        # Use last 'window' prices
        price_window = prices[-window:]

        if len(price_window) < 50:
            raise ValueError(f"Insufficient data: {len(price_window)} < 50")

        # Calculate Hurst exponent
        H, c, data = compute_Hc(price_window, kind='price', simplified=True)

        # Interpret
        if H > 0.55:
            market_type = 'trending'
            interpretation = f"Strong trend persistence (H={H:.3f}). Momentum strategy favorable."
            strategy = 'FOLLOW_TREND'
        elif H < 0.45:
            market_type = 'mean_reverting'
            interpretation = f"Mean-reverting behavior (H={H:.3f}). Reversal strategy favorable."
            strategy = 'REVERSION'
        else:
            market_type = 'random'
            interpretation = f"Random walk (H={H:.3f}). Neutral market, avoid aggressive trading."
            strategy = 'NEUTRAL'

        # Confidence (distance from 0.5)
        confidence = abs(H - 0.5) * 2  # 0 = uncertain, 1 = very certain

        return {
            'theory': 'hurst_exponent',
            'value': float(H),
            'c_value': float(c),
            'market_type': market_type,
            'strategy': strategy,
            'interpretation': interpretation,
            'confidence': float(confidence),
            'window': window
        }

    except Exception as e:
        return {
            'theory': 'hurst_exponent',
            'value': 0.5,
            'market_type': 'unknown',
            'strategy': 'NEUTRAL',
            'interpretation': f"Error calculating Hurst: {str(e)}",
            'confidence': 0.0,
            'window': window
        }

# Test function
if __name__ == '__main__':
    # Generate test data
    np.random.seed(42)

    # Trending series
    trending = np.cumsum(np.random.randn(200) * 0.5 + 0.1) + 100
    print("Trending series:")
    print(analyze_hurst(trending))

    # Mean-reverting series
    mean_reverting = np.random.randn(200) * 2 + 100
    print("\nMean-reverting series:")
    print(analyze_hurst(mean_reverting))
```

**Testing**:
```bash
# 3.1 Run unit test
.venv/bin/python3 libs/theories/hurst_exponent.py

# Expected output:
# Trending series:
#   {'value': 0.6-0.8, 'market_type': 'trending', ...}
# Mean-reverting series:
#   {'value': 0.2-0.4, 'market_type': 'mean_reverting', ...}

# 3.2 Test with real data
.venv/bin/python3 -c "
import pandas as pd
from libs.theories.hurst_exponent import analyze_hurst

df = pd.read_parquet('data/features/features_BTC-USD_1m_latest.parquet')
result = analyze_hurst(df['close'].values)
print(f'BTC-USD Hurst: {result}')
"
```

**Success Criteria**:
- [ ] Module imports without errors
- [ ] Test data produces expected Hurst values (trending >0.5, reverting <0.5)
- [ ] Real BTC data produces reasonable Hurst value (0.4-0.7)
- [ ] Execution time < 200ms

**Estimated Time**: 2-3 hours

---

### STEP 4: Implement Shannon Entropy Theory

**Objective**: Create Shannon entropy module for market predictability

**File**: `libs/theories/shannon_entropy.py`

**Implementation**:
```python
"""
Shannon Entropy Theory
Measures market randomness and predictability
"""
import numpy as np
from scipy.stats import entropy
from typing import Dict, Any

def calculate_shannon_entropy(returns: np.ndarray, bins: int = 50) -> Dict[str, Any]:
    """
    Calculate Shannon entropy of price returns

    Args:
        returns: Array of price returns
        bins: Number of histogram bins (default 50)

    Returns:
        dict: Entropy analysis results
    """
    try:
        # Remove NaN and infinite values
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

        if len(returns) < 20:
            raise ValueError(f"Insufficient data: {len(returns)} < 20")

        # Calculate histogram
        hist, bin_edges = np.histogram(returns, bins=bins, density=True)

        # Remove zero probabilities (log(0) undefined)
        hist = hist[hist > 0]

        # Calculate Shannon entropy
        ent = entropy(hist)

        # Normalize to 0-1 range (max entropy = log(bins))
        max_entropy = np.log(bins)
        normalized_entropy = ent / max_entropy if max_entropy > 0 else 0

        # Interpret
        if normalized_entropy < 0.3:
            predictability = 'high'
            interpretation = f"Low entropy ({normalized_entropy:.3f}). Market is highly predictable. TRADE."
            signal = 'FAVORABLE'
        elif normalized_entropy < 0.6:
            predictability = 'moderate'
            interpretation = f"Moderate entropy ({normalized_entropy:.3f}). Market somewhat predictable. CAUTIOUS."
            signal = 'NEUTRAL'
        else:
            predictability = 'low'
            interpretation = f"High entropy ({normalized_entropy:.3f}). Market is random. AVOID."
            signal = 'UNFAVORABLE'

        return {
            'theory': 'shannon_entropy',
            'entropy': float(ent),
            'normalized_entropy': float(normalized_entropy),
            'predictability': predictability,
            'signal': signal,
            'interpretation': interpretation,
            'bins': bins,
            'data_points': len(returns)
        }

    except Exception as e:
        return {
            'theory': 'shannon_entropy',
            'entropy': 0.0,
            'normalized_entropy': 0.5,
            'predictability': 'unknown',
            'signal': 'NEUTRAL',
            'interpretation': f"Error calculating entropy: {str(e)}",
            'bins': bins,
            'data_points': 0
        }

# Test function
if __name__ == '__main__':
    np.random.seed(42)

    # Predictable series (low entropy)
    predictable = np.random.randn(200) * 0.1
    print("Predictable series:")
    print(calculate_shannon_entropy(predictable))

    # Random series (high entropy)
    random = np.random.randn(200) * 5
    print("\nRandom series:")
    print(calculate_shannon_entropy(random))
```

**Testing**:
```bash
# Test module
.venv/bin/python3 libs/theories/shannon_entropy.py

# Test with real data
.venv/bin/python3 -c "
import pandas as pd
from libs/theories.shannon_entropy import calculate_shannon_entropy

df = pd.read_parquet('data/features/features_BTC-USD_1m_latest.parquet')
returns = df['close'].pct_change().dropna().values
result = calculate_shannon_entropy(returns)
print(f'BTC-USD Entropy: {result}')
"
```

**Success Criteria**:
- [ ] Predictable data shows low entropy (<0.3)
- [ ] Random data shows high entropy (>0.6)
- [ ] Real BTC data shows moderate entropy (0.3-0.6)
- [ ] Execution time < 50ms

**Estimated Time**: 1-2 hours

---

### STEP 5: Implement Markov Regime Detection

**Objective**: Create HMM-based market regime detection

**File**: `libs/theories/markov_regime.py`

**Implementation** (simplified version first):
```python
"""
Markov Regime Detection
Detects bull/bear/sideways market states
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

def detect_market_regime_simple(df: pd.DataFrame, window: int = 50) -> Dict[str, Any]:
    """
    Simple threshold-based regime detection (fast, low memory)

    For production HMM implementation later (requires more data/training)

    Args:
        df: DataFrame with OHLCV data
        window: Lookback window

    Returns:
        dict: Regime detection results
    """
    try:
        # Calculate metrics
        recent = df.tail(window)

        # Trend strength (SMA slope)
        sma = recent['close'].rolling(20).mean()
        trend = (sma.iloc[-1] - sma.iloc[0]) / sma.iloc[0]

        # Volatility
        returns = recent['close'].pct_change()
        volatility = returns.std()

        # Volume trend
        vol_ratio = recent['volume'].tail(10).mean() / recent['volume'].head(10).mean()

        # Regime classification
        if trend > 0.02 and volatility < 0.03:
            regime = 'STRONG_BULL'
            confidence = min(abs(trend) * 20, 0.9)
        elif trend > 0.01:
            regime = 'WEAK_BULL'
            confidence = min(abs(trend) * 15, 0.7)
        elif trend < -0.02 and volatility < 0.03:
            regime = 'STRONG_BEAR'
            confidence = min(abs(trend) * 20, 0.9)
        elif trend < -0.01:
            regime = 'WEAK_BEAR'
            confidence = min(abs(trend) * 15, 0.7)
        elif volatility < 0.02:
            regime = 'SIDEWAYS'
            confidence = 0.6
        else:
            regime = 'CHOPPY'
            confidence = 0.4

        # Trading recommendations
        if regime in ['STRONG_BULL', 'WEAK_BULL']:
            strategy = 'LONG_BIAS'
            interpretation = f"{regime} market detected. Favor long positions."
        elif regime in ['STRONG_BEAR', 'WEAK_BEAR']:
            strategy = 'SHORT_BIAS'
            interpretation = f"{regime} market detected. Favor short positions."
        elif regime == 'SIDEWAYS':
            strategy = 'RANGE_TRADING'
            interpretation = "Sideways market. Trade ranges, fade extremes."
        else:
            strategy = 'AVOID'
            interpretation = "Choppy market. Reduce position sizes or avoid."

        return {
            'theory': 'markov_regime',
            'regime': regime,
            'strategy': strategy,
            'confidence': float(confidence),
            'trend': float(trend),
            'volatility': float(volatility),
            'volume_ratio': float(vol_ratio),
            'interpretation': interpretation
        }

    except Exception as e:
        return {
            'theory': 'markov_regime',
            'regime': 'UNKNOWN',
            'strategy': 'AVOID',
            'confidence': 0.0,
            'interpretation': f"Error: {str(e)}"
        }

# Test
if __name__ == '__main__':
    # Create test data
    dates = pd.date_range('2024-01-01', periods=100, freq='T')

    # Bull market
    prices_bull = np.cumsum(np.random.randn(100) * 0.5 + 0.2) + 100
    df_bull = pd.DataFrame({
        'timestamp': dates,
        'close': prices_bull,
        'volume': np.random.randint(1000, 2000, 100)
    })
    print("Bull market:")
    print(detect_market_regime_simple(df_bull))

    # Sideways market
    prices_side = np.random.randn(100) * 0.5 + 100
    df_side = pd.DataFrame({
        'timestamp': dates,
        'close': prices_side,
        'volume': np.random.randint(1000, 2000, 100)
    })
    print("\nSideways market:")
    print(detect_market_regime_simple(df_side))
```

**Success Criteria**:
- [ ] Correctly identifies bull markets
- [ ] Correctly identifies bear markets
- [ ] Correctly identifies sideways markets
- [ ] Execution time < 100ms

**Estimated Time**: 2-3 hours

---

### STEP 6: Implement Kalman Filter

**Objective**: Price denoising and momentum estimation

**File**: `libs/theories/kalman_filter.py`

**Implementation**:
```python
"""
Kalman Filter Theory
Denoises price data and estimates momentum
"""
from pykalman import KalmanFilter
import numpy as np
from typing import Dict, Any, Tuple

def apply_kalman_filter(prices: np.ndarray) -> Dict[str, Any]:
    """
    Apply Kalman filter to denoise prices and estimate momentum

    Args:
        prices: Array of prices

    Returns:
        dict: Kalman filter results
    """
    try:
        if len(prices) < 10:
            raise ValueError(f"Insufficient data: {len(prices)} < 10")

        # Initialize Kalman filter
        # State: [price, velocity]
        kf = KalmanFilter(
            transition_matrices=[[1, 1], [0, 1]],  # price(t+1) = price(t) + velocity(t)
            observation_matrices=[[1, 0]],  # We observe only price
            initial_state_mean=[prices[0], 0],
            initial_state_covariance=[[1, 0], [0, 1]],
            transition_covariance=[[0.01, 0], [0, 0.01]],
            observation_covariance=[[1]]
        )

        # Filter
        state_means, state_covariances = kf.filter(prices)

        # Extract filtered price and velocity
        filtered_price = state_means[:, 0]
        velocity = state_means[:, 1]

        # Current values
        current_filtered = filtered_price[-1]
        current_velocity = velocity[-1]

        # Momentum score (normalized velocity)
        momentum_score = current_velocity / prices[-1] if prices[-1] > 0 else 0

        # Interpret momentum
        if momentum_score > 0.001:
            momentum_strength = 'STRONG_UP'
            interpretation = f"Strong upward momentum (velocity={current_velocity:.4f}). BULLISH."
        elif momentum_score > 0.0002:
            momentum_strength = 'WEAK_UP'
            interpretation = f"Weak upward momentum (velocity={current_velocity:.4f}). Mildly bullish."
        elif momentum_score < -0.001:
            momentum_strength = 'STRONG_DOWN'
            interpretation = f"Strong downward momentum (velocity={current_velocity:.4f}). BEARISH."
        elif momentum_score < -0.0002:
            momentum_strength = 'WEAK_DOWN'
            interpretation = f"Weak downward momentum (velocity={current_velocity:.4f}). Mildly bearish."
        else:
            momentum_strength = 'NEUTRAL'
            interpretation = f"No significant momentum (velocity={current_velocity:.4f}). NEUTRAL."

        return {
            'theory': 'kalman_filter',
            'filtered_price': float(current_filtered),
            'raw_price': float(prices[-1]),
            'velocity': float(current_velocity),
            'momentum_score': float(momentum_score),
            'momentum_strength': momentum_strength,
            'interpretation': interpretation,
            'noise_reduction': float(abs(current_filtered - prices[-1]))
        }

    except Exception as e:
        return {
            'theory': 'kalman_filter',
            'filtered_price': float(prices[-1]) if len(prices) > 0 else 0.0,
            'velocity': 0.0,
            'momentum_score': 0.0,
            'momentum_strength': 'UNKNOWN',
            'interpretation': f"Error: {str(e)}"
        }

# Test
if __name__ == '__main__':
    np.random.seed(42)

    # Noisy price series with upward trend
    clean_signal = np.linspace(100, 110, 100)
    noise = np.random.randn(100) * 0.5
    noisy_prices = clean_signal + noise

    result = apply_kalman_filter(noisy_prices)
    print("Upward trending (noisy):")
    print(result)
```

**Success Criteria**:
- [ ] Filtered price smoother than raw price
- [ ] Positive velocity detected in uptrend
- [ ] Negative velocity detected in downtrend
- [ ] Execution time < 100ms

**Estimated Time**: 2-3 hours

---

### STEP 7: Implement Monte Carlo VaR/CVaR

**Objective**: Risk simulation and downside risk estimation

**File**: `libs/theories/monte_carlo.py`

**Implementation**:
```python
"""
Monte Carlo Risk Simulation
Calculates VaR and CVaR for risk management
"""
import numpy as np
from typing import Dict, Any

def monte_carlo_var(
    returns: np.ndarray,
    num_simulations: int = 10000,
    time_horizon: int = 1,
    confidence: float = 0.95,
    initial_value: float = 1.0
) -> Dict[str, Any]:
    """
    Calculate VaR and CVaR using Monte Carlo simulation

    Args:
        returns: Historical returns
        num_simulations: Number of Monte Carlo paths
        time_horizon: Days to simulate
        confidence: Confidence level (default 95%)
        initial_value: Initial portfolio value

    Returns:
        dict: VaR/CVaR results
    """
    try:
        # Remove NaN values
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

        if len(returns) < 20:
            raise ValueError(f"Insufficient data: {len(returns)} < 20")

        # Calculate parameters
        mean = np.mean(returns)
        std = np.std(returns)

        # Simulate future returns
        simulated_returns = np.random.normal(
            mean,
            std,
            (num_simulations, time_horizon)
        )

        # Calculate portfolio values
        portfolio_values = initial_value * (1 + simulated_returns).prod(axis=1)

        # Calculate returns
        portfolio_returns = (portfolio_values - initial_value) / initial_value

        # Calculate VaR (Value at Risk)
        var = np.percentile(portfolio_returns, (1 - confidence) * 100)

        # Calculate CVaR (Conditional VaR / Expected Shortfall)
        cvar = portfolio_returns[portfolio_returns <= var].mean()

        # Risk interpretation
        if abs(cvar) < 0.02:
            risk_level = 'LOW'
            interpretation = f"Low risk: CVaR={cvar:.2%}. Maximum expected loss is small."
        elif abs(cvar) < 0.05:
            risk_level = 'MODERATE'
            interpretation = f"Moderate risk: CVaR={cvar:.2%}. Manageable downside."
        elif abs(cvar) < 0.10:
            risk_level = 'HIGH'
            interpretation = f"High risk: CVaR={cvar:.2%}. Significant downside possible."
        else:
            risk_level = 'EXTREME'
            interpretation = f"Extreme risk: CVaR={cvar:.2%}. Large losses possible. REDUCE EXPOSURE."

        return {
            'theory': 'monte_carlo',
            'var_95': float(var),
            'cvar_95': float(cvar),
            'max_loss': float(portfolio_returns.min()),
            'max_gain': float(portfolio_returns.max()),
            'mean_return': float(portfolio_returns.mean()),
            'risk_level': risk_level,
            'interpretation': interpretation,
            'num_simulations': num_simulations,
            'confidence': confidence
        }

    except Exception as e:
        return {
            'theory': 'monte_carlo',
            'var_95': 0.0,
            'cvar_95': 0.0,
            'risk_level': 'UNKNOWN',
            'interpretation': f"Error: {str(e)}"
        }

# Test
if __name__ == '__main__':
    np.random.seed(42)

    # Low volatility returns
    low_vol_returns = np.random.randn(100) * 0.01 + 0.0005
    print("Low volatility:")
    print(monte_carlo_var(low_vol_returns))

    # High volatility returns
    high_vol_returns = np.random.randn(100) * 0.05 - 0.001
    print("\nHigh volatility:")
    print(monte_carlo_var(high_vol_returns))
```

**Success Criteria**:
- [ ] VaR and CVaR calculated correctly
- [ ] Low volatility data shows low risk
- [ ] High volatility data shows high risk
- [ ] Execution time < 500ms (10k simulations)

**Estimated Time**: 2-3 hours

---

### STEP 8: Integrate All Theories into V7 Runtime

**Objective**: Modify V7 runtime to call all 5 core theories

**File**: `apps/runtime/v7_runtime.py`

**Modifications**:
```python
# Add imports at top
from libs.theories.hurst_exponent import analyze_hurst
from libs.theories.shannon_entropy import calculate_shannon_entropy
from libs.theories.markov_regime import detect_market_regime_simple
from libs.theories.kalman_filter import apply_kalman_filter
from libs.theories.monte_carlo import monte_carlo_var

# In signal generation method
def generate_signal(self, symbol: str, df: pd.DataFrame):
    """Generate signal with full mathematical analysis"""

    # ... existing code ...

    # RUN ALL THEORIES
    logger.info(f"Running mathematical theories for {symbol}")

    theories = {}

    # 1. Hurst Exponent
    try:
        theories['hurst'] = analyze_hurst(df['close'].values)
        logger.info(f"Hurst: {theories['hurst']['value']:.3f} ({theories['hurst']['market_type']})")
    except Exception as e:
        logger.error(f"Hurst failed: {e}")
        theories['hurst'] = {'error': str(e)}

    # 2. Shannon Entropy
    try:
        returns = df['close'].pct_change().dropna().values
        theories['shannon'] = calculate_shannon_entropy(returns)
        logger.info(f"Entropy: {theories['shannon']['normalized_entropy']:.3f}")
    except Exception as e:
        logger.error(f"Shannon failed: {e}")
        theories['shannon'] = {'error': str(e)}

    # 3. Markov Regime
    try:
        theories['markov'] = detect_market_regime_simple(df)
        logger.info(f"Regime: {theories['markov']['regime']}")
    except Exception as e:
        logger.error(f"Markov failed: {e}")
        theories['markov'] = {'error': str(e)}

    # 4. Kalman Filter
    try:
        theories['kalman'] = apply_kalman_filter(df['close'].values)
        logger.info(f"Momentum: {theories['kalman']['momentum_strength']}")
    except Exception as e:
        logger.error(f"Kalman failed: {e}")
        theories['kalman'] = {'error': str(e)}

    # 5. Monte Carlo
    try:
        theories['monte_carlo'] = monte_carlo_var(returns)
        logger.info(f"Risk: {theories['monte_carlo']['risk_level']} (CVaR: {theories['monte_carlo']['cvar_95']:.2%})")
    except Exception as e:
        logger.error(f"Monte Carlo failed: {e}")
        theories['monte_carlo'] = {'error': str(e)}

    # ... pass theories to DeepSeek ...
    signal = self.signal_generator.generate(
        symbol=symbol,
        price_data=df,
        theories=theories,  # <-- NEW
        market_context=coingecko_context
    )

    return signal
```

**Testing**:
```bash
# Test V7 with theories
.venv/bin/python3 apps/runtime/v7_runtime.py --iterations 1 --symbols BTC-USD

# Check log output
tail -100 /tmp/v7_with_theories.log | grep -E "Hurst|Entropy|Regime|Momentum|Risk"

# Should see:
# Hurst: 0.XXX (trending/mean_reverting/random)
# Entropy: 0.XXX
# Regime: BULL/BEAR/SIDEWAYS
# Momentum: STRONG_UP/WEAK_UP/etc
# Risk: LOW/MODERATE/HIGH
```

**Success Criteria**:
- [ ] All 5 theories execute without errors
- [ ] Log shows theory outputs
- [ ] Signal generated within 2 seconds
- [ ] Memory usage < 1 GB

**Estimated Time**: 3-4 hours

---

### STEP 9: Update DeepSeek Prompt with Theory Data

**Objective**: Format theory outputs for LLM consumption

**File**: `libs/llm/signal_synthesizer.py`

**Add method**:
```python
def _format_theory_analysis(self, theories: dict) -> str:
    """Format mathematical theory results for DeepSeek LLM"""

    sections = []

    # Hurst Exponent
    if 'hurst' in theories and 'error' not in theories['hurst']:
        h = theories['hurst']
        sections.append(
            f"**Trend Persistence (Hurst)**: {h['value']:.3f}\n"
            f"  - Market Type: {h['market_type']}\n"
            f"  - Strategy: {h['strategy']}\n"
            f"  - Confidence: {h['confidence']:.1%}\n"
            f"  - {h['interpretation']}"
        )

    # Shannon Entropy
    if 'shannon' in theories and 'error' not in theories['shannon']:
        s = theories['shannon']
        sections.append(
            f"**Market Predictability (Shannon Entropy)**: {s['normalized_entropy']:.3f}\n"
            f"  - Predictability: {s['predictability']}\n"
            f"  - Signal: {s['signal']}\n"
            f"  - {s['interpretation']}"
        )

    # Markov Regime
    if 'markov' in theories and 'error' not in theories['markov']:
        m = theories['markov']
        sections.append(
            f"**Market Regime (Markov)**: {m['regime']}\n"
            f"  - Strategy: {m['strategy']}\n"
            f"  - Confidence: {m['confidence']:.1%}\n"
            f"  - Trend: {m['trend']:+.2%}, Volatility: {m['volatility']:.3f}\n"
            f"  - {m['interpretation']}"
        )

    # Kalman Filter
    if 'kalman' in theories and 'error' not in theories['kalman']:
        k = theories['kalman']
        sections.append(
            f"**Momentum (Kalman Filter)**: {k['momentum_strength']}\n"
            f"  - Filtered Price: ${k['filtered_price']:.2f} (Raw: ${k['raw_price']:.2f})\n"
            f"  - Velocity: {k['velocity']:.4f}\n"
            f"  - {k['interpretation']}"
        )

    # Monte Carlo
    if 'monte_carlo' in theories and 'error' not in theories['monte_carlo']:
        mc = theories['monte_carlo']
        sections.append(
            f"**Risk Assessment (Monte Carlo)**: {mc['risk_level']}\n"
            f"  - VaR (95%): {mc['var_95']:.2%}\n"
            f"  - CVaR (95%): {mc['cvar_95']:.2%}\n"
            f"  - Max Potential Loss: {mc['max_loss']:.2%}\n"
            f"  - {mc['interpretation']}"
        )

    return "\n\n".join(sections)

# Update create_prompt method to use this
def create_prompt(self, symbol, price_data, theories, market_context):
    theory_section = self._format_theory_analysis(theories)

    prompt = f"""
You are an expert quantitative trader analyzing {symbol}.

=== MATHEMATICAL ANALYSIS ===

{theory_section}

=== MARKET CONTEXT ===
{self._format_market_context(market_context)}

=== CURRENT PRICE ===
{price_data['close'].iloc[-1]:.2f}

Based on the mathematical theories and market context above, generate a trading signal.

IMPORTANT:
- Only suggest BUY if multiple theories align bullishly
- Only suggest SELL if multiple theories align bearishly
- Suggest HOLD if theories are mixed or inconclusive
- Consider risk assessment in your decision

Output format:
{{
  "signal": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation referencing theories"
}}
"""

    return prompt
```

**Success Criteria**:
- [ ] Theory data formatted correctly in prompt
- [ ] DeepSeek receives all theory outputs
- [ ] LLM reasoning references theory data

**Estimated Time**: 2-3 hours

---

### STEP 10: Monitor & Validate Signal Distribution

**Objective**: Verify theories improve signal quality

**Monitoring Script**: `scripts/monitor_v7_theories.py`

```python
#!/usr/bin/env python3
"""Monitor V7 theory-enhanced signal distribution"""

import sqlite3
from datetime import datetime, timedelta

def monitor_signals(hours=2):
    """Monitor signal distribution over last N hours"""

    conn = sqlite3.connect('tradingai.db')
    cursor = conn.cursor()

    cutoff = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')

    # Signal distribution
    query = """
    SELECT
        direction,
        COUNT(*) as count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage,
        ROUND(AVG(confidence), 3) as avg_confidence
    FROM signals
    WHERE timestamp > ?
    GROUP BY direction
    ORDER BY count DESC
    """

    cursor.execute(query, (cutoff,))
    results = cursor.fetchall()

    print(f"\n=== Signal Distribution (Last {hours} hours) ===\n")
    print(f"{'Direction':<10} {'Count':<8} {'Percentage':<12} {'Avg Confidence'}")
    print("-" * 50)

    for direction, count, pct, conf in results:
        print(f"{direction:<10} {count:<8} {pct:>5}% {conf:>15.1%}")

    # Check if improvement from 98.5% HOLD baseline
    total = sum(r[1] for r in results)
    hold_count = next((r[1] for r in results if r[0] == 'hold'), 0)
    hold_pct = (hold_count / total * 100) if total > 0 else 100

    print(f"\n{'='*50}")
    if hold_pct < 70:
        print(f"✅ SUCCESS: HOLD rate reduced to {hold_pct:.1%} (was 98.5%)")
    elif hold_pct < 85:
        print(f"⚠️  PROGRESS: HOLD rate at {hold_pct:.1%} (target <70%)")
    else:
        print(f"❌ NO IMPROVEMENT: HOLD still at {hold_pct:.1%}")

    conn.close()

if __name__ == '__main__':
    import sys
    hours = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    monitor_signals(hours)
```

**Run monitoring**:
```bash
# After 2 hours of V7 running
.venv/bin/python3 scripts/monitor_v7_theories.py 2

# Expected output:
# Direction   Count    Percentage   Avg Confidence
# --------------------------------------------------
# hold        120      60.0%        65.2%
# long        55       27.5%        68.5%
# short       25       12.5%        66.8%
#
# ✅ SUCCESS: HOLD rate reduced to 60.0% (was 98.5%)
```

**Success Criteria**:
- [ ] HOLD signals < 70% (down from 98.5%)
- [ ] BUY/SELL signals > 25% (up from 1.5%)
- [ ] Average confidence > 60%

**Estimated Time**: 1 hour (+ 2-4 hours monitoring)

---

## 📊 RESOURCE MONITORING

### Monitor During Implementation

```bash
# Terminal 1: Watch CPU usage
watch -n 5 'ps aux | grep v7_runtime | grep -v grep'

# Terminal 2: Watch memory
watch -n 5 'free -h'

# Terminal 3: Watch theory execution times
tail -f /tmp/v7_with_theories.log | grep -E "Hurst|Entropy|Regime|Kalman|Monte"
```

**Expected Resource Usage**:
- CPU: 20-30% average (1 core spike during signal generation)
- RAM: 1.5-2.0 GB total
- Theory calculation: 360-820ms per signal
- Signals per hour: 30-60 (with 10 symbols, 900s interval)

**If Issues**:
- CPU > 80%: Increase sleep interval (900s → 1200s)
- RAM > 4 GB: Reduce data window size (100 → 50 candles)
- Theories > 2 seconds: Optimize or parallelize

---

## ✅ FINAL CHECKLIST

Before considering implementation complete:

**Core Theories**:
- [ ] Hurst exponent module working
- [ ] Shannon entropy module working
- [ ] Markov regime module working
- [ ] Kalman filter module working
- [ ] Monte Carlo VaR/CVaR module working

**Integration**:
- [ ] All theories called in V7 runtime
- [ ] Theory outputs passed to DeepSeek
- [ ] DeepSeek prompt includes theory data
- [ ] LLM reasoning references theories

**Performance**:
- [ ] HOLD signals < 70% (baseline: 98.5%)
- [ ] BUY/SELL signals > 25% (baseline: 1.5%)
- [ ] Theory execution < 1 second per signal
- [ ] Memory usage < 2 GB
- [ ] CPU usage < 50% average

**Monitoring**:
- [ ] Signal distribution dashboard working
- [ ] Theory outputs logged correctly
- [ ] Resource usage within limits
- [ ] No crashes or OOM errors for 24 hours

---

## 🚀 NEXT PHASES (After Core Complete)

**Phase 2: ML Enhancement** (Add Later):
- XGBoost/LightGBM ensemble
- Prophet forecasting
- Advanced technical indicators
- Pattern recognition

**Phase 3: Risk Management** (Add Later):
- Portfolio optimization (Riskfolio-Lib)
- Kelly Criterion position sizing
- Enhanced VaR/CVaR metrics

**Phase 4: Validation** (Add Later):
- Backtrader integration
- Walk-forward analysis
- Performance reporting

---

## 📞 SUPPORT & TROUBLESHOOTING

**If theories are slow**:
- Reduce window sizes
- Use simplified implementations
- Parallelize if needed

**If memory issues**:
- Add swap (Step 1)
- Use float32 instead of float64
- Process data in chunks

**If signal distribution doesn't improve**:
- Check DeepSeek is using theory data
- Verify prompt formatting
- Lower confidence threshold
- Review theory interpretation logic

---

**STATUS**: ⏳ READY TO START STEP 1
**ESTIMATED TOTAL TIME**: 20-30 hours for Steps 1-10
**GOAL**: Transform V7 from 98.5% HOLD → 50-60% HOLD with mathematical foundation
