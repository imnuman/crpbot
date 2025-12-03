# V7 Ultimate - Mathematical Theories Documentation

**Version**: 1.0
**Last Updated**: 2025-11-19
**Purpose**: Detailed explanation of the 6 mathematical theories used in V7

---

## Table of Contents

1. [Overview](#overview)
2. [Shannon Entropy](#1-shannon-entropy)
3. [Hurst Exponent](#2-hurst-exponent)
4. [Kolmogorov Complexity](#3-kolmogorov-complexity)
5. [Market Regime Detection](#4-market-regime-detection)
6. [Risk Metrics](#5-risk-metrics)
7. [Fractal Dimension](#6-fractal-dimension)
8. [Theory Integration](#theory-integration)
9. [Implementation Details](#implementation-details)

---

## Overview

V7 Ultimate uses **6 mathematical theories** to analyze market conditions before generating trading signals. Each theory provides a different perspective on market behavior.

### Why Multiple Theories?

**Diversification of Insight**:
- No single indicator is perfect
- Multiple theories reduce false signals
- Cross-validation improves accuracy

**Inspired by Renaissance Technologies**:
- Jim Simons' Medallion Fund uses mathematical models
- Focus on statistical edges, not predictions
- Systematic, emotion-free decision making

###Combined Score

All 6 theories feed into DeepSeek LLM which:
1. Weights each theory by reliability
2. Synthesizes conflicting signals
3. Generates final recommendation
4. Provides confidence score

---

## 1. Shannon Entropy

### What It Measures

**Market Randomness** - How predictable or chaotic the market is.

**Formula**:
```
H(X) = -Σ P(x) * log₂(P(x))
```

Where:
- H(X) = Entropy
- P(x) = Probability of price state x
- Σ = Sum over all possible states

### Implementation

**V7 Calculation**:
1. Take last 100 price returns
2. Discretize into bins (e.g., -2% to -1%, -1% to 0%, 0% to 1%, etc.)
3. Calculate probability distribution
4. Apply Shannon formula

**Code** (`libs/theories/shannon_entropy.py`):
```python
def calculate_shannon_entropy(returns, bins=10):
    """Calculate Shannon entropy of return distribution."""
    hist, _ = np.histogram(returns, bins=bins)
    probs = hist / np.sum(hist)
    probs = probs[probs > 0]  # Remove zeros
    entropy = -np.sum(probs * np.log2(probs))
    # Normalize to 0-1
    max_entropy = np.log2(bins)
    return entropy / max_entropy
```

### Interpretation

| Entropy | Meaning | Trading Implication |
|---------|---------|-------------------|
| 0.0-0.3 | Very predictable | **Ideal for trading** - clear patterns |
| 0.3-0.5 | Moderately predictable | Good for trading |
| 0.5-0.7 | Somewhat random | Be cautious |
| 0.7-1.0 | Highly random | **Avoid trading** - too chaotic |

**Example**:
- BTC trending: Entropy = 0.35 (predictable)
- BTC ranging with high volatility: Entropy = 0.75 (random)

### Trading Application

**Use entropy to filter signals**:
- ✅ **Low entropy (<0.5)**: Market has patterns, signals reliable
- ❌ **High entropy (>0.7)**: Market chaotic, skip signals

**V7 Decision**:
```python
if entropy > 0.7:
    return "HOLD"  # Too unpredictable
elif entropy < 0.4 and other_signals_bullish:
    return "BUY"  # Predictable + bullish = good signal
```

---

## 2. Hurst Exponent

### What It Measures

**Trend Persistence** - Does price continue trends or mean-revert?

**Formula** (R/S Analysis):
```
H = log(R/S) / log(n)
```

Where:
- R = Range of cumulative deviations
- S = Standard deviation
- n = Number of observations
- H = Hurst exponent

### Implementation

**V7 Calculation**:
1. Take last 100 price points
2. Calculate rescaled range (R/S) for multiple time lags
3. Fit log(R/S) vs log(n) using linear regression
4. Slope = Hurst exponent

**Code** (`libs/theories/hurst_exponent.py`):
```python
def calculate_hurst_exponent(prices, lags=range(2, 100)):
    """Calculate Hurst exponent using R/S analysis."""
    tau = []
    lagvec = []
    
    for lag in lags:
        # Calculate rescaled range
        ts = prices - np.mean(prices)
        cumsum = np.cumsum(ts)
        R = np.max(cumsum[:lag]) - np.min(cumsum[:lag])
        S = np.std(prices[:lag])
        
        if S > 0:
            tau.append(R / S)
            lagvec.append(lag)
    
    # Linear regression to find Hurst
    poly = np.polyfit(np.log(lagvec), np.log(tau), 1)
    return poly[0]  # Slope = Hurst exponent
```

### Interpretation

| Hurst | Meaning | Trading Strategy |
|-------|---------|-----------------|
| 0.0-0.4 | Mean-reverting | Range trading, sell highs/buy lows |
| 0.4-0.6 | Random walk | No clear edge |
| 0.6-0.8 | Trending | Momentum trading, follow trends |
| 0.8-1.0 | Strong trending | Strong momentum strategies |

**Example**:
- Bull market: Hurst = 0.68 (trending - buy dips)
- Ranging market: Hurst = 0.42 (mean-revert - fade extremes)

### Trading Application

**Adapt strategy to Hurst**:

**Trending (H > 0.6)**:
- Buy breakouts
- Hold winners
- Trail stop losses

**Mean-Reverting (H < 0.4)**:
- Sell resistance
- Buy support
- Take quick profits

**V7 Decision**:
```python
if hurst > 0.6 and price_breaking_resistance:
    return "BUY"  # Trending market, breakout valid
elif hurst < 0.4 and price_at_support:
    return "BUY"  # Mean-revert from oversold
```

---

## 3. Kolmogorov Complexity

### What It Measures

**Pattern Complexity** - How complex is the price pattern?

**Approximation** (Lempel-Ziv Complexity):
Since Kolmogorov complexity is uncomputable, V7 uses Lempel-Ziv compression as a proxy.

### Implementation

**V7 Calculation**:
1. Convert price movements to binary string (up=1, down=0)
2. Apply Lempel-Ziv compression
3. Measure compression ratio
4. Normalize to 0-1

**Code** (`libs/theories/kolmogorov_complexity.py`):
```python
def lempel_ziv_complexity(binary_string):
    """Calculate Lempel-Ziv complexity."""
    i, k, l = 0, 1, 1
    c, k_max = 1, 1
    n = len(binary_string)
    
    while True:
        if binary_string[i + k - 1] == binary_string[l + k - 1]:
            k += 1
            if l + k >= n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    
    return c

def calculate_kolmogorov_complexity(prices):
    """Calculate Kolmogorov complexity proxy."""
    # Convert to binary (up/down movements)
    binary = ''.join(['1' if prices[i] > prices[i-1] else '0' 
                     for i in range(1, len(prices))])
    
    complexity = lempel_ziv_complexity(binary)
    # Normalize
    max_complexity = len(binary) / 2  # Theoretical max
    return min(complexity / max_complexity, 1.0)
```

### Interpretation

| Complexity | Meaning | Trading Implication |
|------------|---------|-------------------|
| 0.0-0.3 | Simple patterns | Easy to predict - good for trading |
| 0.3-0.6 | Moderate complexity | Reasonable predictability |
| 0.6-0.8 | Complex patterns | Harder to predict |
| 0.8-1.0 | Highly complex | Very difficult - avoid |

**Example**:
- Clean uptrend: Complexity = 0.25 (simple)
- Choppy range: Complexity = 0.72 (complex)

### Trading Application

**Use with other signals**:
- Low complexity + low entropy = **Very predictable**
- High complexity + high entropy = **Avoid completely**

**V7 Decision**:
```python
if kolmogorov < 0.4 and shannon < 0.5:
    confidence *= 1.2  # Boost confidence for simple, predictable market
elif kolmogorov > 0.7:
    return "HOLD"  # Too complex to trade
```

---

## 4. Market Regime Detection

### What It Measures

**Current Market State** - Bullish, bearish, or sideways?

**Approach**: Hidden Markov Model (HMM) or simple statistical classification

### Implementation

**V7 Calculation**:
1. Calculate 20-period and 50-period moving averages
2. Measure price position relative to MAs
3. Calculate volatility
4. Classify regime

**Code** (`libs/theories/market_regime.py`):
```python
def detect_market_regime(prices, fast_period=20, slow_period=50):
    """Detect current market regime."""
    ma_fast = prices[-fast_period:].mean()
    ma_slow = prices[-slow_period:].mean()
    current_price = prices[-1]
    volatility = prices[-20:].std() / prices[-20:].mean()
    
    # Bullish conditions
    if (current_price > ma_fast > ma_slow and 
        ma_fast > ma_slow * 1.01):  # 1% separation
        return "bullish"
    
    # Bearish conditions
    elif (current_price < ma_fast < ma_slow and 
          ma_fast < ma_slow * 0.99):
        return "bearish"
    
    # Sideways/ranging
    else:
        return "sideways"
```

### Interpretation

| Regime | Characteristics | Trading Strategy |
|--------|----------------|-----------------|
| Bullish | Price > MAs, MAs ascending | Buy dips, hold longs |
| Bearish | Price < MAs, MAs descending | Sell rallies, hold shorts |
| Sideways | Price oscillating around MAs | Range trade, reduce size |

### Trading Application

**Align signals with regime**:
- ✅ BUY signal in bullish regime = **Strong**
- ⚠️ BUY signal in bearish regime = **Weak/Risky**
- ❌ BUY signal in sideways = **Flip a coin**

**V7 Decision**:
```python
if regime == "bullish" and signal == "BUY":
    confidence *= 1.3  # Boost confidence
elif regime == "bearish" and signal == "BUY":
    confidence *= 0.7  # Reduce confidence
elif regime == "sideways":
    return "HOLD"  # Avoid ranging markets
```

---

## 5. Risk Metrics

### What It Measures

**Risk-Adjusted Returns** - Is the potential profit worth the risk?

**Metrics Calculated**:
1. Value at Risk (VaR)
2. Sharpe Ratio
3. Maximum Drawdown
4. Profit Probability

### Implementation

**5.1 Value at Risk (VaR)**

**95% VaR** = Maximum expected loss with 95% confidence

```python
def calculate_var(returns, confidence=0.95):
    """Calculate Value at Risk."""
    sorted_returns = np.sort(returns)
    index = int((1 - confidence) * len(sorted_returns))
    var = abs(sorted_returns[index])
    return var
```

**Interpretation**:
- VaR = 2% → 95% chance loss won't exceed 2%
- VaR = 8% → High risk, be cautious

**5.2 Sharpe Ratio**

**Formula**:
```
Sharpe = (Expected Return - Risk-Free Rate) / Standard Deviation
```

```python
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio (annualized)."""
    mean_return = returns.mean() * 252  # Annualize
    std_return = returns.std() * np.sqrt(252)
    sharpe = (mean_return - risk_free_rate) / std_return
    return sharpe
```

**Interpretation**:
- Sharpe < 0: Losing money
- Sharpe 0-1: Poor risk/reward
- Sharpe 1-2: Good risk/reward
- Sharpe > 2: Excellent risk/reward

**5.3 Monte Carlo Simulation**

Run 10,000 price simulations to estimate:
- Profit probability
- Expected value
- Worst-case scenarios

```python
def monte_carlo_simulation(current_price, volatility, days=5, sims=10000):
    """Run Monte Carlo simulation for price prediction."""
    results = []
    for _ in range(sims):
        price = current_price
        for _ in range(days):
            change = np.random.normal(0, volatility)
            price *= (1 + change)
        results.append(price)
    
    profit_prob = np.mean([r > current_price for r in results])
    expected_price = np.mean(results)
    
    return {
        'profit_probability': profit_prob,
        'expected_price': expected_price,
        'var_95': np.percentile(results, 5)
    }
```

### Trading Application

**Risk filters**:
```python
if var_95 > 0.05:  # VaR > 5%
    return "HOLD"  # Too risky
    
if sharpe < 1.0:
    return "HOLD"  # Poor risk/reward
    
if profit_probability < 0.55:
    return "HOLD"  # < 55% chance of profit
```

---

## 6. Fractal Dimension

### What It Measures

**Market Structure** - How "rough" or "smooth" is the price chart?

**Formula** (Box-Counting Method):
```
D = lim(ε→0) [log(N(ε)) / log(1/ε)]
```

Where:
- D = Fractal dimension
- N(ε) = Number of boxes needed to cover the curve
- ε = Box size

### Implementation

**V7 Calculation**:
```python
def calculate_fractal_dimension(prices, max_box_size=10):
    """Calculate fractal dimension using box-counting."""
    # Normalize prices
    normalized = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))
    
    counts = []
    sizes = []
    
    for box_size in range(1, max_box_size + 1):
        grid_x = int(len(prices) / box_size) + 1
        grid_y = int(1.0 / box_size) + 1
        
        boxes = set()
        for i, price in enumerate(normalized):
            box_x = int(i / box_size)
            box_y = int(price / box_size)
            boxes.add((box_x, box_y))
        
        counts.append(len(boxes))
        sizes.append(box_size)
    
    # Linear regression to find dimension
    poly = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -poly[0]  # Negative slope = dimension
```

### Interpretation

| Dimension | Meaning | Market Type |
|-----------|---------|-------------|
| 1.0-1.2 | Very smooth | Strong trend, low volatility |
| 1.2-1.5 | Moderate roughness | Typical trending market |
| 1.5-1.8 | Rough | Volatile trending or ranging |
| 1.8-2.0 | Very rough | Highly volatile, chaotic |

**Example**:
- Smooth uptrend: D = 1.3
- Choppy range: D = 1.8

### Trading Application

**Adjust strategy to fractal dimension**:
```python
if fractal < 1.3:
    # Smooth trend - hold longer
    tp_multiplier = 1.5
elif fractal > 1.7:
    # Rough/choppy - take quick profits
    tp_multiplier = 0.8
    return "HOLD"  # Or avoid entirely
```

---

## Theory Integration

### How V7 Combines All 6 Theories

**Step 1: Individual Calculations**
Each theory runs independently:
```python
shannon = calculate_shannon_entropy(returns)
hurst = calculate_hurst_exponent(prices)
kolmogorov = calculate_kolmogorov_complexity(prices)
regime = detect_market_regime(prices)
sharpe, var = calculate_risk_metrics(returns)
fractal = calculate_fractal_dimension(prices)
```

**Step 2: Quality Filters**
Hard cutoffs for bad conditions:
```python
if shannon > 0.7:
    return {"signal": "HOLD", "reason": "High entropy"}
if var > 0.05:
    return {"signal": "HOLD", "reason": "High risk"}
if regime == "sideways":
    return {"signal": "HOLD", "reason": "No clear trend"}
```

**Step 3: LLM Synthesis**
Pass all theory values to DeepSeek:
```python
prompt = f"""
Analyze these market conditions and recommend BUY/SELL/HOLD:

Shannon Entropy: {shannon} (0=predictable, 1=random)
Hurst Exponent: {hurst} (>0.5=trending, <0.5=mean-revert)
Kolmogorov Complexity: {kolmogorov} (low=simple patterns)
Market Regime: {regime}
Sharpe Ratio: {sharpe}
VaR 95%: {var}
Fractal Dimension: {fractal}

Current price: ${current_price}
Provide: signal, confidence, entry_price, sl_price, tp_price, reasoning
"""

response = deepseek_api.complete(prompt)
```

**Step 4: Confidence Scoring**
LLM provides 0-1 confidence, which V7 validates:
```python
if response.confidence < 0.60:
    return "HOLD"  # Low confidence
elif response.confidence >= 0.75:
    tier = "high"
elif response.confidence >= 0.65:
    tier = "medium"
else:
    tier = "low"
```

---

## Implementation Details

### Code Organization

```
libs/theories/
├── shannon_entropy.py
├── hurst_exponent.py
├── kolmogorov_complexity.py
├── market_regime.py
├── risk_metrics.py
└── fractal_dimension.py
```

### Performance Optimization

**Caching**: Theory calculations cached for 60 seconds per symbol
**Batch Processing**: Calculate all theories in parallel
**Incremental Updates**: Only recalculate when new data arrives

### Testing

Each theory module includes:
- Unit tests with known data
- Integration tests with historical prices
- Performance benchmarks

**Example Test**:
```python
def test_shannon_entropy_deterministic():
    # Perfect deterministic series
    perfect = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    entropy = calculate_shannon_entropy(np.diff(perfect))
    assert entropy < 0.1  # Should be very low

def test_shannon_entropy_random():
    # Random series
    random = np.random.randn(100)
    entropy = calculate_shannon_entropy(random)
    assert entropy > 0.7  # Should be high
```

---

## Further Reading

**Shannon Entropy**:
- "A Mathematical Theory of Communication" - Claude Shannon (1948)

**Hurst Exponent**:
- "Long-Term Storage Capacity of Reservoirs" - H.E. Hurst (1951)
- "Fractal Market Analysis" - Edgar Peters (1994)

**Kolmogorov Complexity**:
- "Three Approaches to the Quantitative Definition of Information" - Kolmogorov (1965)

**Market Regimes**:
- "Regime Switching Models" - James Hamilton (1989)

**Risk Metrics**:
- "Value at Risk" - Philippe Jorion (2006)
- "The Sharpe Ratio" - William Sharpe (1966)

**Fractals**:
- "The Misbehavior of Markets" - Benoit Mandelbrot (2004)

---

**Last Updated**: 2025-11-19
**Version**: 1.0
**For V7 Ultimate**: Mathematical Framework
