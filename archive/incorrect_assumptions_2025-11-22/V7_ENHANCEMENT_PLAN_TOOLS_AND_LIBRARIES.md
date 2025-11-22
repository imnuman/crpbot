# V7 Enhancement Plan: Tools & Libraries for Performance Improvement

**Date**: 2025-11-21
**From**: QC Claude (Research & Planning)
**To**: Builder Claude (Implementation)
**Purpose**: Comprehensive tool recommendations to improve V7 prediction accuracy, timing, and risk management

---

## 🎯 OBJECTIVES

**Goal**: Transform V7 from 98.5% HOLD signals → High-performance prediction system

**Requirements**:
1. ✅ Predict coin prices accurately (short-term trends)
2. ✅ Determine optimal buy/sell timing (entry/exit points)
3. ✅ Minimize risk (drawdown, VaR, position sizing)
4. ✅ Maximize profit (risk-adjusted returns, Sharpe ratio)

**Approach**: Implement battle-tested open-source libraries + mathematical theories

---

## 📦 RECOMMENDED TOOLS & LIBRARIES

### Category 1: Mathematical Theory Implementation

#### 1.1 Shannon Entropy & Information Theory

**Library**: `EntropyHub`
- **GitHub**: https://pypi.org/project/EntropyHub/
- **Purpose**: 40+ entropy functions for time series analysis
- **Use Case**: Measure market predictability and uncertainty
- **Installation**: `pip install EntropyHub`

**Alternative**: Manual implementation
```python
import numpy as np
from scipy.stats import entropy

def shannon_entropy(returns):
    """Calculate Shannon entropy of price returns"""
    hist, bin_edges = np.histogram(returns, bins=50, density=True)
    hist = hist[hist > 0]  # Remove zero probabilities
    return entropy(hist)
```

**Integration Point**: `libs/theories/shannon_entropy.py`

---

#### 1.2 Hurst Exponent (Trend Persistence)

**Library**: `GenHurst` + Custom Implementation
- **GitHub**: https://github.com/PTRRupprecht/GenHurst
- **Purpose**: Detect trend persistence vs mean reversion
- **Output**: H > 0.5 (trending), H < 0.5 (mean-reverting)

**Better Alternative**: `hurst` package
```bash
pip install hurst
```

**Usage**:
```python
from hurst import compute_Hc

# Compute Hurst exponent
H, c, data = compute_Hc(prices, kind='price', simplified=True)

# Interpretation:
# H > 0.5: Trending market (momentum strategy)
# H < 0.5: Mean-reverting (reversal strategy)
# H ≈ 0.5: Random walk (avoid trading)
```

**Integration Point**: `libs/theories/hurst_exponent.py`

---

#### 1.3 Kalman Filter (State Estimation & Denoising)

**Library**: `pykalman` (Best for finance)
- **GitHub**: https://github.com/pykalman/pykalman
- **Purpose**: Denoise prices, estimate velocity/momentum
- **Installation**: `pip install pykalman`

**Alternative**: `FilterPy` (more flexible)
```bash
pip install filterpy
```

**Usage Example**:
```python
from pykalman import KalmanFilter

# Define Kalman filter
kf = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[1],
    initial_state_mean=0,
    initial_state_covariance=1,
    observation_covariance=1,
    transition_covariance=0.01
)

# Apply to price series
state_means, _ = kf.filter(prices)
denoised_prices = state_means.flatten()
```

**Integration Point**: `libs/theories/kalman_filter.py`

---

#### 1.4 Monte Carlo Simulation (Risk Analysis)

**Library**: Built-in with `numpy` (sufficient)
- **Purpose**: Simulate 10,000 future price paths
- **Output**: VaR (Value at Risk), CVaR (Conditional VaR)

**Implementation**:
```python
import numpy as np

def monte_carlo_var(returns, num_simulations=10000, time_horizon=1, confidence=0.95):
    """Calculate VaR using Monte Carlo simulation"""

    # Calculate mean and std of returns
    mean = np.mean(returns)
    std = np.std(returns)

    # Simulate future returns
    simulated_returns = np.random.normal(mean, std, (num_simulations, time_horizon))

    # Calculate portfolio values
    portfolio_values = (1 + simulated_returns).prod(axis=1)

    # Calculate VaR
    var = np.percentile(portfolio_values, (1 - confidence) * 100)

    # Calculate CVaR (Expected Shortfall)
    cvar = portfolio_values[portfolio_values <= var].mean()

    return var, cvar
```

**Integration Point**: `libs/theories/monte_carlo.py`

---

#### 1.5 Markov Regime Detection

**Library**: `hmmlearn` (Hidden Markov Models)
- **GitHub**: https://github.com/hmmlearn/hmmlearn
- **Purpose**: Detect market regimes (bull/bear/sideways)
- **Installation**: `pip install hmmlearn`

**Usage**:
```python
from hmmlearn import hmm
import numpy as np

# Define 3-state HMM (bull, bear, sideways)
model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000)

# Train on returns + volatility
features = np.column_stack([returns, volatility])
model.fit(features)

# Predict current regime
current_regime = model.predict(features[-1:])

# Decode regime states
# 0 = Bull, 1 = Bear, 2 = Sideways (based on mean returns)
```

**Integration Point**: `libs/theories/markov_regime.py`

---

### Category 2: Technical Analysis Enhancement

#### 2.1 Advanced Technical Indicators

**Library**: `pandas-ta` (Most Comprehensive)
- **GitHub**: https://github.com/twopirllc/pandas-ta
- **Purpose**: 130+ technical indicators
- **Installation**: `pip install pandas-ta`

**Key Indicators to Add**:
```python
import pandas_ta as ta

# Add missing indicators
df.ta.adx()         # Average Directional Index (trend strength)
df.ta.cci()         # Commodity Channel Index
df.ta.dpo()         # Detrended Price Oscillator
df.ta.kc()          # Keltner Channels
df.ta.psar()        # Parabolic SAR (stop and reverse)
df.ta.supertrend()  # SuperTrend (trend following)
df.ta.vwap()        # Volume Weighted Average Price
df.ta.vortex()      # Vortex Indicator
```

**Integration**: Add to `libs/features/technical_indicators.py`

---

#### 2.2 Pattern Recognition

**Library**: `ta-lib` (Industry Standard)
- **GitHub**: https://github.com/TA-Lib/ta-lib-python
- **Purpose**: Candlestick pattern recognition
- **Installation**: Complex (requires C library)

**Simpler Alternative**: `mplfinance` + manual patterns
```bash
pip install mplfinance
```

**Pattern Detection**:
```python
def detect_patterns(df):
    """Detect common candlestick patterns"""
    patterns = {}

    # Doji
    patterns['doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low'])) < 0.1

    # Hammer
    patterns['hammer'] = (
        (df['close'] > df['open']) &
        ((df['close'] - df['low']) > 2 * (df['high'] - df['close']))
    )

    # Engulfing
    patterns['bullish_engulfing'] = (
        (df['close'] > df['open']) &
        (df['open'].shift(1) > df['close'].shift(1)) &
        (df['close'] > df['open'].shift(1))
    )

    return patterns
```

**Integration**: `libs/features/pattern_recognition.py`

---

### Category 3: Machine Learning Enhancement

#### 3.1 Ensemble Models for Predictions

**Library**: `scikit-learn` (Already have) + `XGBoost` + `LightGBM`

**Installation**:
```bash
pip install xgboost lightgbm
```

**Usage - Price Direction Prediction**:
```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

# Create ensemble
xgb = XGBClassifier(n_estimators=100, max_depth=5)
lgb = LGBMClassifier(n_estimators=100, max_depth=5)

ensemble = VotingClassifier(
    estimators=[('xgb', xgb), ('lgb', lgb)],
    voting='soft'
)

# Train
ensemble.fit(X_train, y_train)

# Predict direction (UP/DOWN/NEUTRAL)
predictions = ensemble.predict_proba(X_test)
```

**Integration**: `libs/ml/ensemble_predictor.py`

---

#### 3.2 Time Series Forecasting

**Library**: `Prophet` (Facebook)
- **GitHub**: https://github.com/facebook/prophet
- **Purpose**: Robust time series forecasting
- **Installation**: `pip install prophet`

**Usage**:
```python
from prophet import Prophet

# Prepare data
df_prophet = pd.DataFrame({
    'ds': df['timestamp'],
    'y': df['close']
})

# Create and fit model
model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    changepoint_prior_scale=0.05
)
model.fit(df_prophet)

# Forecast next 24 hours
future = model.make_future_dataframe(periods=1440, freq='T')  # 1440 minutes
forecast = model.predict(future)
```

**Integration**: `libs/ml/prophet_forecaster.py`

---

### Category 4: Risk Management & Portfolio Optimization

#### 4.1 Portfolio Risk Management

**Library**: `Riskfolio-Lib` (Best for Crypto)
- **GitHub**: https://github.com/dcajasn/Riskfolio-Lib
- **Purpose**: 20+ risk measures, portfolio optimization
- **Installation**: `pip install riskfolio-lib`

**Key Features**:
- CVaR (Conditional Value at Risk)
- CDaR (Conditional Drawdown at Risk)
- Maximum Drawdown
- Ulcer Index
- Efficient frontier

**Usage**:
```python
import riskfolio as rp

# Create portfolio object
port = rp.Portfolio(returns=returns_df)

# Calculate optimal weights (minimize CVaR)
port.assets_stats(method_mu='hist', method_cov='hist')
weights = port.optimization(
    model='Classic',
    rm='CVaR',
    obj='Sharpe',
    rf=0,
    l=0,
    hist=True
)
```

**Integration**: `libs/risk/portfolio_optimizer.py`

---

#### 4.2 Position Sizing

**Library**: Custom implementation using Kelly Criterion

**Implementation**:
```python
def kelly_position_size(win_rate, avg_win, avg_loss, max_risk=0.25):
    """
    Calculate optimal position size using Kelly Criterion

    Args:
        win_rate: Historical win rate (0-1)
        avg_win: Average win percentage
        avg_loss: Average loss percentage
        max_risk: Maximum fraction of capital to risk (default 25%)

    Returns:
        Optimal position size as fraction of capital
    """
    if win_rate <= 0 or win_rate >= 1:
        return 0

    # Kelly fraction
    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

    # Apply safety factor (use 25-50% of Kelly)
    safe_kelly = kelly_fraction * 0.5

    # Cap at max_risk
    position_size = min(safe_kelly, max_risk)

    return max(0, position_size)
```

**Integration**: `libs/risk/position_sizing.py`

---

### Category 5: Backtesting & Validation

#### 5.1 Professional Backtesting Engine

**Library**: `Backtrader` (Most Popular)
- **GitHub**: https://github.com/mementum/backtrader
- **Purpose**: Realistic backtesting with slippage, fees, multiple strategies
- **Installation**: `pip install backtrader`

**Alternative**: `VectorBT` (Faster)
```bash
pip install vectorbt
```

**Usage Example**:
```python
import backtrader as bt

class V7Strategy(bt.Strategy):
    def __init__(self):
        self.signal = None

    def next(self):
        # Get V7 signal
        signal = self.get_v7_signal()

        if signal == 'BUY' and not self.position:
            self.buy()
        elif signal == 'SELL' and self.position:
            self.sell()

# Run backtest
cerebro = bt.Cerebro()
cerebro.addstrategy(V7Strategy)
cerebro.adddata(data_feed)
cerebro.broker.setcash(100000)
cerebro.broker.setcommission(commission=0.001)
cerebro.run()
```

**Integration**: `apps/backtester/v7_backtest.py`

---

#### 5.2 Walk-Forward Analysis

**Library**: Custom implementation

**Purpose**: Test strategy on rolling windows to avoid overfitting

**Implementation**:
```python
def walk_forward_analysis(data, train_period=30, test_period=7, step=7):
    """
    Perform walk-forward analysis

    Args:
        data: Historical data
        train_period: Days to train on
        test_period: Days to test on
        step: Days to step forward
    """
    results = []

    for i in range(0, len(data) - train_period - test_period, step):
        # Split data
        train_data = data[i:i+train_period]
        test_data = data[i+train_period:i+train_period+test_period]

        # Train model
        model = train_model(train_data)

        # Test model
        performance = test_model(model, test_data)
        results.append(performance)

    return results
```

**Integration**: `apps/backtester/walk_forward.py`

---

## 🎯 IMPLEMENTATION PRIORITY

### Phase 1: Core Mathematical Theories (Days 1-2)

**Priority 1A - Trend & Regime Detection**:
1. ✅ Hurst Exponent (`hurst` package) - 2 hours
2. ✅ Markov Regime Detection (`hmmlearn`) - 3 hours
3. ✅ Shannon Entropy (manual implementation) - 2 hours

**Priority 1B - Risk & Filtering**:
4. ✅ Kalman Filter (`pykalman`) - 2 hours
5. ✅ Monte Carlo VaR/CVaR (numpy) - 2 hours

**Expected Impact**: Signal distribution 98.5% HOLD → 60-70% HOLD

---

### Phase 2: ML Enhancement (Days 3-4)

**Priority 2A - Prediction Models**:
1. ✅ XGBoost/LightGBM Ensemble - 4 hours
2. ✅ Prophet Time Series Forecasting - 3 hours

**Priority 2B - Feature Enhancement**:
3. ✅ Advanced Technical Indicators (`pandas-ta`) - 2 hours
4. ✅ Pattern Recognition (manual) - 2 hours

**Expected Impact**: Prediction accuracy improvement 10-15%

---

### Phase 3: Risk Management (Day 5)

**Priority 3A - Portfolio Optimization**:
1. ✅ Riskfolio-Lib integration - 3 hours
2. ✅ Kelly Criterion Position Sizing - 2 hours

**Priority 3B - Risk Metrics**:
3. ✅ Enhanced VaR/CVaR calculation - 2 hours
4. ✅ Drawdown analysis - 2 hours

**Expected Impact**: Risk-adjusted returns improvement 20-30%

---

### Phase 4: Validation (Days 6-7)

**Priority 4A - Backtesting**:
1. ✅ Backtrader integration - 4 hours
2. ✅ Walk-forward analysis - 3 hours

**Priority 4B - Performance Analysis**:
3. ✅ Sharpe ratio, Sortino ratio, Calmar ratio - 2 hours
4. ✅ Monte Carlo strategy validation - 2 hours

**Expected Impact**: Confidence in system reliability

---

## 📋 DETAILED IMPLEMENTATION PLAN

### Day 1: Hurst + Shannon + Markov

**Morning (4 hours)**:
```bash
# 1. Install libraries
pip install hurst hmmlearn EntropyHub

# 2. Create theory modules
touch libs/theories/hurst_exponent.py
touch libs/theories/shannon_entropy.py
touch libs/theories/markov_regime.py

# 3. Implement Hurst Exponent
```

**`libs/theories/hurst_exponent.py`**:
```python
"""Hurst Exponent - Trend Persistence Analysis"""
from hurst import compute_Hc
import numpy as np

def analyze_hurst(prices, window=100):
    """
    Calculate Hurst exponent for trend analysis

    Returns:
        dict: {
            'value': H value (0-1),
            'interpretation': str,
            'market_type': 'trending'|'mean_reverting'|'random'
        }
    """
    try:
        H, c, data = compute_Hc(prices[-window:], kind='price', simplified=True)

        # Interpret
        if H > 0.55:
            market_type = 'trending'
            interpretation = f"Strong trend persistence (H={H:.2f}). Momentum strategy favorable."
        elif H < 0.45:
            market_type = 'mean_reverting'
            interpretation = f"Mean-reverting behavior (H={H:.2f}). Reversal strategy favorable."
        else:
            market_type = 'random'
            interpretation = f"Random walk (H={H:.2f}). Avoid trading or use neutral strategy."

        return {
            'value': float(H),
            'c': float(c),
            'interpretation': interpretation,
            'market_type': market_type,
            'confidence': abs(H - 0.5) * 2  # 0=uncertain, 1=very certain
        }

    except Exception as e:
        return {
            'value': 0.5,
            'interpretation': f"Error: {e}",
            'market_type': 'unknown',
            'confidence': 0.0
        }
```

**Afternoon (4 hours)**:
```python
# 4. Implement Shannon Entropy
# 5. Implement Markov Regime
# 6. Test all three theories
# 7. Integrate into v7_runtime.py
```

**Test Command**:
```bash
python -c "
from libs.theories.hurst_exponent import analyze_hurst
import pandas as pd

# Load test data
df = pd.read_parquet('data/features/features_BTC-USD_1m_latest.parquet')
result = analyze_hurst(df['close'].values)
print(f'Hurst: {result}')
"
```

---

### Day 2: Kalman + Monte Carlo + Integration

**Morning (4 hours)**:
```bash
# 1. Install libraries
pip install pykalman filterpy

# 2. Implement Kalman Filter
# 3. Implement Monte Carlo
```

**Afternoon (4 hours)**:
```bash
# 4. Integrate all 5 theories into V7 runtime
# 5. Update signal_synthesizer to format theory outputs
# 6. Test signal generation with full theory analysis
# 7. Deploy updated V7
```

**Integration in `apps/runtime/v7_runtime.py`**:
```python
def generate_signal(self, symbol: str, df: pd.DataFrame):
    """Generate trading signal with full mathematical analysis"""

    # Run all theories
    theories = {
        'hurst': analyze_hurst(df['close'].values),
        'shannon': calculate_shannon_entropy(df['close'].pct_change().dropna()),
        'markov': detect_market_regime(df),
        'kalman': apply_kalman_filter(df['close'].values),
        'monte_carlo': monte_carlo_var(df['close'].pct_change().dropna())
    }

    # Pass to DeepSeek LLM
    signal = self.signal_generator.generate(
        symbol=symbol,
        price_data=df,
        theories=theories,
        market_context=coingecko_context
    )

    return signal
```

---

### Day 3: ML Ensemble + Prophet

**Morning (4 hours)**:
```bash
# 1. Install ML libraries
pip install xgboost lightgbm prophet

# 2. Create ML prediction module
touch libs/ml/ensemble_predictor.py
touch libs/ml/prophet_forecaster.py

# 3. Implement XGBoost + LightGBM ensemble
# 4. Train on historical signals and outcomes
```

**Afternoon (4 hours)**:
```bash
# 5. Implement Prophet forecasting
# 6. Create price prediction pipeline
# 7. Add predictions to signal generation
# 8. Test ML-enhanced signals
```

---

### Day 4: Advanced Technical Indicators + Patterns

**Morning (3 hours)**:
```bash
# 1. Install pandas-ta
pip install pandas-ta

# 2. Add 20+ new technical indicators
# 3. Update feature engineering pipeline
```

**Afternoon (3 hours)**:
```bash
# 4. Implement pattern recognition
# 5. Add pattern signals to theory analysis
# 6. Test combined system (theories + ML + patterns)
```

---

### Day 5: Risk Management + Position Sizing

**Morning (4 hours)**:
```bash
# 1. Install Riskfolio-Lib
pip install riskfolio-lib

# 2. Implement portfolio optimization
touch libs/risk/portfolio_optimizer.py

# 3. Implement Kelly Criterion position sizing
touch libs/risk/position_sizing.py

# 4. Calculate optimal position sizes for each signal
```

**Afternoon (4 hours)**:
```bash
# 5. Add risk metrics to signals (VaR, CVaR, Max DD)
# 6. Implement stop-loss and take-profit calculation
# 7. Update signal output with risk parameters
# 8. Test risk-managed signals
```

---

### Days 6-7: Backtesting + Validation

**Day 6 (6 hours)**:
```bash
# 1. Install Backtrader
pip install backtrader

# 2. Create V7 backtest strategy
touch apps/backtester/v7_strategy.py

# 3. Run backtest on 90 days historical data
# 4. Calculate performance metrics (Sharpe, Sortino, Calmar, Win Rate)
# 5. Generate backtest report
```

**Day 7 (6 hours)**:
```bash
# 6. Implement walk-forward analysis
# 7. Run 10 walk-forward periods
# 8. Validate strategy robustness
# 9. Generate final performance report
# 10. Document results
```

---

## 🎯 EXPECTED IMPROVEMENTS

### Quantitative Targets

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| HOLD signals | 98.5% | 50-60% | -40% |
| BUY/SELL signals | 1.5% | 30-40% | +2500% |
| Prediction accuracy | Unknown | 60-65% | N/A |
| Sharpe ratio | Unknown | 1.5-2.0 | Target |
| Max drawdown | Unknown | <10% | Target |
| Win rate | 58.3% (22 trades) | 60-65% | +5% |
| Profit factor | Unknown | >1.5 | Target |
| Signals per day | ~100 | 50-80 | Focused |

### Qualitative Improvements

**Better Price Predictions**:
- Prophet forecasts provide 24-hour ahead predictions
- ML ensemble confirms direction with confidence
- Kalman filter smooths noise for clearer trends

**Smarter Buy/Sell Timing**:
- Hurst exponent identifies optimal entry points (trend confirmations)
- Pattern recognition catches reversal signals
- Market regime detection prevents counter-trend trades

**Risk Minimization**:
- Monte Carlo VaR/CVaR quantifies downside risk
- Kelly Criterion optimizes position sizes
- Portfolio optimization balances multi-symbol exposure
- Stop-loss levels calculated mathematically

**Profit Maximization**:
- Risk-adjusted position sizing (bigger positions in high-confidence trades)
- Take-profit levels based on volatility and target risk/reward ratios
- Avoids low-quality signals (filters out uncertain conditions)

---

## 📊 SUCCESS METRICS & VALIDATION

### Phase 1 Success Criteria (Theories Implemented)

✅ **Pass**:
- Signal distribution: 50-70% HOLD (down from 98.5%)
- At least 100 actionable signals (BUY/SELL) generated in 24 hours
- Theory outputs passing validation tests

❌ **Fail**:
- Still >90% HOLD signals → Theory integration issue
- DeepSeek not using theory data → Prompt formatting issue
- Theories returning errors → Implementation bugs

---

### Phase 2 Success Criteria (ML Enhanced)

✅ **Pass**:
- ML ensemble achieves >55% directional accuracy on test set
- Prophet forecasts within 5% of actual prices (24h ahead)
- Feature importance analysis shows new indicators are useful

❌ **Fail**:
- ML accuracy <50% → Overfitting or bad features
- Prophet forecasts wildly off → Model misconfiguration
- New features have zero importance → Not adding value

---

### Phase 3 Success Criteria (Risk Management)

✅ **Pass**:
- VaR accurately predicts 95th percentile losses
- Kelly position sizing produces positive expected value
- Portfolio optimization reduces correlation-based risk

❌ **Fail**:
- VaR underestimates actual losses → Bad simulation
- Kelly sizes are unrealistically large → Bad parameters
- Portfolio optimization produces nonsensical weights

---

### Phase 4 Success Criteria (Backtesting)

✅ **Pass**:
- Backtest Sharpe ratio >1.5
- Win rate >55%
- Maximum drawdown <15%
- Profit factor >1.5
- Walk-forward analysis shows consistent performance

❌ **Fail**:
- Sharpe <1.0 → Strategy not profitable after costs
- Win rate <50% → Losing more trades than winning
- Max drawdown >20% → Too risky
- Walk-forward performance deteriorates → Overfitting

---

## 🚀 DEPLOYMENT STRATEGY

### Parallel Deployment (Recommended)

Run both old V7 (minimal theories) and new V7 (full theories) side-by-side:

```bash
# Terminal 1: Old V7 (baseline)
nohup .venv/bin/python3 apps/runtime/v7_runtime.py \
  --iterations -1 \
  --sleep-seconds 900 \
  > /tmp/v7_baseline.log 2>&1 &

# Terminal 2: New V7 (enhanced)
nohup .venv/bin/python3 apps/runtime/v7_enhanced.py \
  --iterations -1 \
  --sleep-seconds 900 \
  > /tmp/v7_enhanced.log 2>&1 &
```

**Compare after 48 hours**:
```sql
-- Signal distribution comparison
SELECT
  'Baseline' as version,
  direction,
  COUNT(*) as count,
  ROUND(AVG(confidence), 2) as avg_conf
FROM signals
WHERE timestamp > datetime('now', '-48 hours')
  AND runtime_version = 'baseline'
GROUP BY direction

UNION ALL

SELECT
  'Enhanced' as version,
  direction,
  COUNT(*) as count,
  ROUND(AVG(confidence), 2) as avg_conf
FROM signals
WHERE timestamp > datetime('now', '-48 hours')
  AND runtime_version = 'enhanced'
GROUP BY direction;
```

---

## 📝 DOCUMENTATION REQUIREMENTS

### Code Documentation

Each theory module must include:
```python
"""
Theory: [Name]
Purpose: [What it measures]
Output: [Format and interpretation]
References: [Academic papers or GitHub repos]
"""
```

### Performance Documentation

Create `V7_ENHANCED_PERFORMANCE_REPORT.md`:
- Before/after signal distribution
- Theory contribution analysis
- ML model accuracy metrics
- Risk-adjusted returns
- Backtest results
- Walk-forward analysis results

---

## 🎓 LEARNING RESOURCES

**For Shannon Entropy**:
- Paper: "Shannon Entropy in Financial Time Series"
- Tutorial: https://risk-engineering.org/VaR/

**For Hurst Exponent**:
- Tutorial: https://towardsdatascience.com/introduction-to-the-hurst-exponent-with-code-in-python-4da0414ca52e/
- Paper: "The Hurst Exponent in Finance"

**For Kalman Filters**:
- Book: "Kalman and Bayesian Filters in Python" (free online)
- Tutorial: https://blog.quantinsti.com/kalman-filter/

**For Monte Carlo**:
- Tutorial: https://www.pyquantnews.com/the-pyquant-newsletter/quickly-compute-value-at-risk-with-monte-carlo

**For Portfolio Optimization**:
- Riskfolio-Lib docs: https://riskfolio-lib.readthedocs.io/
- Paper: "Portfolio Optimization with CVaR"

---

## ✅ NEXT STEPS FOR BUILDER CLAUDE

**Immediate Actions** (Today):

1. **Review This Plan** (30 min)
   - [ ] Understand all recommended libraries
   - [ ] Confirm Day 1-7 timeline is feasible
   - [ ] Ask questions if anything unclear

2. **Environment Setup** (30 min)
   ```bash
   # Install Day 1 libraries
   pip install hurst hmmlearn EntropyHub

   # Test installations
   python -c "from hurst import compute_Hc; print('✅ hurst')"
   python -c "from hmmlearn import hmm; print('✅ hmmlearn')"
   python -c "import EntropyHub as eh; print('✅ EntropyHub')"
   ```

3. **Start Day 1 Implementation** (4-6 hours)
   - [ ] Create `libs/theories/hurst_exponent.py`
   - [ ] Implement Hurst analysis function
   - [ ] Test with BTC-USD data
   - [ ] Integrate into V7 runtime
   - [ ] Commit and push

**Expected Deliverable Today**:
- Working Hurst exponent module
- Tested on sample data
- Integrated into V7
- Committed to GitHub

**Report Back**:
- Paste Hurst analysis output for BTC-USD
- Share any issues encountered
- Confirm Day 2 plan

---

## 🎯 FINAL GOALS (7 Days)

By end of Day 7, V7 should:
- ✅ Generate 30-40% actionable signals (not 1.5%)
- ✅ Have 5 core mathematical theories implemented
- ✅ Use ML ensemble for direction prediction
- ✅ Calculate optimal position sizes (Kelly Criterion)
- ✅ Provide VaR/CVaR risk metrics
- ✅ Show 60-65% win rate in backtests
- ✅ Achieve Sharpe ratio >1.5
- ✅ Keep max drawdown <15%

**This transforms V7 from incomplete prototype → Professional trading system**

---

**Status**: ⏳ AWAITING BUILDER CLAUDE TO START DAY 1
**Next**: Environment setup + Hurst exponent implementation
**Timeline**: 7 days to complete all enhancements
