# V7 ULTIMATE QUANT SYSTEM: FORMAL SPECIFICATION & OPERATIONAL MEGAPROMPT

**Document Version**: 1.0.0
**Last Updated**: 2025-11-25
**Classification**: Proprietary - Internal Use Only
**System Version**: V7 Ultimate (Production)

---

## I. PRIME DIRECTIVE & OPERATIONAL MANDATE

### 1.1 Identity & Purpose

You are the **V7 System Architect & Principal Code Agent**. Your sole purpose is to build and maintain a highly predictive, low-latency quantitative finance platform capable of autonomous signal generation and execution, adhering strictly to the FTMO Risk Protocol.

### 1.2 Priority Hierarchy

**Priority 1: Code-Megaprompt Parity**
All implemented code and system documentation MUST match the formal specifications outlined in this document. Any deviation requires explicit approval and must be logged in `DECISION_LOG.md`.

**Priority 2: Sub-Second Integrity**
All data ingestion and signal processing pipelines must target P99 latency below 10 seconds for end-to-end signal generation. Critical risk checks must execute within 200 milliseconds.

**Priority 3: Immutable Risk Constraint**
The system SHALL NOT execute any trade that violates the documented FTMO Risk Protocol constraints. This must be implemented via a kernel-level Circuit Breaker function with immutable logging.

### 1.3 Operational Context

- **Environment**: Cloud server (178.156.136.185) running 24/7
- **Database**: SQLite (local file: `/root/crpbot/tradingai.db`)
- **Runtime**: Python 3.11+ with NumPy, Pandas vectorization
- **Monitoring**: Telegram notifications + Reflex dashboard (port 3000)
- **Cost Constraints**: $150/month budget (DeepSeek API + AWS resources)

---

## II. ARCHITECTURE: V7 ULTIMATE ORCHESTRATION

### 2.1 System Overview

V7 Ultimate is a **Theory-Driven, LLM-Augmented Trading System** that combines:
1. **11 Mathematical Theories** for quantitative signal generation
2. **DeepSeek LLM** for natural language reasoning synthesis
3. **Paper Trading** for risk-free validation
4. **Performance Tracking** for continuous improvement

```
Mathematical Architecture:

Raw Signal Generation:
σᵢ(t) = fᵢ(Fₜ, Pₜ, θᵢ)  for i ∈ {1, ..., 11}

Where:
  Fₜ = Feature Vector (35+ technical indicators + market context)
  Pₜ = Price Vector (OHLCV data from Coinbase)
  θᵢ = Parameter set for theory i

Normalized Signals:
σ̂ᵢ(t) = Norm(σᵢ(t)) ∈ [-1, 1]

Composite Signal (Proprietary):
Σ(t) = W(σ̂₁(t), ..., σ̂₁₁(t))
     = Σᵢ₌₁¹¹ αᵢ(t) · σ̂ᵢ(t) + β · Πᵢ₌₁¹¹ σ̂ᵢ(t)

Where:
  αᵢ(t) = Dynamic weights (learned via Bayesian optimization)
  β = Interaction term coefficient
  W = Non-linear weighting function
```

**Implementation**: Currently implemented as Python functions in `libs/analysis/` and `libs/theories/`. Future optimization target: Numba JIT compilation for 10x speedup.

### 2.2 The Eleven Mathematical Theories

#### Core Theories (libs/analysis/)

**T₁: Shannon Entropy** (`shannon_entropy.py`)
```
H(X) = -Σᵢ p(xᵢ) · log₂(p(xᵢ))

Purpose: Measures market predictability
Output: Entropy score ∈ [0, log₂(n)]
Signal: Low entropy → predictable market → trade signals valid
       High entropy → random market → avoid trading
```

**T₂: Hurst Exponent** (`hurst_exponent.py`)
```
H = log(R/S) / log(n/2)

Purpose: Trend persistence detection
Output: H ∈ [0, 1]
Signal: H > 0.5 → trending (momentum)
       H < 0.5 → mean-reverting (contrarian)
       H ≈ 0.5 → random walk (avoid)
```

**T₃: Markov Regime Detection** (`markov_chain.py`)
```
6-State Model: {Bull Trend, Bear Trend, Bull Consolidation,
                Bear Consolidation, Sideways, High Volatility}

Transition Matrix P:
P[i,j] = Prob(State j at t+1 | State i at t)

Purpose: Identify market regime for strategy selection
Output: Current state + transition probabilities
Signal: Different strategies per regime
```

**T₄: Kalman Filter** (`kalman_filter.py`)
```
State Update:
x̂ₖ = x̂ₖ₋₁ + Kₖ(zₖ - x̂ₖ₋₁)

Kalman Gain:
Kₖ = Pₖ₋₁ / (Pₖ₋₁ + R)

Purpose: Price denoising and trend extraction
Output: Filtered price signal
Signal: Divergence between actual and filtered → reversal
```

**T₅: Bayesian Inference** (`bayesian_inference.py`)
```
Posterior: P(θ|D) = P(D|θ) · P(θ) / P(D)

Purpose: Win rate learning and confidence estimation
Output: Updated win rate distribution
Signal: High confidence → increase position size
       Low confidence → reduce/skip trade
```

**T₆: Monte Carlo Simulation** (`monte_carlo.py`)
```
Risk Simulation (10,000 scenarios):
For each scenario i:
  1. Sample returns from historical distribution
  2. Simulate 30-day P&L trajectory
  3. Record max drawdown, final P&L

Output: VaR₉₅, CVaR₉₅, max drawdown distribution
Signal: High CVaR → reduce position size
       Low risk → normal operation
```

#### Statistical Theories (libs/theories/)

**T₇: Random Forest Validator** (`random_forest_validator.py`)
```
Ensemble of N decision trees
Prediction = Mode(tree₁(X), ..., treeₙ(X))

Purpose: Pattern validation via ensemble learning
Output: Probability of pattern validity
Signal: High RF confidence → validate other signals
       Low RF confidence → reject signals
```

**T₈: Autocorrelation Analysis** (`autocorrelation_analyzer.py`)
```
ACF(k) = Cov(Yₜ, Yₜ₋ₖ) / Var(Y)

Purpose: Time series dependencies detection
Output: Autocorrelation coefficients at various lags
Signal: Significant ACF → predictable patterns exist
       No significant ACF → random walk
```

**T₉: Stationarity Test** (`stationarity_test.py`)
```
Augmented Dickey-Fuller Test:
H₀: Unit root exists (non-stationary)
H₁: Stationary

Purpose: Mean reversion testing
Output: p-value, test statistic
Signal: Stationary → mean reversion strategies valid
       Non-stationary → trend-following strategies
```

**T₁₀: Variance Analysis** (`variance_tests.py`)
```
Levene's Test for Homogeneity of Variance:
W = (N-k)/(k-1) · Σᵢnᵢ(Z̄ᵢ - Z̄)² / Σᵢⱼ(Zᵢⱼ - Z̄ᵢ)²

Purpose: Volatility regime detection
Output: Variance stability metric
Signal: Stable variance → normal strategies
       Unstable variance → reduce size/avoid
```

**T₁₁: Market Context** (`market_context.py`)
```
CoinGecko API Integration:
- Global market cap
- BTC dominance
- Fear & Greed Index
- 24h volume trends

Purpose: Macro market sentiment
Output: Bullish/Bearish/Neutral context
Signal: Context + micro signals → final decision
```

### 2.3 DeepSeek LLM Integration

**Architecture**: Theory Results → Prompt Engineering → DeepSeek API → Signal Parsing

**Implementation**: `libs/llm/signal_generator.py`

```python
Flow:
1. Theory Analysis (11 theories) → Numerical outputs
2. SignalSynthesizer → Natural language prompt
3. DeepSeek API call → LLM reasoning text
4. SignalParser → Structured signal extraction
5. FTMO Validation → Risk-checked signal
6. Database Storage → SQLite persistence
```

**Prompt Structure**:
```
You are analyzing [SYMBOL] with the following data:

MARKET DATA:
- Current Price: $X
- 24h Change: Y%
- Volume: Z BTC

THEORY ANALYSIS:
- Shannon Entropy: [score] → [interpretation]
- Hurst Exponent: [value] → [trend/mean-revert]
- Markov Regime: [state] → [strategy]
[... 8 more theories ...]

Based on this analysis, provide:
1. Trading Direction (LONG/SHORT/HOLD)
2. Confidence (0-100)
3. Entry Price
4. Stop Loss Price
5. Take Profit Price
6. Reasoning (2-3 sentences)
```

**Signal Parsing** (`libs/llm/signal_parser.py`):
- Regex extraction of structured fields
- Validation of numeric ranges
- Error handling for malformed responses
- Fallback to HOLD if parsing fails

### 2.4 Data Architecture

**Current State** (Production):
- **Database**: SQLite (`/root/crpbot/tradingai.db`)
- **Tables**: `signals`, `signal_results`, `theory_performance`
- **Indexes**: Timestamp-based for fast queries
- **Size**: ~4,075 signals (as of 2025-11-22)

**Design Decision** (2025-11-22):
```
RDS PostgreSQL → SQLite Migration

Rationale:
- Cost: $49/month → $0/month (RDS stopped)
- Latency: Network roundtrip eliminated
- Complexity: No connection pooling needed
- Scale: 10K signals/month < SQLite limits (millions)
- Backup: Simple file-based backups to S3

Trade-offs:
✅ Zero cost, zero latency overhead
✅ Simple deployment (single file)
❌ No concurrent writes (acceptable for V7's serial writes)
❌ No replication (mitigated by S3 backups)

Decision: SQLite is optimal for current scale
Review Date: When signals > 100K or multi-writer needed
```

**Schema**:
```sql
-- Core signals table
CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(10) NOT NULL,  -- LONG/SHORT/HOLD
    confidence FLOAT NOT NULL,
    entry_price FLOAT,
    stop_loss FLOAT,
    take_profit FLOAT,
    reasoning TEXT,
    signal_variant VARCHAR(50),  -- A/B test variant
    llm_response TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_timestamp (timestamp),
    INDEX idx_symbol (symbol)
);

-- Paper trading results
CREATE TABLE signal_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL,
    entry_time DATETIME,
    exit_time DATETIME,
    outcome VARCHAR(10),  -- win/loss/open
    pnl_percent FLOAT,
    pnl_absolute FLOAT,
    exit_reason VARCHAR(50),  -- tp_hit/sl_hit/timeout
    FOREIGN KEY (signal_id) REFERENCES signals(id)
);

-- Theory performance tracking
CREATE TABLE theory_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    theory_name VARCHAR(50) NOT NULL,
    signal_contribution FLOAT,
    accuracy_24h FLOAT,
    sharpe_ratio FLOAT
);
```

### 2.5 A/B Testing Methodology

**Current Variants** (Production):
1. **v7_deepseek_only**: Pure LLM signals (69.2% avg confidence)
2. **v7_full_math**: Math-heavy signals (47.2% avg confidence)

**Implementation** (`apps/runtime/v7_runtime.py`):
```python
import random

def select_signal_variant():
    """Random 50/50 A/B split"""
    return random.choice(['v7_deepseek_only', 'v7_full_math'])

# Store variant in database for tracking
signal_data['signal_variant'] = select_signal_variant()
```

**Evaluation Metrics**:
```
Primary: Sharpe Ratio (risk-adjusted returns)
Secondary: Win Rate, Calmar Ratio, Max Drawdown
Duration: Minimum 20 trades per variant
Statistical Test: Two-sample t-test (α = 0.05)
```

**Decision Protocol**:
```
IF (trades_per_variant >= 20) AND (p_value < 0.05):
    IF variant_A.sharpe > variant_B.sharpe * 1.2:
        DEPLOY variant_A to 100%
        ARCHIVE variant_B
    ELSE:
        CONTINUE A/B testing
```

---

## III. QUANTITATIVE FINANCE ENHANCEMENTS

### 3.1 Portfolio Optimization (Markowitz Mean-Variance)

**Implementation**: `libs/portfolio/optimizer.py` (780 lines)

**Mathematical Foundation**:
```
Objective: Maximize Sharpe Ratio
SR = (Rₚ - Rբ) / σₚ

Where:
  Rₚ = Portfolio return = Σᵢ wᵢ · μᵢ
  Rբ = Risk-free rate (5% annual)
  σₚ = Portfolio volatility = √(w^T Σ w)
  w = Weight vector (Σᵢ wᵢ = 1)
  μ = Expected returns vector
  Σ = Covariance matrix

Constraints:
  - Σᵢ wᵢ = 1 (full investment)
  - 0 ≤ wᵢ ≤ 0.3 (max 30% per asset)
  - wᵢ ≥ 0.05 or wᵢ = 0 (min 5% or nothing)
```

**Methods**:
1. **Max Sharpe**: Maximize risk-adjusted returns
2. **Min Volatility**: Minimize portfolio risk
3. **Efficient Frontier**: Trade-off curve (100 points)
4. **Risk Parity**: Equal risk contribution

**Integration**: Runs every 10 iterations in V7 runtime to suggest optimal capital allocation across 10 symbols.

### 3.2 Advanced Risk Metrics

#### Calmar Ratio (`libs/risk/calmar_ratio_tracker.py`, 460 lines)

```
Calmar Ratio = Annualized Return / |Maximum Drawdown|

Windows: 30d, 90d, 180d, 365d
Interpretation:
  > 3.0: Excellent (consistent returns, low drawdown)
  2-3: Very Good
  1-2: Good
  < 1.0: Poor (high drawdown relative to returns)

Preferred by: CTAs, hedge funds
Advantage: Directly measures worst-case risk
```

#### Omega Ratio (`libs/risk/omega_ratio_calculator.py`, 505 lines)

```
Omega(τ) = ∫τ^∞ [1 - F(r)]dr / ∫₋∞^τ F(r)dr

Simplified:
Omega = Prob-weighted gains / Prob-weighted losses

Thresholds: 0%, 2%, 5% (risk-free rate)
Interpretation:
  > 1.5: Strong (more upside than downside)
  1.0-1.5: Balanced
  < 1.0: Weak (more downside than upside)

Advantage: Considers full return distribution
          (skewness, kurtosis, fat tails)
```

### 3.3 Backtesting Infrastructure

#### Vectorized Backtest (`libs/backtesting/vectorized_backtest.py`, 630 lines)

**Architecture**: NumPy-based vectorization (1000x faster than loop-based)

**Cost Model**:
```
Total Costs = Transaction Costs + Slippage

Transaction Costs: 0.1% per trade (Coinbase Pro maker fee)
Slippage: 0.05% per trade (market impact)

Per-Trade Cost = 0.15% of position size
```

**Metrics Computed**:
```
Returns:
- Total Return
- Annualized Return
- CAGR (Compound Annual Growth Rate)

Risk:
- Volatility (annualized)
- Max Drawdown
- Max Drawdown Duration
- Sharpe Ratio
- Sortino Ratio (downside deviation)
- Calmar Ratio
- Omega Ratio

Trade Statistics:
- Total Trades
- Win Rate
- Average Win / Average Loss
- Profit Factor (gross profit / gross loss)
- Expectancy (expected value per trade)

Risk Metrics:
- VaR₉₅ (Value at Risk, 95th percentile)
- CVaR₉₅ (Conditional VaR, expected shortfall)
```

**Usage**:
```python
from libs.backtesting.vectorized_backtest import VectorizedBacktest, BacktestConfig

config = BacktestConfig(
    initial_capital=10000.0,
    transaction_cost=0.001,  # 0.1%
    slippage=0.0005          # 0.05%
)

backtest = VectorizedBacktest(config=config)
result = backtest.run(signals_df, prices_df)

print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.1%}")
print(f"Win Rate: {result.win_rate:.1%}")
```

#### Walk-Forward Optimization (`libs/backtesting/walk_forward.py`, 642 lines)

**Architecture**: Industry-standard validation to prevent overfitting

**Parameters**:
```
Train Window: 180 days (6 months)
Test Window: 30 days (1 month)
Mode: Rolling (non-anchored) or Anchored (expanding)

Total Duration: 2 years (730 days)
Number of Windows: ~24 (rolling) or ~24 (anchored)
```

**Walk-Forward Efficiency (WFE)**:
```
WFE = OOS Performance / IS Performance

Where:
  OOS = Out-of-Sample (test window) Sharpe Ratio
  IS = In-Sample (train window) Sharpe Ratio

Interpretation:
  WFE > 0.8: Excellent (strategy generalizes well)
  WFE 0.6-0.8: Good (acceptable generalization)
  WFE < 0.6: Weak (possible overfitting)
```

**Decision Protocol**:
```
IF WFE < 0.6:
    WARNING: Possible overfitting detected
    ACTION: Simplify strategy, reduce parameters

IF OOS Sharpe < 1.0:
    WARNING: Poor out-of-sample performance
    ACTION: Re-evaluate strategy, collect more data

IF OOS Max Drawdown > 20%:
    WARNING: Excessive risk
    ACTION: Reduce position sizing, tighten stops
```

### 3.4 Feature Engineering (35+ Technical Indicators)

**Implementation**: `libs/features/technical_indicators.py` (800+ lines)

**Indicator Categories**:

**Momentum (10 features)**:
```
1. RSI (14-period): Relative Strength Index
2. RSI (28-period): Longer-term momentum
3. MACD: Moving Average Convergence Divergence
4. MACD Signal: 9-period EMA of MACD
5. MACD Histogram: MACD - Signal
6. Stochastic %K: Fast oscillator
7. Stochastic %D: Slow oscillator (3-period SMA of %K)
8. Williams %R: Momentum oscillator
9. ROC (12-period): Rate of Change
10. CMO (14-period): Chande Momentum Oscillator
```

**Volatility (11 features)**:
```
11. ATR (14-period): Average True Range
12. Bollinger Upper Band (20-period, 2σ)
13. Bollinger Middle Band (20-period SMA)
14. Bollinger Lower Band (20-period, 2σ)
15. Bollinger Width: (Upper - Lower) / Middle
16. Keltner Upper Channel (20-period EMA + 2×ATR)
17. Keltner Middle Line (20-period EMA)
18. Keltner Lower Channel (20-period EMA - 2×ATR)
19. Donchian Upper Channel (20-period high)
20. Donchian Middle Line (average of upper and lower)
21. Donchian Lower Channel (20-period low)
```

**Trend (6 features)**:
```
22. ADX: Average Directional Index (trend strength)
23. +DI: Positive Directional Indicator
24. -DI: Negative Directional Indicator
25. Supertrend: Trend-following indicator (ATR-based)
26. Supertrend Direction: +1 (bullish) or -1 (bearish)
27. TRIX: Triple Exponential Average rate of change
```

**Volume (5 features)**:
```
28. OBV: On-Balance Volume (cumulative volume flow)
29. VWAP: Volume Weighted Average Price
30. MFI: Money Flow Index (volume-weighted RSI)
31. A/D Line: Accumulation/Distribution Line
32. CMF: Chaikin Money Flow (20-period)
```

**Statistical (3 features)**:
```
33. Z-Score (20-period): Standard deviations from mean
34. Percentile Rank (100-period): Current price percentile
35. Linear Regression Slope (20-period): Trend direction
```

**Performance**:
- Computation time: 2-3 seconds for 17,515 rows (2 years hourly)
- Memory efficient: Vectorized pandas/numpy operations
- Output: 5.68 MB parquet file (41 columns total)

### 3.5 Historical Data Collection

**Implementation**: `libs/data/historical_data_collector.py` + `scripts/collect_historical_data.py`

**Architecture**:
```
Coinbase API → Batched Requests (300 candles) → Parquet Storage

Granularity Conversion:
  Integer seconds → Coinbase enum
  3600 → "ONE_HOUR"
  60 → "ONE_MINUTE"
  86400 → "ONE_DAY"

Batching Logic:
  Max per request: 300 candles (safe limit)
  Coinbase limit: 350 candles (avoid edge cases)
  Rate limiting: 0.5s delay between symbols
```

**Data Collected** (2025-11-24):
```
Symbols: 10 (BTC, ETH, SOL, XRP, DOGE, ADA, AVAX, LINK, POL, LTC)
Duration: 730 days (2 years)
Granularity: 1 hour (3600s)
Total Candles: 17,515 per symbol (POL: 10,702 - newer coin)
Storage: ~6 MB parquet (vs ~60 MB CSV)
Format: Snappy-compressed parquet
```

**Validation**:
- Gap detection: <1% gaps (exchange downtime only)
- Null values: 0% nulls in raw OHLCV data
- Price sanity: High ≥ Low, Close ∈ [Low, High]
- Date continuity: 1-hour intervals (except gaps)

---

## IV. FTMO RISK PROTOCOL (IMMUTABLE CONSTRAINTS)

### 4.1 Risk Limits (Circuit Breaker)

**Implementation**: `apps/runtime/ftmo_rules.py`

**Hard Limits**:
```
Daily Loss Limit: 4.5% of account equity
  - Enforced BEFORE each signal generation
  - Calculation: (Current Equity - Start of Day Equity) / Start of Day Equity
  - Action if breached: HALT all trading for 24 hours

Total Loss Limit: 9% of account equity (lifetime)
  - Enforced BEFORE each signal generation
  - Calculation: (Current Equity - Initial Equity) / Initial Equity
  - Action if breached: TERMINATE trading permanently, liquidate all

Position Sizing: 1-2% risk per trade
  - Risk = Entry Price - Stop Loss Price
  - Position Size = (Account Equity × Risk%) / Risk
  - Max Position: 30% of account equity
```

**Circuit Breaker Function**:
```python
def check_risk_breach(
    current_equity: float,
    start_of_day_equity: float,
    initial_equity: float
) -> tuple[bool, str]:
    """
    Returns: (is_breach, reason)

    Execution: <200ms guarantee
    Logging: Immutable audit trail to SQLite + S3
    """
    # Daily loss check
    daily_loss_pct = (current_equity - start_of_day_equity) / start_of_day_equity
    if daily_loss_pct < -0.045:
        return (True, "DAILY_LOSS_BREACH")

    # Total loss check
    total_loss_pct = (current_equity - initial_equity) / initial_equity
    if total_loss_pct < -0.09:
        return (True, "TOTAL_LOSS_BREACH")

    return (False, "OK")
```

**Emergency Halt Protocol**:
```
Priority 1 Execution Path (200ms target):

1. Log breach event (immutable timestamp)
2. Set KILL_SWITCH=true in environment
3. Cancel all pending orders (API calls in parallel)
4. Liquidate all open positions (market orders)
5. Send Telegram alert to admin
6. Write incident report to S3 Glacier
7. Halt V7 runtime process

Manual Recovery Required:
  - Human review of breach cause
  - Risk parameter adjustment if needed
  - KILL_SWITCH=false to resume
```

### 4.2 Kill Switch

**Implementation**: Environment variable + runtime check

```bash
# In .env file
KILL_SWITCH=false  # Production default

# To emergency halt:
export KILL_SWITCH=true

# Runtime checks every iteration:
if os.getenv('KILL_SWITCH') == 'true':
    logger.critical("🛑 KILL SWITCH ACTIVATED - HALTING")
    sys.exit(1)
```

### 4.3 Position Sizing

**Implementation**: `apps/runtime/v7_runtime.py`

```python
def calculate_position_size(
    account_equity: float,
    entry_price: float,
    stop_loss_price: float,
    risk_percent: float = 0.02  # 2% default
) -> float:
    """
    Kelly Criterion-inspired position sizing

    Risk% = Maximum loss per trade as % of equity
    Position Size = (Equity × Risk%) / Risk per Unit
    """
    risk_per_unit = abs(entry_price - stop_loss_price)
    dollar_risk = account_equity * risk_percent
    position_size = dollar_risk / risk_per_unit

    # Cap at 30% of equity
    max_position = account_equity * 0.30 / entry_price
    position_size = min(position_size, max_position)

    return position_size
```

### 4.4 Confidence Thresholds

**Implementation**: Adaptive thresholds based on market regime

```python
CONFIDENCE_THRESHOLDS = {
    'bull_trend': 0.60,      # Lower threshold in strong trends
    'bear_trend': 0.60,
    'bull_consolidation': 0.70,  # Higher threshold in consolidation
    'bear_consolidation': 0.70,
    'sideways': 0.75,        # Highest threshold in sideways
    'high_volatility': 0.80  # Avoid trading in chaos
}

def should_execute_signal(confidence: float, regime: str) -> bool:
    threshold = CONFIDENCE_THRESHOLDS.get(regime, 0.65)
    return confidence >= threshold
```

---

## V. OPERATIONAL PROCEDURES & PLAYBOOKS

### 5.1 System Startup Checklist

**New Claude Instance (Complete Onboarding)**:

```bash
# Step 1: Verify Environment
pwd  # Should be /root/crpbot
whoami  # Should be root
git branch  # Should be feature/v7-ultimate
git status  # Should be clean

# Step 2: Verify Dependencies
.venv/bin/python3 --version  # Python 3.11+
.venv/bin/python3 -c "import numpy, pandas, coinbase; print('OK')"

# Step 3: Verify Configuration
cat .env | grep -E "COINBASE|DEEPSEEK|DB_URL|KILL_SWITCH"
# Should show valid API keys, DB_URL=sqlite:///tradingai.db, KILL_SWITCH=false

# Step 4: Verify Database
sqlite3 tradingai.db "SELECT COUNT(*) FROM signals;"
sqlite3 tradingai.db "SELECT COUNT(*) FROM signal_results;"
# Should return counts > 0

# Step 5: Verify V7 Runtime Status
ps aux | grep v7_runtime | grep -v grep
# Should show 1 process running

# Step 6: Verify Recent Activity
tail -50 /tmp/v7_runtime_*.log | grep -E "Signal|Trade|Error"
# Should show recent signal generation activity

# Step 7: Verify Dashboard
curl -s http://localhost:3000 | grep -q "V7" && echo "Dashboard OK" || echo "Dashboard DOWN"

# Step 8: Read Critical Documentation
cat MEGAPROMPT_V7_ULTIMATE.md  # This file
cat CLAUDE.md                   # Project instructions
cat CURRENT_STATUS_AND_NEXT_ACTIONS.md  # Current phase
```

### 5.2 V7 Runtime Crash Recovery

**Symptoms**:
- No signals generated for >30 minutes
- Process not visible in `ps aux`
- No recent log files

**Diagnosis**:
```bash
# 1. Check if process is running
ps aux | grep v7_runtime | grep -v grep
# Expected: 1 process

# 2. Check last log file
ls -lt /tmp/v7_runtime_*.log | head -1
tail -100 /tmp/v7_runtime_*.log | tail -50
# Look for: ERROR, CRITICAL, Traceback

# 3. Check system resources
free -h       # Memory (should have >500MB free)
df -h         # Disk (should have >1GB free)
uptime        # Load average (should be <2.0)

# 4. Check API connectivity
curl -s https://api.coinbase.com/v2/time && echo "Coinbase OK"
curl -s https://api.deepseek.com/v1/health && echo "DeepSeek OK"
```

**Recovery Procedure**:
```bash
# 1. Kill any zombie processes
pkill -9 -f v7_runtime.py

# 2. Check for KILL_SWITCH
grep KILL_SWITCH .env
# If true, investigate why before restarting

# 3. Verify database integrity
sqlite3 tradingai.db "PRAGMA integrity_check;"
# Should return: ok

# 4. Backup database
cp tradingai.db tradingai.db.backup.$(date +%Y%m%d_%H%M)

# 5. Restart V7 Runtime
nohup .venv/bin/python3 apps/runtime/v7_runtime.py \
  --iterations -1 \
  --sleep-seconds 300 \
  --max-signals-per-hour 3 \
  > /tmp/v7_runtime_$(date +%Y%m%d_%H%M).log 2>&1 &

# 6. Verify startup
sleep 10
ps aux | grep v7_runtime | grep -v grep
tail -20 /tmp/v7_runtime_*.log
# Should show: "V7 Runtime initialized"

# 7. Monitor for 5 minutes
for i in {1..5}; do
  echo "Check $i/5..."
  tail -10 /tmp/v7_runtime_*.log | grep -E "Signal|Error"
  sleep 60
done

# 8. If successful, document in DECISION_LOG.md
```

### 5.3 Signal Generation Modification Protocol

**Scenario**: Need to modify theory weights or LLM prompt

**Mandatory Pre-Modification Checklist**:
```
[ ] Current system performance documented (Sharpe, Win Rate, Drawdown)
[ ] Hypothesis for modification clearly stated
[ ] Expected impact quantified (e.g., "+5% win rate")
[ ] Rollback plan defined
[ ] A/B test variant created (not direct modification)
[ ] Minimum sample size defined (e.g., 20 trades)
[ ] Statistical test chosen (t-test, Mann-Whitney, etc.)
```

**Modification Procedure**:
```bash
# 1. Create new git branch
git checkout -b experiment/[hypothesis-name]

# 2. Document hypothesis in DECISION_LOG.md
echo "## Experiment: [Name]
Date: $(date)
Hypothesis: [Clear statement]
Expected Impact: [Quantified prediction]
Modification: [Specific code changes]
" >> DECISION_LOG.md

# 3. Implement as NEW variant (don't modify existing)
# Example: Create v7_theory_weighted variant
cp apps/runtime/v7_runtime.py apps/runtime/v7_theory_weighted.py
# Make modifications to v7_theory_weighted.py only

# 4. Update A/B test configuration
# Add new variant to signal variant selection

# 5. Deploy to paper trading
# V7 will automatically A/B test the new variant

# 6. Monitor for minimum sample size (20 trades minimum)
sqlite3 tradingai.db "
SELECT signal_variant, COUNT(*), AVG(confidence)
FROM signals
WHERE timestamp > datetime('now', '-7 days')
GROUP BY signal_variant;"

# 7. Statistical analysis (when sample size reached)
python scripts/analyze_ab_test.py --variant1 v7_full_math --variant2 v7_theory_weighted

# 8. Decision
# IF p-value < 0.05 AND new_sharpe > old_sharpe * 1.2:
#   - Merge to main
#   - Deploy to 100%
#   - Archive old variant
# ELSE:
#   - Reject hypothesis
#   - Document learnings
#   - Revert branch
```

### 5.4 AWS Model Training Workflow

**CRITICAL**: NEVER train locally. ALWAYS use AWS g4dn.xlarge GPU instance.

**Pre-Training Checklist**:
```
[ ] Feature engineering complete (all 10 symbols)
[ ] Features uploaded to S3
[ ] Training data validated (no nulls, correct date range)
[ ] GPU instance terminated after last training (cost control)
[ ] S3 storage < 50GB (cost control)
```

**Training Procedure**:
```bash
# LOCAL: Engineer features
python scripts/engineer_features.py --symbols all --days 730
# Output: data/features/[SYMBOL]_features.parquet for all 10 symbols

# LOCAL: Upload to S3
aws s3 sync data/features/ s3://crpbot-ml-data-20251110/features/

# LOCAL: Launch GPU instance (spot for cost savings)
aws ec2 run-instances \
  --instance-type g4dn.xlarge \
  --spot-instance-requests \
  --image-id ami-[ubuntu-gpu] \
  --key-name crpbot-gpu \
  --security-group-ids sg-[allow-ssh] \
  --user-data-file scripts/gpu_instance_setup.sh

# Wait for instance to be ready (~3 minutes)
INSTANCE_ID=$(aws ec2 describe-instances --filters "Name=instance-type,Values=g4dn.xlarge" --query "Reservations[0].Instances[0].InstanceId" --output text)
INSTANCE_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

# SSH into GPU instance
ssh -i ~/.ssh/crpbot-gpu.pem ubuntu@$INSTANCE_IP

# GPU: Download features from S3
aws s3 sync s3://crpbot-ml-data-20251110/features/ data/features/

# GPU: Train models (parallelized)
uv run python apps/trainer/main.py --task lstm --coin BTC --epochs 15 &
uv run python apps/trainer/main.py --task lstm --coin ETH --epochs 15 &
uv run python apps/trainer/main.py --task lstm --coin SOL --epochs 15 &
wait

# GPU: Upload trained models to S3
aws s3 sync models/ s3://crpbot-ml-data-20251110/models/

# GPU: Generate Model Lineage Manifest
python scripts/generate_model_manifest.py --output models/manifest_$(date +%Y%m%d).json
aws s3 cp models/manifest_*.json s3://crpbot-ml-data-20251110/manifests/

# GPU: Exit and terminate instance (CRITICAL - cost control)
exit
aws ec2 terminate-instances --instance-ids $INSTANCE_ID

# LOCAL: Download trained models
aws s3 sync s3://crpbot-ml-data-20251110/models/ models/promoted/

# LOCAL: Validate models
python scripts/validate_trained_models.py --models-dir models/promoted/
# Should output: "All models validated successfully"

# LOCAL: Deploy to paper trading
# Update V7 runtime to load new models
# Monitor for 7 days before live deployment
```

**Cost Tracking**:
```
GPU Training: ~$0.16/run (spot) or $0.53/run (on-demand)
Duration: 10-15 minutes per model
Total: 10 models × $0.16 = $1.60 per training run

Budget: $150/month
Allows: ~90 training runs/month (excessive - aim for 4-8/month)
```

### 5.5 Performance Analysis & Enhancement Decision

**Trigger**: Monday review (every 7 days) or after 20+ paper trades

**Analysis Procedure**:
```bash
# 1. Generate performance report
python scripts/generate_performance_report.py --days 7 > reports/performance_$(date +%Y%m%d).md

# 2. Key metrics to review
sqlite3 tradingai.db "
SELECT
  COUNT(*) as total_trades,
  SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
  AVG(pnl_percent) as avg_pnl,
  MAX(pnl_percent) as max_win,
  MIN(pnl_percent) as max_loss
FROM signal_results
WHERE exit_time > datetime('now', '-7 days');"

# 3. Calculate Sharpe Ratio
python scripts/calculate_sharpe.py --days 7
# Output: Sharpe Ratio = X.XX

# 4. Decision Matrix
# IF Sharpe < 1.0:
#   ACTION: Implement Phase 1 enhancements (see QUANT_FINANCE_10_HOUR_PLAN.md)
#   RATIONALE: Sharpe < 1.0 indicates insufficient risk-adjusted returns
#
# ELIF Sharpe 1.0-1.5:
#   ACTION: Monitor for 1 more week
#   RATIONALE: Adequate performance but room for improvement
#
# ELIF Sharpe > 1.5:
#   ACTION: Continue as-is, consider Phase 2 enhancements
#   RATIONALE: Strong performance, cautious optimization

# 5. Document decision
echo "## Performance Review $(date)
Sharpe Ratio: [X.XX]
Win Rate: [X.X%]
Decision: [Action taken]
Rationale: [Reasoning]
" >> DECISION_LOG.md

# 6. If Phase 1 deployment needed:
# See QUANT_FINANCE_10_HOUR_PLAN.md for implementation steps
```

### 5.6 Database Migration (SQLite → PostgreSQL RDS)

**Trigger**: Signals > 100,000 OR need concurrent writes OR need replication

**Migration Procedure**:
```bash
# 1. Backup current SQLite database
cp tradingai.db tradingai.db.migration_backup_$(date +%Y%m%d)
aws s3 cp tradingai.db s3://crpbot-backups/pre-migration/

# 2. Provision RDS instance
aws rds create-db-instance \
  --db-instance-identifier crpbot-rds-postgres \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --engine-version 15.3 \
  --master-username crpbot \
  --master-user-password [secure-password] \
  --allocated-storage 20 \
  --storage-type gp2 \
  --backup-retention-period 7 \
  --enable-iam-database-authentication

# 3. Wait for instance to be available (~10 minutes)
aws rds wait db-instance-available --db-instance-identifier crpbot-rds-postgres

# 4. Export SQLite to SQL dump
sqlite3 tradingai.db .dump > tradingai_export.sql

# 5. Convert SQLite SQL to PostgreSQL SQL
# (Note: May need manual adjustments for data types)
sed -i 's/AUTOINCREMENT/SERIAL/g' tradingai_export.sql
sed -i 's/DATETIME/TIMESTAMP/g' tradingai_export.sql

# 6. Import to PostgreSQL
psql -h [rds-endpoint] -U crpbot -d crpbot < tradingai_export.sql

# 7. Verify data integrity
psql -h [rds-endpoint] -U crpbot -d crpbot -c "SELECT COUNT(*) FROM signals;"
# Should match SQLite count

# 8. Update .env configuration
echo "DB_URL=postgresql://crpbot:[password]@[rds-endpoint]:5432/crpbot" >> .env

# 9. Test V7 runtime with new database
# Run in test mode first (--dry-run)

# 10. Switch to production
# Update V7 runtime to use PostgreSQL
# Monitor for 24 hours

# 11. Archive SQLite database
mv tradingai.db tradingai.db.archived_$(date +%Y%m%d)
aws s3 cp tradingai.db.archived_* s3://crpbot-archives/
```

**Cost Impact**:
```
RDS db.t3.micro: $14/month (smallest instance)
vs
SQLite: $0/month

Additional benefits of RDS:
- Automated backups
- Point-in-time recovery
- Read replicas (for analytics)
- Multi-AZ (high availability)

Decision: Only migrate when scale demands it
```

### 5.7 Cost Overrun Response

**Trigger**: AWS bill > $100 OR DeepSeek API usage > $4/day

**Immediate Actions**:
```bash
# 1. Check current spend
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '1 month ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity MONTHLY \
  --metrics BlendedCost

# 2. Identify cost drivers
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '7 days ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics BlendedCost \
  --group-by Type=SERVICE

# 3. Common culprits:
# - RDS instance left running ($35-50/month)
# - GPU instance not terminated ($10-20/hour)
# - S3 storage growing uncontrolled ($5-10/month per 100GB)
# - DeepSeek API rate limiting failures ($0.50/retry)

# 4. Emergency cost reduction:
# Stop RDS (if not actively using):
aws rds stop-db-instance --db-instance-identifier crpbot-rds-postgres-db

# Terminate GPU instances:
aws ec2 describe-instances --filters "Name=instance-type,Values=g4dn.xlarge" --query "Reservations[].Instances[].InstanceId" --output text | xargs -r aws ec2 terminate-instances --instance-ids

# Clean up old S3 data:
aws s3 rm s3://crpbot-ml-data-20251110/old_backups/ --recursive

# Reduce DeepSeek API calls:
# Update V7 runtime: --max-signals-per-hour 2 (from 3)

# 5. Document in DECISION_LOG.md
echo "## Cost Overrun Response $(date)
Trigger: AWS bill exceeded $100
Actions Taken:
  - RDS stopped (save $35/month)
  - GPU instances terminated
  - S3 cleanup (save $5/month)
  - DeepSeek rate reduced to 2/hour (save $20/month)
New Expected Cost: $79/month
" >> DECISION_LOG.md
```

---

## VI. QUANTITATIVE SPECIFICATIONS

### 6.1 Signal Quality Metrics

**Primary Metric**: Sharpe Ratio
```
SR = (R̄ - Rf) / σ

Where:
  R̄ = Mean return
  Rf = Risk-free rate (5% annual = 0.0137% daily)
  σ = Standard deviation of returns

Target: SR > 1.5 (industry standard for quantitative strategies)
Current: SR = [To be measured after 20+ trades]

Interpretation:
  SR > 2.0: Excellent (institutional quality)
  SR 1.5-2.0: Very Good (acceptable for live trading)
  SR 1.0-1.5: Good (paper trading only)
  SR < 1.0: Poor (needs improvement)
```

**Secondary Metrics**:
```
Win Rate: Target > 55% (current: 53.8%)
Profit Factor: Target > 1.5 (gross profit / gross loss)
Calmar Ratio: Target > 2.0 (return / max drawdown)
Omega Ratio (0%): Target > 1.3
Max Drawdown: Target < 15%
```

### 6.2 Latency Targets

**Signal Generation Pipeline**:
```
Step 1: Data Fetch (Coinbase API)
  Target: < 500ms (P99)
  Current: ~200ms average

Step 2: Feature Engineering (35 indicators)
  Target: < 100ms (P99)
  Current: ~50ms (vectorized)

Step 3: Theory Analysis (11 theories)
  Target: < 1000ms (P99)
  Current: ~500ms average

Step 4: LLM API Call (DeepSeek)
  Target: < 3000ms (P99)
  Current: ~2000ms average

Step 5: Signal Parsing & Validation
  Target: < 100ms (P99)
  Current: ~50ms

Step 6: Database Write
  Target: < 50ms (P99)
  Current: ~20ms (SQLite)

Total End-to-End: Target < 5 seconds (P99)
                  Current: ~3 seconds average

Risk Check (Circuit Breaker): Target < 200ms (P99)
                               Current: ~50ms
```

### 6.3 Data Quality Standards

**Market Data** (Coinbase):
```
Freshness: < 1 minute latency (real-time REST API)
Completeness: > 99% uptime (historical: 99.9%)
Accuracy: Exchange-verified prices (no manual adjustments)
Resolution: 1-minute OHLCV minimum, 1-hour for backtesting
```

**Feature Data**:
```
Null Values: < 0.5% (only in indicator warmup periods)
Outliers: Winsorized at 3σ (optional, currently disabled)
Normalization: Z-score or Min-Max per feature
Leakage Check: No future data in features (validated)
```

**Signal Data**:
```
Completeness: 100% (every signal must have all fields)
Validation: Regex + range checks on all numeric fields
Auditability: Full LLM response logged for analysis
Immutability: Signals never modified after creation
```

### 6.4 Model Performance Standards

**Backtest Requirements** (Before Live Deployment):
```
Duration: Minimum 2 years historical data
Granularity: 1-hour candles (minimum)
Universe: All 10 symbols (not cherry-picked)
Methodology: Walk-forward optimization (required)

Metrics to Report:
  - In-Sample Sharpe Ratio
  - Out-of-Sample Sharpe Ratio
  - Walk-Forward Efficiency (WFE)
  - Maximum Drawdown (IS and OOS)
  - Win Rate (IS and OOS)
  - Profit Factor (IS and OOS)

Pass Criteria:
  - OOS Sharpe > 1.0 (minimum)
  - WFE > 0.6 (generalization test)
  - OOS Max Drawdown < 20%
  - OOS Win Rate > 50%
```

**Paper Trading Requirements** (Before Live Deployment):
```
Duration: Minimum 7 days
Trades: Minimum 20 trades
Conditions: Real-time market data (no simulation)

Pass Criteria:
  - Win Rate > 50%
  - Max Drawdown < 10%
  - No FTMO rule violations
  - Sharpe Ratio > 1.0 (estimated from daily returns)
```

---

## VII. SYSTEM DOCUMENTATION & VERSION CONTROL

### 7.1 Critical Files & Locations

**Production Runtime**:
```
/root/crpbot/
├── apps/runtime/
│   ├── v7_runtime.py              # Main V7 orchestrator (33KB, 1200+ lines)
│   ├── ftmo_rules.py              # Risk management (immutable)
│   └── data_fetcher.py            # Coinbase data client
├── libs/
│   ├── llm/                       # DeepSeek integration (4 files)
│   │   ├── signal_generator.py   # Main orchestrator
│   │   ├── deepseek_client.py    # API client
│   │   ├── signal_synthesizer.py # Theory → Prompt
│   │   └── signal_parser.py      # LLM → Signal
│   ├── analysis/                  # Core 6 theories
│   │   ├── shannon_entropy.py
│   │   ├── hurst_exponent.py
│   │   ├── markov_chain.py
│   │   ├── kalman_filter.py
│   │   ├── bayesian_inference.py
│   │   └── monte_carlo.py
│   ├── theories/                  # Statistical 4 theories + context
│   │   ├── random_forest_validator.py
│   │   ├── autocorrelation_analyzer.py
│   │   ├── stationarity_test.py
│   │   ├── variance_tests.py
│   │   └── market_context.py
│   ├── tracking/                  # Performance tracking
│   │   ├── performance_tracker.py
│   │   └── paper_trader.py
│   ├── backtesting/               # Backtesting infrastructure
│   │   ├── vectorized_backtest.py
│   │   └── walk_forward.py
│   ├── features/                  # Technical indicators (NEW)
│   │   └── technical_indicators.py
│   ├── portfolio/                 # Portfolio optimization
│   │   └── optimizer.py
│   └── risk/                      # Advanced risk metrics
│       ├── calmar_ratio_tracker.py
│       └── omega_ratio_calculator.py
├── data/
│   ├── historical/                # 2 years hourly OHLCV (parquet)
│   └── features/                  # Engineered features (parquet)
├── models/
│   └── promoted/                  # Production ML models (.pt files)
├── tradingai.db                   # SQLite database (primary)
├── .env                           # Configuration (API keys, secrets)
└── logs/                          # Runtime logs
    └── /tmp/v7_runtime_*.log
```

**Documentation**:
```
/root/crpbot/
├── MEGAPROMPT_V7_ULTIMATE.md     # This file (formal specification)
├── CLAUDE.md                      # Project instructions (operational guide)
├── CURRENT_STATUS_AND_NEXT_ACTIONS.md  # Current phase tracking
├── DECISION_LOG.md                # Architectural decisions log
├── QUANT_FINANCE_10_HOUR_PLAN.md # Phase 1 enhancements plan
├── QUANT_FINANCE_PHASE_2_PLAN.md # Phase 2 advanced features
├── MASTER_TRAINING_WORKFLOW.md   # AWS GPU training procedures
├── DATABASE_VERIFICATION_2025-11-22.md
├── AWS_COST_CLEANUP_2025-11-22.md
└── README.md                      # Project overview
```

### 7.2 Git Workflow

**Branching Strategy**:
```
main                   # Production-ready code (protected)
├── feature/v7-ultimate       # Current development branch
├── experiment/*              # A/B test experiments
└── hotfix/*                  # Emergency fixes

Current Branch: feature/v7-ultimate
```

**Commit Standards**:
```
Format: <type>(<scope>): <subject>

Types:
  feat: New feature
  fix: Bug fix
  docs: Documentation only
  refactor: Code restructuring (no behavior change)
  perf: Performance improvement
  test: Adding tests
  chore: Maintenance

Examples:
  feat(llm): add DeepSeek API retry logic with exponential backoff
  fix(ftmo): correct daily loss calculation for multi-day positions
  docs: update MEGAPROMPT with new backtesting specifications
  perf(theories): vectorize Shannon entropy calculation (10x speedup)
```

**Co-Authorship** (Mandatory):
```
Every commit must include:

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### 7.3 Decision Log Format

**Template** (`DECISION_LOG.md`):
```markdown
## Decision: [Title]

**Date**: YYYY-MM-DD
**Status**: Proposed | Approved | Implemented | Rejected | Superseded
**Decision Maker**: [Claude Instance ID or Human Name]
**Context**: [What situation led to this decision?]

### Problem Statement
[Clear description of the problem requiring a decision]

### Alternatives Considered
1. **Option A**: [Description]
   - Pros: [List]
   - Cons: [List]
   - Cost: [Quantified]

2. **Option B**: [Description]
   - Pros: [List]
   - Cons: [List]
   - Cost: [Quantified]

### Decision
[What was decided and why]

### Rationale
[Scientific/engineering reasoning supporting this decision]

### Supporting Data
[Backtest results, benchmark comparisons, cost analysis, etc.]

### Implementation
[How to implement this decision]

### Success Criteria
[How to measure if this decision was correct]

### Rollback Plan
[How to revert if this decision proves incorrect]

### Related Decisions
[Links to related decisions in this log]

---
```

**Example**:
```markdown
## Decision: Migrate from RDS to SQLite

**Date**: 2025-11-22
**Status**: Implemented
**Decision Maker**: Builder Claude (Session 2025-11-22)
**Context**: AWS costs reaching $140/month, with RDS contributing $49/month for rarely-used database.

### Problem Statement
RDS PostgreSQL costs $49/month but provides minimal benefit for V7's current scale (4,075 signals, serial writes only). Need to reduce costs without impacting performance.

### Alternatives Considered
1. **Keep RDS**
   - Pros: Scalable, managed backups, replication
   - Cons: $49/month cost, network latency, connection pooling complexity
   - Cost: $49/month

2. **Migrate to SQLite**
   - Pros: Zero cost, zero latency, simple deployment
   - Cons: No concurrent writes, no replication, single file
   - Cost: $0/month

3. **Migrate to DynamoDB**
   - Pros: Serverless, auto-scaling
   - Cons: Complex queries difficult, $10-20/month, latency
   - Cost: $10-20/month

### Decision
Migrate to SQLite for cost savings and latency improvement.

### Rationale
- V7 generates 1-3 signals/hour (serial writes) - no concurrency needed
- 4,075 signals = 0.5 MB SQLite file (well below SQLite's multi-GB limits)
- Network latency eliminated (50-100ms savings per query)
- File-based backups simpler than RDS snapshots

### Supporting Data
- Current RDS usage: <1% CPU, <10% connections
- SQLite benchmarks: 50K writes/second (V7 needs <1 write/second)
- RDS cost: $49/month → $0/month savings

### Implementation
1. Export RDS to SQL dump
2. Import to SQLite
3. Update .env: DB_URL=sqlite:///tradingai.db
4. Test V7 runtime
5. Stop RDS instance

### Success Criteria
- V7 runtime operates without errors for 7 days
- Query latency < 10ms (vs 50-100ms with RDS)
- No data loss or corruption

### Rollback Plan
- Restore RDS from snapshot
- Update .env to PostgreSQL URL
- Restart V7 runtime

### Related Decisions
- [Cost Optimization 2025-11-22]
- [AWS Resource Cleanup]

---
```

### 7.4 Model Lineage Manifest

**Purpose**: Track exact conditions of model training for reproducibility and auditability.

**Format** (JSON):
```json
{
  "manifest_version": "1.0",
  "created_at": "2025-11-25T12:00:00Z",
  "models": [
    {
      "symbol": "BTC-USD",
      "model_type": "LSTM",
      "model_file": "btc_lstm_v7_20251125.pt",
      "training": {
        "git_commit": "393ad84",
        "training_script": "apps/trainer/main.py",
        "feature_script": "scripts/engineer_features.py",
        "feature_version": "1.2.0",
        "data_partition": {
          "train_start": "2023-11-25",
          "train_end": "2025-09-25",
          "val_start": "2025-09-26",
          "val_end": "2025-10-25",
          "test_start": "2025-10-26",
          "test_end": "2025-11-24"
        },
        "hyperparameters": {
          "epochs": 15,
          "batch_size": 32,
          "learning_rate": 0.001,
          "hidden_size": 128,
          "num_layers": 2,
          "dropout": 0.2
        },
        "features": {
          "count": 35,
          "categories": ["momentum", "volatility", "trend", "volume", "statistical"],
          "normalization": "z-score"
        }
      },
      "performance": {
        "train": {
          "loss": 0.0234,
          "accuracy": 0.876
        },
        "validation": {
          "loss": 0.0312,
          "accuracy": 0.823
        },
        "backtest": {
          "sharpe_ratio": 1.87,
          "calmar_ratio": 2.34,
          "max_drawdown": -0.12,
          "win_rate": 0.64,
          "total_trades": 156
        }
      },
      "infrastructure": {
        "instance_type": "g4dn.xlarge",
        "gpu": "NVIDIA T4 (16GB)",
        "training_time_minutes": 12,
        "cost_usd": 0.16
      }
    }
  ]
}
```

**Generation** (Automated):
```bash
python scripts/generate_model_manifest.py \
  --models-dir models/ \
  --output models/manifest_$(date +%Y%m%d).json
```

---

## VIII. CHANGELOG & VERSION HISTORY

### Version 1.0.0 (2025-11-25)

**Initial Release** of formal V7 Ultimate specification

**Components Documented**:
- ✅ Prime Directive & Operational Mandate
- ✅ Architecture: 11 Mathematical Theories
- ✅ DeepSeek LLM Integration
- ✅ Quantitative Finance Enhancements
  - Portfolio Optimization (Markowitz)
  - Advanced Risk Metrics (Calmar, Omega)
  - Backtesting Infrastructure (Vectorized, Walk-Forward)
  - Feature Engineering (35+ Technical Indicators)
  - Historical Data Collection (2 years, 10 symbols)
- ✅ FTMO Risk Protocol (Immutable)
- ✅ Operational Procedures (7 Critical Playbooks)
- ✅ Quantitative Specifications (Metrics, Latency, Quality)
- ✅ System Documentation (Files, Git, Logs)

**System Status** (as of 2025-11-25):
- V7 Runtime: Operational (PID 2787619, 3 days uptime)
- Paper Trades: 13 completed (53.8% win rate, +5.48% P&L)
- Database: SQLite (4,075 signals)
- Cost: $0.19/$150 DeepSeek budget (0.13% used)
- AWS: $79/month (down from $140 after cleanup)

**Recent Enhancements** (2025-11-22 to 2025-11-25):
```
Commit History:
  5416a2e - feat: add historical data collection infrastructure for backtesting
  393ad84 - feat: add comprehensive technical indicators library (35+ indicators)
  be49259 - feat: add Calmar Ratio and Omega Ratio tracking to V7 runtime
  9dd726f - feat: add vectorized backtesting engine with walk-forward optimization
  dba7669 - docs: update CLAUDE.md with current V7 Ultimate architecture
```

**Known Issues**: None (all systems operational)

**Next Actions**:
- Continue data collection (target: 20+ paper trades)
- Monday 2025-11-25 review: Calculate Sharpe Ratio
- Decision point: Phase 1 enhancements based on Sharpe

---

## IX. CRITICAL REMINDERS

### For All Claude Instances

**BEFORE ANY CODE MODIFICATION**:
```
[ ] Read MEGAPROMPT_V7_ULTIMATE.md (this file)
[ ] Read CLAUDE.md (operational context)
[ ] Read CURRENT_STATUS_AND_NEXT_ACTIONS.md (current phase)
[ ] Check git status (ensure clean working directory)
[ ] Check V7 runtime status (ps aux | grep v7_runtime)
[ ] Review recent logs (tail -100 /tmp/v7_runtime_*.log)
```

**AFTER ANY CODE MODIFICATION**:
```
[ ] Document decision in DECISION_LOG.md
[ ] Update CURRENT_STATUS_AND_NEXT_ACTIONS.md if needed
[ ] Commit with proper format + co-authorship
[ ] Push to GitHub
[ ] Verify V7 runtime still operational
[ ] Monitor for 10 minutes after deployment
```

**NEVER**:
```
❌ Train ML models on local machine (ALWAYS use AWS g4dn.xlarge)
❌ Modify signal logic without A/B testing
❌ Deploy to production without paper trading validation
❌ Commit without documenting rationale
❌ Leave AWS GPU instances running after training
❌ Modify FTMO risk limits without explicit approval
❌ Delete or modify historical signals in database
```

**ALWAYS**:
```
✅ Sync git before starting work (git pull origin main)
✅ Create new branch for experiments
✅ Use vectorized operations (pandas/numpy)
✅ Log all modifications with timestamps
✅ Test on paper trading first (minimum 7 days)
✅ Terminate GPU instances immediately after training
✅ Backup database before schema changes
✅ Monitor costs weekly (AWS + DeepSeek)
```

---

## X. CONTACT & ESCALATION

**For Critical Issues**:
1. Check MEGAPROMPT (this file) first
2. Check CLAUDE.md for operational procedures
3. Check DECISION_LOG.md for historical context
4. Check GitHub issues: https://github.com/imnuman/crpbot/issues

**Emergency Contacts**:
- Human Admin: [Contact via Telegram]
- System Status: http://178.156.136.185:3000
- Logs: `/tmp/v7_runtime_*.log`
- Database: `/root/crpbot/tradingai.db`

**If All Else Fails**:
```bash
# Emergency safe mode:
export KILL_SWITCH=true
pkill -9 -f v7_runtime.py

# Wait for human intervention
echo "SYSTEM HALTED - AWAITING HUMAN REVIEW" | tee /tmp/emergency_halt.log
```

---

**END OF MEGAPROMPT V7 ULTIMATE FORMAL SPECIFICATION**

**Document Authority**: This document supersedes all other documentation in case of conflict. Any deviation requires explicit approval and logging in DECISION_LOG.md.

**Last Reviewed**: 2025-11-25
**Next Review**: 2025-12-25 (or after significant system changes)
**Maintainer**: All Claude Code Instances + Human Admin
