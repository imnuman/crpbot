# Monte Carlo Methods Implementation Plan

## Overview
Monte Carlo simulations for trading signal validation, risk analysis, and confidence estimation.

## Planned Applications

### 1. Signal Confidence Intervals (Priority 1)
**Purpose**: Estimate uncertainty in model predictions
**Method**:
- Run 1000+ forward simulations with different market scenarios
- Generate confidence bands around predicted price movements
- Calculate probability distributions for signal outcomes

**Implementation**:
```python
def monte_carlo_confidence(signal, n_simulations=1000):
    """
    Run Monte Carlo simulations to estimate signal confidence.

    Returns:
        - mean_prediction: Expected outcome
        - confidence_interval: (lower_bound, upper_bound)
        - probability_of_success: P(profitable)
    """
    pass
```

### 2. Risk Simulation (Priority 2)
**Purpose**: Simulate potential P&L outcomes
**Method**:
- Simulate 10,000 possible price paths using GBM (Geometric Brownian Motion)
- Calculate P&L for each path
- Estimate VaR (Value at Risk) and CVaR (Conditional VaR)

**Implementation**:
```python
def simulate_risk(entry_price, position_size, holding_period, volatility):
    """
    Simulate risk metrics using Monte Carlo.

    Returns:
        - expected_pnl: Mean P&L
        - var_95: 95% Value at Risk
        - cvar_95: 95% Conditional VaR
        - max_drawdown: Maximum drawdown
    """
    pass
```

### 3. Feature Imputation (Priority 3)
**Purpose**: Estimate missing multi-timeframe features
**Method**:
- Use historical correlations to simulate missing features
- Bootstrap from similar market conditions
- Validate imputed features against actual data when available

### 4. Backtest Validation (Priority 4)
**Purpose**: Validate strategy robustness
**Method**:
- Resample historical data with replacement (bootstrap)
- Run backtests on resampled data
- Estimate confidence in backtest metrics

### 5. Portfolio Optimization (Priority 5)
**Purpose**: Optimize position sizing
**Method**:
- Simulate different position sizing strategies
- Maximize Sharpe ratio while respecting FTMO limits
- Find optimal risk per trade

## Dependencies
```bash
uv add numpy scipy pandas matplotlib seaborn
```

## Integration Points
- `apps/runtime/monte_carlo.py` - Core MC engine
- `apps/runtime/main.py` - Call MC before emitting signals
- `apps/runtime/risk_analysis.py` - Risk metrics using MC

## Status
- [x] Documented plan
- [ ] Implement core MC engine
- [ ] Integrate with signal generation
- [ ] Add to runtime pipeline
- [ ] Test with live signals

## Notes
- Start with Signal Confidence Intervals (simplest, highest value)
- Use MC to filter low-quality signals before FTMO checks
- Can reduce false positives significantly
