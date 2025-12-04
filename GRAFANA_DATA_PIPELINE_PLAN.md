# HYDRA 4.0 → Grafana Data Pipeline Plan

## Current State

### What's Working
- Prometheus scraping HYDRA metrics every 15s
- Basic metrics exported (17 metrics)
- Grafana dashboards provisioned (Overview + Risk)

### What's Exported Now
```
Performance:     pnl_total, pnl_daily, win_rate_24h, win_rate_total
Engine:          engine_rank, engine_weight, engine_points, engine_win_rate (per engine)
Risk:            daily_drawdown, total_drawdown, kill_switch, guardian_status
Market:          asset_price (per asset)
System:          uptime, cycle_duration, errors_total
```

### What's Missing (50+ metrics available but not exported)

| Category | Available Data | Source |
|----------|---------------|--------|
| Per-Asset Performance | Win rate, P&L, trade count by BTC/ETH/SOL | `paper_trader.get_stats_by_asset()` |
| Per-Regime Performance | Performance by TRENDING/RANGING/VOLATILE | `paper_trader.get_stats_by_regime()` |
| Technical Indicators | ADX, ATR, BB Width, Volatility | `regime_detector.detect_regime()` |
| Guardian Full State | Account balance, peak balance, circuit breaker | `guardian.get_status()` |
| Trade Statistics | Sharpe ratio, avg R:R, trade distribution | `paper_trader.get_overall_stats()` |
| Engine Details | Per-asset performance, vote history | `tournament_tracker.get_engine_stats()` |

---

## Implementation Plan

### Phase 1: Core Metrics (Priority: HIGH)
**Goal**: Get essential trading data into Grafana

#### 1.1 Add Per-Asset Metrics
```python
# New metrics in libs/monitoring/metrics.py
asset_pnl = Gauge('hydra_asset_pnl_percent', 'P&L by asset', ['asset'])
asset_win_rate = Gauge('hydra_asset_win_rate', 'Win rate by asset', ['asset'])
asset_trades = Counter('hydra_asset_trades_total', 'Trades by asset', ['asset', 'direction'])
```

**Data Source**: `paper_trader.get_stats_by_asset()`
**Update Location**: `hydra_runtime.py:_update_prometheus_metrics()`

#### 1.2 Add Technical Indicators
```python
# New metrics
regime_current = Gauge('hydra_regime_current', 'Current market regime', ['asset'])
indicator_adx = Gauge('hydra_indicator_adx', 'ADX value', ['asset'])
indicator_atr = Gauge('hydra_indicator_atr', 'ATR value', ['asset'])
indicator_volatility = Gauge('hydra_indicator_volatility', 'Volatility', ['asset'])
```

**Data Source**: `regime_detector.detect_regime()` returns dict with these values
**Update Location**: Add call in `_update_prometheus_metrics()`

#### 1.3 Complete Guardian Metrics
```python
# New metrics
account_balance = Gauge('hydra_account_balance_usd', 'Current account balance')
peak_balance = Gauge('hydra_peak_balance_usd', 'Peak account balance')
daily_pnl_usd = Gauge('hydra_daily_pnl_usd', 'Daily P&L in USD')
circuit_breaker = Gauge('hydra_circuit_breaker_active', 'Circuit breaker status')
```

**Data Source**: `guardian.get_status()`
**Update Location**: `_update_prometheus_metrics()` - guardian call exists but incomplete

---

### Phase 2: Regime & Strategy Analytics (Priority: MEDIUM)
**Goal**: Understand performance across market conditions

#### 2.1 Per-Regime Performance
```python
# New metrics
regime_pnl = Gauge('hydra_regime_pnl_percent', 'P&L by regime', ['regime'])
regime_win_rate = Gauge('hydra_regime_win_rate', 'Win rate by regime', ['regime'])
regime_trades = Counter('hydra_regime_trades_total', 'Trades by regime', ['regime'])
```

**Regimes**: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, CHOPPY
**Data Source**: `paper_trader.get_stats_by_regime()`

#### 2.2 Regime Distribution
```python
regime_time_percent = Gauge('hydra_regime_time_percent', 'Time spent in regime', ['regime', 'asset'])
```

**Data Source**: Database `regime_history` table

---

### Phase 3: Engine Deep Analytics (Priority: MEDIUM)
**Goal**: Understand which engines perform best in which conditions

#### 3.1 Engine Per-Asset Performance
```python
engine_asset_pnl = Gauge('hydra_engine_asset_pnl', 'Engine P&L by asset', ['engine', 'asset'])
engine_asset_accuracy = Gauge('hydra_engine_asset_accuracy', 'Engine accuracy by asset', ['engine', 'asset'])
```

**Data Source**: `tournament_tracker.get_engine_stats()`

#### 3.2 Engine Voting Metrics
```python
engine_votes_total = Counter('hydra_engine_votes_total', 'Total votes by engine', ['engine', 'direction'])
engine_agreement_rate = Gauge('hydra_engine_agreement_rate', 'How often engines agree')
```

**Data Source**: `vote_tracker.get_voting_history()`

---

### Phase 4: Advanced Analytics (Priority: LOW)
**Goal**: Statistical trading metrics

#### 4.1 Risk-Adjusted Metrics
```python
sharpe_ratio = Gauge('hydra_sharpe_ratio', 'Sharpe ratio')
sortino_ratio = Gauge('hydra_sortino_ratio', 'Sortino ratio')
max_drawdown = Gauge('hydra_max_drawdown_percent', 'Maximum drawdown')
calmar_ratio = Gauge('hydra_calmar_ratio', 'Calmar ratio')
```

#### 4.2 Trade Quality Metrics
```python
avg_risk_reward = Gauge('hydra_avg_risk_reward', 'Average risk/reward ratio')
profit_factor = Gauge('hydra_profit_factor', 'Profit factor')
expectancy = Gauge('hydra_expectancy', 'Trade expectancy')
```

---

## Dashboard Updates Required

### Phase 1 Dashboard Additions
1. **Asset Performance Panel** - Bar chart showing P&L by asset
2. **Technical Indicators Panel** - Time series of ADX, ATR, Volatility
3. **Current Regime Indicator** - Stat panel showing current regime per asset
4. **Account Balance Panel** - Time series with peak balance line

### Phase 2 Dashboard Additions
1. **Regime Performance Heatmap** - Win rate by regime
2. **Regime Distribution Pie** - Time spent in each regime
3. **Regime Transitions** - State diagram or timeline

### Phase 3 Dashboard Additions
1. **Engine x Asset Matrix** - Heatmap of engine performance by asset
2. **Engine Agreement Gauge** - How often engines align
3. **Vote Distribution** - Sankey or bar chart of votes

---

## Implementation Order

```
Week 1: Phase 1 (Core Metrics)
├─ Day 1: Add metrics definitions to libs/monitoring/metrics.py
├─ Day 2: Update _update_prometheus_metrics() to collect all data
├─ Day 3: Update Grafana dashboards with new panels
└─ Day 4: Test and verify data flow

Week 2: Phase 2 (Regime Analytics)
├─ Add regime metrics
├─ Create regime dashboard panels
└─ Test regime tracking

Week 3: Phase 3 (Engine Analytics)
├─ Add engine detailed metrics
├─ Create engine analytics dashboard
└─ Test engine tracking

Week 4: Phase 4 (Advanced Analytics)
├─ Add risk-adjusted metrics
├─ Create advanced analytics dashboard
└─ Final testing and optimization
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `libs/monitoring/metrics.py` | Add 30+ new metric definitions |
| `apps/runtime/hydra_runtime.py` | Expand `_update_prometheus_metrics()` |
| `monitoring/grafana/dashboards/hydra-overview.json` | Add new panels |
| `monitoring/grafana/dashboards/hydra-asset-performance.json` | NEW dashboard |
| `monitoring/grafana/dashboards/hydra-regime-analytics.json` | NEW dashboard |
| `monitoring/grafana/dashboards/hydra-engine-analytics.json` | NEW dashboard |

---

## Quick Start: Phase 1 Implementation

### Step 1: Add Metrics (libs/monitoring/metrics.py)
```python
# Per-Asset Metrics
asset_pnl = Gauge('hydra_asset_pnl_percent', 'P&L percentage by asset', ['asset'])
asset_win_rate = Gauge('hydra_asset_win_rate', 'Win rate by asset', ['asset'])
asset_trade_count = Gauge('hydra_asset_trade_count', 'Trade count by asset', ['asset'])

# Technical Indicators
regime_current = Gauge('hydra_regime_current', 'Current regime (enum)', ['asset'])
indicator_adx = Gauge('hydra_indicator_adx', 'ADX indicator', ['asset'])
indicator_atr = Gauge('hydra_indicator_atr', 'ATR indicator', ['asset'])
indicator_bb_width = Gauge('hydra_indicator_bb_width', 'Bollinger Band width', ['asset'])

# Guardian Full State
account_balance = Gauge('hydra_account_balance_usd', 'Account balance in USD')
peak_balance = Gauge('hydra_peak_balance_usd', 'Peak account balance')
daily_pnl_usd = Gauge('hydra_daily_pnl_usd', 'Daily P&L in USD')
```

### Step 2: Update Export (hydra_runtime.py)
```python
def _update_prometheus_metrics(self):
    # ... existing code ...

    # NEW: Per-asset metrics
    asset_stats = self.paper_trader.get_stats_by_asset()
    for asset, stats in asset_stats.items():
        HydraMetrics.asset_pnl.labels(asset=asset).set(stats['pnl_percent'])
        HydraMetrics.asset_win_rate.labels(asset=asset).set(stats['win_rate'])
        HydraMetrics.asset_trade_count.labels(asset=asset).set(stats['trade_count'])

    # NEW: Technical indicators per asset
    for asset in self.assets:
        regime_info = self.regime_detector.detect_regime(asset)
        HydraMetrics.indicator_adx.labels(asset=asset).set(regime_info['adx'])
        HydraMetrics.indicator_atr.labels(asset=asset).set(regime_info['atr'])
```

### Step 3: Add Dashboard Panel (Grafana JSON)
```json
{
  "title": "P&L by Asset",
  "type": "barchart",
  "targets": [{
    "expr": "hydra_asset_pnl_percent",
    "legendFormat": "{{asset}}"
  }]
}
```

---

## Success Criteria

Phase 1 Complete When:
- [ ] All 8 assets show individual P&L in Grafana
- [ ] Technical indicators (ADX, ATR) visible per asset
- [ ] Current regime displayed for each asset
- [ ] Account balance tracking working

Phase 2 Complete When:
- [ ] Performance breakdown by regime visible
- [ ] Can see which regimes are most profitable
- [ ] Regime transitions tracked over time

Phase 3 Complete When:
- [ ] Engine performance visible per asset
- [ ] Can identify best engine for each asset
- [ ] Vote distribution and agreement visible

Phase 4 Complete When:
- [ ] Sharpe ratio calculated and displayed
- [ ] All risk-adjusted metrics available
- [ ] Complete trading analytics dashboard
