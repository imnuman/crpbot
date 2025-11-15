# V5 Feature Reference - 53 Features

**Version**: V5
**Total Features**: 53
**Data Source**: Tardis.dev professional tick data + order book
**Target Accuracy**: 65-75% (vs V4's 50% ceiling)

---

## Feature Breakdown

### Category 1: OHLCV (5 features) - REUSED from V4

| # | Feature | Description | Source |
|---|---------|-------------|--------|
| 1 | `open` | 1-min candle open price | Tardis tick aggregation |
| 2 | `high` | 1-min candle high price | Tardis tick aggregation |
| 3 | `low` | 1-min candle low price | Tardis tick aggregation |
| 4 | `close` | 1-min candle close price | Tardis tick aggregation |
| 5 | `volume` | 1-min candle volume | Tardis tick aggregation |

---

### Category 2: Session Features (5 features) - REUSED from V4

| # | Feature | Description | Calculation |
|---|---------|-------------|-------------|
| 6 | `session_tokyo` | Tokyo session (0/1) | 00:00-09:00 UTC |
| 7 | `session_london` | London session (0/1) | 08:00-17:00 UTC |
| 8 | `session_ny` | New York session (0/1) | 13:00-22:00 UTC |
| 9 | `day_of_week` | Day of week (0-6) | Monday=0, Sunday=6 |
| 10 | `is_weekend` | Weekend flag (0/1) | Saturday/Sunday=1 |

---

### Category 3: Spread Features (4 features) - REUSED from V4

| # | Feature | Description | Calculation |
|---|---------|-------------|-------------|
| 11 | `spread` | Absolute bid-ask spread | ask - bid |
| 12 | `spread_pct` | Spread as % of mid | spread / mid_price |
| 13 | `atr` | Average True Range (14-period) | max(H-L, H-C_prev, C_prev-L) |
| 14 | `spread_atr_ratio` | Spread relative to volatility | spread / atr |

---

### Category 4: Volume Features (3 features) - REUSED from V4

| # | Feature | Description | Calculation |
|---|---------|-------------|-------------|
| 15 | `volume_ma` | Volume moving average (20-period) | SMA(volume, 20) |
| 16 | `volume_ratio` | Current volume vs average | volume / volume_ma |
| 17 | `volume_trend` | Volume trend direction | sign(volume_ma - volume_ma.shift(5)) |

---

### Category 5: Moving Averages (8 features) - REUSED from V4

| # | Feature | Description | Calculation |
|---|---------|-------------|-------------|
| 18 | `sma_7` | 7-period simple moving average | SMA(close, 7) |
| 19 | `sma_14` | 14-period simple moving average | SMA(close, 14) |
| 20 | `sma_21` | 21-period simple moving average | SMA(close, 21) |
| 21 | `sma_50` | 50-period simple moving average | SMA(close, 50) |
| 22 | `price_sma7_ratio` | Price relative to SMA7 | close / sma_7 |
| 23 | `price_sma14_ratio` | Price relative to SMA14 | close / sma_14 |
| 24 | `price_sma21_ratio` | Price relative to SMA21 | close / sma_21 |
| 25 | `price_sma50_ratio` | Price relative to SMA50 | close / sma_50 |

---

### Category 6: Technical Indicators (8 features) - REUSED from V4

| # | Feature | Description | Calculation |
|---|---------|-------------|-------------|
| 26 | `rsi` | Relative Strength Index (14-period) | RSI(close, 14) |
| 27 | `macd` | MACD line | EMA(12) - EMA(26) |
| 28 | `macd_signal` | MACD signal line | EMA(macd, 9) |
| 29 | `macd_diff` | MACD histogram | macd - macd_signal |
| 30 | `bb_upper` | Bollinger Band upper | SMA(20) + 2*STD(20) |
| 31 | `bb_lower` | Bollinger Band lower | SMA(20) - 2*STD(20) |
| 32 | `bb_width` | Bollinger Band width | bb_upper - bb_lower |
| 33 | `bb_position` | Price position in bands | (close - bb_lower) / bb_width |

**Total V4 features**: 33

---

## NEW V5 Features (20 Microstructure Features)

### Category 7: Order Book Features (8 features) - NEW

| # | Feature | Description | Calculation | Source |
|---|---------|-------------|-------------|--------|
| 34 | `bid_ask_spread` | Immediate spread (bps) | (ask - bid) / mid * 10000 | Tardis order book |
| 35 | `spread_volatility` | Spread volatility (10-min) | STD(bid_ask_spread, 10) | Tardis order book |
| 36 | `order_book_imbalance` | Volume imbalance at L1 | (bid_vol - ask_vol) / (bid_vol + ask_vol) | Tardis order book |
| 37 | `book_pressure_5` | Volume imbalance (5 levels) | Sum imbalance over 5 levels | Tardis order book |
| 38 | `book_pressure_10` | Volume imbalance (10 levels) | Sum imbalance over 10 levels | Tardis order book |
| 39 | `weighted_mid_price` | Volume-weighted mid | (bid*ask_vol + ask*bid_vol) / (bid_vol + ask_vol) | Tardis order book |
| 40 | `microprice` | Order flow microprice | bid + (ask-bid) * ask_vol/(bid_vol+ask_vol) | Tardis order book |
| 41 | `effective_spread` | Realized spread from trades | 2 * \|trade_price - mid_price\| | Tardis trades |

**References**:
- Order book imbalance: Cont, Stoikov, Talreja (2010)
- Microprice: Stoikov (2018)
- Effective spread: Harris (2003)

---

### Category 8: Order Flow Features (6 features) - NEW

| # | Feature | Description | Calculation | Source |
|---|---------|-------------|-------------|--------|
| 42 | `trade_intensity` | Trades per minute | COUNT(trades) per 1-min | Tardis trades |
| 43 | `buy_volume_ratio` | Buy-initiated trades | buy_volume / total_volume | Tardis trades (aggressor) |
| 44 | `sell_volume_ratio` | Sell-initiated trades | sell_volume / total_volume | Tardis trades (aggressor) |
| 45 | `trade_size_mean` | Average trade size | MEAN(trade_sizes) per 1-min | Tardis trades |
| 46 | `trade_size_std` | Trade size volatility | STD(trade_sizes) per 1-min | Tardis trades |
| 47 | `large_trade_ratio` | Large trades (>2σ) | COUNT(size > mean+2σ) / total_trades | Tardis trades |

**References**:
- Order flow toxicity: Easley, O'Hara, López de Prado (2012) - VPIN
- Trade classification: Lee-Ready algorithm (1991)

---

### Category 9: Tick-Level Volatility (4 features) - NEW

| # | Feature | Description | Calculation | Source |
|---|---------|-------------|-------------|--------|
| 48 | `tick_volatility` | Tick price volatility (1-min) | STD(tick_prices) per 1-min | Tardis ticks |
| 49 | `tick_volatility_5m` | Tick price volatility (5-min) | STD(tick_prices) per 5-min | Tardis ticks |
| 50 | `realized_volatility` | Sum of squared returns | SUM((p_t - p_{t-1})^2) | Tardis ticks |
| 51 | `jump_indicator` | Large price jumps (>3σ) | 1 if \|return\| > 3*σ else 0 | Tardis ticks |

**References**:
- Realized volatility: Andersen, Bollerslev (1998)
- Jump detection: Barndorff-Nielsen, Shephard (2006)

---

### Category 10: Execution Quality (2 features) - NEW

| # | Feature | Description | Calculation | Source |
|---|---------|-------------|-------------|--------|
| 52 | `vwap_distance` | Distance from VWAP (bps) | (close - vwap) / vwap * 10000 | Tardis trades |
| 53 | `arrival_price_impact` | Market impact | (mid_now - mid_before) / mid_before | Tardis order book |

**References**:
- VWAP: Standard execution benchmark
- Price impact: Almgren, Chriss (2001) - Optimal execution

---

## Feature Groups Summary

| Group | Count | Type | Data Source |
|-------|-------|------|-------------|
| OHLCV | 5 | V4 Reuse | Tardis ticks (aggregated) |
| Session | 5 | V4 Reuse | Timestamp |
| Spread | 4 | V4 Reuse | Tardis order book |
| Volume | 3 | V4 Reuse | Tardis ticks |
| Moving Averages | 8 | V4 Reuse | Derived from OHLCV |
| Technical Indicators | 8 | V4 Reuse | Derived from OHLCV |
| **Order Book** | **8** | **V5 NEW** | **Tardis order book** |
| **Order Flow** | **6** | **V5 NEW** | **Tardis trades** |
| **Tick Volatility** | **4** | **V5 NEW** | **Tardis ticks** |
| **Execution Quality** | **2** | **V5 NEW** | **Tardis trades + book** |
| **TOTAL** | **53** | | |

---

## Why These Features?

### V4 Problem: Noisy Free Data
- Free Coinbase OHLCV: 1-minute aggregated bars
- Missing: Tick-level dynamics, order book depth, trade flow
- Result: Models stuck at 50% accuracy (coin flip)

### V5 Solution: Microstructure Signals
- **Order book**: Reveals supply/demand imbalance before price moves
- **Order flow**: Detects institutional activity (large trades, aggression)
- **Tick volatility**: Captures intraday volatility regime changes
- **Execution quality**: Measures market liquidity and impact

### Expected Impact
- **Conservative**: +15 percentage points (50% → 65%)
- **Optimistic**: +25 percentage points (50% → 75%)
- **Rationale**: Professional data reduces noise, microstructure features capture alpha

---

## Feature Engineering Implementation

### Data Flow

```
Tardis API
   ↓
Raw tick data (trades + order book snapshots)
   ↓
1. Aggregate ticks → 1-min OHLCV (5 features)
2. Compute V4 features → (28 derived features)
3. Compute microstructure features → (20 NEW features)
   ↓
53-feature dataset (1-min resolution)
   ↓
Train LSTM (60-min lookback) + Transformer (100-min lookback)
```

### Code Structure

**New script**: `scripts/engineer_v5_features.py`

```python
def engineer_v5_features(tardis_trades: pd.DataFrame,
                         tardis_book: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all 53 features for V5.

    Args:
        tardis_trades: Tick-level trades with columns [timestamp, price, size, side]
        tardis_book: Order book snapshots with columns [timestamp, bids[], asks[]]

    Returns:
        DataFrame with 53 features, 1-minute resolution
    """
    # Step 1: OHLCV (5 features)
    ohlcv = aggregate_ohlcv(tardis_trades)

    # Step 2: V4 features (28 features)
    v4_features = compute_v4_features(ohlcv)

    # Step 3: Microstructure features (20 features)
    book_features = compute_order_book_features(tardis_book)      # 8 features
    flow_features = compute_order_flow_features(tardis_trades)    # 6 features
    vol_features = compute_tick_volatility_features(tardis_trades) # 4 features
    exec_features = compute_execution_features(tardis_trades, tardis_book) # 2 features

    # Step 4: Combine all 53 features
    all_features = pd.concat([
        ohlcv,           # 5
        v4_features,     # 28
        book_features,   # 8
        flow_features,   # 6
        vol_features,    # 4
        exec_features    # 2
    ], axis=1)

    assert all_features.shape[1] == 53, f"Expected 53 features, got {all_features.shape[1]}"

    return all_features
```

---

## Validation Checklist

### Data Quality
- [ ] No NaN values in features
- [ ] No infinite values
- [ ] Feature ranges are reasonable (no outliers >10σ)
- [ ] Timestamp alignment correct (all features at same 1-min intervals)

### Feature Quality
- [ ] Order book imbalance in [-1, 1]
- [ ] Buy/sell volume ratios sum to ~1
- [ ] Spread > 0 always
- [ ] Volatility measures > 0
- [ ] VWAP distance in reasonable range (< 1000 bps)

### Correlation Analysis
- [ ] Features not perfectly correlated (check correlation matrix)
- [ ] New microstructure features provide unique signal
- [ ] Feature importance scores make sense (plot)

---

## Timeline

**Week 2** (Nov 22-29):
- Implement all 53 features
- Validate feature quality
- Generate final parquet files

**Expected Output**:
- 3 files: `features_BTC-USD_v5_1m.parquet`, `features_ETH-USD_v5_1m.parquet`, `features_SOL-USD_v5_1m.parquet`
- Each file: ~1M rows × 53 columns = ~400-500 MB
- Total: ~1.5 GB (3 symbols)

---

## References

### Academic Papers
1. Cont, Stoikov, Talreja (2010) - "A Stochastic Model for Order Book Dynamics"
2. Easley, O'Hara, López de Prado (2012) - "Flow Toxicity and Liquidity in a High-Frequency World"
3. Almgren, Chriss (2001) - "Optimal Execution of Portfolio Transactions"
4. Andersen, Bollerslev (1998) - "Answering the Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts"

### Books
- Lehalle & Laruelle (2013) - "Market Microstructure in Practice"
- Cartea, Jaimungal, Penalva (2015) - "Algorithmic and High-Frequency Trading"
- Harris (2003) - "Trading and Exchanges"

### Data Provider
- Tardis.dev Documentation: https://docs.tardis.dev/

---

**Last Updated**: 2025-11-15
**Next Review**: After Week 2 feature engineering complete
**Owner**: Cloud Claude (Builder)
