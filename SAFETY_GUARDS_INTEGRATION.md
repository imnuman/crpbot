# Safety Guards & Data Enhancers Integration
## Layer 2 (Core Engine) & Layer 3 (Data & Memory) Enhancements

**Date**: 2025-11-24
**Purpose**: Add critical risk mitigation and feedback loops to V7 Ultimate
**Integration**: Fits between existing Risk Management (Phase 1) and Agent Layer (Phase 6)

---

## 🎯 Overview

These modules add **production-grade safety** and **comprehensive feedback loops** that professional trading systems require:

**Problem Solved**:
1. Trading in choppy/ranging markets (kills win rate)
2. Drawdown spirals (series of losses compound)
3. Correlated position stacking (multiple LONGs on BTC-like assets)
4. No feedback on rejected signals (can't learn what NOT to trade)

**Impact**:
- **Win Rate**: +5-10 points (avoiding bad market conditions)
- **Max Drawdown**: -30-50% reduction (circuit breaker protection)
- **Risk Management**: Better position diversification
- **System Intelligence**: Learn from rejections, not just trades

---

## 🛡️ Layer 2: Safety Guards (Core Engine)

These modules **block bad trades before they execute** - acting as the final gatekeepers.

### Module 1: Market Regime Detector (New)

**File**: `libs/safety/market_regime_detector.py`

**Purpose**: Detect unfavorable market conditions and block signals

**Problem**:
- Current system trades in all market conditions
- Choppy/ranging markets destroy win rate (45% of losses in V7 review)
- R:R Sniper signals get whipsawed in sideways action

**Solution**:
```
Market Condition Analysis:
├── Trend Strength: ADX, Hurst Exponent
├── Volatility: ATR, Bollinger Band Width
├── Directional Bias: EMA slope, MACD
└── Market Structure: Higher highs/lows

If "Choppy/Ranging" detected:
  → Block all signals
  → Log: "Market chop detected, trade blocked"
  → Wait for trend emergence
```

**Implementation**:

```python
class MarketRegimeDetector:
    """
    Detect market regime and block signals in unfavorable conditions

    Regimes:
    1. Strong Trend (GOOD) - Clear direction, high momentum
    2. Weak Trend (OK) - Directional but low confidence
    3. Ranging/Chop (BAD) - No clear direction, whipsaws
    4. High Volatility Breakout (GOOD) - Explosive moves
    5. Low Volatility Compression (BAD) - Boring, no edge
    """

    def __init__(
        self,
        timeframe: str = '5m',  # 5-minute charts for regime detection
        lookback_periods: int = 100  # 100 periods = ~8 hours
    ):
        self.timeframe = timeframe
        self.lookback_periods = lookback_periods

    def detect_regime(
        self,
        symbol: str,
        candles_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect current market regime

        Returns:
            {
                'regime': 'Strong Trend' | 'Ranging' | 'Breakout' | ...,
                'quality': 'good' | 'ok' | 'bad',
                'should_trade': True/False,
                'confidence': 0.85,
                'metrics': {
                    'adx': 35.2,
                    'atr_percentile': 0.68,
                    'bb_width_percentile': 0.45,
                    'trend_strength': 0.72
                },
                'reason': 'Strong uptrend detected, ADX > 25'
            }
        """

        # 1. Calculate indicators
        adx = self._calculate_adx(candles_df)
        atr = self._calculate_atr(candles_df)
        bb_width = self._calculate_bb_width(candles_df)
        trend_slope = self._calculate_trend_slope(candles_df)

        # 2. Normalize to percentiles (0-1)
        adx_percentile = self._percentile(adx, candles_df['adx_history'])
        atr_percentile = self._percentile(atr, candles_df['atr_history'])
        bb_percentile = self._percentile(bb_width, candles_df['bb_history'])

        # 3. Classify regime
        regime = self._classify_regime(
            adx=adx,
            adx_pct=adx_percentile,
            atr_pct=atr_percentile,
            bb_pct=bb_percentile,
            trend_slope=trend_slope
        )

        return regime

    def _classify_regime(
        self,
        adx: float,
        adx_pct: float,
        atr_pct: float,
        bb_pct: float,
        trend_slope: float
    ) -> Dict[str, Any]:
        """
        Classify market regime from indicators

        Classification Logic:
        - Strong Trend: ADX > 25, clear slope, expanding BB
        - Weak Trend: ADX 20-25, moderate slope
        - Ranging: ADX < 20, flat slope, tight BB
        - Breakout: ADX rising, BB expanding rapidly
        - Compression: ADX falling, BB contracting
        """

        # Strong Trend (GOOD - Trade this)
        if adx > 25 and abs(trend_slope) > 0.0015 and bb_pct > 0.5:
            return {
                'regime': 'Strong Trend',
                'quality': 'good',
                'should_trade': True,
                'confidence': 0.9,
                'reason': f'Strong trend: ADX={adx:.1f}, slope={trend_slope:.4f}'
            }

        # Breakout (GOOD - High profit potential)
        elif atr_pct > 0.8 and bb_pct > 0.7 and adx > 20:
            return {
                'regime': 'Volatility Breakout',
                'quality': 'good',
                'should_trade': True,
                'confidence': 0.85,
                'reason': f'Breakout: ATR pct={atr_pct:.2f}, expanding volatility'
            }

        # Weak Trend (OK - Trade with caution)
        elif adx >= 20 and abs(trend_slope) > 0.0008:
            return {
                'regime': 'Weak Trend',
                'quality': 'ok',
                'should_trade': True,
                'confidence': 0.7,
                'reason': f'Weak trend: ADX={adx:.1f}, moderate directional bias'
            }

        # Ranging/Chop (BAD - DO NOT TRADE)
        elif adx < 20 and abs(trend_slope) < 0.0008 and bb_pct < 0.4:
            return {
                'regime': 'Ranging/Chop',
                'quality': 'bad',
                'should_trade': False,  # BLOCK SIGNALS
                'confidence': 0.8,
                'reason': f'Chop detected: ADX={adx:.1f} < 20, flat slope'
            }

        # Low Volatility Compression (BAD - Wait for breakout)
        elif atr_pct < 0.3 and bb_pct < 0.3:
            return {
                'regime': 'Low Volatility Compression',
                'quality': 'bad',
                'should_trade': False,  # BLOCK SIGNALS
                'confidence': 0.75,
                'reason': f'Low vol: ATR pct={atr_pct:.2f}, BB compression'
            }

        # Uncertain (OK - Proceed with extra caution)
        else:
            return {
                'regime': 'Uncertain',
                'quality': 'ok',
                'should_trade': True,
                'confidence': 0.6,
                'reason': 'Mixed signals, trade with reduced size'
            }

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average Directional Index (ADX)

        ADX > 25 = Strong trend
        ADX < 20 = Ranging market
        """
        # Implementation using TA-Lib or manual calculation
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # Calculate +DI and -DI
        # Calculate ADX
        # Return latest ADX value

        return adx_value

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR)

        High ATR = High volatility (good for trends)
        Low ATR = Low volatility (bad for breakouts)
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        # ATR = EMA of True Range

        return atr_value

    def _calculate_bb_width(self, df: pd.DataFrame, period: int = 20) -> float:
        """
        Calculate Bollinger Band Width

        Wide BB = High volatility, trending
        Narrow BB = Low volatility, ranging or about to break out
        """
        close = df['close'].values
        sma = close[-period:].mean()
        std = close[-period:].std()

        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)

        bb_width = (upper_band - lower_band) / sma

        return bb_width

    def _calculate_trend_slope(self, df: pd.DataFrame, period: int = 20) -> float:
        """
        Calculate trend slope using EMA

        Positive slope = Uptrend
        Negative slope = Downtrend
        Near-zero slope = Ranging
        """
        close = df['close'].values[-period:]
        ema = self._ema(close, period)

        # Linear regression slope of EMA
        x = np.arange(len(ema))
        slope, _ = np.polyfit(x, ema, 1)

        # Normalize by price
        slope_normalized = slope / close[-1]

        return slope_normalized
```

**Integration Point**:
```python
# In libs/llm/signal_generator.py (before LLM call)

regime = self.regime_detector.detect_regime(symbol, candles_df)

if not regime['should_trade']:
    logger.info(f"{symbol}: Signal blocked by regime detector")
    logger.info(f"  Regime: {regime['regime']}")
    logger.info(f"  Reason: {regime['reason']}")

    # Log rejection
    self.rejection_logger.log_rejection(
        symbol=symbol,
        reason='market_regime',
        details=regime
    )

    return SignalGenerationResult(
        direction='hold',
        confidence=0.0,
        reasoning=f"Market regime unfavorable: {regime['reason']}"
    )
```

**Configuration** (`.env`):
```bash
# Market Regime Detector
REGIME_DETECTOR_ENABLED=true
REGIME_MIN_ADX=20              # Minimum ADX to trade
REGIME_MIN_TREND_SLOPE=0.0008  # Minimum trend strength
REGIME_TIMEFRAME=5m            # Regime detection timeframe
```

**Expected Impact**:
- **Win Rate**: +5-8 points (avoiding 30-40% of choppy losses)
- **Trade Frequency**: -20-30% (more selective)
- **Sharpe Ratio**: +0.3-0.5 (better risk-adjusted returns)

---

### Module 2: Drawdown Circuit Breaker (New)

**File**: `libs/safety/drawdown_circuit_breaker.py`

**Purpose**: Emergency stop on excessive drawdowns

**Problem**:
- Series of losses can compound (3 losses = -2.4%, 5 losses = -4%)
- Emotional/algorithmic spiral (revenge trading, desperation)
- No automatic kill switch in current system

**Solution**:
```
Real-time Equity Monitoring:
├── Track session starting balance
├── Calculate running P&L
├── Compare to thresholds (3%, 5%, 9%)
└── Trigger actions at each level

Level 1 (Daily -3%): Warning + Reduce size 50%
Level 2 (Daily -5%): Emergency stop + Alert
Level 3 (Total -9%): Full shutdown + FTMO breach
```

**Implementation**:

```python
class DrawdownCircuitBreaker:
    """
    Monitor drawdown and enforce emergency stops

    Thresholds (configurable):
    - Level 1 (Warning): -3% daily loss
    - Level 2 (Emergency): -5% daily loss
    - Level 3 (Shutdown): -9% total loss (FTMO)

    Actions:
    - Level 1: Log warning, reduce position size 50%
    - Level 2: Stop all trading, send alert
    - Level 3: Kill switch, human intervention required
    """

    def __init__(
        self,
        starting_balance: float,
        daily_loss_warning: float = 0.03,      # 3%
        daily_loss_emergency: float = 0.05,    # 5%
        total_loss_shutdown: float = 0.09      # 9% (FTMO)
    ):
        self.starting_balance = starting_balance
        self.session_start_balance = starting_balance
        self.daily_loss_warning = daily_loss_warning
        self.daily_loss_emergency = daily_loss_emergency
        self.total_loss_shutdown = total_loss_shutdown

        # State tracking
        self.current_balance = starting_balance
        self.peak_balance = starting_balance
        self.session_pnl = 0.0
        self.total_pnl = 0.0

        # Circuit breaker state
        self.level = 0  # 0=Normal, 1=Warning, 2=Emergency, 3=Shutdown
        self.is_trading_allowed = True
        self.last_reset = datetime.now()

    def check_drawdown(self) -> Dict[str, Any]:
        """
        Check current drawdown levels

        Returns:
            {
                'level': 0-3,
                'is_trading_allowed': True/False,
                'daily_drawdown_pct': -0.028,
                'total_drawdown_pct': -0.045,
                'session_pnl': -140.0,
                'total_pnl': -450.0,
                'action_required': 'warning' | 'emergency' | 'shutdown' | None,
                'message': 'Daily loss at -2.8%, approaching warning threshold'
            }
        """

        # Calculate drawdowns
        session_dd_pct = self.session_pnl / self.session_start_balance
        total_dd_pct = (self.current_balance - self.starting_balance) / self.starting_balance

        # Determine level
        if abs(total_dd_pct) >= self.total_loss_shutdown:
            # Level 3: SHUTDOWN (FTMO breach)
            self.level = 3
            self.is_trading_allowed = False
            action = 'shutdown'
            message = f"🚨 SHUTDOWN: Total loss {total_dd_pct:.1%} >= {self.total_loss_shutdown:.1%}"

        elif abs(session_dd_pct) >= self.daily_loss_emergency:
            # Level 2: EMERGENCY STOP
            self.level = 2
            self.is_trading_allowed = False
            action = 'emergency'
            message = f"⛔ EMERGENCY: Daily loss {session_dd_pct:.1%} >= {self.daily_loss_emergency:.1%}"

        elif abs(session_dd_pct) >= self.daily_loss_warning:
            # Level 1: WARNING
            self.level = 1
            # Still allow trading but reduce size
            action = 'warning'
            message = f"⚠️  WARNING: Daily loss {session_dd_pct:.1%} >= {self.daily_loss_warning:.1%}"

        else:
            # Level 0: Normal
            self.level = 0
            action = None
            message = f"✅ Normal: Daily {session_dd_pct:.1%}, Total {total_dd_pct:.1%}"

        return {
            'level': self.level,
            'is_trading_allowed': self.is_trading_allowed,
            'daily_drawdown_pct': session_dd_pct,
            'total_drawdown_pct': total_dd_pct,
            'session_pnl': self.session_pnl,
            'total_pnl': self.total_pnl,
            'current_balance': self.current_balance,
            'action_required': action,
            'message': message,
            'position_size_multiplier': self._get_size_multiplier()
        }

    def _get_size_multiplier(self) -> float:
        """
        Get position size multiplier based on drawdown level

        Returns:
            1.0 = Normal (Level 0)
            0.5 = Warning (Level 1) - Half size
            0.0 = Emergency/Shutdown (Level 2/3) - No trading
        """
        if self.level == 0:
            return 1.0  # Normal size
        elif self.level == 1:
            return 0.5  # Half size (warning)
        else:
            return 0.0  # No trading (emergency/shutdown)

    def update_balance(self, new_balance: float):
        """Update current balance after trade"""
        pnl = new_balance - self.current_balance

        self.current_balance = new_balance
        self.session_pnl += pnl
        self.total_pnl = new_balance - self.starting_balance

        # Update peak
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

    def reset_daily(self):
        """Reset daily stats (call at start of new trading day)"""
        self.session_start_balance = self.current_balance
        self.session_pnl = 0.0
        self.last_reset = datetime.now()

        # Reset level if not in total shutdown
        if self.level < 3:
            self.level = 0
            self.is_trading_allowed = True

    def send_alert(self, level: int, message: str):
        """
        Send emergency alert (Telegram, email, SMS)

        Args:
            level: 1=Warning, 2=Emergency, 3=Shutdown
            message: Alert message
        """
        # Telegram notification
        from libs.notifications.telegram import send_alert

        urgency = ['INFO', 'WARNING', 'EMERGENCY', 'CRITICAL'][level]

        send_alert(
            title=f"{urgency}: Drawdown Circuit Breaker",
            message=message,
            priority=level
        )

        # Could also: SMS, email, webhook, etc.
```

**Integration Point**:
```python
# In apps/runtime/v7_runtime.py (main loop)

# At start of each iteration
dd_check = self.circuit_breaker.check_drawdown()

if not dd_check['is_trading_allowed']:
    logger.critical(dd_check['message'])

    # Send alert
    self.circuit_breaker.send_alert(
        level=dd_check['level'],
        message=dd_check['message']
    )

    # Stop processing signals
    if dd_check['level'] >= 2:
        logger.critical("Trading halted by circuit breaker")
        break  # Exit main loop

# Apply size multiplier to all signals
position_size *= dd_check['position_size_multiplier']

# After each trade closes
self.circuit_breaker.update_balance(new_balance)
```

**Configuration** (`.env`):
```bash
# Drawdown Circuit Breaker
CIRCUIT_BREAKER_ENABLED=true
DAILY_LOSS_WARNING=0.03        # 3% warning threshold
DAILY_LOSS_EMERGENCY=0.05      # 5% emergency stop
TOTAL_LOSS_SHUTDOWN=0.09       # 9% full shutdown (FTMO)
STARTING_BALANCE=5000.0        # Initial balance
```

**Expected Impact**:
- **Max Drawdown**: -30-50% reduction (stops spiral)
- **Risk Management**: Automatic position sizing adjustment
- **FTMO Compliance**: Enforces daily/total loss limits
- **Peace of Mind**: System can't blow up account

---

## 📊 Layer 3: Data Enhancers (Data & Memory)

These modules provide **comprehensive feedback loops** for the agent to learn and improve.

### Module 3: Correlation Manager (Enhanced)

**File**: `libs/risk/correlation_manager.py` (replaces correlation_analyzer.py)

**Purpose**: Prevent correlated position stacking

**Problem**:
- Current correlation analyzer is basic (30-day rolling)
- Doesn't dynamically adjust to market conditions
- No real-time correlation monitoring
- Doesn't account for volatility regime

**Enhanced Solution**:
```
Multi-Timeframe Correlation:
├── 30-day rolling (long-term relationship)
├── 7-day rolling (recent regime)
├── 1-day rolling (current session)
└── Volatility-adjusted correlation

Dynamic Threshold:
├── High vol regime: Lower threshold (0.6)
├── Normal regime: Standard threshold (0.7)
├── Low vol regime: Higher threshold (0.8)

Position Overlap Check:
├── Count open positions per asset class
├── Calculate portfolio beta (BTC exposure)
├── Block if >70% capital in correlated assets
```

**Enhanced Implementation**:

```python
class CorrelationManager:
    """
    Enhanced correlation management with dynamic thresholds

    Features:
    - Multi-timeframe correlation (1d, 7d, 30d)
    - Volatility-adjusted thresholds
    - Portfolio beta calculation
    - Asset class diversification
    - Real-time correlation monitoring
    """

    def __init__(
        self,
        base_threshold: float = 0.7,
        lookback_periods: List[int] = [1, 7, 30]  # Days
    ):
        self.base_threshold = base_threshold
        self.lookback_periods = lookback_periods

        # Correlation matrices (multi-timeframe)
        self.corr_matrices = {
            f'{period}d': None
            for period in lookback_periods
        }

        # Asset class mapping
        self.asset_classes = {
            'BTC-USD': 'crypto_large_cap',
            'ETH-USD': 'crypto_large_cap',
            'SOL-USD': 'crypto_mid_cap',
            'XRP-USD': 'crypto_mid_cap',
            'DOGE-USD': 'crypto_meme',
            'ADA-USD': 'crypto_mid_cap',
            'AVAX-USD': 'crypto_mid_cap',
            'LINK-USD': 'crypto_mid_cap',
            'POL-USD': 'crypto_mid_cap',
            'LTC-USD': 'crypto_large_cap'
        }

    def check_new_position(
        self,
        new_symbol: str,
        new_direction: str,
        open_positions: List[Dict[str, Any]],
        market_volatility: str = 'normal'
    ) -> Dict[str, Any]:
        """
        Check if new position is acceptable given correlations

        Returns:
            {
                'allowed': True/False,
                'reason': 'High correlation with BTC-USD (0.95)',
                'correlation_details': {
                    'BTC-USD': {
                        '1d_corr': 0.98,
                        '7d_corr': 0.95,
                        '30d_corr': 0.92,
                        'avg_corr': 0.95
                    }
                },
                'portfolio_beta': 0.85,
                'asset_class_exposure': {
                    'crypto_large_cap': 0.65,
                    'crypto_mid_cap': 0.20
                },
                'recommendation': 'Block - Too correlated with BTC'
            }
        """

        # Get dynamic threshold based on volatility
        threshold = self._get_dynamic_threshold(market_volatility)

        # Calculate correlations with open positions
        corr_details = {}
        max_correlation = 0.0
        blocking_symbol = None

        for pos in open_positions:
            if pos['direction'] != new_direction:
                continue  # Opposite direction = hedge, allow

            # Multi-timeframe correlation
            corr_1d = self._get_correlation(new_symbol, pos['symbol'], '1d')
            corr_7d = self._get_correlation(new_symbol, pos['symbol'], '7d')
            corr_30d = self._get_correlation(new_symbol, pos['symbol'], '30d')

            # Weighted average (recent > long-term)
            avg_corr = (0.5 * corr_1d + 0.3 * corr_7d + 0.2 * corr_30d)

            corr_details[pos['symbol']] = {
                '1d_corr': corr_1d,
                '7d_corr': corr_7d,
                '30d_corr': corr_30d,
                'avg_corr': avg_corr
            }

            if avg_corr > max_correlation:
                max_correlation = avg_corr
                blocking_symbol = pos['symbol']

        # Check threshold
        if max_correlation > threshold:
            return {
                'allowed': False,
                'reason': f'High correlation with {blocking_symbol} ({max_correlation:.2f} > {threshold:.2f})',
                'correlation_details': corr_details,
                'max_correlation': max_correlation,
                'threshold': threshold,
                'recommendation': f'Block - Diversify away from {blocking_symbol}'
            }

        # Check asset class exposure
        asset_class_check = self._check_asset_class_exposure(
            new_symbol, open_positions
        )

        if not asset_class_check['allowed']:
            return asset_class_check

        # Check portfolio beta (BTC exposure)
        beta_check = self._check_portfolio_beta(
            new_symbol, new_direction, open_positions
        )

        if not beta_check['allowed']:
            return beta_check

        # All checks passed
        return {
            'allowed': True,
            'reason': 'Diversification criteria met',
            'correlation_details': corr_details,
            'max_correlation': max_correlation,
            'threshold': threshold,
            'recommendation': 'Approve - Good diversification'
        }

    def _get_dynamic_threshold(self, market_volatility: str) -> float:
        """
        Adjust correlation threshold based on market volatility

        High vol: Lower threshold (0.6) - Allow more correlated trades
        Normal: Base threshold (0.7)
        Low vol: Higher threshold (0.8) - Require more diversification
        """
        if market_volatility == 'high':
            return self.base_threshold - 0.1  # 0.6
        elif market_volatility == 'low':
            return self.base_threshold + 0.1  # 0.8
        else:
            return self.base_threshold  # 0.7

    def _check_asset_class_exposure(
        self,
        new_symbol: str,
        open_positions: List[Dict]
    ) -> Dict[str, Any]:
        """
        Check if adding position violates asset class limits

        Limits:
        - Max 70% in single asset class
        - Max 50% in meme coins
        """
        # Calculate current exposure
        exposure = {}
        total_positions = len(open_positions)

        for pos in open_positions:
            asset_class = self.asset_classes.get(pos['symbol'], 'unknown')
            exposure[asset_class] = exposure.get(asset_class, 0) + 1

        # Add new position
        new_class = self.asset_classes.get(new_symbol, 'unknown')
        exposure[new_class] = exposure.get(new_class, 0) + 1

        # Calculate percentages
        exposure_pct = {
            k: v / (total_positions + 1)
            for k, v in exposure.items()
        }

        # Check limits
        if exposure_pct.get(new_class, 0) > 0.7:
            return {
                'allowed': False,
                'reason': f'{new_class} exposure would be {exposure_pct[new_class]:.1%} > 70%',
                'asset_class_exposure': exposure_pct,
                'recommendation': 'Block - Over-concentrated in asset class'
            }

        if new_class == 'crypto_meme' and exposure_pct.get(new_class, 0) > 0.5:
            return {
                'allowed': False,
                'reason': f'Meme coin exposure would be {exposure_pct[new_class]:.1%} > 50%',
                'asset_class_exposure': exposure_pct,
                'recommendation': 'Block - Too much meme coin exposure'
            }

        return {
            'allowed': True,
            'asset_class_exposure': exposure_pct
        }

    def _check_portfolio_beta(
        self,
        new_symbol: str,
        new_direction: str,
        open_positions: List[Dict]
    ) -> Dict[str, Any]:
        """
        Check portfolio beta (exposure to BTC)

        Beta = Sum(Position_i * Correlation_i_with_BTC)

        Limit: Total portfolio beta < 2.0 (200% BTC exposure)
        """
        # Calculate current beta
        current_beta = 0.0

        for pos in open_positions:
            btc_corr = self._get_correlation(pos['symbol'], 'BTC-USD', '7d')

            # Beta contribution
            if pos['direction'] == 'long':
                current_beta += btc_corr
            else:  # short
                current_beta -= btc_corr

        # Add new position beta
        new_btc_corr = self._get_correlation(new_symbol, 'BTC-USD', '7d')

        if new_direction == 'long':
            projected_beta = current_beta + new_btc_corr
        else:
            projected_beta = current_beta - new_btc_corr

        # Check limit
        if abs(projected_beta) > 2.0:
            return {
                'allowed': False,
                'reason': f'Portfolio beta would be {projected_beta:.2f} > 2.0',
                'current_beta': current_beta,
                'projected_beta': projected_beta,
                'recommendation': 'Block - Over-leveraged to BTC'
            }

        return {
            'allowed': True,
            'portfolio_beta': projected_beta
        }
```

**Expected Impact**:
- **Diversification**: Better spread across assets
- **Risk Reduction**: Lower portfolio volatility
- **Win Rate**: +2-3 points (avoiding correlated losses)

---

### Module 4: Rejection Logger (New)

**File**: `libs/tracking/rejection_logger.py`

**Purpose**: Log every rejected signal for learning

**Problem**:
- Current system only tracks executed trades
- No data on WHY signals were rejected
- Can't learn from "what not to trade"
- No analysis of false positives

**Solution**:
```
Rejection Database:
├── Every rejected signal logged
├── Rejection reason (regime, R:R, correlation, etc.)
├── Market conditions at rejection
├── Counterfactual tracking (what would have happened?)
└── Pattern analysis (most common rejections)

Learning Loop:
1. Log rejection with full context
2. Track hypothetical outcome
3. Analyze if rejection was correct
4. Tune filters based on results
```

**Implementation**:

```python
class RejectionLogger:
    """
    Log all rejected signals for analysis and learning

    Purpose:
    - Track why signals are rejected
    - Analyze if rejections were correct (counterfactual)
    - Identify over-filtering (missed opportunities)
    - Provide feedback for filter tuning
    """

    def __init__(self, db_path: str = "tradingai.db"):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        """Create rejections table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS signal_rejections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            confidence REAL NOT NULL,

            -- Rejection details
            rejection_reason TEXT NOT NULL,
            rejection_category TEXT NOT NULL,
            rejection_details JSON,

            -- Market context
            regime TEXT,
            volatility REAL,
            trend_strength REAL,

            -- Counterfactual (what would have happened)
            entry_price REAL,
            hypothetical_sl REAL,
            hypothetical_tp REAL,

            -- Outcome tracking
            tracked BOOLEAN DEFAULT 1,
            outcome TEXT,  -- 'would_win', 'would_lose', 'unknown'
            hypothetical_pnl REAL,
            rejection_correct BOOLEAN,

            -- Theory scores (for debugging)
            theory_scores JSON,

            INDEX idx_symbol_timestamp ON signal_rejections(symbol, timestamp),
            INDEX idx_rejection_reason ON signal_rejections(rejection_reason),
            INDEX idx_outcome ON signal_rejections(outcome)
        )
        """)

        conn.commit()
        conn.close()

    def log_rejection(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        rejection_reason: str,
        rejection_category: str,
        rejection_details: Dict[str, Any],
        market_context: Dict[str, Any],
        theory_scores: Dict[str, float],
        hypothetical_prices: Dict[str, float]
    ):
        """
        Log a rejected signal

        Args:
            symbol: Trading pair
            direction: LONG/SHORT
            confidence: Signal confidence
            rejection_reason: Why it was rejected
            rejection_category: 'regime' | 'correlation' | 'rr_ratio' | 'drawdown'
            rejection_details: Full details dict
            market_context: Regime, volatility, etc.
            theory_scores: All 11 theory scores
            hypothetical_prices: Entry, SL, TP if it had traded
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO signal_rejections (
            symbol, direction, confidence,
            rejection_reason, rejection_category, rejection_details,
            regime, volatility, trend_strength,
            entry_price, hypothetical_sl, hypothetical_tp,
            theory_scores
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            direction,
            confidence,
            rejection_reason,
            rejection_category,
            json.dumps(rejection_details),
            market_context.get('regime'),
            market_context.get('volatility'),
            market_context.get('trend_strength'),
            hypothetical_prices.get('entry'),
            hypothetical_prices.get('sl'),
            hypothetical_prices.get('tp'),
            json.dumps(theory_scores)
        ))

        rejection_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"Logged rejection #{rejection_id}: {symbol} {direction} - {rejection_reason}")

        return rejection_id

    def track_counterfactual(
        self,
        rejection_id: int,
        outcome: str,
        hypothetical_pnl: float
    ):
        """
        Update rejection with what actually happened

        Args:
            rejection_id: ID from log_rejection
            outcome: 'would_win' | 'would_lose' | 'would_hold'
            hypothetical_pnl: What P&L would have been
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Determine if rejection was correct
        # Rejection correct if: would_lose OR would_hold (no clear outcome)
        rejection_correct = outcome in ['would_lose', 'would_hold']

        cursor.execute("""
        UPDATE signal_rejections
        SET outcome = ?,
            hypothetical_pnl = ?,
            rejection_correct = ?
        WHERE id = ?
        """, (outcome, hypothetical_pnl, rejection_correct, rejection_id))

        conn.commit()
        conn.close()

    def analyze_rejections(
        self,
        time_period: str = '7d'
    ) -> Dict[str, Any]:
        """
        Analyze rejection patterns

        Returns:
            {
                'total_rejections': 45,
                'by_category': {
                    'regime': 20,
                    'correlation': 10,
                    'rr_ratio': 8,
                    'drawdown': 7
                },
                'rejection_accuracy': {
                    'regime': 0.85,  # 85% of regime rejections were correct
                    'correlation': 0.70,
                    'rr_ratio': 0.60,  # 60% correct (may be over-filtering)
                    'drawdown': 0.95
                },
                'missed_opportunities': [
                    {
                        'rejection_id': 123,
                        'symbol': 'BTC-USD',
                        'reason': 'R:R too low (3.2:1)',
                        'hypothetical_pnl': +1.5,  # Would have won +1.5%
                        'lesson': 'R:R threshold too strict for BTC'
                    }
                ],
                'correct_rejections': [
                    {
                        'rejection_id': 125,
                        'symbol': 'ETH-USD',
                        'reason': 'Chop detected (ADX=15)',
                        'hypothetical_pnl': -0.8,  # Would have lost
                        'lesson': 'Regime filter working well'
                    }
                ],
                'recommendations': [
                    'Lower R:R threshold for BTC (high accuracy asset)',
                    'Regime filter is working excellently (85% accuracy)',
                    'Consider relaxing correlation for opposite directions'
                ]
            }
        """
        conn = sqlite3.connect(self.db_path)

        # Get rejections
        df = pd.read_sql(f"""
        SELECT *
        FROM signal_rejections
        WHERE timestamp > datetime('now', '-{time_period}')
        AND outcome IS NOT NULL
        """, conn)

        conn.close()

        if df.empty:
            return {'total_rejections': 0}

        # Analyze by category
        by_category = df['rejection_category'].value_counts().to_dict()

        # Calculate accuracy by category
        rejection_accuracy = {}
        for category in by_category.keys():
            cat_df = df[df['rejection_category'] == category]
            accuracy = cat_df['rejection_correct'].mean()
            rejection_accuracy[category] = accuracy

        # Find missed opportunities (rejection wrong, would have won)
        missed = df[
            (df['rejection_correct'] == False) &
            (df['hypothetical_pnl'] > 0)
        ].sort_values('hypothetical_pnl', ascending=False)

        missed_opportunities = []
        for _, row in missed.head(5).iterrows():
            missed_opportunities.append({
                'rejection_id': row['id'],
                'symbol': row['symbol'],
                'reason': row['rejection_reason'],
                'hypothetical_pnl': row['hypothetical_pnl'],
                'lesson': f'{row["rejection_category"]} may be over-filtering {row["symbol"]}'
            })

        # Find correct rejections (saved from losses)
        correct = df[
            (df['rejection_correct'] == True) &
            (df['hypothetical_pnl'] < 0)
        ].sort_values('hypothetical_pnl')

        correct_rejections = []
        for _, row in correct.head(5).iterrows():
            correct_rejections.append({
                'rejection_id': row['id'],
                'symbol': row['symbol'],
                'reason': row['rejection_reason'],
                'hypothetical_pnl': row['hypothetical_pnl'],
                'lesson': f'{row["rejection_category"]} saved from {row["hypothetical_pnl"]:.1%} loss'
            })

        # Generate recommendations
        recommendations = self._generate_recommendations(
            rejection_accuracy,
            missed_opportunities,
            correct_rejections
        )

        return {
            'total_rejections': len(df),
            'by_category': by_category,
            'rejection_accuracy': rejection_accuracy,
            'missed_opportunities': missed_opportunities,
            'correct_rejections': correct_rejections,
            'recommendations': recommendations
        }

    def _generate_recommendations(
        self,
        accuracy: Dict[str, float],
        missed: List[Dict],
        correct: List[Dict]
    ) -> List[str]:
        """Generate tuning recommendations"""
        recommendations = []

        # Check each category accuracy
        for category, acc in accuracy.items():
            if acc < 0.6:
                # Over-filtering (less than 60% correct)
                recommendations.append(
                    f"⚠️  {category} filter may be too strict ({acc:.0%} accuracy). "
                    f"Consider relaxing threshold."
                )
            elif acc > 0.85:
                # Working well
                recommendations.append(
                    f"✅ {category} filter working excellently ({acc:.0%} accuracy)"
                )

        # Check for asset-specific patterns
        if missed:
            common_symbols = {}
            for m in missed:
                symbol = m['symbol']
                common_symbols[symbol] = common_symbols.get(symbol, 0) + 1

            for symbol, count in common_symbols.items():
                if count >= 2:
                    recommendations.append(
                        f"💡 Consider relaxing filters for {symbol} "
                        f"({count} missed opportunities)"
                    )

        return recommendations
```

**Integration**:
```python
# In signal_generator.py, whenever rejecting a signal

self.rejection_logger.log_rejection(
    symbol=symbol,
    direction=direction,
    confidence=confidence,
    rejection_reason="Chop detected (ADX=15)",
    rejection_category="regime",
    rejection_details=regime_check,
    market_context={
        'regime': regime_check['regime'],
        'volatility': atr_value,
        'trend_strength': adx_value
    },
    theory_scores=all_theory_scores,
    hypothetical_prices={
        'entry': current_price,
        'sl': calculate_sl(current_price),
        'tp': calculate_tp(current_price)
    }
)

# Background task: Track counterfactuals
# (Check if hypothetical SL/TP would have been hit)
```

**Expected Impact**:
- **Filter Tuning**: Data-driven threshold optimization
- **Learning**: Understand what not to trade
- **Missed Opportunities**: Identify over-filtering
- **Agent Intelligence**: Feed rejection data to agent for self-improvement

---

## 📊 Integration Architecture

```
                    Signal Generated
                         ↓
           ┌─────────────────────────┐
           │   Safety Guard Layer    │
           └─────────────────────────┘
                         │
            ┌────────────┼────────────┐
            │                         │
    ┌───────▼────────┐    ┌──────────▼──────────┐
    │ Market Regime  │    │ Drawdown Circuit   │
    │ Detector       │    │ Breaker            │
    │                │    │                    │
    │ Check: ADX,    │    │ Check: Daily loss  │
    │  ATR, BB width │    │   Total drawdown   │
    │                │    │                    │
    │ Block if chop  │    │ Block if exceed    │
    └────────┬───────┘    └──────────┬─────────┘
             │ Passed                │ Passed
             └────────────┬──────────┘
                          │
                ┌─────────▼──────────┐
                │ Correlation        │
                │ Manager            │
                │                    │
                │ Check: Multi-TF    │
                │   correlation      │
                │ Asset class limits │
                │ Portfolio beta     │
                │                    │
                │ Block if too       │
                │ correlated         │
                └─────────┬──────────┘
                          │ Passed
                          │
                ┌─────────▼──────────┐
                │ Execute Signal     │
                │ (Paper Trading)    │
                └─────────┬──────────┘
                          │
                ┌─────────▼──────────┐
                │ Track Result       │
                │ (Performance DB)   │
                └────────────────────┘

   Rejected at any stage?
              ↓
   ┌──────────────────────┐
   │ Rejection Logger     │
   │                      │
   │ - Log reason         │
   │ - Track hypothetical │
   │ - Analyze patterns   │
   │ - Tune filters       │
   └──────────────────────┘
```

---

## 📈 Expected Performance Impact

### Current V7 (Before Safety Guards)
```
Win Rate: 53.8%
Max Drawdown: ~15%+
Trade Frequency: 3-10/day
Choppy market losses: 30-40%
Correlated losses: 15-20%
```

### V7 + Safety Guards
```
Win Rate: 60-65% (+6-12 points)
  - Market Regime Filter: +5-8 points
  - Correlation Manager: +2-3 points
  - Circuit Breaker: -2-3 max DD points

Max Drawdown: <8% (-50% reduction)
  - Circuit breaker prevents spirals
  - Position sizing adjustment at -3%

Trade Frequency: 2-7/day (-30%)
  - More selective (quality > quantity)

Risk-Adjusted Returns:
  - Sharpe: 1.0-1.2 → 2.0-2.5
  - Expectancy: +0.42% → +1.0-1.5%
```

### Learning Loop Enhancement
```
Rejection Database:
- 100+ rejections/week logged
- Counterfactual tracking
- Filter accuracy metrics
- Data-driven tuning

Agent Enhancement:
- Feed rejection data to agent
- Auto-tune thresholds
- Identify over-filtering
- Continuous improvement
```

---

## ✅ Implementation Checklist

### Week 1: Safety Guards
- [ ] Market Regime Detector (~400 lines)
  - ADX, ATR, BB width calculations
  - Regime classification logic
  - Integration with signal generator

- [ ] Drawdown Circuit Breaker (~300 lines)
  - Real-time balance monitoring
  - Multi-level thresholds
  - Alert system integration

### Week 2: Data Enhancers
- [ ] Correlation Manager Enhanced (~500 lines)
  - Multi-timeframe correlation
  - Dynamic thresholds
  - Portfolio beta calculation
  - Asset class limits

- [ ] Rejection Logger (~400 lines)
  - Database schema
  - Rejection logging
  - Counterfactual tracking
  - Analysis & recommendations

### Week 3: Integration & Testing
- [ ] Integrate all modules into V7
- [ ] End-to-end testing
- [ ] Backtest with historical data
- [ ] A/B test vs current V7

### Week 4: Production & Monitoring
- [ ] Deploy to production
- [ ] Monitor rejection patterns
- [ ] Tune thresholds based on data
- [ ] Document learnings

**Total Lines**: ~1,600 lines
**Timeline**: 4 weeks
**Priority**: High (critical for production safety)

---

## 🎯 Success Criteria

**Minimum Targets** (to validate safety guards):
- Win rate: > 58% (vs 53.8% baseline)
- Max drawdown: < 10% (vs 15%+ baseline)
- Filter accuracy: > 70% (rejections are correct)
- No circuit breaker Level 3 triggers (no account blown)

**Optimal Targets**:
- Win rate: 60-65%
- Max drawdown: < 8%
- Filter accuracy: > 80%
- Sharpe ratio: > 2.0

**Go/No-Go Decision** (after 30+ trades):
```
IF win_rate > 58% AND max_dd < 10%:
    → Success, keep all filters
ELSE IF win_rate > 55%:
    → Partial success, tune thresholds
ELSE:
    → Review filters, may be over-restricting
```

---

**Last Updated**: 2025-11-24
**Status**: Design Complete, Implementation Pending
**Priority**: High (Production Safety Critical)
**Integration**: Fits between Phase 1 (Risk Mgmt) and Phase 6 (Agent)
**Timeline**: 4 weeks to full integration
