"""
Base FTMO Bot Class

Abstract base class for all FTMO challenge bots.
Provides common functionality:
- MT5 signal routing via Windows VPS (ZMQ)
- Risk management (1.5% per trade)
- Position sizing
- Trade logging
- Telegram alerts
"""

import os
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger

# Import ZMQ client for MT5 communication
try:
    from libs.brokers.mt5_zmq_client import get_mt5_client, MT5ZMQClient
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    logger.warning("MT5 ZMQ client not available")

# Import knowledge query for sentiment/risk analysis
try:
    from libs.knowledge.query import get_trading_context, RiskLevel
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False
    logger.info("Knowledge system not available - trading without sentiment data")

# Import Certainty Engine for multi-factor confidence scoring
try:
    from libs.hydra.certainty_engine import get_certainty_engine, CertaintyResult
    CERTAINTY_AVAILABLE = True
except ImportError:
    CERTAINTY_AVAILABLE = False
    logger.info("Certainty Engine not available - trading without certainty scoring")

# Import Technical Utils for RSI/Z-score calculation (Phase 3 - 2025-12-11)
try:
    from libs.hydra.ftmo_bots.technical_utils import TechnicalIndicators, get_market_indicators
    TECHNICAL_UTILS_AVAILABLE = True
except ImportError:
    TECHNICAL_UTILS_AVAILABLE = False
    logger.info("Technical Utils not available - trading without RSI/z-score")


@dataclass
class BotConfig:
    """Configuration for FTMO bot."""
    bot_name: str
    symbol: str  # MT5 symbol (e.g., "XAUUSD", "EURUSD")
    # Phase 9 (2025-12-11): Reduced from 1.5% to 0.75% per research
    # Source: Prop firm research shows 0.5-1% risk = 75% pass rate improvement
    risk_percent: float = 0.0075  # 0.75% risk per trade (was 1.5%)
    max_daily_trades: int = 3
    stop_loss_pips: float = 50.0
    take_profit_pips: float = 90.0
    max_hold_hours: float = 2.0
    enabled: bool = True
    paper_mode: bool = True  # Set False for live trading
    turbo_mode: bool = False  # Turbo mode: loosen thresholds for more trades

    # ATR-based dynamic stop losses (Phase 4 - 2025-12-11)
    use_atr_stops: bool = True  # Use ATR for SL/TP instead of fixed pips
    atr_sl_multiplier: float = 1.5  # SL = 1.5x ATR
    atr_tp_multiplier: float = 2.0  # TP = 2.0x ATR (R:R = 1.33)
    atr_min_sl_pips: float = 20.0  # Minimum SL in pips (floor)
    atr_max_sl_pips: float = 100.0  # Maximum SL in pips (ceiling)

    def get_turbo_multiplier(self) -> float:
        """Get threshold multiplier for turbo mode (looser = more trades)."""
        return 0.5 if self.turbo_mode else 1.0

    def get_turbo_max_trades(self) -> int:
        """Get max daily trades for turbo mode (3x normal)."""
        return self.max_daily_trades * 3 if self.turbo_mode else self.max_daily_trades


@dataclass
class TradeSignal:
    """Trade signal from bot."""
    bot_name: str
    symbol: str
    direction: str  # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    reason: str
    confidence: float = 0.70
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Technical indicators for filtered shorts (Phase 3 - 2025-12-11)
    rsi: Optional[float] = None  # RSI value (0-100)
    zscore: Optional[float] = None  # Price z-score (std devs from mean)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bot_name": self.bot_name,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "lot_size": self.lot_size,
            "reason": self.reason,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "rsi": self.rsi,
            "zscore": self.zscore,
        }


class BaseFTMOBot(ABC):
    """
    Abstract base class for FTMO trading bots.

    All bots must implement:
    - analyze(): Analyze market conditions
    - should_trade(): Check if trading conditions are met
    - generate_signal(): Generate trade signal if conditions met
    """

    # ZMQ client (singleton) for MT5 communication
    _zmq_client: Optional['MT5ZMQClient'] = None

    # Certainty Engine integration (added 2025-12-11)
    USE_CERTAINTY_ENGINE = True  # Set False to bypass certainty checks
    CERTAINTY_THRESHOLD = 0.65  # Lower than engine's 80% - bot-level confidence matters too

    # Session-aware position sizing (added 2025-12-11)
    # Session overlap (13:00-16:00 UTC) has 42.9% WR - reduce exposure
    USE_SESSION_SIZING = True
    SESSION_MULTIPLIERS = {
        "london": 1.0,       # London (08:00-13:00 UTC) - best session
        "overlap": 0.8,      # London-NY overlap (13:00-16:00 UTC) - reduce 20%
        "ny": 0.9,           # NY session (16:00-21:00 UTC) - slight reduction
        "asian": 0.7,        # Asian session (22:00-08:00 UTC) - lower volume
        "off_hours": 0.5,    # Off hours (21:00-22:00 UTC) - avoid if possible
    }

    # Phase 8: R:R Validation (2025-12-11)
    # Reject trades with risk/reward ratio < 1.5:1
    USE_RR_VALIDATION = True
    MIN_RR_RATIO = 1.5  # Minimum acceptable risk-reward ratio

    def __init__(self, config: BotConfig):
        self.config = config
        self._lock = threading.Lock()
        self._daily_trades: list = []
        self._last_signal: Optional[TradeSignal] = None
        self._last_trade_time: Optional[datetime] = None

        logger.info(f"[{config.bot_name}] Bot initialized (symbol: {config.symbol}, risk: {config.risk_percent*100:.1f}%)")

    def validate_rr_ratio(
        self,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> Tuple[bool, float, str]:
        """
        Validate risk/reward ratio for a potential trade.

        Phase 8 (2025-12-11): Ensures minimum 1.5:1 R:R ratio to improve expectancy.

        Args:
            direction: Trade direction ("BUY" or "SELL")
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Tuple of (is_valid, rr_ratio, reason)
        """
        if not self.USE_RR_VALIDATION:
            return True, 0.0, "R:R validation disabled"

        direction = direction.upper()

        # Calculate risk and reward based on direction
        if direction == "BUY":
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:  # SELL
            risk = stop_loss - entry_price
            reward = entry_price - take_profit

        # Sanity checks
        if risk <= 0:
            return False, 0.0, f"Invalid risk: {risk:.5f} (SL on wrong side)"

        if reward <= 0:
            return False, 0.0, f"Invalid reward: {reward:.5f} (TP on wrong side)"

        rr_ratio = reward / risk

        if rr_ratio < self.MIN_RR_RATIO:
            return False, rr_ratio, f"R:R {rr_ratio:.2f} < {self.MIN_RR_RATIO} minimum"

        return True, rr_ratio, f"R:R {rr_ratio:.2f} >= {self.MIN_RR_RATIO} OK"

    @classmethod
    def get_zmq_client(cls) -> Optional['MT5ZMQClient']:
        """Get or create ZMQ client singleton."""
        if not ZMQ_AVAILABLE:
            return None
        if cls._zmq_client is None:
            cls._zmq_client = get_mt5_client()
        return cls._zmq_client

    @abstractmethod
    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market conditions for trading opportunity.

        Args:
            market_data: Dict with OHLCV data, indicators, etc.

        Returns:
            Analysis result dict
        """
        pass

    @abstractmethod
    def should_trade(self, analysis: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if trading conditions are met.

        Args:
            analysis: Result from analyze()

        Returns:
            Tuple of (should_trade, reason)
        """
        pass

    @abstractmethod
    def generate_signal(self, analysis: Dict[str, Any], current_price: float) -> Optional[TradeSignal]:
        """
        Generate trade signal if conditions are met.

        Args:
            analysis: Result from analyze()
            current_price: Current market price

        Returns:
            TradeSignal if trade should be taken, None otherwise
        """
        pass

    def get_session_multiplier(self) -> float:
        """
        Get position size multiplier based on current trading session.

        Returns:
            Multiplier (0.5 - 1.0) based on session quality
        """
        if not self.USE_SESSION_SIZING:
            return 1.0

        now = datetime.now(timezone.utc)
        hour = now.hour

        # Determine current session
        if 8 <= hour < 13:
            session = "london"
        elif 13 <= hour < 16:
            session = "overlap"
        elif 16 <= hour < 21:
            session = "ny"
        elif hour == 21:
            session = "off_hours"
        else:  # 22:00 - 08:00
            session = "asian"

        multiplier = self.SESSION_MULTIPLIERS.get(session, 1.0)

        if multiplier < 1.0:
            logger.debug(f"[{self.config.bot_name}] Session sizing: {session} = {multiplier}x")

        return multiplier

    def calculate_lot_size(self, account_balance: float, stop_loss_pips: float) -> float:
        """
        Calculate lot size based on risk percentage.

        For Gold (XAUUSD): 1 pip = $10 per standard lot (1.0)
        For Forex pairs: 1 pip = $10 per standard lot
        For Indices: varies by contract

        Args:
            account_balance: Current account balance
            stop_loss_pips: Stop loss distance in pips

        Returns:
            Lot size (e.g., 0.1 for mini lot)
        """
        risk_amount = account_balance * self.config.risk_percent

        # Apply session multiplier (added 2025-12-11)
        session_mult = self.get_session_multiplier()
        risk_amount *= session_mult

        # Pip value calculation (simplified - adjust for specific symbols)
        if self.config.symbol.startswith("XAU"):
            pip_value_per_lot = 10.0  # $10 per pip for 1.0 lot on gold
        elif self.config.symbol.endswith("USD"):
            pip_value_per_lot = 10.0  # $10 per pip for 1.0 lot on USD pairs
        elif self.config.symbol.startswith("US30") or self.config.symbol.startswith("NAS"):
            pip_value_per_lot = 1.0  # $1 per point for indices
        else:
            pip_value_per_lot = 10.0  # Default

        if stop_loss_pips <= 0:
            logger.warning(f"[{self.config.bot_name}] Invalid stop loss: {stop_loss_pips}")
            return 0.01  # Minimum lot

        lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)

        # Round to 2 decimal places and enforce min/max
        # SAFETY: Max 0.5 lots after $503 loss on 2025-12-10
        lot_size = round(lot_size, 2)
        lot_size = max(0.01, min(lot_size, 0.5))  # Min 0.01, Max 0.5 lots (conservative)

        logger.debug(
            f"[{self.config.bot_name}] Lot size: {lot_size} "
            f"(risk: ${risk_amount:.2f}, SL: {stop_loss_pips} pips)"
        )

        return lot_size

    def calculate_atr_stops(
        self,
        candles: List[Dict[str, float]],
        entry_price: float,
        direction: str
    ) -> Tuple[float, float, float]:
        """
        Calculate ATR-based stop loss and take profit (Phase 4 - 2025-12-11).

        Uses ATR to adapt SL/TP to current volatility conditions.

        Args:
            candles: List of candle dicts with 'high', 'low', 'close' keys
            entry_price: Proposed entry price
            direction: "BUY" or "SELL"

        Returns:
            Tuple of (stop_loss_price, take_profit_price, sl_pips)
        """
        if not self.config.use_atr_stops:
            # Fall back to fixed pips
            pip_multiplier = self._get_pip_multiplier()
            sl_distance = self.config.stop_loss_pips * pip_multiplier
            tp_distance = self.config.take_profit_pips * pip_multiplier

            if direction.upper() == "BUY":
                return entry_price - sl_distance, entry_price + tp_distance, self.config.stop_loss_pips
            else:
                return entry_price + sl_distance, entry_price - tp_distance, self.config.stop_loss_pips

        # Calculate ATR using technical_utils
        try:
            if TECHNICAL_UTILS_AVAILABLE:
                from libs.hydra.ftmo_bots.technical_utils import calculate_atr
                atr = calculate_atr(candles, period=14)
            else:
                atr = None
        except Exception as e:
            logger.debug(f"[{self.config.bot_name}] ATR calculation failed: {e}")
            atr = None

        if atr is None or atr <= 0:
            # Fall back to fixed pips if ATR unavailable
            logger.debug(f"[{self.config.bot_name}] ATR unavailable, using fixed stops")
            pip_multiplier = self._get_pip_multiplier()
            sl_distance = self.config.stop_loss_pips * pip_multiplier
            tp_distance = self.config.take_profit_pips * pip_multiplier

            if direction.upper() == "BUY":
                return entry_price - sl_distance, entry_price + tp_distance, self.config.stop_loss_pips
            else:
                return entry_price + sl_distance, entry_price - tp_distance, self.config.stop_loss_pips

        # Calculate ATR-based distances
        sl_distance = atr * self.config.atr_sl_multiplier
        tp_distance = atr * self.config.atr_tp_multiplier

        # Convert to pips for logging/lot sizing
        pip_multiplier = self._get_pip_multiplier()
        sl_pips = sl_distance / pip_multiplier if pip_multiplier > 0 else self.config.stop_loss_pips

        # Apply min/max limits
        sl_pips = max(self.config.atr_min_sl_pips, min(sl_pips, self.config.atr_max_sl_pips))
        sl_distance = sl_pips * pip_multiplier
        tp_distance = sl_distance * (self.config.atr_tp_multiplier / self.config.atr_sl_multiplier)

        if direction.upper() == "BUY":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        logger.debug(
            f"[{self.config.bot_name}] ATR stops: ATR={atr:.4f}, "
            f"SL={sl_pips:.1f} pips, TP={sl_pips * self.config.atr_tp_multiplier / self.config.atr_sl_multiplier:.1f} pips"
        )

        return stop_loss, take_profit, sl_pips

    def _get_pip_multiplier(self) -> float:
        """Get pip multiplier for the symbol (price distance per pip)."""
        symbol = self.config.symbol.upper()
        if symbol.startswith("XAU"):
            return 0.1  # Gold: 1 pip = $0.10 price move
        elif symbol.startswith("US30") or symbol.startswith("US100"):
            return 1.0  # Indices: 1 pip = 1 point
        elif "JPY" in symbol:
            return 0.01  # JPY pairs: 1 pip = 0.01
        else:
            return 0.0001  # Standard forex: 1 pip = 0.0001

    def can_trade_today(self) -> Tuple[bool, str]:
        """Check if bot can place more trades today."""
        # Reset daily trades at midnight UTC
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        self._daily_trades = [
            t for t in self._daily_trades
            if t.get("timestamp", datetime.min) > today_start
        ]

        max_trades = self.config.get_turbo_max_trades()
        if len(self._daily_trades) >= max_trades:
            return False, f"Max daily trades reached ({max_trades})"

        return True, "Can trade"

    def execute_signal(self, signal: TradeSignal) -> Dict[str, Any]:
        """
        Execute trade signal via MT5 on Windows VPS.

        Args:
            signal: Trade signal to execute

        Returns:
            Execution result dict
        """
        if self.config.paper_mode:
            return self._execute_paper(signal)
        else:
            return self._execute_live(signal)

    def _execute_paper(self, signal: TradeSignal) -> Dict[str, Any]:
        """Execute in paper trading mode (local simulation)."""
        logger.info(
            f"[{self.config.bot_name}] PAPER TRADE: "
            f"{signal.direction} {signal.symbol} @ {signal.entry_price:.2f} "
            f"(SL: {signal.stop_loss:.2f}, TP: {signal.take_profit:.2f}, lot: {signal.lot_size})"
        )

        # Record paper trade
        trade_record = {
            "timestamp": datetime.now(timezone.utc),
            "signal": signal.to_dict(),
            "mode": "paper",
            "status": "filled",
        }
        self._daily_trades.append(trade_record)
        self._last_signal = signal
        self._last_trade_time = datetime.now(timezone.utc)

        return {
            "success": True,
            "mode": "paper",
            "ticket": f"PAPER_{int(datetime.now().timestamp())}",
            "signal": signal.to_dict(),
        }

    def _execute_live(self, signal: TradeSignal) -> Dict[str, Any]:
        """Execute via MT5 on Windows VPS using ZMQ."""
        try:
            # KNOWLEDGE CHECK: Get sentiment, central bank risk, fear/greed
            knowledge_modifier = 1.0
            if KNOWLEDGE_AVAILABLE:
                try:
                    ctx = get_trading_context(signal.symbol, signal.direction)

                    # CRITICAL: Skip trade if central bank meeting imminent
                    if ctx.get("central_bank") and ctx["central_bank"].get("risk_level") == "critical":
                        logger.warning(
                            f"[{self.config.bot_name}] [Knowledge] Trade blocked: "
                            f"Central bank meeting imminent ({ctx['central_bank']['bank']})"
                        )
                        return {"success": False, "error": "Central bank risk too high"}

                    knowledge_modifier = ctx.get("confidence_modifier", 1.0)

                    # Log knowledge context
                    if ctx.get("warnings"):
                        logger.warning(f"[{self.config.bot_name}] [Knowledge] Warnings: {ctx['warnings']}")
                    if ctx.get("recommendations"):
                        logger.info(f"[{self.config.bot_name}] [Knowledge] Recommendations: {ctx['recommendations']}")

                    logger.info(
                        f"[{self.config.bot_name}] [Knowledge] Modifier: {knowledge_modifier:.2f} "
                        f"(sentiment: {ctx.get('sentiment', {}).get('bias', 'N/A')}, "
                        f"CB: {ctx.get('central_bank', {}).get('risk_level', 'N/A')}, "
                        f"F&G: {ctx.get('fear_greed', {}).get('classification', 'N/A')})"
                    )

                    # Adjust lot size based on knowledge
                    if knowledge_modifier < 1.0:
                        original_lot = signal.lot_size
                        signal.lot_size = round(signal.lot_size * knowledge_modifier, 2)
                        signal.lot_size = max(0.01, signal.lot_size)  # Minimum 0.01
                        logger.info(
                            f"[{self.config.bot_name}] [Knowledge] Lot adjusted: "
                            f"{original_lot} → {signal.lot_size} (mod: {knowledge_modifier:.2f})"
                        )

                except Exception as e:
                    logger.debug(f"[{self.config.bot_name}] Knowledge check failed: {e}")

            # SELL FILTER: Block SELL for most bots, but allow proven performers
            # gold_london has 71% WR on SELL (7 trades) - allow it
            SELL_WHITELIST = ["GoldLondonReversal"]  # Bots with proven SELL performance

            if signal.direction.upper() in ("SELL", "SHORT"):
                if self.config.bot_name not in SELL_WHITELIST:
                    logger.warning(
                        f"[{self.config.bot_name}] [SellFilter] Trade blocked: "
                        f"{signal.direction} {signal.symbol} - not in SELL whitelist"
                    )
                    return {"success": False, "error": "SELL trades disabled for this bot"}
                else:
                    logger.info(
                        f"[{self.config.bot_name}] [SellFilter] SELL allowed (whitelisted)"
                    )

            # POSITION LIMIT: Max 1 position per symbol to prevent concentration
            # Added 2025-12-10 after $503 loss from 4x EURUSD positions
            MAX_POSITIONS_PER_SYMBOL = 1
            client = self.get_zmq_client()
            if client:
                try:
                    # get_positions() returns a LIST, not a dict
                    positions = client.get_positions()
                    if positions and isinstance(positions, list) and len(positions) > 0:
                        symbol_count = sum(
                            1 for p in positions
                            if isinstance(p, dict) and p.get("symbol") == signal.symbol
                        )
                        if symbol_count >= MAX_POSITIONS_PER_SYMBOL:
                            logger.warning(
                                f"[{self.config.bot_name}] [PositionLimit] Trade blocked: "
                                f"Already {symbol_count} position(s) on {signal.symbol} (max {MAX_POSITIONS_PER_SYMBOL})"
                            )
                            return {"success": False, "error": f"Max positions reached for {signal.symbol}"}
                except Exception as e:
                    logger.warning(f"[{self.config.bot_name}] Position check failed: {e}")

            client = self.get_zmq_client()
            if not client:
                logger.error(f"[{self.config.bot_name}] ZMQ client not available")
                return {"success": False, "error": "ZMQ client not available"}

            logger.info(
                f"[{self.config.bot_name}] LIVE TRADE: "
                f"{signal.direction} {signal.symbol} @ {signal.entry_price:.2f}"
            )

            result = client.trade(
                symbol=signal.symbol,
                direction=signal.direction,
                volume=signal.lot_size,
                sl=signal.stop_loss,
                tp=signal.take_profit,
                comment=f"HYDRA_{signal.bot_name}"
            )

            if result.get("success"):
                logger.success(
                    f"[{self.config.bot_name}] Trade executed: ticket {result.get('ticket')}"
                )
                self._daily_trades.append({
                    "timestamp": datetime.now(timezone.utc),
                    "signal": signal.to_dict(),
                    "result": result,
                    "mode": "live",
                })
                self._last_signal = signal
                self._last_trade_time = datetime.now(timezone.utc)
            else:
                logger.error(
                    f"[{self.config.bot_name}] Trade failed: {result.get('error')}"
                )

            return result

        except Exception as e:
            logger.error(f"[{self.config.bot_name}] MT5 execution error: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def get_account_info(self) -> Dict[str, Any]:
        """Get account info from MT5 via ZMQ."""
        try:
            client = self.get_zmq_client()
            if client:
                result = client.get_account()
                if result and result.get("balance"):
                    return result
            # Do NOT use hardcoded fallback - better to skip trade than use wrong balance
            logger.warning("ZMQ client not available or account info unavailable - trade will be skipped")
            return None  # Caller must check for None and skip trade
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None  # Safer to skip trade than use wrong balance

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from MT5 via ZMQ."""
        try:
            client = self.get_zmq_client()
            if client:
                result = client.get_price(symbol)
                if result:
                    return result.get("bid")
            logger.warning(f"ZMQ client not available for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None

    def run_cycle(self, market_data: Optional[Dict[str, Any]] = None) -> Optional[TradeSignal]:
        """
        Run one trading cycle.

        Args:
            market_data: Optional market data (fetches if not provided)

        Returns:
            TradeSignal if trade was generated, None otherwise
        """
        if not self.config.enabled:
            return None

        # Check if we can trade today
        can_trade, reason = self.can_trade_today()
        if not can_trade:
            logger.debug(f"[{self.config.bot_name}] {reason}")
            return None

        # Get market data if not provided
        if market_data is None:
            market_data = self._fetch_market_data()

        # Analyze market
        analysis = self.analyze(market_data)

        # Check if should trade
        should_trade, reason = self.should_trade(analysis)
        if not should_trade:
            logger.debug(f"[{self.config.bot_name}] No trade: {reason}")
            return None

        # Get current price
        current_price = self.get_current_price(self.config.symbol)
        if current_price is None:
            logger.warning(f"[{self.config.bot_name}] Could not get price")
            return None

        # Generate signal
        signal = self.generate_signal(analysis, current_price)
        if signal is None:
            return None

        # Calculate and attach RSI/Z-score for filtered shorts (Phase 3 - 2025-12-11)
        if TECHNICAL_UTILS_AVAILABLE:
            try:
                candles = market_data.get("candles", [])
                if candles and len(candles) >= 20:
                    indicators = get_market_indicators(candles)
                    signal.rsi = indicators.rsi
                    signal.zscore = indicators.zscore

                    # Log for SELL signals (helps debug filtered shorts)
                    if signal.direction.upper() == "SELL":
                        logger.info(
                            f"[{self.config.bot_name}] SELL signal indicators: "
                            f"RSI={signal.rsi}, z-score={signal.zscore}"
                        )
            except Exception as e:
                logger.debug(f"[{self.config.bot_name}] Technical indicator calc failed: {e}")

        # Certainty Engine check (added 2025-12-11)
        if self.USE_CERTAINTY_ENGINE and CERTAINTY_AVAILABLE:
            try:
                candles = market_data.get("candles", [])
                if candles and len(candles) >= 20:
                    certainty_engine = get_certainty_engine()
                    certainty_result = certainty_engine.calculate_certainty(
                        symbol=signal.symbol,
                        direction=signal.direction,
                        candles=candles,
                        current_price=current_price,
                    )

                    # Log certainty score
                    logger.info(
                        f"[{self.config.bot_name}] [Certainty] Score: {certainty_result.total_score:.0%} "
                        f"(tech={certainty_result.factors.technical_confluence:.0%}, "
                        f"struct={certainty_result.factors.market_structure:.0%})"
                    )

                    # Check if certainty is too low
                    if certainty_result.total_score < self.CERTAINTY_THRESHOLD:
                        logger.warning(
                            f"[{self.config.bot_name}] [Certainty] Trade blocked: "
                            f"{certainty_result.total_score:.0%} < {self.CERTAINTY_THRESHOLD:.0%} threshold"
                        )
                        return None

                    # Update signal confidence with certainty score
                    signal.confidence = min(signal.confidence, certainty_result.total_score)
            except Exception as e:
                logger.debug(f"[{self.config.bot_name}] Certainty check failed: {e}")

        # Execute signal
        result = self.execute_signal(signal)

        if result.get("success"):
            return signal

        return None

    def _fetch_market_data(self) -> Dict[str, Any]:
        """Fetch market data from MT5 or data provider."""
        # Subclasses can override for specific data needs
        return {}

    def get_status(self) -> Dict[str, Any]:
        """Get bot status summary."""
        return {
            "bot_name": self.config.bot_name,
            "symbol": self.config.symbol,
            "enabled": self.config.enabled,
            "paper_mode": self.config.paper_mode,
            "daily_trades": len(self._daily_trades),
            "max_daily_trades": self.config.max_daily_trades,
            "last_trade_time": self._last_trade_time.isoformat() if self._last_trade_time else None,
        }
