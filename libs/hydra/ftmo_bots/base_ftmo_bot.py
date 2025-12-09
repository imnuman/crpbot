"""
Base FTMO Bot Class

Abstract base class for all FTMO challenge bots.
Provides common functionality:
- MT5 signal routing via Windows VPS
- Risk management (1.5% per trade)
- Position sizing
- Trade logging
- Telegram alerts
"""

import os
import requests
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class BotConfig:
    """Configuration for FTMO bot."""
    bot_name: str
    symbol: str  # MT5 symbol (e.g., "XAUUSD", "EURUSD")
    risk_percent: float = 0.015  # 1.5% risk per trade
    max_daily_trades: int = 3
    stop_loss_pips: float = 50.0
    take_profit_pips: float = 90.0
    max_hold_hours: float = 2.0
    enabled: bool = True
    paper_mode: bool = True  # Set False for live trading
    turbo_mode: bool = False  # Turbo mode: loosen thresholds for more trades

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
        }


class BaseFTMOBot(ABC):
    """
    Abstract base class for FTMO trading bots.

    All bots must implement:
    - analyze(): Analyze market conditions
    - should_trade(): Check if trading conditions are met
    - generate_signal(): Generate trade signal if conditions met
    """

    # MT5 Executor service on Windows VPS
    MT5_EXECUTOR_URL = os.getenv("MT5_EXECUTOR_URL", "http://45.82.167.195:5000")
    MT5_API_SECRET = os.getenv("MT5_API_SECRET", "hydra_secret_2024")

    def __init__(self, config: BotConfig):
        self.config = config
        self._lock = threading.Lock()
        self._daily_trades: list = []
        self._last_signal: Optional[TradeSignal] = None
        self._last_trade_time: Optional[datetime] = None

        logger.info(f"[{config.bot_name}] Bot initialized (symbol: {config.symbol}, risk: {config.risk_percent*100:.1f}%)")

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
        lot_size = round(lot_size, 2)
        lot_size = max(0.01, min(lot_size, 5.0))  # Min 0.01, Max 5.0 lots

        logger.debug(
            f"[{self.config.bot_name}] Lot size: {lot_size} "
            f"(risk: ${risk_amount:.2f}, SL: {stop_loss_pips} pips)"
        )

        return lot_size

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
        """Execute via MT5 on Windows VPS."""
        try:
            url = f"{self.MT5_EXECUTOR_URL}/trade"
            headers = {
                "Authorization": f"Bearer {self.MT5_API_SECRET}",
                "Content-Type": "application/json",
            }

            payload = {
                "action": "open",
                "symbol": signal.symbol,
                "direction": signal.direction,
                "lot_size": signal.lot_size,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "comment": f"HYDRA_{signal.bot_name}",
            }

            logger.info(
                f"[{self.config.bot_name}] LIVE TRADE: "
                f"{signal.direction} {signal.symbol} @ {signal.entry_price:.2f}"
            )

            response = requests.post(url, json=payload, headers=headers, timeout=30)
            result = response.json()

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
        """Get account info from MT5."""
        try:
            url = f"{self.MT5_EXECUTOR_URL}/account"
            headers = {"Authorization": f"Bearer {self.MT5_API_SECRET}"}
            response = requests.get(url, headers=headers, timeout=10)
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {"balance": 15000, "equity": 15000}  # Fallback

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from MT5."""
        try:
            url = f"{self.MT5_EXECUTOR_URL}/price/{symbol}"
            headers = {"Authorization": f"Bearer {self.MT5_API_SECRET}"}
            response = requests.get(url, headers=headers, timeout=10)
            data = response.json()
            return data.get("bid")
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
