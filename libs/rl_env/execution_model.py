"""Empirical FTMO execution model using measured spreads and slippage."""
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from apps.trainer.features import get_trading_session


@dataclass
class ExecutionMetrics:
    """Execution metrics for a symbol/session pair."""

    spread_bps_mean: float
    spread_bps_p50: float
    spread_bps_p90: float
    slippage_bps_mean: float
    slippage_bps_p50: float
    slippage_bps_p90: float
    sample_count: int
    last_updated: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "spread_bps": {
                "mean": self.spread_bps_mean,
                "p50": self.spread_bps_p50,
                "p90": self.spread_bps_p90,
            },
            "slippage_bps": {
                "mean": self.slippage_bps_mean,
                "p50": self.slippage_bps_p50,
                "p90": self.slippage_bps_p90,
            },
            "sample_count": self.sample_count,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionMetrics":
        """Create from dictionary."""
        return cls(
            spread_bps_mean=data["spread_bps"]["mean"],
            spread_bps_p50=data["spread_bps"]["p50"],
            spread_bps_p90=data["spread_bps"]["p90"],
            slippage_bps_mean=data["slippage_bps"]["mean"],
            slippage_bps_p50=data["slippage_bps"]["p50"],
            slippage_bps_p90=data["slippage_bps"]["p90"],
            sample_count=data.get("sample_count", 0),
            last_updated=data.get("last_updated", ""),
        )


class ExecutionModel:
    """
    Empirical execution model that samples from measured spread/slippage distributions.

    Uses session-specific and pair-specific metrics from FTMO bridge measurements.
    """

    def __init__(self, metrics_file: str | Path | None = None):
        """
        Initialize execution model.

        Args:
            metrics_file: Path to execution metrics JSON file (default: latest symlink)
        """
        if metrics_file is None:
            metrics_file = Path("data/execution_metrics.json")
        else:
            metrics_file = Path(metrics_file)

        self.metrics_file = metrics_file
        self.metrics: dict[str, dict[str, ExecutionMetrics]] = {}
        self.load_metrics()

    def load_metrics(self) -> None:
        """Load execution metrics from JSON file."""
        if not self.metrics_file.exists():
            logger.warning(
                f"Execution metrics file not found: {self.metrics_file}. "
                "Using default fallback values."
            )
            self._load_default_metrics()
            return

        try:
            with open(self.metrics_file) as f:
                data = json.load(f)

            # Parse metrics structure: {symbol: {session: ExecutionMetrics}}
            for symbol, sessions in data.items():
                self.metrics[symbol] = {}
                for session, metrics_data in sessions.items():
                    self.metrics[symbol][session] = ExecutionMetrics.from_dict(metrics_data)

            logger.info(f"Loaded execution metrics from {self.metrics_file}")
            logger.info(f"  Symbols: {list(self.metrics.keys())}")
            logger.info(
                f"  Sessions: {list(next(iter(self.metrics.values())).keys()) if self.metrics else []}"
            )

        except Exception as e:
            logger.error(f"Failed to load execution metrics: {e}. Using default fallback values.")
            self._load_default_metrics()

    def _load_default_metrics(self) -> None:
        """Load default fallback metrics (used when no FTMO data available)."""
        # Default conservative estimates (will be replaced by real measurements)
        default_metrics = ExecutionMetrics(
            spread_bps_mean=12.0,  # 12 bps default spread
            spread_bps_p50=12.0,
            spread_bps_p90=18.0,
            slippage_bps_mean=3.0,  # 3 bps default slippage
            slippage_bps_p50=3.0,
            slippage_bps_p90=6.0,
            sample_count=0,
            last_updated="",
        )

        # Use same defaults for all symbols/sessions initially
        for symbol in ["BTC-USD", "ETH-USD", "BNB-USD"]:
            self.metrics[symbol] = {}
            for session in ["tokyo", "london", "new_york"]:
                self.metrics[symbol][session] = default_metrics

        logger.warning("Using default execution metrics (not measured from FTMO)")

    def get_metrics(
        self, symbol: str, session: str, timestamp: datetime | None = None
    ) -> ExecutionMetrics:
        """
        Get execution metrics for a symbol/session pair.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            session: Trading session ('tokyo', 'london', 'new_york')
            timestamp: Timestamp to determine session if not provided

        Returns:
            ExecutionMetrics for the symbol/session
        """
        # Determine session from timestamp if not provided
        if timestamp is not None:
            session = get_trading_session(timestamp)

        # Normalize symbol format (e.g., BTC-USD, ETH-USD)
        symbol = symbol.upper().replace("_", "-")

        # Get metrics for symbol/session, fallback to defaults
        if symbol in self.metrics and session in self.metrics[symbol]:
            return self.metrics[symbol][session]

        # Fallback to BTC-USD if symbol not found
        if "BTC-USD" in self.metrics and session in self.metrics["BTC-USD"]:
            logger.debug(f"Using BTC-USD metrics for {symbol} (fallback)")
            return self.metrics["BTC-USD"][session]

        # Final fallback: use default metrics
        logger.warning(f"Using default metrics for {symbol}/{session}")
        return ExecutionMetrics(
            spread_bps_mean=12.0,
            spread_bps_p50=12.0,
            spread_bps_p90=18.0,
            slippage_bps_mean=3.0,
            slippage_bps_p50=3.0,
            slippage_bps_p90=6.0,
            sample_count=0,
            last_updated="",
        )

    def sample_spread(
        self,
        symbol: str,
        session: str,
        timestamp: datetime | None = None,
        use_p90: bool = False,
    ) -> float:
        """
        Sample spread in basis points from distribution.

        Args:
            symbol: Trading pair
            session: Trading session
            timestamp: Timestamp (optional, for session detection)
            use_p90: If True, use p90 (conservative), else sample from distribution

        Returns:
            Spread in basis points
        """
        metrics = self.get_metrics(symbol, session, timestamp)

        if use_p90:
            return metrics.spread_bps_p90

        # Sample from distribution (use triangular distribution between p50 and p90)
        # This is a simple approximation; could be improved with actual distribution
        spread = np.random.triangular(
            metrics.spread_bps_p50, metrics.spread_bps_mean, metrics.spread_bps_p90
        )
        return max(0.0, spread)  # Ensure non-negative

    def sample_slippage(
        self,
        symbol: str,
        session: str,
        timestamp: datetime | None = None,
        use_p90: bool = False,
        latency_ms: float | None = None,
        latency_budget_ms: float = 500,
    ) -> float:
        """
        Sample slippage in basis points from distribution.

        Args:
            symbol: Trading pair
            session: Trading session
            timestamp: Timestamp (optional, for session detection)
            use_p90: If True, use p90 (conservative), else sample from distribution
            latency_ms: Decision latency in milliseconds (if > budget, use p90)
            latency_budget_ms: Latency budget (default: 500ms)

        Returns:
            Slippage in basis points
        """
        metrics = self.get_metrics(symbol, session, timestamp)

        # If latency exceeds budget, degrade with p90 slippage
        if latency_ms is not None and latency_ms > latency_budget_ms:
            logger.debug(
                f"Latency {latency_ms}ms exceeds budget {latency_budget_ms}ms, "
                f"using p90 slippage {metrics.slippage_bps_p90}bps"
            )
            return metrics.slippage_bps_p90

        if use_p90:
            return metrics.slippage_bps_p90

        # Sample from distribution
        slippage = np.random.triangular(
            metrics.slippage_bps_p50, metrics.slippage_bps_mean, metrics.slippage_bps_p90
        )
        return max(0.0, slippage)  # Ensure non-negative

    def calculate_execution_cost(
        self,
        entry_price: float,
        symbol: str,
        session: str,
        timestamp: datetime | None = None,
        latency_ms: float | None = None,
        latency_budget_ms: float = 500,
    ) -> float:
        """
        Calculate total execution cost (spread + slippage) in price units.

        Args:
            entry_price: Entry price
            symbol: Trading pair
            session: Trading session
            timestamp: Timestamp (optional)
            latency_ms: Decision latency in milliseconds
            latency_budget_ms: Latency budget (default: 500ms)

        Returns:
            Total execution cost in price units (not basis points)
        """
        spread_bps = self.sample_spread(symbol, session, timestamp)
        slippage_bps = self.sample_slippage(
            symbol, session, timestamp, latency_ms=latency_ms, latency_budget_ms=latency_budget_ms
        )

        total_cost_bps = spread_bps + slippage_bps
        total_cost = entry_price * (total_cost_bps / 10000.0)  # Convert bps to price units

        return total_cost

    def apply_execution_cost(
        self,
        entry_price: float,
        symbol: str,
        session: str,
        timestamp: datetime | None = None,
        latency_ms: float | None = None,
        latency_budget_ms: float = 500,
        direction: str = "long",
    ) -> float:
        """
        Apply execution cost to entry price.

        Args:
            entry_price: Theoretical entry price
            symbol: Trading pair
            session: Trading session
            timestamp: Timestamp (optional)
            latency_ms: Decision latency in milliseconds
            latency_budget_ms: Latency budget (default: 500ms)
            direction: 'long' or 'short'

        Returns:
            Actual entry price after execution costs
        """
        cost = self.calculate_execution_cost(
            entry_price, symbol, session, timestamp, latency_ms, latency_budget_ms
        )

        # For long: add cost (buy higher), for short: subtract cost (sell lower)
        if direction.lower() == "long":
            return entry_price + cost
        else:
            return entry_price - cost
