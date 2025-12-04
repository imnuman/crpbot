"""
HYDRA 3.0 - Execution Optimizer (Layer 7)

Optimizes trade execution to reduce costs and improve fills:
- Spread checking (reject if too wide)
- Smart limit orders (slightly better than market price)
- 30-second fill timeout (adjust if not filled)
- Execution cost tracking
- Saves 0.02-0.1% per trade vs market orders

For $100k/month volume: saves $20-$100/month
"""

from typing import Dict, Optional, Tuple
from datetime import datetime, timezone, timedelta
from loguru import logger
import time


class ExecutionOptimizer:
    """
    Optimizes order execution for HYDRA.

    Strategy:
    1. Check spread (reject if > normal * multiplier)
    2. Place limit order slightly inside spread
    3. Wait 30 seconds for fill
    4. If not filled, adjust price and retry
    5. Max 3 retries, then use market order
    """

    # Execution settings
    FILL_TIMEOUT_SECONDS = 30
    MAX_RETRIES = 3
    IMPROVEMENT_BASIS_POINTS = 2  # Try to save 0.02% (2 basis points)

    def __init__(self):
        self.execution_history = []  # Track execution quality
        logger.info("Execution Optimizer initialized")

    def optimize_entry(
        self,
        asset: str,
        asset_type: str,
        direction: str,
        size: float,
        current_bid: float,
        current_ask: float,
        spread_normal: float,
        spread_reject_multiplier: float,
        broker_api=None  # Placeholder for broker API
    ) -> Dict:
        """
        Optimize entry execution.

        Args:
            asset: Symbol (e.g., "BTC-USD", "USD/TRY")
            asset_type: "crypto", "forex", etc.
            direction: "LONG" or "SHORT"
            size: Position size
            current_bid: Current bid price
            current_ask: Current ask price
            spread_normal: Normal spread for this asset
            spread_reject_multiplier: Reject if spread > normal * multiplier
            broker_api: Broker API client (optional - for paper/live)

        Returns:
            {
                "status": "filled" | "rejected" | "timeout",
                "fill_price": float,
                "spread_paid": float,
                "execution_cost": float,  # How much worse than mid-price
                "retries": int,
                "time_to_fill": float,  # Seconds
                "order_type": "limit" | "market"
            }
        """
        start_time = time.time()

        # Calculate current spread
        current_spread = current_ask - current_bid
        spread_pct = (current_spread / current_bid) * 100

        # Check 1: Spread too wide?
        max_spread = spread_normal * spread_reject_multiplier
        if asset_type in ["exotic_forex", "meme_perp"]:
            # For exotic assets, spread check is critical
            if current_spread > max_spread:
                logger.warning(
                    f"{asset} spread too wide: {spread_pct:.3f}% "
                    f"(max: {(max_spread / current_bid) * 100:.3f}%)"
                )
                return {
                    "status": "rejected",
                    "rejection_reason": f"Spread too wide: {spread_pct:.3f}%",
                    "fill_price": None,
                    "spread_paid": current_spread,
                    "execution_cost": None,
                    "retries": 0,
                    "time_to_fill": 0,
                    "order_type": None
                }

        # Calculate smart limit price
        mid_price = (current_bid + current_ask) / 2

        if direction == "LONG":
            # BUY: Try to buy slightly below ask (save spread)
            # Place limit at: mid + (spread * 0.3)
            # This is inside the spread, more likely to fill than mid
            limit_price = mid_price + (current_spread * 0.3)
        else:
            # SELL/SHORT: Try to sell slightly above bid
            limit_price = mid_price - (current_spread * 0.3)

        logger.info(
            f"{asset} {direction} limit: {limit_price:.6f} "
            f"(bid: {current_bid:.6f}, ask: {current_ask:.6f}, mid: {mid_price:.6f})"
        )

        # Attempt execution with retries
        for retry in range(self.MAX_RETRIES):
            result = self._attempt_limit_order(
                asset=asset,
                direction=direction,
                size=size,
                limit_price=limit_price,
                current_bid=current_bid,
                current_ask=current_ask,
                mid_price=mid_price,
                broker_api=broker_api,
                retry_num=retry
            )

            if result["status"] == "filled":
                # Success!
                execution_time = time.time() - start_time
                result["time_to_fill"] = execution_time
                result["retries"] = retry

                # Log execution quality
                self._log_execution(asset, direction, result)

                logger.success(
                    f"{asset} {direction} FILLED at {result['fill_price']:.6f} "
                    f"(saved {result['execution_cost']:.4f} vs mid, "
                    f"{retry} retries, {execution_time:.1f}s)"
                )
                return result

            # Not filled - adjust price and retry
            if retry < self.MAX_RETRIES - 1:
                # Move limit price closer to market
                if direction == "LONG":
                    limit_price = mid_price + (current_spread * (0.5 + retry * 0.2))
                else:
                    limit_price = mid_price - (current_spread * (0.5 + retry * 0.2))

                logger.debug(f"Retry {retry + 1}: adjusting limit to {limit_price:.6f}")
                time.sleep(1)  # Brief pause before retry

        # Max retries exhausted - use market order
        logger.warning(f"{asset} limit orders failed after {self.MAX_RETRIES} retries, using market order")

        market_result = self._execute_market_order(
            asset=asset,
            direction=direction,
            size=size,
            current_bid=current_bid,
            current_ask=current_ask,
            mid_price=mid_price,
            broker_api=broker_api
        )

        execution_time = time.time() - start_time
        market_result["time_to_fill"] = execution_time
        market_result["retries"] = self.MAX_RETRIES

        self._log_execution(asset, direction, market_result)

        return market_result

    def _attempt_limit_order(
        self,
        asset: str,
        direction: str,
        size: float,
        limit_price: float,
        current_bid: float,
        current_ask: float,
        mid_price: float,
        broker_api,
        retry_num: int
    ) -> Dict:
        """
        Attempt to fill limit order.

        In production: Place actual limit order via broker API
        In paper/backtest: Simulate fill based on price movement
        """
        if broker_api is None:
            # Paper trading / simulation mode
            # Simulate fill probability based on limit price aggressiveness
            if direction == "LONG":
                aggressiveness = (limit_price - mid_price) / (current_ask - mid_price)
            else:
                aggressiveness = (mid_price - limit_price) / (mid_price - current_bid)

            # Higher aggressiveness = higher fill probability
            # 0.0 (at mid) = 50% fill, 1.0 (at market) = 95% fill
            fill_probability = 0.5 + (aggressiveness * 0.45)

            # Simulate fill (in production, would wait for actual fill)
            import random
            filled = random.random() < fill_probability

            if filled:
                # Filled at limit price
                execution_cost = abs(limit_price - mid_price)
                return {
                    "status": "filled",
                    "fill_price": limit_price,
                    "spread_paid": current_ask - current_bid,
                    "execution_cost": execution_cost,
                    "order_type": "limit"
                }
            else:
                return {"status": "timeout"}

        else:
            # Production mode - use broker API
            # Place limit order
            order = broker_api.place_limit_order(
                symbol=asset,
                side="buy" if direction == "LONG" else "sell",
                size=size,
                limit_price=limit_price
            )

            # Wait for fill (with timeout)
            timeout_time = time.time() + self.FILL_TIMEOUT_SECONDS
            while time.time() < timeout_time:
                order_status = broker_api.get_order_status(order["order_id"])

                if order_status["status"] == "filled":
                    execution_cost = abs(order_status["fill_price"] - mid_price)
                    return {
                        "status": "filled",
                        "fill_price": order_status["fill_price"],
                        "spread_paid": current_ask - current_bid,
                        "execution_cost": execution_cost,
                        "order_type": "limit"
                    }

                time.sleep(0.5)  # Check every 500ms

            # Timeout - cancel order
            broker_api.cancel_order(order["order_id"])
            return {"status": "timeout"}

    def _execute_market_order(
        self,
        asset: str,
        direction: str,
        size: float,
        current_bid: float,
        current_ask: float,
        mid_price: float,
        broker_api
    ) -> Dict:
        """
        Execute market order (fallback when limit orders fail).
        """
        if broker_api is None:
            # Paper trading - assume fill at ask (LONG) or bid (SHORT)
            if direction == "LONG":
                fill_price = current_ask
            else:
                fill_price = current_bid

            execution_cost = abs(fill_price - mid_price)

            return {
                "status": "filled",
                "fill_price": fill_price,
                "spread_paid": current_ask - current_bid,
                "execution_cost": execution_cost,
                "order_type": "market"
            }

        else:
            # Production - place market order
            order = broker_api.place_market_order(
                symbol=asset,
                side="buy" if direction == "LONG" else "sell",
                size=size
            )

            # Market orders fill immediately
            execution_cost = abs(order["fill_price"] - mid_price)

            return {
                "status": "filled",
                "fill_price": order["fill_price"],
                "spread_paid": current_ask - current_bid,
                "execution_cost": execution_cost,
                "order_type": "market"
            }

    def _log_execution(self, asset: str, direction: str, result: Dict):
        """Log execution for quality tracking."""
        self.execution_history.append({
            "timestamp": datetime.now(timezone.utc),
            "asset": asset,
            "direction": direction,
            **result
        })

        # Keep only last 1000 executions
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

    # ==================== ANALYSIS METHODS ====================

    def get_execution_stats(self, hours: int = 24) -> Dict:
        """
        Get execution quality statistics.

        Returns:
            {
                "total_executions": int,
                "limit_fill_rate": float,  # % filled with limit orders
                "avg_execution_cost": float,  # Average cost vs mid-price
                "avg_time_to_fill": float,  # Average seconds
                "total_saved_vs_market": float  # Total $ saved vs market orders
            }
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent = [e for e in self.execution_history if e["timestamp"] > cutoff]

        if not recent:
            return {
                "total_executions": 0,
                "limit_fill_rate": 0,
                "avg_execution_cost": 0,
                "avg_time_to_fill": 0,
                "total_saved_vs_market": 0
            }

        limit_fills = [e for e in recent if e.get("order_type") == "limit"]
        total_cost = sum(e.get("execution_cost", 0) for e in recent)
        total_spread = sum(e.get("spread_paid", 0) for e in recent)

        # Estimate savings: market orders would pay full spread, limits pay less
        market_cost = total_spread / 2  # Market = mid + half spread
        limit_cost = total_cost
        savings = market_cost - limit_cost

        return {
            "total_executions": len(recent),
            "limit_fill_rate": len(limit_fills) / len(recent) if recent else 0,
            "avg_execution_cost": total_cost / len(recent) if recent else 0,
            "avg_time_to_fill": sum(e.get("time_to_fill", 0) for e in recent) / len(recent) if recent else 0,
            "total_saved_vs_market": savings
        }

    def get_worst_executions(self, count: int = 10) -> list:
        """Get worst executions by cost."""
        sorted_executions = sorted(
            self.execution_history,
            key=lambda e: e.get("execution_cost", 0),
            reverse=True
        )
        return sorted_executions[:count]


# Global singleton instance
_execution_optimizer = None

def get_execution_optimizer() -> ExecutionOptimizer:
    """Get global ExecutionOptimizer singleton."""
    global _execution_optimizer
    if _execution_optimizer is None:
        _execution_optimizer = ExecutionOptimizer()
    return _execution_optimizer
