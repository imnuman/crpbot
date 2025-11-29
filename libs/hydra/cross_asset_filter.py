"""
HYDRA 3.0 - Cross-Asset Filter (Upgrade D)

Prevents trading against major macro forces.

Examples of what this blocks:
- Buying EUR/USD while DXY is surging upward
- Buying altcoins while BTC is dumping -5%
- Shorting USD/TRY when EM currencies rallying
- Going long crypto while stock market crashing

The market is bigger than your strategy. Don't fight macro.
"""

from typing import Dict, List, Optional, Tuple
from loguru import logger
from datetime import datetime, timezone


class CrossAssetFilter:
    """
    Checks if trade aligns with major cross-asset correlations.

    Key correlations:
    1. DXY (US Dollar Index) - Affects all USD pairs
    2. BTC - Affects all altcoins (correlation ~0.7-0.9)
    3. EM Sentiment - Affects all emerging market currencies
    4. Risk-On/Risk-Off - Affects risk assets vs safe havens
    """

    # Thresholds
    DXY_STRONG_MOVE = 0.005  # 0.5% move in DXY
    BTC_STRONG_MOVE = 0.03   # 3% move in BTC
    EM_DIVERGENCE_THRESHOLD = 0.02  # 2% divergence from EM basket

    def __init__(self):
        self.filter_history = []
        logger.info("Cross-Asset Filter initialized")

    def check_cross_asset_alignment(
        self,
        asset: str,
        asset_type: str,
        direction: str,
        market_data: Dict,
        dxy_data: Optional[Dict] = None,
        btc_data: Optional[Dict] = None,
        em_basket_data: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Check if trade aligns with macro forces.

        Args:
            asset: Symbol being traded
            asset_type: "exotic_forex", "meme_perp", "standard", etc.
            direction: "LONG" or "SHORT"
            market_data: Current market data for the asset
            dxy_data: DXY (US Dollar Index) data
            btc_data: BTC price data (for crypto correlation)
            em_basket_data: EM currency basket data

        Returns:
            (passed: bool, reason: str)
        """
        # Check 1: USD pair vs DXY
        if self._is_usd_pair(asset) and dxy_data:
            passed, reason = self._check_dxy_correlation(asset, direction, dxy_data)
            if not passed:
                self._log_filter(asset, "DXY Conflict", reason)
                return False, reason

        # Check 2: Altcoin vs BTC
        if asset_type == "meme_perp" and btc_data:
            passed, reason = self._check_btc_correlation(asset, direction, btc_data, market_data)
            if not passed:
                self._log_filter(asset, "BTC Conflict", reason)
                return False, reason

        # Check 3: EM currency vs EM basket
        if asset_type == "exotic_forex" and em_basket_data:
            passed, reason = self._check_em_basket_correlation(asset, direction, em_basket_data, market_data)
            if not passed:
                self._log_filter(asset, "EM Sentiment Conflict", reason)
                return False, reason

        # All checks passed
        self._log_filter(asset, "Passed", "Cross-asset alignment confirmed")
        return True, "Cross-asset alignment confirmed"

    # ==================== CORRELATION CHECKS ====================

    def _check_dxy_correlation(
        self,
        asset: str,
        direction: str,
        dxy_data: Dict
    ) -> Tuple[bool, str]:
        """
        Check USD pair vs DXY movement.

        Rules:
        - DXY up strongly → USD pairs should go UP (not down)
        - DXY down strongly → USD pairs should go DOWN (not up)
        """
        dxy_change = dxy_data.get("change_pct_1h", 0.0)

        if abs(dxy_change) < self.DXY_STRONG_MOVE:
            # DXY not moving strongly, no conflict
            return True, "DXY neutral"

        # Determine if this is a USD-base or USD-quote pair
        is_usd_base = asset.startswith("USD/")  # USD/TRY, USD/ZAR, etc.

        if is_usd_base:
            # USD/TRY - when DXY up, USD/TRY should go up
            if dxy_change > self.DXY_STRONG_MOVE and direction == "SHORT":
                return False, f"DXY up {dxy_change:.2%}, don't SHORT {asset}"
            if dxy_change < -self.DXY_STRONG_MOVE and direction == "LONG":
                return False, f"DXY down {dxy_change:.2%}, don't LONG {asset}"
        else:
            # EUR/USD - when DXY up, EUR/USD should go down
            if dxy_change > self.DXY_STRONG_MOVE and direction == "LONG":
                return False, f"DXY up {dxy_change:.2%}, don't LONG {asset}"
            if dxy_change < -self.DXY_STRONG_MOVE and direction == "SHORT":
                return False, f"DXY down {dxy_change:.2%}, don't SHORT {asset}"

        return True, f"DXY aligned ({dxy_change:+.2%})"

    def _check_btc_correlation(
        self,
        asset: str,
        direction: str,
        btc_data: Dict,
        asset_data: Dict
    ) -> Tuple[bool, str]:
        """
        Check altcoin vs BTC movement.

        Rules:
        - BTC dumping -3%+ → Don't LONG altcoins
        - BTC pumping +3%+ → Don't SHORT altcoins
        - Altcoins typically 1.5-2x more volatile than BTC
        """
        btc_change = btc_data.get("change_pct_1h", 0.0)

        if abs(btc_change) < self.BTC_STRONG_MOVE:
            return True, "BTC neutral"

        # BTC dumping hard
        if btc_change < -self.BTC_STRONG_MOVE:
            if direction == "LONG":
                return False, f"BTC dumping {btc_change:.2%}, don't LONG {asset}"

        # BTC pumping hard
        if btc_change > self.BTC_STRONG_MOVE:
            if direction == "SHORT":
                return False, f"BTC pumping {btc_change:.2%}, don't SHORT {asset}"

        # Check if asset is moving WITH BTC
        asset_change = asset_data.get("change_pct_1h", 0.0)

        # If BTC and asset moving in same direction, that's good
        if (btc_change > 0 and asset_change > 0) or (btc_change < 0 and asset_change < 0):
            return True, f"BTC ({btc_change:+.2%}) and {asset} ({asset_change:+.2%}) aligned"

        # If asset moving opposite to BTC strongly, that's a red flag
        if abs(asset_change) > abs(btc_change):
            logger.warning(f"{asset} diverging from BTC: BTC {btc_change:+.2%}, {asset} {asset_change:+.2%}")

        return True, "BTC correlation acceptable"

    def _check_em_basket_correlation(
        self,
        asset: str,
        direction: str,
        em_basket_data: Dict,
        asset_data: Dict
    ) -> Tuple[bool, str]:
        """
        Check EM currency vs EM basket.

        EM currencies tend to move together (risk-on/risk-off).
        If EM basket rallying but our EM currency dumping, that's suspicious.
        """
        em_basket_change = em_basket_data.get("change_pct_1h", 0.0)
        asset_change = asset_data.get("change_pct_1h", 0.0)

        # Calculate divergence
        divergence = abs(em_basket_change - asset_change)

        if divergence > self.EM_DIVERGENCE_THRESHOLD:
            # Large divergence - check if it makes sense
            if em_basket_change > 0.01 and asset_change < -0.01:
                # EM rallying but this currency dumping
                if direction == "SHORT":
                    # We're shorting while EM rallying - conflict
                    return False, f"EM basket up {em_basket_change:.2%} but {asset} down {asset_change:.2%}"

            if em_basket_change < -0.01 and asset_change > 0.01:
                # EM dumping but this currency rallying
                if direction == "LONG":
                    # We're buying while EM dumping - conflict
                    return False, f"EM basket down {em_basket_change:.2%} but {asset} up {asset_change:.2%}"

        return True, f"EM basket aligned (basket: {em_basket_change:+.2%}, {asset}: {asset_change:+.2%})"

    # ==================== HELPER METHODS ====================

    def _is_usd_pair(self, asset: str) -> bool:
        """Check if asset is a USD pair."""
        return "USD" in asset.upper()

    def _log_filter(self, asset: str, filter_type: str, reason: str):
        """Log filter result."""
        self.filter_history.append({
            "timestamp": datetime.now(timezone.utc),
            "asset": asset,
            "filter_type": filter_type,
            "reason": reason
        })

        # Keep only last 1000 filter results
        if len(self.filter_history) > 1000:
            self.filter_history = self.filter_history[-1000:]

    # ==================== MOCK DATA (for testing without external APIs) ====================

    def get_mock_dxy_data(self) -> Dict:
        """Mock DXY data for testing."""
        return {
            "symbol": "DXY",
            "price": 104.5,
            "change_pct_1h": 0.002,  # +0.2% (neutral)
            "change_pct_24h": 0.008
        }

    def get_mock_btc_data(self, scenario: str = "neutral") -> Dict:
        """Mock BTC data for testing."""
        scenarios = {
            "neutral": {"change_pct_1h": 0.005},  # +0.5%
            "dumping": {"change_pct_1h": -0.04},  # -4%
            "pumping": {"change_pct_1h": 0.05}    # +5%
        }

        return {
            "symbol": "BTC-USD",
            "price": 97000,
            **scenarios.get(scenario, scenarios["neutral"]),
            "change_pct_24h": 0.02
        }

    def get_mock_em_basket_data(self) -> Dict:
        """Mock EM basket data for testing."""
        return {
            "symbol": "EM_BASKET",
            "change_pct_1h": -0.003,  # -0.3% (slight risk-off)
            "change_pct_24h": -0.015
        }

    # ==================== STATISTICS ====================

    def get_filter_stats(self, hours: int = 24) -> Dict:
        """Get cross-asset filter statistics."""
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent = [f for f in self.filter_history if f["timestamp"] > cutoff]

        if not recent:
            return {
                "total_checks": 0,
                "passed": 0,
                "blocked": 0,
                "block_rate": 0.0,
                "block_reasons": {}
            }

        passed = sum(1 for f in recent if f["filter_type"] == "Passed")
        blocked = len(recent) - passed

        # Count block reasons
        block_reasons = {}
        for f in recent:
            if f["filter_type"] != "Passed":
                reason_type = f["filter_type"]
                block_reasons[reason_type] = block_reasons.get(reason_type, 0) + 1

        return {
            "total_checks": len(recent),
            "passed": passed,
            "blocked": blocked,
            "block_rate": blocked / len(recent) if recent else 0.0,
            "block_reasons": block_reasons
        }


# Global singleton instance
_cross_asset_filter = None

def get_cross_asset_filter() -> CrossAssetFilter:
    """Get global CrossAssetFilter singleton."""
    global _cross_asset_filter
    if _cross_asset_filter is None:
        _cross_asset_filter = CrossAssetFilter()
    return _cross_asset_filter
