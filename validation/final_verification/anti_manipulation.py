"""
HYDRA 3.0 - Anti-Manipulation Filter (Layer 9)

7-layer filter system to catch:
- Bad strategies from gladiators (hallucinations)
- Market manipulation (fake volume, whale dumps, spoofing)
- Correlation conflicts (fighting macro forces)

Every trade must pass ALL 7 filters. If ANY filter fails → NO TRADE.
"""

from typing import Dict, List, Optional, Tuple
from loguru import logger
from datetime import datetime, timezone
import re


class AntiManipulationFilter:
    """
    7-layer filter system for HYDRA 3.0.

    Filter 1: Logic Validator - Catches contradictory rules
    Filter 2: Backtest Reality - Claimed vs actual performance
    Filter 3: Live Confirmation - Paper vs backtest degradation
    Filter 4: Cross-Agent Audit - Majority approval required
    Filter 5: Sanity Rules - Hard-coded minimums
    Filter 6: Manipulation Detection - Fake volume, whale dumps
    Filter 7: Cross-Asset Correlation - Macro force alignment
    """

    # Filter 5: Sanity rules (hard-coded)
    MIN_BACKTEST_TRADES = 100
    MAX_WIN_RATE = 0.85  # >85% = likely overfit
    MIN_WIN_RATE = 0.45  # <45% = worse than random
    MIN_SHARPE = 0.5
    MIN_REGIMES_TESTED = 2
    MAX_PAPER_DEGRADATION = 0.20  # 20% WR drop = overfit

    # Filter 6: Manipulation thresholds
    VOLUME_SPIKE_MULTIPLIER = 5.0  # 5x volume
    VOLUME_SPIKE_MAX_MOVE = 0.01  # 1% max price move
    ORDER_BOOK_IMBALANCE_THRESHOLD = 0.90  # 90% one side
    SPREAD_SPIKE_MULTIPLIER = 3.0  # 3x normal spread

    def __init__(self):
        logger.info("Anti-Manipulation Filter initialized (7 layers)")

    # ==================== FILTER 1: LOGIC VALIDATOR ====================

    def filter_1_logic_validator(self, strategy: Dict) -> Tuple[bool, str]:
        """
        Filter 1: Check for contradictory or nonsensical logic.

        Examples of bad logic:
        - "Buy when RSI > 70" (overbought = bearish, not bullish)
        - "Sell when price is rising"
        - Entry and exit contradict each other

        Args:
            strategy: Strategy dict with entry_rules, exit_rules, etc.

        Returns:
            (passed: bool, reason: str)
        """
        entry_rules = strategy.get("entry_rules", "").lower()
        exit_rules = strategy.get("exit_rules", "").lower()

        # Check for banned retail patterns (from HYDRA spec)
        banned_patterns = [
            "rsi", "macd", "bollinger", "moving average crossover",
            "support", "resistance", "fibonacci", "candlestick pattern",
            "head and shoulders", "double top", "double bottom"
        ]

        for pattern in banned_patterns:
            if pattern in entry_rules or pattern in exit_rules:
                return False, f"BANNED PATTERN: {pattern} (retail indicator)"

        # Check for overbought/oversold contradictions
        if "overbought" in entry_rules and "buy" in entry_rules:
            return False, "Logic error: Buy when overbought (should be sell)"
        if "oversold" in entry_rules and "sell" in entry_rules:
            return False, "Logic error: Sell when oversold (should be buy)"

        # Check for contradictory entry/exit
        if "rising" in entry_rules and "sell" in entry_rules:
            return False, "Logic error: Sell when price rising"
        if "falling" in entry_rules and "buy" in entry_rules:
            return False, "Logic error: Buy when price falling"

        # Must have actual structural edge (not pattern)
        structural_keywords = [
            "funding", "liquidation", "session", "volatility", "carry",
            "correlation", "gap", "central bank", "news", "spread",
            "order book", "whale", "exchange"
        ]

        has_structural_edge = any(kw in entry_rules for kw in structural_keywords)
        if not has_structural_edge:
            return False, "No structural edge detected (must use market mechanics)"

        return True, "Logic valid"

    # ==================== FILTER 2: BACKTEST REALITY CHECK ====================

    def filter_2_backtest_reality(
        self,
        claimed_wr: float,
        claimed_sharpe: float,
        actual_backtest_wr: float,
        actual_backtest_sharpe: float,
        actual_trades: int
    ) -> Tuple[bool, str]:
        """
        Filter 2: Compare gladiator's claims vs actual backtest results.

        LLMs hallucinate performance. Always verify.

        Args:
            claimed_wr: What gladiator claimed (0.0-1.0)
            claimed_sharpe: What gladiator claimed
            actual_backtest_wr: Real backtest win rate
            actual_backtest_sharpe: Real backtest Sharpe
            actual_trades: Number of backtest trades

        Returns:
            (passed: bool, reason: str)
        """
        # Check if enough trades
        if actual_trades < self.MIN_BACKTEST_TRADES:
            return False, f"Insufficient backtest trades: {actual_trades} < {self.MIN_BACKTEST_TRADES}"

        # Check WR mismatch
        wr_diff = abs(claimed_wr - actual_backtest_wr)
        if wr_diff > 0.10:  # >10% difference = hallucination
            return False, f"WR mismatch: Claimed {claimed_wr:.1%}, Actual {actual_backtest_wr:.1%}"

        # Check Sharpe mismatch
        sharpe_diff = abs(claimed_sharpe - actual_backtest_sharpe)
        if sharpe_diff > 0.5:
            return False, f"Sharpe mismatch: Claimed {claimed_sharpe:.2f}, Actual {actual_backtest_sharpe:.2f}"

        # Use actual results (not claims) going forward
        return True, f"Backtest verified: {actual_backtest_wr:.1%} WR, {actual_backtest_sharpe:.2f} Sharpe"

    # ==================== FILTER 3: LIVE CONFIRMATION ====================

    def filter_3_live_confirmation(
        self,
        backtest_wr: float,
        paper_wr: float,
        paper_trades: int
    ) -> Tuple[bool, str]:
        """
        Filter 3: Check if paper trading matches backtest.

        If paper WR degrades >20% from backtest → Strategy is overfit.

        Args:
            backtest_wr: Backtest win rate
            paper_wr: Paper trading win rate
            paper_trades: Number of paper trades

        Returns:
            (passed: bool, reason: str)
        """
        # Need minimum paper trades to judge
        if paper_trades < 20:
            # Not enough data yet - allow to continue paper trading
            return True, f"Paper trading ({paper_trades} trades) - collecting data"

        # Check degradation
        degradation = (backtest_wr - paper_wr) / backtest_wr if backtest_wr > 0 else 0

        if degradation > self.MAX_PAPER_DEGRADATION:
            return False, f"Overfit detected: Paper WR {paper_wr:.1%} vs Backtest {backtest_wr:.1%} ({degradation:.1%} drop)"

        return True, f"Paper confirms backtest: {paper_wr:.1%} WR ({paper_trades} trades)"

    # ==================== FILTER 4: CROSS-AGENT AUDIT ====================

    def filter_4_cross_agent_audit(
        self,
        strategy: Dict,
        auditor_approvals: List[bool]
    ) -> Tuple[bool, str]:
        """
        Filter 4: Other gladiators review and approve/reject.

        If majority disapproves → Strategy has flaws.

        Args:
            strategy: Strategy to audit
            auditor_approvals: List of True/False from auditing gladiators

        Returns:
            (passed: bool, reason: str)
        """
        if not auditor_approvals:
            return True, "No auditors available (skip)"

        approval_rate = sum(auditor_approvals) / len(auditor_approvals)

        if approval_rate < 0.5:  # Majority disapproves
            return False, f"Cross-agent audit failed: {approval_rate:.0%} approval"

        return True, f"Cross-agent audit passed: {approval_rate:.0%} approval"

    # ==================== FILTER 5: SANITY RULES ====================

    def filter_5_sanity_rules(
        self,
        backtest_trades: int,
        win_rate: float,
        sharpe: float,
        regimes_tested: int
    ) -> Tuple[bool, str]:
        """
        Filter 5: Hard-coded sanity checks.

        These are absolute minimums that cannot be bypassed.

        Args:
            backtest_trades: Number of backtest trades
            win_rate: Win rate (0.0-1.0)
            sharpe: Sharpe ratio
            regimes_tested: Number of different regimes tested

        Returns:
            (passed: bool, reason: str)
        """
        # Check 1: Minimum trades
        if backtest_trades < self.MIN_BACKTEST_TRADES:
            return False, f"Too few trades: {backtest_trades} < {self.MIN_BACKTEST_TRADES}"

        # Check 2: Win rate bounds
        if win_rate > self.MAX_WIN_RATE:
            return False, f"Win rate too high: {win_rate:.1%} (likely overfit)"
        if win_rate < self.MIN_WIN_RATE:
            return False, f"Win rate too low: {win_rate:.1%} (worse than random)"

        # Check 3: Minimum Sharpe
        if sharpe < self.MIN_SHARPE:
            return False, f"Sharpe too low: {sharpe:.2f} < {self.MIN_SHARPE}"

        # Check 4: Multi-regime tested
        if regimes_tested < self.MIN_REGIMES_TESTED:
            return False, f"Only tested in {regimes_tested} regime(s), need {self.MIN_REGIMES_TESTED}+"

        return True, "All sanity checks passed"

    # ==================== FILTER 6: MANIPULATION DETECTION ====================

    def filter_6_manipulation_detection(
        self,
        asset: str,
        asset_type: str,
        volume_24h: float,
        volume_1h: float,
        price_change_1h: float,
        order_book_buy_percent: Optional[float] = None,
        whale_alert_usd: Optional[float] = None,
        current_spread: Optional[float] = None,
        normal_spread: Optional[float] = None,
        price_direction: Optional[str] = None,
        volume_direction: Optional[str] = None,
        funding_rate: Optional[float] = None,
        funding_threshold: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Filter 6: Detect market manipulation.

        Catches:
        - Fake volume (5x spike with <1% move)
        - Order book spoofing (90%+ one side)
        - Whale dumps ($1M+ to exchange)
        - Spread manipulation (3x spike)
        - Price/volume divergence (price up, volume down)
        - Funding extremes (>0.3% for BTC, >0.5% for memes)

        Args:
            asset: Symbol
            asset_type: "exotic_forex" or "meme_perp"
            volume_24h: 24-hour average volume
            volume_1h: Current 1-hour volume
            price_change_1h: Price change last hour (0.01 = 1%)
            order_book_buy_percent: % of buy orders (0.0-1.0)
            whale_alert_usd: Whale movement in USD
            current_spread: Current bid-ask spread
            normal_spread: Normal spread for this asset
            price_direction: "up" or "down"
            volume_direction: "up" or "down"
            funding_rate: Current funding rate
            funding_threshold: Extreme funding threshold for this asset

        Returns:
            (passed: bool, reason: str)
        """
        # Check 1: Volume spike with no price movement (FAKE VOLUME)
        if volume_24h > 0 and volume_1h > 0:
            volume_ratio = volume_1h / (volume_24h / 24)
            if volume_ratio > self.VOLUME_SPIKE_MULTIPLIER and abs(price_change_1h) < self.VOLUME_SPIKE_MAX_MOVE:
                return False, f"Fake volume: {volume_ratio:.1f}x spike with {price_change_1h:.2%} move"

        # Check 2: Order book imbalance (SPOOFING) - Crypto only
        if asset_type == "meme_perp" and order_book_buy_percent is not None:
            if order_book_buy_percent > self.ORDER_BOOK_IMBALANCE_THRESHOLD:
                return False, f"Order book spoofing: {order_book_buy_percent:.1%} buy side"
            if order_book_buy_percent < (1 - self.ORDER_BOOK_IMBALANCE_THRESHOLD):
                return False, f"Order book spoofing: {(1-order_book_buy_percent):.1%} sell side"

        # Check 3: Whale alert (DUMP INCOMING) - Crypto only
        if asset_type == "meme_perp" and whale_alert_usd is not None:
            if whale_alert_usd > 0:  # Any whale movement to exchange
                return False, f"Whale alert: ${whale_alert_usd:,.0f} moved to exchange"

        # Check 4: Spread spike (LIQUIDITY CRISIS)
        if current_spread is not None and normal_spread is not None and normal_spread > 0:
            spread_ratio = current_spread / normal_spread
            if spread_ratio > self.SPREAD_SPIKE_MULTIPLIER:
                return False, f"Spread spike: {spread_ratio:.1f}x normal ({current_spread} vs {normal_spread})"

        # Check 5: Price/volume divergence (WEAK MOVE)
        if price_direction and volume_direction:
            if price_direction == "up" and volume_direction == "down":
                return False, "Price/volume divergence: Price up but volume declining"
            if price_direction == "down" and volume_direction == "up" and asset_type == "meme_perp":
                # This is OK for crypto (dump on volume is normal)
                pass

        # Check 6: Funding rate extreme (OVERCROWDED TRADE) - Crypto only
        if asset_type == "meme_perp" and funding_rate is not None and funding_threshold is not None:
            if abs(funding_rate) > funding_threshold:
                direction = "long" if funding_rate > 0 else "short"
                return False, f"Funding extreme: {funding_rate:.2%} ({direction} overcrowded)"

        return True, "No manipulation detected"

    # ==================== FILTER 7: CROSS-ASSET CORRELATION ====================

    def filter_7_cross_asset_check(
        self,
        asset: str,
        direction: str,
        dxy_change_24h: Optional[float] = None,
        btc_change_24h: Optional[float] = None,
        us10y_change_24h: Optional[float] = None,
        risk_sentiment: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Filter 7: Check if trade aligns with macro forces.

        Don't fight the tide:
        - EUR/USD long while DXY surging = fighting macro
        - Altcoin long while BTC dumping = fighting correlation
        - EM currency long during risk-off = fighting flows

        Args:
            asset: Symbol
            direction: "LONG" or "SHORT"
            dxy_change_24h: Dollar index change (0.01 = 1%)
            btc_change_24h: Bitcoin change (0.01 = 1%)
            us10y_change_24h: US 10-year yield change
            risk_sentiment: "risk-on" or "risk-off"

        Returns:
            (passed: bool, reason: str)
        """
        # EUR/USD vs DXY
        if asset == "EUR/USD" and dxy_change_24h is not None:
            if direction == "LONG" and dxy_change_24h > 0.005:  # DXY up >0.5%
                return False, f"Fighting DXY: EUR long while DXY +{dxy_change_24h:.2%}"
            if direction == "SHORT" and dxy_change_24h < -0.005:
                return False, f"Fighting DXY: EUR short while DXY {dxy_change_24h:.2%}"

        # Gold vs DXY + Yields
        if asset == "XAUUSD" and dxy_change_24h is not None and us10y_change_24h is not None:
            if direction == "LONG" and dxy_change_24h > 0.005 and us10y_change_24h > 0.02:
                return False, f"Fighting macro: Gold long while DXY +{dxy_change_24h:.2%} and yields +{us10y_change_24h:.2%}"

        # Altcoins vs BTC
        if asset in ["BONK", "WIF", "PEPE", "FLOKI", "SUI", "INJ"] and btc_change_24h is not None:
            if direction == "LONG" and btc_change_24h < -0.02:  # BTC down >2%
                return False, f"Fighting BTC: {asset} long while BTC {btc_change_24h:.1%}"

        # EM currencies vs Risk sentiment
        if asset in ["USD/TRY", "USD/ZAR", "USD/MXN", "EUR/TRY"] and risk_sentiment:
            if direction == "LONG" and risk_sentiment == "risk-off" and asset.startswith("USD"):
                # USD/TRY long during risk-off = OK (TRY weakens)
                pass
            elif direction == "SHORT" and risk_sentiment == "risk-on" and asset.startswith("USD"):
                # USD/TRY short during risk-on = OK (TRY strengthens)
                pass
            elif direction == "LONG" and risk_sentiment == "risk-off" and asset.startswith("EUR"):
                return False, f"Fighting flows: {asset} long during risk-off"

        return True, "Cross-asset aligned"

    # ==================== MAIN FILTER PIPELINE ====================

    def run_all_filters(
        self,
        strategy: Dict,
        backtest_results: Dict,
        paper_results: Optional[Dict] = None,
        auditor_approvals: Optional[List[bool]] = None,
        market_data: Optional[Dict] = None,
        cross_asset_data: Optional[Dict] = None
    ) -> Tuple[bool, List[str]]:
        """
        Run all 7 filters in sequence.

        If ANY filter fails → BLOCK trade.

        Args:
            strategy: Strategy dict with entry/exit rules
            backtest_results: Backtest performance
            paper_results: Paper trading results (optional)
            auditor_approvals: List of approvals from other gladiators
            market_data: Current market conditions
            cross_asset_data: DXY, BTC, yields, sentiment

        Returns:
            (all_passed: bool, filter_results: List[str])
        """
        results = []

        # Filter 1: Logic Validator
        passed, reason = self.filter_1_logic_validator(strategy)
        results.append(f"Filter 1 (Logic): {reason}")
        if not passed:
            return False, results

        # Filter 2: Backtest Reality
        passed, reason = self.filter_2_backtest_reality(
            claimed_wr=strategy.get("expected_wr", 0.5),
            claimed_sharpe=backtest_results.get("claimed_sharpe", 1.0),
            actual_backtest_wr=backtest_results.get("win_rate", 0.0),
            actual_backtest_sharpe=backtest_results.get("sharpe", 0.0),
            actual_trades=backtest_results.get("total_trades", 0)
        )
        results.append(f"Filter 2 (Backtest): {reason}")
        if not passed:
            return False, results

        # Filter 3: Live Confirmation (if paper data available)
        if paper_results:
            passed, reason = self.filter_3_live_confirmation(
                backtest_wr=backtest_results.get("win_rate", 0.0),
                paper_wr=paper_results.get("win_rate", 0.0),
                paper_trades=paper_results.get("total_trades", 0)
            )
            results.append(f"Filter 3 (Paper): {reason}")
            if not passed:
                return False, results
        else:
            results.append("Filter 3 (Paper): No paper data yet")

        # Filter 4: Cross-Agent Audit
        if auditor_approvals:
            passed, reason = self.filter_4_cross_agent_audit(strategy, auditor_approvals)
            results.append(f"Filter 4 (Audit): {reason}")
            if not passed:
                return False, results
        else:
            results.append("Filter 4 (Audit): No auditors")

        # Filter 5: Sanity Rules
        passed, reason = self.filter_5_sanity_rules(
            backtest_trades=backtest_results.get("total_trades", 0),
            win_rate=backtest_results.get("win_rate", 0.0),
            sharpe=backtest_results.get("sharpe", 0.0),
            regimes_tested=backtest_results.get("regimes_tested", 0)
        )
        results.append(f"Filter 5 (Sanity): {reason}")
        if not passed:
            return False, results

        # Filter 6: Manipulation Detection (if market data available)
        if market_data:
            passed, reason = self.filter_6_manipulation_detection(
                asset=market_data.get("asset", ""),
                asset_type=market_data.get("asset_type", ""),
                volume_24h=market_data.get("volume_24h", 0),
                volume_1h=market_data.get("volume_1h", 0),
                price_change_1h=market_data.get("price_change_1h", 0),
                order_book_buy_percent=market_data.get("order_book_buy_percent"),
                whale_alert_usd=market_data.get("whale_alert_usd"),
                current_spread=market_data.get("current_spread"),
                normal_spread=market_data.get("normal_spread"),
                price_direction=market_data.get("price_direction"),
                volume_direction=market_data.get("volume_direction"),
                funding_rate=market_data.get("funding_rate"),
                funding_threshold=market_data.get("funding_threshold")
            )
            results.append(f"Filter 6 (Manipulation): {reason}")
            if not passed:
                return False, results
        else:
            results.append("Filter 6 (Manipulation): No market data")

        # Filter 7: Cross-Asset Check (if cross-asset data available)
        if cross_asset_data:
            passed, reason = self.filter_7_cross_asset_check(
                asset=cross_asset_data.get("asset", ""),
                direction=cross_asset_data.get("direction", ""),
                dxy_change_24h=cross_asset_data.get("dxy_change_24h"),
                btc_change_24h=cross_asset_data.get("btc_change_24h"),
                us10y_change_24h=cross_asset_data.get("us10y_change_24h"),
                risk_sentiment=cross_asset_data.get("risk_sentiment")
            )
            results.append(f"Filter 7 (Cross-Asset): {reason}")
            if not passed:
                return False, results
        else:
            results.append("Filter 7 (Cross-Asset): No cross-asset data")

        # All filters passed!
        return True, results



# ==================== SINGLETON PATTERN ====================

_anti_manipulation_filter = None

def get_anti_manipulation_filter() -> AntiManipulationFilter:
    """Get singleton instance of AntiManipulationFilter."""
    global _anti_manipulation_filter
    if _anti_manipulation_filter is None:
        _anti_manipulation_filter = AntiManipulationFilter()
    return _anti_manipulation_filter
