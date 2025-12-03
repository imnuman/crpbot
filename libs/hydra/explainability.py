"""
HYDRA 3.0 - Explainability Logger (Upgrade A)

Every trade decision is logged with FULL context:
- Which gladiators voted (and what they voted)
- Consensus level (2/4, 3/4, 4/4)
- All 7 anti-manipulation filters (passed/blocked)
- Guardian decision (approved/rejected + adjustments)
- Structural edge identified
- Entry/exit reasoning
- Position sizing adjustments

This makes HYDRA a transparent, auditable system.
NO BLACK BOXES.
"""

import json
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from pathlib import Path
from loguru import logger


class ExplainabilityLogger:
    """
    Logs every trade decision with full context for post-mortem analysis.

    Every log entry answers:
    1. WHAT happened? (trade details)
    2. WHY happened? (structural edge + reasoning)
    3. WHO decided? (gladiators + votes)
    4. HOW validated? (filters + Guardian)
    5. WHEN happened? (timestamp + regime)
    """

    def __init__(self, log_dir: str = "data/hydra/explainability"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        logger.info(f"Explainability Logger initialized: {self.log_dir}")

    def log_trade_decision(
        self,
        trade_id: str,
        asset: str,
        asset_type: str,
        regime: str,
        # Gladiator votes
        gladiator_votes: List[Dict],  # [{"name": "A", "vote": "BUY", "confidence": 0.7, "reasoning": "..."}]
        consensus_level: float,  # 0.5, 0.75, 1.0
        # Strategy info
        strategy_id: str,
        structural_edge: str,
        entry_reasoning: str,
        exit_reasoning: str,
        # Filters
        filters_passed: Dict[str, bool],  # {"filter_1_logic": True, "filter_2_backtest": False, ...}
        filter_block_reasons: List[str],  # Reasons for blocked filters
        # Guardian
        guardian_approved: bool,
        guardian_reason: str,
        position_size_original: float,
        position_size_final: float,
        adjustment_reason: str,
        # Trade details
        direction: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        risk_reward_ratio: float,
        # Metadata
        timestamp: Optional[datetime] = None,
        additional_context: Optional[Dict] = None
    ) -> str:
        """
        Log complete trade decision with all context.

        Returns:
            Path to log file
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Build comprehensive log entry
        log_entry = {
            # Basic info
            "trade_id": trade_id,
            "timestamp": timestamp.isoformat(),
            "session_id": self.session_id,

            # Market context
            "asset": asset,
            "asset_type": asset_type,
            "regime": regime,

            # Strategy
            "strategy_id": strategy_id,
            "structural_edge": structural_edge,
            "entry_reasoning": entry_reasoning,
            "exit_reasoning": exit_reasoning,

            # Gladiator consensus
            "gladiator_votes": gladiator_votes,
            "consensus_level": consensus_level,
            "votes_summary": self._summarize_votes(gladiator_votes),

            # Filters
            "filters_passed": filters_passed,
            "filters_summary": self._summarize_filters(filters_passed),
            "filter_block_reasons": filter_block_reasons,
            "all_filters_passed": all(filters_passed.values()),

            # Guardian
            "guardian_approved": guardian_approved,
            "guardian_reason": guardian_reason,
            "position_size_original": position_size_original,
            "position_size_final": position_size_final,
            "adjustment_reason": adjustment_reason,
            "position_size_reduction_pct": round((1 - position_size_final / position_size_original) * 100, 1) if position_size_original > 0 else 0,

            # Trade details
            "direction": direction,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "risk_reward_ratio": risk_reward_ratio,

            # Additional context
            "additional_context": additional_context or {}
        }

        # Save to JSON file
        log_file = self._save_log(trade_id, log_entry)

        # Log summary to console
        self._log_summary(log_entry)

        return str(log_file)

    def log_rejected_trade(
        self,
        asset: str,
        asset_type: str,
        regime: str,
        gladiator_votes: List[Dict],
        consensus_level: float,
        rejection_reason: str,
        filters_passed: Optional[Dict[str, bool]] = None,
        guardian_reason: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Log rejected trade with reason.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        log_entry = {
            "trade_id": f"REJECTED-{timestamp.strftime('%Y%m%d%H%M%S')}",
            "timestamp": timestamp.isoformat(),
            "session_id": self.session_id,
            "status": "REJECTED",

            "asset": asset,
            "asset_type": asset_type,
            "regime": regime,

            "gladiator_votes": gladiator_votes,
            "consensus_level": consensus_level,
            "votes_summary": self._summarize_votes(gladiator_votes),

            "rejection_reason": rejection_reason,
            "filters_passed": filters_passed or {},
            "guardian_reason": guardian_reason
        }

        log_file = self._save_log(log_entry["trade_id"], log_entry)
        logger.warning(f"Trade REJECTED: {asset} - {rejection_reason}")

        return str(log_file)

    def _summarize_votes(self, votes: List[Dict]) -> Dict:
        """Summarize engine votes."""
        buy_votes = [v for v in votes if v.get("vote") == "BUY"]
        sell_votes = [v for v in votes if v.get("vote") == "SELL"]
        hold_votes = [v for v in votes if v.get("vote") == "HOLD"]

        return {
            "total_votes": len(votes),
            "buy": len(buy_votes),
            "sell": len(sell_votes),
            "hold": len(hold_votes),
            "avg_confidence": round(sum(v.get("confidence", 0) for v in votes) / len(votes), 3) if votes else 0
        }

    def _summarize_filters(self, filters: Dict[str, bool]) -> Dict:
        """Summarize filter results."""
        return {
            "total": len(filters),
            "passed": sum(1 for v in filters.values() if v),
            "blocked": sum(1 for v in filters.values() if not v),
            "pass_rate": round(sum(1 for v in filters.values() if v) / len(filters), 3) if filters else 0
        }

    def _save_log(self, trade_id: str, log_entry: Dict) -> Path:
        """Save log entry to JSON file."""
        # Daily log file
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = self.log_dir / f"explainability_{date_str}.jsonl"

        # Append to JSONL (one JSON object per line)
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return log_file

    def _log_summary(self, entry: Dict):
        """Log human-readable summary to console."""
        logger.info(f"""
╔══════════════════════════════════════════════════════════════
║ TRADE DECISION: {entry['trade_id']}
╠══════════════════════════════════════════════════════════════
║ Asset: {entry['asset']} ({entry['asset_type']})
║ Regime: {entry['regime']}
║ Direction: {entry['direction']}
║ Entry: {entry['entry_price']:.6f} | SL: {entry['sl_price']:.6f} | TP: {entry['tp_price']:.6f}
║ R:R = {entry['risk_reward_ratio']:.2f}
╠══════════════════════════════════════════════════════════════
║ STRUCTURAL EDGE: {entry['structural_edge']}
╠══════════════════════════════════════════════════════════════
║ GLADIATOR CONSENSUS: {entry['consensus_level']:.0%}
║   • BUY: {entry['votes_summary']['buy']}/4
║   • SELL: {entry['votes_summary']['sell']}/4
║   • HOLD: {entry['votes_summary']['hold']}/4
║   • Avg Confidence: {entry['votes_summary']['avg_confidence']:.1%}
╠══════════════════════════════════════════════════════════════
║ FILTERS: {entry['filters_summary']['passed']}/{entry['filters_summary']['total']} passed
║   {self._format_filters(entry['filters_passed'])}
╠══════════════════════════════════════════════════════════════
║ GUARDIAN: {"✅ APPROVED" if entry['guardian_approved'] else "❌ REJECTED"}
║   • Reason: {entry['guardian_reason']}
║   • Position Size: ${entry['position_size_original']:.0f} → ${entry['position_size_final']:.0f} ({entry['position_size_reduction_pct']:+.1f}%)
║   • Adjustment: {entry['adjustment_reason']}
╚══════════════════════════════════════════════════════════════
        """)

    def _format_filters(self, filters: Dict[str, bool]) -> str:
        """Format filters for console display."""
        lines = []
        for name, passed in filters.items():
            status = "✅" if passed else "❌"
            lines.append(f"{status} {name}")
        return "\n║   ".join(lines)

    # ==================== QUERY METHODS ====================

    def get_trades_by_asset(self, asset: str, days: int = 7) -> List[Dict]:
        """Get all trade logs for specific asset in last N days."""
        logs = []

        for i in range(days):
            date_str = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            log_file = self.log_dir / f"explainability_{date_str}.jsonl"

            if log_file.exists():
                with open(log_file, "r") as f:
                    for line in f:
                        entry = json.loads(line)
                        if entry.get("asset") == asset:
                            logs.append(entry)

        return logs

    def get_rejected_trades(self, days: int = 7) -> List[Dict]:
        """Get all rejected trades in last N days."""
        logs = []

        for i in range(days):
            date_str = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            log_file = self.log_dir / f"explainability_{date_str}.jsonl"

            if log_file.exists():
                with open(log_file, "r") as f:
                    for line in f:
                        entry = json.loads(line)
                        if entry.get("status") == "REJECTED":
                            logs.append(entry)

        return logs

    def get_filter_failure_stats(self, days: int = 7) -> Dict[str, int]:
        """Get stats on which filters block trades most often."""
        filter_blocks = {}

        for i in range(days):
            date_str = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            log_file = self.log_dir / f"explainability_{date_str}.jsonl"

            if log_file.exists():
                with open(log_file, "r") as f:
                    for line in f:
                        entry = json.loads(line)
                        for filter_name, passed in entry.get("filters_passed", {}).items():
                            if not passed:
                                filter_blocks[filter_name] = filter_blocks.get(filter_name, 0) + 1

        return dict(sorted(filter_blocks.items(), key=lambda x: x[1], reverse=True))

    def get_consensus_breakdown(self, days: int = 7) -> Dict[str, int]:
        """Get breakdown of consensus levels."""
        consensus_counts = {"100%": 0, "75%": 0, "50%": 0, "rejected": 0}

        for i in range(days):
            date_str = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            log_file = self.log_dir / f"explainability_{date_str}.jsonl"

            if log_file.exists():
                with open(log_file, "r") as f:
                    for line in f:
                        entry = json.loads(line)
                        if entry.get("status") == "REJECTED":
                            consensus_counts["rejected"] += 1
                        else:
                            level = entry.get("consensus_level", 0)
                            if level == 1.0:
                                consensus_counts["100%"] += 1
                            elif level == 0.75:
                                consensus_counts["75%"] += 1
                            elif level == 0.5:
                                consensus_counts["50%"] += 1

        return consensus_counts


# Utility imports for query methods
from datetime import timedelta


# Global singleton instance
_explainability_logger = None

def get_explainability_logger() -> ExplainabilityLogger:
    """Get global ExplainabilityLogger singleton."""
    global _explainability_logger
    if _explainability_logger is None:
        _explainability_logger = ExplainabilityLogger()
    return _explainability_logger
