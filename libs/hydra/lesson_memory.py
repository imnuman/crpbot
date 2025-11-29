"""
HYDRA 3.0 - Lesson Memory (Upgrade C)

Never repeat the same mistake twice.

Lesson Storage:
- Every losing trade analyzed
- Failure pattern extracted
- Lesson permanently stored
- Future trades checked against lessons

Lesson Types:
1. Specific: "Never trade USD/TRY during Turkish elections"
2. Structural: "Session opens lose 70% in ranging regimes"
3. Timing: "London open strategies fail on Fridays"
4. Correlation: "BTC dumps always kill meme coins"

This is HYDRA's long-term memory - it learns and remembers.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from loguru import logger
from pathlib import Path
import json


class Lesson:
    """
    A single lesson learned from a failure.

    Structure:
    - lesson_id: Unique ID
    - failure_pattern: What went wrong
    - context: Market conditions when it failed
    - severity: How bad was the loss (1-10)
    - occurrences: How many times we've seen this
    - first_seen: When we first learned this
    - last_seen: Most recent occurrence
    - prevention_rule: How to avoid this in future
    """

    def __init__(
        self,
        lesson_id: str,
        failure_pattern: str,
        context: Dict,
        severity: int,
        prevention_rule: str
    ):
        self.lesson_id = lesson_id
        self.failure_pattern = failure_pattern
        self.context = context
        self.severity = severity  # 1-10 (10 = catastrophic)
        self.prevention_rule = prevention_rule
        self.occurrences = 1
        self.first_seen = datetime.now(timezone.utc)
        self.last_seen = datetime.now(timezone.utc)

    def to_dict(self) -> Dict:
        """Convert to dict for storage."""
        return {
            "lesson_id": self.lesson_id,
            "failure_pattern": self.failure_pattern,
            "context": self.context,
            "severity": self.severity,
            "prevention_rule": self.prevention_rule,
            "occurrences": self.occurrences,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Lesson':
        """Load from dict."""
        lesson = cls(
            lesson_id=data["lesson_id"],
            failure_pattern=data["failure_pattern"],
            context=data["context"],
            severity=data["severity"],
            prevention_rule=data["prevention_rule"]
        )
        lesson.occurrences = data.get("occurrences", 1)
        lesson.first_seen = datetime.fromisoformat(data["first_seen"])
        lesson.last_seen = datetime.fromisoformat(data["last_seen"])
        return lesson


class LessonMemory:
    """
    Long-term memory system that learns from failures.

    Workflow:
    1. Trade closes as loss
    2. Analyze: What went wrong?
    3. Extract pattern
    4. Store as lesson
    5. Before future trades: Check lessons
    6. If match found: REJECT trade
    """

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            storage_path = Path("/root/crpbot/data/hydra/lessons.jsonl")

        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory lesson storage
        self.lessons: Dict[str, Lesson] = {}

        # Load existing lessons
        self._load_lessons()

        logger.info(f"Lesson Memory initialized ({len(self.lessons)} lessons loaded)")

    # ==================== LEARNING (FROM FAILURES) ====================

    def learn_from_failure(
        self,
        trade_id: str,
        asset: str,
        regime: str,
        strategy: Dict,
        signal: Dict,
        trade_result: Dict,
        market_context: Dict
    ) -> Optional[Lesson]:
        """
        Analyze losing trade and extract lesson.

        Args:
            trade_result: {
                "outcome": "loss",
                "pnl_percent": -0.015,
                "exit_reason": "stop_loss",
                "entry_timestamp": datetime,
                "exit_timestamp": datetime,
                "slippage": 0.0005
            }
            market_context: {
                "dxy_change": 0.008,
                "btc_change": -0.03,
                "news_events": ["CB meeting"],
                "session": "London",
                "day_of_week": "Friday"
            }

        Returns:
            Lesson object (or None if no clear pattern)
        """
        if trade_result.get("outcome") != "loss":
            return None

        loss_pct = abs(trade_result.get("pnl_percent", 0))

        # Calculate severity (1-10)
        if loss_pct >= 0.03:  # 3%+ loss
            severity = 10  # Catastrophic
        elif loss_pct >= 0.02:  # 2-3% loss
            severity = 8
        elif loss_pct >= 0.015:  # 1.5-2% loss
            severity = 6
        elif loss_pct >= 0.01:  # 1-1.5% loss
            severity = 4
        else:  # <1% loss
            severity = 2

        # Analyze WHY it failed
        failure_analysis = self._analyze_failure(
            asset=asset,
            regime=regime,
            strategy=strategy,
            signal=signal,
            trade_result=trade_result,
            market_context=market_context
        )

        if not failure_analysis:
            logger.warning(f"Could not analyze failure for {trade_id}")
            return None

        # Create lesson
        lesson_id = f"LESSON_{len(self.lessons):04d}_{asset}_{regime}"

        lesson = Lesson(
            lesson_id=lesson_id,
            failure_pattern=failure_analysis["pattern"],
            context=failure_analysis["context"],
            severity=severity,
            prevention_rule=failure_analysis["prevention"]
        )

        # Check if we already learned this lesson
        existing = self._find_similar_lesson(lesson)
        if existing:
            # Increment occurrences
            existing.occurrences += 1
            existing.last_seen = datetime.now(timezone.utc)
            existing.severity = max(existing.severity, severity)  # Update if worse

            logger.warning(
                f"Lesson reinforced: {existing.lesson_id} (now {existing.occurrences}x, "
                f"severity {existing.severity}/10)"
            )

            self._save_lessons()
            return existing
        else:
            # New lesson
            self.lessons[lesson_id] = lesson

            logger.error(
                f"NEW LESSON LEARNED: {lesson_id}\n"
                f"  Pattern: {lesson.failure_pattern}\n"
                f"  Prevention: {lesson.prevention_rule}\n"
                f"  Severity: {severity}/10"
            )

            self._save_lessons()
            return lesson

    def _analyze_failure(
        self,
        asset: str,
        regime: str,
        strategy: Dict,
        signal: Dict,
        trade_result: Dict,
        market_context: Dict
    ) -> Optional[Dict]:
        """
        Analyze why trade failed and extract actionable pattern.

        Returns:
            {
                "pattern": "USD/TRY long during DXY surge",
                "context": {"asset": "USD/TRY", "dxy_change": 0.008, ...},
                "prevention": "Block USD/TRY LONG when DXY up >0.5%"
            }
        """
        patterns = []

        # Pattern 1: Cross-asset conflict
        dxy_change = market_context.get("dxy_change", 0)
        btc_change = market_context.get("btc_change", 0)

        if "USD" in asset and abs(dxy_change) > 0.005:
            direction = signal.get("direction", "UNKNOWN")
            if dxy_change > 0 and direction == "SHORT" and asset.startswith("USD/"):
                patterns.append({
                    "pattern": f"{asset} SHORT during DXY surge (+{dxy_change:.2%})",
                    "context": {
                        "asset": asset,
                        "direction": direction,
                        "dxy_change": dxy_change,
                        "regime": regime
                    },
                    "prevention": f"Block {asset} SHORT when DXY up >{0.005:.2%}"
                })

        # Pattern 2: BTC correlation (for crypto)
        if asset.endswith("-USD") and abs(btc_change) > 0.03:
            direction = signal.get("direction", "UNKNOWN")
            if btc_change < -0.03 and direction == "LONG":
                patterns.append({
                    "pattern": f"{asset} LONG during BTC dump ({btc_change:.2%})",
                    "context": {
                        "asset": asset,
                        "direction": direction,
                        "btc_change": btc_change,
                        "regime": regime
                    },
                    "prevention": f"Block {asset} LONG when BTC down >{0.03:.2%}"
                })

        # Pattern 3: News events
        news_events = market_context.get("news_events", [])
        if news_events:
            patterns.append({
                "pattern": f"{asset} trade during {', '.join(news_events)}",
                "context": {
                    "asset": asset,
                    "news_events": news_events,
                    "regime": regime
                },
                "prevention": f"Block {asset} trades 1hr before/after: {', '.join(news_events)}"
            })

        # Pattern 4: Session timing
        session = market_context.get("session", "")
        day_of_week = market_context.get("day_of_week", "")
        structural_edge = strategy.get("structural_edge", "")

        if "session" in structural_edge.lower() and session:
            patterns.append({
                "pattern": f"{structural_edge} failed in {regime} regime ({session}, {day_of_week})",
                "context": {
                    "asset": asset,
                    "regime": regime,
                    "session": session,
                    "day_of_week": day_of_week,
                    "structural_edge": structural_edge
                },
                "prevention": f"Avoid '{structural_edge}' in {regime} regime"
            })

        # Pattern 5: Regime mismatch
        if "trending" in structural_edge.lower() and regime == "RANGING":
            patterns.append({
                "pattern": f"Trend-following strategy in RANGING regime",
                "context": {
                    "asset": asset,
                    "regime": regime,
                    "structural_edge": structural_edge
                },
                "prevention": f"Block trend strategies in RANGING regime"
            })

        # Pattern 6: Spread blowout
        slippage = trade_result.get("slippage", 0)
        if slippage > 0.002:  # 0.2%+ slippage
            patterns.append({
                "pattern": f"{asset} excessive slippage ({slippage:.2%})",
                "context": {
                    "asset": asset,
                    "slippage": slippage,
                    "session": session
                },
                "prevention": f"Block {asset} when spread >2x normal"
            })

        # Return strongest pattern
        if patterns:
            return patterns[0]  # Most specific pattern
        else:
            # Generic fallback
            return {
                "pattern": f"{asset} {signal.get('direction', 'trade')} in {regime} regime",
                "context": {
                    "asset": asset,
                    "regime": regime,
                    "strategy_id": strategy.get("strategy_id", "unknown")
                },
                "prevention": f"Review {asset} strategy performance in {regime}"
            }

    def _find_similar_lesson(self, lesson: Lesson) -> Optional[Lesson]:
        """Find existing lesson that matches this pattern."""
        for existing in self.lessons.values():
            # Check if patterns match (fuzzy)
            if self._patterns_match(existing.failure_pattern, lesson.failure_pattern):
                return existing

        return None

    def _patterns_match(self, pattern1: str, pattern2: str) -> bool:
        """Check if two patterns are similar enough to be the same lesson."""
        # Simple substring matching (could be improved with NLP)
        p1_lower = pattern1.lower()
        p2_lower = pattern2.lower()

        # Extract key components
        p1_words = set(p1_lower.split())
        p2_words = set(p2_lower.split())

        # Calculate Jaccard similarity
        intersection = p1_words.intersection(p2_words)
        union = p1_words.union(p2_words)

        if not union:
            return False

        similarity = len(intersection) / len(union)

        return similarity > 0.6  # 60% word overlap = same lesson

    # ==================== PREVENTION (BEFORE TRADES) ====================

    def check_lessons(
        self,
        asset: str,
        regime: str,
        strategy: Dict,
        signal: Dict,
        market_context: Dict
    ) -> Tuple[bool, Optional[Lesson]]:
        """
        Check if this trade matches any learned lessons.

        Returns:
            (should_reject: bool, matching_lesson: Optional[Lesson])
        """
        for lesson in self.lessons.values():
            # Skip low-severity lessons with only 1 occurrence
            if lesson.severity < 5 and lesson.occurrences == 1:
                continue

            # Check if context matches
            if self._context_matches(lesson, asset, regime, strategy, signal, market_context):
                logger.warning(
                    f"LESSON TRIGGERED: {lesson.lesson_id}\n"
                    f"  Pattern: {lesson.failure_pattern}\n"
                    f"  Prevention: {lesson.prevention_rule}\n"
                    f"  Occurrences: {lesson.occurrences}x, Severity: {lesson.severity}/10\n"
                    f"  → REJECTING TRADE"
                )
                return True, lesson

        return False, None

    def _context_matches(
        self,
        lesson: Lesson,
        asset: str,
        regime: str,
        strategy: Dict,
        signal: Dict,
        market_context: Dict
    ) -> bool:
        """Check if current trade context matches lesson context."""
        context = lesson.context

        # Check asset
        if context.get("asset") and context["asset"] != asset:
            return False

        # Check regime
        if context.get("regime") and context["regime"] != regime:
            return False

        # Check direction
        if context.get("direction") and context["direction"] != signal.get("direction"):
            return False

        # Check DXY (if applicable)
        if "dxy_change" in context:
            lesson_dxy = context["dxy_change"]
            current_dxy = market_context.get("dxy_change", 0)

            # Same direction and magnitude?
            if (lesson_dxy > 0 and current_dxy > 0.005) or (lesson_dxy < 0 and current_dxy < -0.005):
                return True

        # Check BTC (if applicable)
        if "btc_change" in context:
            lesson_btc = context["btc_change"]
            current_btc = market_context.get("btc_change", 0)

            if (lesson_btc < -0.03 and current_btc < -0.03) or (lesson_btc > 0.03 and current_btc > 0.03):
                return True

        # Check news events
        if "news_events" in context:
            current_news = market_context.get("news_events", [])
            if any(event in current_news for event in context["news_events"]):
                return True

        # Check structural edge
        if "structural_edge" in context:
            if context["structural_edge"] in strategy.get("structural_edge", ""):
                return True

        # Check session/day
        if "session" in context and context["session"] == market_context.get("session"):
            return True

        return False

    # ==================== PERSISTENCE ====================

    def _save_lessons(self):
        """Save lessons to JSONL file."""
        with open(self.storage_path, "w") as f:
            for lesson in self.lessons.values():
                f.write(json.dumps(lesson.to_dict()) + "\n")

        logger.debug(f"Saved {len(self.lessons)} lessons to {self.storage_path}")

    def _load_lessons(self):
        """Load lessons from JSONL file."""
        if not self.storage_path.exists():
            logger.info("No existing lessons found, starting fresh")
            return

        count = 0
        with open(self.storage_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        lesson = Lesson.from_dict(data)
                        self.lessons[lesson.lesson_id] = lesson
                        count += 1
                    except Exception as e:
                        logger.error(f"Failed to load lesson: {e}")

        logger.info(f"Loaded {count} lessons from {self.storage_path}")

    # ==================== STATISTICS & REPORTING ====================

    def get_lesson_stats(self) -> Dict:
        """Get statistics about learned lessons."""
        if not self.lessons:
            return {
                "total_lessons": 0,
                "avg_severity": 0,
                "total_occurrences": 0
            }

        severities = [l.severity for l in self.lessons.values()]
        occurrences = [l.occurrences for l in self.lessons.values()]

        return {
            "total_lessons": len(self.lessons),
            "avg_severity": sum(severities) / len(severities),
            "max_severity": max(severities),
            "total_occurrences": sum(occurrences),
            "most_common_lesson": max(self.lessons.values(), key=lambda l: l.occurrences).lesson_id,
            "most_severe_lesson": max(self.lessons.values(), key=lambda l: l.severity).lesson_id
        }

    def get_top_lessons(self, n: int = 10) -> List[Dict]:
        """Get top N most important lessons."""
        # Sort by (severity * occurrences)
        ranked = sorted(
            self.lessons.values(),
            key=lambda l: l.severity * l.occurrences,
            reverse=True
        )

        return [
            {
                "lesson_id": l.lesson_id,
                "pattern": l.failure_pattern,
                "prevention": l.prevention_rule,
                "severity": l.severity,
                "occurrences": l.occurrences,
                "importance": l.severity * l.occurrences
            }
            for l in ranked[:n]
        ]


# Global singleton instance
_lesson_memory = None

def get_lesson_memory() -> LessonMemory:
    """Get global LessonMemory singleton."""
    global _lesson_memory
    if _lesson_memory is None:
        _lesson_memory = LessonMemory()
    return _lesson_memory
