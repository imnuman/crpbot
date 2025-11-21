"""
Signal Synthesizer for V7 Ultimate

Combines outputs from 6 mathematical theories into a structured prompt
for DeepSeek LLM to generate trading signals.

Theories:
1. Shannon Entropy - Market predictability
2. Hurst Exponent - Trend vs mean-reversion
3. Markov Chain - Regime detection
4. Kalman Filter - Price denoising
5. Bayesian Inference - Win rate learning
6. Monte Carlo - Risk simulation
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MarketContext:
    """Market context for signal generation"""
    symbol: str
    current_price: float
    timeframe: str
    timestamp: datetime
    recent_prices: Optional[np.ndarray] = None
    volume: Optional[float] = None
    spread: Optional[float] = None


@dataclass
class TheoryAnalysis:
    """Analysis results from all 6 theories"""
    # Shannon Entropy
    entropy: float
    entropy_interpretation: Dict[str, Any]

    # Hurst Exponent
    hurst: float
    hurst_interpretation: str

    # Markov Chain
    current_regime: str
    regime_probabilities: Dict[str, float]

    # Kalman Filter
    denoised_price: float
    price_momentum: float

    # Bayesian Inference
    win_rate_estimate: float
    win_rate_confidence: float

    # Monte Carlo
    risk_metrics: Dict[str, float]


class SignalSynthesizer:
    """
    Synthesize trading signals using 6 mathematical theories + DeepSeek LLM

    Workflow:
    1. Collect analysis from all 6 theories
    2. Format into structured prompt for DeepSeek
    3. Request LLM signal generation
    4. Parse and validate LLM response

    Usage:
        synthesizer = SignalSynthesizer()

        # Prepare market context
        context = MarketContext(
            symbol="BTC-USD",
            current_price=45000.0,
            timeframe="1h",
            timestamp=datetime.now()
        )

        # Prepare theory analysis
        analysis = TheoryAnalysis(
            entropy=0.65,
            entropy_interpretation={...},
            hurst=0.72,
            # ... other theories
        )

        # Generate prompt
        prompt = synthesizer.build_prompt(context, analysis)
    """

    # System prompt for DeepSeek
    SYSTEM_PROMPT = """You are an expert quantitative trading analyst specializing in cryptocurrency markets. Your role is to synthesize complex mathematical analysis into actionable trading signals.

You analyze market conditions using 7 theories:
1. Shannon Entropy - Market predictability (0=predictable, 1=random)
2. Hurst Exponent - Trend persistence (>0.5=trending, <0.5=mean-reverting)
3. Markov Chain - Market regime classification
4. Kalman Filter - Price denoising and momentum
5. Bayesian Inference - Strategy performance learning
6. Monte Carlo - Risk assessment and scenario analysis
7. Market Context (CoinGecko) - Macro market conditions, sentiment, and liquidity

**CRITICAL OUTPUT FORMAT REQUIREMENT:**
You MUST respond in this EXACT format - do NOT deviate from this structure:

SIGNAL: [BUY/SELL/HOLD]
CONFIDENCE: [0-100]%
ENTRY PRICE: $[number or N/A]
STOP LOSS: $[number or N/A]
TAKE PROFIT: $[number or N/A]
REASONING: [2-3 sentences explaining signal and price levels]

This format is MANDATORY. Your response will be parsed programmatically, so any deviation will cause system errors.

IMPORTANT: This is a MANUAL trading system. You generate signals for a human trader to review and execute. Do NOT execute trades automatically."""

    def __init__(self, conservative_mode: bool = True):
        """
        Initialize Signal Synthesizer

        Args:
            conservative_mode: If True, emphasize risk management in prompts
        """
        self.conservative_mode = conservative_mode
        logger.info(
            f"SignalSynthesizer initialized | "
            f"Conservative: {conservative_mode}"
        )

    def build_prompt(
        self,
        context: MarketContext,
        analysis: TheoryAnalysis,
        additional_context: Optional[str] = None,
        coingecko_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Build DeepSeek chat prompt from market context and theory analysis

        Args:
            context: Market context (symbol, price, timeframe)
            analysis: Results from all 7 theories (6 mathematical + CoinGecko)
            additional_context: Optional additional context (news, etc.)
            coingecko_context: Optional CoinGecko market context (7th theory)

        Returns:
            List of message dicts for DeepSeek chat API
        """
        # Format theory analysis into structured text
        theory_summary = self._format_theory_analysis(analysis)

        # Build user prompt
        user_prompt = f"""**Market Context:**
Symbol: {context.symbol}
Current Price: ${context.current_price:,.2f}
Timeframe: {context.timeframe}
Timestamp: {context.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
"""

        # Add spread if available
        if context.spread is not None:
            user_prompt += f"Spread: {context.spread:.2f}%\n"

        # Add volume if available
        if context.volume is not None:
            user_prompt += f"Volume: {context.volume:,.0f}\n"

        user_prompt += f"\n**Mathematical Analysis:**\n{theory_summary}\n"

        # Add CoinGecko market context if available (7th theory)
        if coingecko_context:
            user_prompt += "\n**7. Market Context (CoinGecko):**\n"
            user_prompt += f"   - Market Cap: ${coingecko_context.get('market_cap_billions', 0):.1f}B\n"
            user_prompt += f"   - Trading Volume: ${coingecko_context.get('volume_billions', 0):.1f}B\n"
            user_prompt += f"   - ATH Distance: {coingecko_context.get('ath_distance_pct', 0):.1f}%\n"
            user_prompt += f"   - Market Sentiment: {coingecko_context.get('sentiment', 'neutral')}\n"
            user_prompt += f"   - Liquidity Score: {coingecko_context.get('liquidity_score', 0):.3f}\n"
            user_prompt += f"   - Market Strength: {coingecko_context.get('market_strength', 0):.1%}\n"
            if coingecko_context.get('notes'):
                user_prompt += f"   - Notes: {coingecko_context['notes']}\n"

        # Add additional context if provided
        if additional_context:
            user_prompt += f"\n**Additional Context:**\n{additional_context}\n"

        # Add conservative mode disclaimer with momentum priority
        if self.conservative_mode:
            user_prompt += """
**Risk Management: FTMO-COMPLIANT**
Apply proper risk management and position sizing.

**CRITICAL SIGNAL GENERATION RULES**:
1. **In Choppy/Ranging Markets** (high entropy >0.85, consolidation regime):
   - PRIORITIZE momentum signals (Kalman momentum, Hurst exponent)
   - Strong momentum (>±15) with trending Hurst (>0.55) = ACTIONABLE SIGNAL
   - Don't let negative Sharpe ratios paralyze you - they're backward-looking

2. **Price Action Override**:
   - If Kalman momentum >+20 and Hurst >0.55: Consider BUY (35-55% confidence)
   - If Kalman momentum <-20 and Hurst <0.45: Consider SELL (35-55% confidence)
   - Clear directional movement >0.5% = tradeable opportunity

3. **Confidence Calibration**:
   - High entropy + strong momentum = 35-45% confidence (ACCEPTABLE in ranging markets)
   - Trending market + aligned theories = 60-75% confidence
   - Conflicting signals = 20-35% confidence or HOLD

Recommend BUY/SELL when momentum is clear, even if other metrics are mixed. HOLD only when truly no edge exists."""

        # Request format with price targets (SIMPLIFIED - system prompt now has format requirement)
        user_prompt += """

**YOUR RESPONSE MUST FOLLOW THIS EXACT FORMAT:**

SIGNAL: [BUY/SELL/HOLD]
CONFIDENCE: [0-100]%
ENTRY PRICE: $[number or N/A for HOLD]
STOP LOSS: $[number or N/A for HOLD]
TAKE PROFIT: $[number or N/A for HOLD]
REASONING: [2-3 sentences - explain signal + justify price levels]

**Examples:**

BUY Signal:
SIGNAL: BUY
CONFIDENCE: 75%
ENTRY PRICE: $91,234
STOP LOSS: $90,500
TAKE PROFIT: $92,800
REASONING: Strong bullish momentum (Hurst 0.72 trending) + bull regime (65%). SL at support $90,500 (0.8% risk), TP at Fib 1.618 $92,800 (1.7% reward, R:R 1:2.1).

HOLD Signal:
SIGNAL: HOLD
CONFIDENCE: 45%
ENTRY PRICE: N/A
STOP LOSS: N/A
TAKE PROFIT: N/A
REASONING: High entropy (0.89) indicates random market. Insufficient edge for trade.

**Price Level Guidelines:**
- Entry: Current price or specific limit level
- Stop Loss: 0.5-2% risk from entry
- Take Profit: Minimum 1:1.5 R:R ratio
- HOLD signals: Use N/A for all prices"""

        # Build messages list
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        theories_count = 7 if coingecko_context else 6
        logger.debug(
            f"Built prompt for {context.symbol} | "
            f"Theories: {theories_count} | "
            f"CoinGecko: {bool(coingecko_context)} | "
            f"Conservative: {self.conservative_mode}"
        )

        return messages

    def build_minimal_prompt(
        self,
        context: MarketContext,
        additional_context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build MINIMAL prompt with ONLY price/volume data - NO mathematical theories

        This is for A/B testing to determine if our mathematical analysis actually helps.
        DeepSeek will use ONLY its own knowledge with basic price data.

        Args:
            context: Market context (symbol, price, timeframe, recent prices)
            additional_context: Optional additional context (news, etc.)

        Returns:
            List of message dicts for DeepSeek chat API
        """
        # Build simple user prompt with only basic data
        user_prompt = f"""**Market Context:**
Symbol: {context.symbol}
Current Price: ${context.current_price:,.2f}
Timeframe: {context.timeframe}
Timestamp: {context.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
"""

        # Add spread if available
        if context.spread is not None:
            user_prompt += f"Spread: {context.spread:.2f}%\n"

        # Add volume if available
        if context.volume is not None:
            user_prompt += f"Volume: {context.volume:,.0f}\n"

        # Add recent price action if available
        if context.recent_prices is not None and len(context.recent_prices) > 0:
            user_prompt += f"\n**Recent Price Action (last {len(context.recent_prices)} candles):**\n"
            # Show last 10 prices to give context
            recent = context.recent_prices[-10:] if len(context.recent_prices) > 10 else context.recent_prices
            user_prompt += f"Prices: " + ", ".join([f"${p:,.2f}" for p in recent]) + "\n"

            # Simple price change
            if len(context.recent_prices) >= 2:
                price_change = ((context.recent_prices[-1] - context.recent_prices[0]) / context.recent_prices[0]) * 100
                user_prompt += f"Price Change: {price_change:+.2f}%\n"

        # Add additional context if provided
        if additional_context:
            user_prompt += f"\n**Additional Context:**\n{additional_context}\n"

        # Simple task instruction
        user_prompt += """
**Your Task:**
Analyze this cryptocurrency market using your knowledge and experience.
Generate a trading signal: BUY, SELL, or HOLD.

**YOUR RESPONSE MUST FOLLOW THIS EXACT FORMAT:**

SIGNAL: [BUY/SELL/HOLD]
CONFIDENCE: [0-100]%
ENTRY PRICE: $[number or N/A for HOLD]
STOP LOSS: $[number or N/A for HOLD]
TAKE PROFIT: $[number or N/A for HOLD]
REASONING: [2-3 sentences explaining signal and price levels]

**Price Level Guidelines:**
- Entry: Current price or specific limit level
- Stop Loss: 0.5-2% risk from entry
- Take Profit: Minimum 1:1.5 R:R ratio
- HOLD signals: Use N/A for all prices"""

        # Simplified system prompt (no theory mentions)
        minimal_system_prompt = """You are an expert cryptocurrency trading analyst. Your role is to analyze market conditions and generate actionable trading signals.

**CRITICAL OUTPUT FORMAT REQUIREMENT:**
You MUST respond in this EXACT format - do NOT deviate from this structure:

SIGNAL: [BUY/SELL/HOLD]
CONFIDENCE: [0-100]%
ENTRY PRICE: $[number or N/A]
STOP LOSS: $[number or N/A]
TAKE PROFIT: $[number or N/A]
REASONING: [2-3 sentences explaining signal and price levels]

This format is MANDATORY. Your response will be parsed programmatically, so any deviation will cause system errors.

IMPORTANT: This is a MANUAL trading system. You generate signals for a human trader to review and execute. Do NOT execute trades automatically."""

        # Build messages list
        messages = [
            {"role": "system", "content": minimal_system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        logger.debug(
            f"Built MINIMAL prompt for {context.symbol} | "
            f"Strategy: v7_deepseek_only (NO math theories) | "
            f"A/B Test Mode"
        )

        return messages

    def _format_theory_analysis(self, analysis: TheoryAnalysis) -> str:
        """
        Format theory analysis into human-readable text

        Args:
            analysis: TheoryAnalysis object

        Returns:
            Formatted string
        """
        lines = []

        # 1. Shannon Entropy
        lines.append(f"1. **Shannon Entropy**: {analysis.entropy:.3f}")
        lines.append(
            f"   - Predictability: {analysis.entropy_interpretation.get('predictability', 'unknown')}"
        )
        lines.append(
            f"   - Market Regime: {analysis.entropy_interpretation.get('regime', 'unknown')}"
        )
        lines.append(
            f"   - Trading Difficulty: {analysis.entropy_interpretation.get('trading_difficulty', 'unknown')}"
        )

        # 2. Hurst Exponent
        lines.append(f"\n2. **Hurst Exponent**: {analysis.hurst:.3f}")
        lines.append(f"   - Interpretation: {analysis.hurst_interpretation}")
        if analysis.hurst > 0.5:
            lines.append(f"   - Market Type: Trending (momentum strategies favored)")
        elif analysis.hurst < 0.5:
            lines.append(f"   - Market Type: Mean-reverting (contrarian strategies favored)")
        else:
            lines.append(f"   - Market Type: Random walk (no clear edge)")

        # 3. Markov Chain
        lines.append(f"\n3. **Markov Chain Regime**: {analysis.current_regime}")
        lines.append(f"   - Regime Confidence:")
        for regime, prob in sorted(
            analysis.regime_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]:  # Top 3 regimes
            lines.append(f"     - {regime}: {prob*100:.1f}%")

        # 4. Kalman Filter
        lines.append(f"\n4. **Kalman Filter**:")
        lines.append(f"   - Denoised Price: ${analysis.denoised_price:,.2f}")
        lines.append(f"   - Price Momentum: {analysis.price_momentum:+.4f}")
        momentum_direction = "bullish" if analysis.price_momentum > 0 else "bearish"
        lines.append(f"   - Momentum Direction: {momentum_direction}")

        # 5. Bayesian Inference
        lines.append(f"\n5. **Bayesian Win Rate**: {analysis.win_rate_estimate*100:.1f}%")
        lines.append(f"   - Confidence Interval: ±{analysis.win_rate_confidence*100:.1f}%")
        if analysis.win_rate_estimate > 0.6:
            lines.append(f"   - Strategy Performance: Strong")
        elif analysis.win_rate_estimate > 0.5:
            lines.append(f"   - Strategy Performance: Moderate")
        else:
            lines.append(f"   - Strategy Performance: Weak")

        # 6. Monte Carlo Risk
        lines.append(f"\n6. **Monte Carlo Risk Assessment**:")
        if 'var_95' in analysis.risk_metrics:
            lines.append(f"   - Value at Risk (95%): {analysis.risk_metrics['var_95']*100:.1f}%")
        if 'cvar_95' in analysis.risk_metrics:
            lines.append(f"   - Expected Shortfall: {analysis.risk_metrics['cvar_95']*100:.1f}%")
        if 'max_drawdown' in analysis.risk_metrics:
            lines.append(f"   - Max Drawdown: {analysis.risk_metrics['max_drawdown']*100:.1f}%")
        if 'sharpe_ratio' in analysis.risk_metrics:
            lines.append(f"   - Sharpe Ratio: {analysis.risk_metrics['sharpe_ratio']:.2f}")
        if 'profit_probability' in analysis.risk_metrics:
            lines.append(
                f"   - Probability of Profit: {analysis.risk_metrics['profit_probability']*100:.1f}%"
            )

        return "\n".join(lines)

    def validate_prompt(self, messages: List[Dict[str, str]]) -> bool:
        """
        Validate prompt structure before sending to LLM

        Args:
            messages: List of message dicts

        Returns:
            True if valid, False otherwise
        """
        if not messages:
            logger.error("Empty messages list")
            return False

        # Check for system message
        if messages[0].get('role') != 'system':
            logger.error("Missing system message")
            return False

        # Check for user message
        if len(messages) < 2 or messages[1].get('role') != 'user':
            logger.error("Missing user message")
            return False

        # Check content not empty
        for msg in messages:
            if not msg.get('content'):
                logger.error(f"Empty content in {msg.get('role')} message")
                return False

        logger.debug(f"Prompt validated successfully ({len(messages)} messages)")
        return True

    def estimate_token_count(self, messages: List[Dict[str, str]]) -> int:
        """
        Estimate token count for prompt (rough approximation)

        Args:
            messages: List of message dicts

        Returns:
            Estimated token count
        """
        # Rough approximation: 1 token ≈ 4 characters
        total_chars = sum(len(msg['content']) for msg in messages)
        estimated_tokens = total_chars // 4

        logger.debug(f"Estimated tokens: {estimated_tokens} (~{total_chars} chars)")
        return estimated_tokens


# Convenience function for quick prompt building
def build_signal_prompt(
    symbol: str,
    current_price: float,
    entropy: float,
    hurst: float,
    regime: str,
    denoised_price: float,
    momentum: float,
    win_rate: float,
    risk_metrics: Dict[str, float],
    conservative: bool = True
) -> List[Dict[str, str]]:
    """
    Quick prompt builder with simplified parameters

    Args:
        symbol: Trading symbol (e.g., "BTC-USD")
        current_price: Current market price
        entropy: Shannon entropy (0-1)
        hurst: Hurst exponent
        regime: Market regime from Markov chain
        denoised_price: Kalman filtered price
        momentum: Kalman price momentum
        win_rate: Bayesian win rate estimate
        risk_metrics: Monte Carlo risk metrics dict
        conservative: Conservative mode flag

    Returns:
        Messages list for DeepSeek API
    """
    synthesizer = SignalSynthesizer(conservative_mode=conservative)

    context = MarketContext(
        symbol=symbol,
        current_price=current_price,
        timeframe="1h",
        timestamp=datetime.now()
    )

    analysis = TheoryAnalysis(
        entropy=entropy,
        entropy_interpretation={
            'predictability': 'high' if entropy < 0.4 else 'low' if entropy > 0.7 else 'medium',
            'regime': 'trending' if entropy < 0.4 else 'random_walk' if entropy > 0.7 else 'mixed',
            'trading_difficulty': 'easy' if entropy < 0.4 else 'hard' if entropy > 0.7 else 'moderate'
        },
        hurst=hurst,
        hurst_interpretation=(
            "Strong trending" if hurst > 0.6 else
            "Strong mean-reversion" if hurst < 0.4 else
            "Random walk"
        ),
        current_regime=regime,
        regime_probabilities={regime: 0.8, 'OTHER': 0.2},
        denoised_price=denoised_price,
        price_momentum=momentum,
        win_rate_estimate=win_rate,
        win_rate_confidence=0.1,
        risk_metrics=risk_metrics
    )

    return synthesizer.build_prompt(context, analysis)


if __name__ == "__main__":
    # Test Signal Synthesizer
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    print("=" * 80)
    print("Signal Synthesizer - Test Run")
    print("=" * 80)

    # Test 1: Build prompt with sample data
    print("\n1. Testing Prompt Building")

    synthesizer = SignalSynthesizer(conservative_mode=True)

    context = MarketContext(
        symbol="BTC-USD",
        current_price=45250.0,
        timeframe="1h",
        timestamp=datetime.now(),
        spread=0.02,
        volume=1_500_000
    )

    analysis = TheoryAnalysis(
        # Shannon Entropy: Medium predictability
        entropy=0.55,
        entropy_interpretation={
            'entropy': 0.55,
            'predictability': 'medium',
            'regime': 'mixed',
            'trading_difficulty': 'moderate'
        },

        # Hurst: Trending market
        hurst=0.72,
        hurst_interpretation="Strong trending behavior",

        # Markov: Bull trend regime
        current_regime="BULL_TREND",
        regime_probabilities={
            'BULL_TREND': 0.65,
            'HIGH_VOL_RANGE': 0.20,
            'BREAKOUT': 0.10,
            'CONSOLIDATION': 0.05
        },

        # Kalman: Positive momentum
        denoised_price=45180.0,
        price_momentum=0.0025,

        # Bayesian: Good win rate
        win_rate_estimate=0.68,
        win_rate_confidence=0.08,

        # Monte Carlo: Moderate risk
        risk_metrics={
            'var_95': 0.12,
            'cvar_95': 0.18,
            'max_drawdown': 0.15,
            'sharpe_ratio': 1.2,
            'profit_probability': 0.72
        }
    )

    messages = synthesizer.build_prompt(context, analysis)

    print(f"\n   Generated {len(messages)} messages:")
    print(f"   - System: {len(messages[0]['content'])} chars")
    print(f"   - User: {len(messages[1]['content'])} chars")

    # Test 2: Validate prompt
    print("\n2. Testing Prompt Validation")
    is_valid = synthesizer.validate_prompt(messages)
    print(f"   Prompt valid: {'✅ Yes' if is_valid else '❌ No'}")

    # Test 3: Token estimation
    print("\n3. Testing Token Estimation")
    tokens = synthesizer.estimate_token_count(messages)
    print(f"   Estimated tokens: {tokens}")
    print(f"   Estimated cost: ${(tokens / 1_000_000) * 0.27:.6f} (input only)")

    # Test 4: Display full prompt
    print("\n4. Sample Prompt Preview")
    print("-" * 80)
    print("SYSTEM MESSAGE:")
    print(messages[0]['content'][:200] + "...")
    print("\nUSER MESSAGE:")
    print(messages[1]['content'])
    print("-" * 80)

    # Test 5: Conservative vs aggressive
    print("\n5. Testing Conservative vs Aggressive Mode")

    aggressive = SignalSynthesizer(conservative_mode=False)
    aggressive_messages = aggressive.build_prompt(context, analysis)

    conservative_len = len(messages[1]['content'])
    aggressive_len = len(aggressive_messages[1]['content'])

    print(f"   Conservative prompt: {conservative_len} chars")
    print(f"   Aggressive prompt: {aggressive_len} chars")
    print(f"   Difference: {abs(conservative_len - aggressive_len)} chars")

    # Test 6: Convenience function
    print("\n6. Testing Convenience Function")

    quick_prompt = build_signal_prompt(
        symbol="ETH-USD",
        current_price=2850.0,
        entropy=0.42,
        hurst=0.58,
        regime="CONSOLIDATION",
        denoised_price=2845.0,
        momentum=-0.0008,
        win_rate=0.64,
        risk_metrics={'var_95': 0.10, 'profit_probability': 0.68}
    )

    print(f"   Quick prompt generated: {len(quick_prompt)} messages")
    print(f"   Token estimate: {synthesizer.estimate_token_count(quick_prompt)}")

    print("\n" + "=" * 80)
    print("Signal Synthesizer Test Complete!")
    print("=" * 80)
    print("\nKey Features Verified:")
    print("  ✅ Prompt building from 6 theories")
    print("  ✅ Market context integration")
    print("  ✅ Conservative/aggressive modes")
    print("  ✅ Prompt validation")
    print("  ✅ Token estimation")
    print("  ✅ Convenience function")
    print("=" * 80)
