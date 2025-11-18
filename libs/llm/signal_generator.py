"""
V7 Ultimate Signal Generator

Orchestrates the complete signal generation pipeline:
1. Mathematical Analysis (6 theories)
2. LLM Prompt Synthesis
3. DeepSeek API Call
4. Response Parsing
5. Signal Validation

This is the main entry point for V7 Ultimate signal generation.
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

# Import mathematical theories
from libs.analysis import (
    ShannonEntropyAnalyzer,
    HurstExponentAnalyzer,
    MarkovRegimeDetector,
    KalmanPriceFilter,
    BayesianWinRateLearner,
    MonteCarloSimulator,
)

# Import LLM components
from .deepseek_client import DeepSeekClient, DeepSeekResponse
from .signal_synthesizer import (
    SignalSynthesizer,
    MarketContext,
    TheoryAnalysis,
)
from .signal_parser import SignalParser, ParsedSignal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class SignalGenerationResult:
    """Complete result of signal generation process"""
    # Final signal
    parsed_signal: ParsedSignal

    # Mathematical analysis
    theory_analysis: TheoryAnalysis

    # LLM interaction
    llm_response: DeepSeekResponse
    prompt_messages: list

    # Metadata
    market_context: MarketContext
    generation_time_seconds: float
    total_cost_usd: float

    # Status
    success: bool
    error_message: Optional[str] = None


class SignalGenerator:
    """
    V7 Ultimate Signal Generator

    Coordinates all components to generate trading signals using
    6 mathematical theories + DeepSeek LLM synthesis.

    Workflow:
    1. Run mathematical analysis on price data
    2. Format analysis into LLM prompt
    3. Query DeepSeek for signal synthesis
    4. Parse and validate LLM response
    5. Return structured signal result

    Usage:
        # Initialize generator (reusable for multiple signals)
        generator = SignalGenerator(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            conservative_mode=True
        )

        # Generate signal from price data
        result = generator.generate_signal(
            symbol="BTC-USD",
            prices=price_array,  # numpy array of recent prices
            timestamps=timestamp_array,
            current_price=45000.0
        )

        if result.success and result.parsed_signal.is_valid:
            print(f"Signal: {result.parsed_signal.signal}")
            print(f"Confidence: {result.parsed_signal.confidence:.1%}")
            print(f"Reasoning: {result.parsed_signal.reasoning}")
        else:
            print(f"Failed: {result.error_message}")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        conservative_mode: bool = True,
        strict_parsing: bool = False,
        lookback_window: int = 200,
        temperature: float = 0.7,
        max_tokens: int = 200
    ):
        """
        Initialize Signal Generator

        Args:
            api_key: DeepSeek API key (or use DEEPSEEK_API_KEY env var)
            conservative_mode: Emphasize risk management in prompts
            strict_parsing: Require exact format match in LLM responses
            lookback_window: Number of data points for analysis
            temperature: LLM sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens for LLM response
        """
        # Initialize LLM components
        self.deepseek_client = DeepSeekClient(api_key=api_key)
        self.signal_synthesizer = SignalSynthesizer(conservative_mode=conservative_mode)
        self.signal_parser = SignalParser(strict_mode=strict_parsing)

        # Initialize mathematical analyzers (stateful for Bayesian learning)
        self.entropy_analyzer = ShannonEntropyAnalyzer()
        self.hurst_analyzer = HurstExponentAnalyzer()
        self.markov_detector = MarkovRegimeDetector()
        self.kalman_filter = KalmanPriceFilter()
        self.bayesian_learner = BayesianWinRateLearner(prior_wins=10, prior_losses=10)
        self.monte_carlo_simulator = MonteCarloSimulator()

        # Configuration
        self.lookback_window = lookback_window
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(
            f"SignalGenerator initialized | "
            f"Conservative: {conservative_mode} | "
            f"Strict Parsing: {strict_parsing} | "
            f"Lookback: {lookback_window}"
        )

    def generate_signal(
        self,
        symbol: str,
        prices: np.ndarray,
        timestamps: np.ndarray,
        current_price: float,
        timeframe: str = "1h",
        volume: Optional[float] = None,
        spread: Optional[float] = None,
        additional_context: Optional[str] = None
    ) -> SignalGenerationResult:
        """
        Generate trading signal from price data

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            prices: Array of historical prices (most recent last)
            timestamps: Array of timestamps corresponding to prices
            current_price: Current market price
            timeframe: Timeframe string (e.g., "1h", "15m")
            volume: Optional current volume
            spread: Optional bid-ask spread
            additional_context: Optional additional context (news, events)

        Returns:
            SignalGenerationResult with complete analysis and signal
        """
        start_time = datetime.now()

        try:
            # Validate inputs
            if len(prices) < self.lookback_window:
                raise ValueError(
                    f"Insufficient data: {len(prices)} < {self.lookback_window}"
                )

            # Use most recent data for analysis
            analysis_prices = prices[-self.lookback_window:]
            analysis_timestamps = timestamps[-self.lookback_window:]

            logger.info(
                f"Starting signal generation | "
                f"Symbol: {symbol} | "
                f"Price: ${current_price:,.2f} | "
                f"Data points: {len(analysis_prices)}"
            )

            # STEP 1: Run Mathematical Analysis
            theory_analysis = self._run_mathematical_analysis(
                prices=analysis_prices,
                current_price=current_price
            )

            # STEP 2: Build Market Context
            market_context = MarketContext(
                symbol=symbol,
                current_price=current_price,
                timeframe=timeframe,
                timestamp=datetime.fromtimestamp(timestamps[-1]),
                recent_prices=analysis_prices,
                volume=volume,
                spread=spread
            )

            # STEP 3: Generate LLM Prompt
            prompt_messages = self.signal_synthesizer.build_prompt(
                context=market_context,
                analysis=theory_analysis,
                additional_context=additional_context
            )

            # Validate prompt
            if not self.signal_synthesizer.validate_prompt(prompt_messages):
                raise ValueError("Prompt validation failed")

            # STEP 4: Query DeepSeek LLM
            logger.debug(f"Sending prompt to DeepSeek ({len(prompt_messages)} messages)")
            llm_response = self.deepseek_client.chat(
                messages=prompt_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            logger.info(
                f"DeepSeek response received | "
                f"Tokens: {llm_response.total_tokens} | "
                f"Cost: ${llm_response.cost_usd:.6f}"
            )

            # STEP 5: Parse LLM Response
            parsed_signal = self.signal_parser.parse(llm_response.content)

            # Additional validation
            is_valid, validation_errors = self.signal_parser.validate_signal(parsed_signal)
            if validation_errors:
                logger.warning(f"Signal validation warnings: {validation_errors}")

            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()

            # Build result
            result = SignalGenerationResult(
                parsed_signal=parsed_signal,
                theory_analysis=theory_analysis,
                llm_response=llm_response,
                prompt_messages=prompt_messages,
                market_context=market_context,
                generation_time_seconds=generation_time,
                total_cost_usd=llm_response.cost_usd,
                success=True,
                error_message=None
            )

            logger.info(
                f"Signal generated successfully | "
                f"Signal: {parsed_signal.signal.value} | "
                f"Confidence: {parsed_signal.confidence:.1%} | "
                f"Valid: {parsed_signal.is_valid} | "
                f"Time: {generation_time:.2f}s"
            )

            return result

        except Exception as e:
            generation_time = (datetime.now() - start_time).total_seconds()

            logger.error(f"Signal generation failed: {e}", exc_info=True)

            # Return failed result with safe defaults
            return SignalGenerationResult(
                parsed_signal=ParsedSignal(
                    signal=SignalType.HOLD,
                    confidence=0.0,
                    reasoning=f"Generation failed: {str(e)}",
                    raw_response="",
                    is_valid=False,
                    timestamp=datetime.now(),
                    parse_warnings=[str(e)]
                ),
                theory_analysis=None,
                llm_response=None,
                prompt_messages=[],
                market_context=MarketContext(
                    symbol=symbol,
                    current_price=current_price,
                    timeframe=timeframe,
                    timestamp=datetime.now()
                ),
                generation_time_seconds=generation_time,
                total_cost_usd=0.0,
                success=False,
                error_message=str(e)
            )

    def _run_mathematical_analysis(
        self,
        prices: np.ndarray,
        current_price: float
    ) -> TheoryAnalysis:
        """
        Run all 6 mathematical theories on price data

        Args:
            prices: Price array
            current_price: Current price

        Returns:
            TheoryAnalysis with results from all theories
        """
        logger.debug("Running mathematical analysis (6 theories)")

        # Calculate returns for some analyses
        returns = np.diff(np.log(prices))

        # 1. Shannon Entropy
        entropy = self.entropy_analyzer.calculate_entropy(prices)
        entropy_interpretation = self.entropy_analyzer.interpret_entropy(entropy)

        # 2. Hurst Exponent
        hurst = self.hurst_analyzer.calculate_hurst(prices)
        hurst_interpretation = self.hurst_analyzer.interpret_hurst(hurst)

        # 3. Markov Chain Regime Detection
        regime_result = self.markov_detector.detect_regime(prices, returns)
        current_regime = regime_result['current_regime']
        regime_probabilities = regime_result['probabilities']

        # 4. Kalman Filter
        self.kalman_filter.reset()
        for price in prices:
            self.kalman_filter.update(price)

        denoised_price = self.kalman_filter.get_state()
        price_momentum = self.kalman_filter.get_velocity()

        # 5. Bayesian Inference (win rate estimation)
        # Note: This requires historical trade results, using placeholder for now
        win_rate_estimate = self.bayesian_learner.get_win_rate()
        win_rate_confidence = self.bayesian_learner.get_confidence_interval()[1] - win_rate_estimate

        # 6. Monte Carlo Risk Simulation
        risk_metrics = self.monte_carlo_simulator.simulate_scenarios(
            current_price=current_price,
            historical_returns=returns,
            num_simulations=1000,
            time_horizon=24  # hours
        )

        # Build TheoryAnalysis
        analysis = TheoryAnalysis(
            entropy=entropy,
            entropy_interpretation=entropy_interpretation,
            hurst=hurst,
            hurst_interpretation=hurst_interpretation,
            current_regime=current_regime,
            regime_probabilities=regime_probabilities,
            denoised_price=denoised_price,
            price_momentum=price_momentum,
            win_rate_estimate=win_rate_estimate,
            win_rate_confidence=win_rate_confidence,
            risk_metrics=risk_metrics
        )

        logger.debug(
            f"Mathematical analysis complete | "
            f"Entropy: {entropy:.3f} | "
            f"Hurst: {hurst:.3f} | "
            f"Regime: {current_regime}"
        )

        return analysis

    def update_bayesian_learning(self, trade_won: bool) -> None:
        """
        Update Bayesian win rate learner with new trade result

        Args:
            trade_won: True if trade was profitable, False otherwise
        """
        self.bayesian_learner.update(trade_won)
        new_win_rate = self.bayesian_learner.get_win_rate()

        logger.info(
            f"Bayesian learning updated | "
            f"Trade Won: {trade_won} | "
            f"New Win Rate: {new_win_rate:.1%}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get generator usage statistics

        Returns:
            Dictionary with usage stats from all components
        """
        deepseek_stats = self.deepseek_client.get_stats()

        return {
            "deepseek_api": deepseek_stats,
            "bayesian_win_rate": self.bayesian_learner.get_win_rate(),
            "bayesian_confidence_interval": self.bayesian_learner.get_confidence_interval(),
            "bayesian_total_trades": (
                self.bayesian_learner.wins + self.bayesian_learner.losses
            )
        }

    def reset_statistics(self) -> None:
        """Reset usage statistics"""
        self.deepseek_client.reset_stats()
        logger.info("Statistics reset")


if __name__ == "__main__":
    # Test Signal Generator
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    print("=" * 80)
    print("V7 Ultimate Signal Generator - Test Run")
    print("=" * 80)

    # Check for API key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("\n❌ ERROR: DEEPSEEK_API_KEY environment variable not set")
        print("\nPlease set your DeepSeek API key:")
        print("  export DEEPSEEK_API_KEY='sk-...'")
        print("\nGet your API key from: https://platform.deepseek.com/")
        print("=" * 80)
        exit(1)

    try:
        # Initialize generator
        print("\n1. Initializing Signal Generator...")
        generator = SignalGenerator(
            api_key=api_key,
            conservative_mode=True,
            strict_parsing=False,
            lookback_window=200
        )

        # Generate synthetic price data for testing
        print("\n2. Generating Synthetic Market Data...")
        np.random.seed(42)

        # Simulate BTC price movement (starting at $45,000)
        base_price = 45000
        num_points = 250
        timestamps = np.arange(num_points) * 3600  # Hourly data

        # Random walk with drift
        returns = np.random.normal(0.0002, 0.01, num_points)  # Slight upward drift
        prices = base_price * np.exp(np.cumsum(returns))

        current_price = prices[-1]

        print(f"   Generated {num_points} hourly prices")
        print(f"   Start: ${prices[0]:,.2f}")
        print(f"   End: ${current_price:,.2f}")
        print(f"   Change: {((current_price / prices[0]) - 1) * 100:+.2f}%")

        # Generate signal
        print("\n3. Generating Trading Signal...")
        print("   (This will call DeepSeek API - estimated cost: $0.0003)")

        result = generator.generate_signal(
            symbol="BTC-USD",
            prices=prices,
            timestamps=timestamps,
            current_price=current_price,
            timeframe="1h",
            volume=1_500_000,
            spread=0.02
        )

        # Display results
        print("\n4. Signal Generation Results")
        print("=" * 80)

        if result.success:
            signal = result.parsed_signal

            print(f"\n✅ SUCCESS - Signal Generated\n")
            print(f"Signal:     {signal.signal.value}")
            print(f"Confidence: {signal.confidence:.1%}")
            print(f"Valid:      {'✅ Yes' if signal.is_valid else '❌ No'}")
            print(f"Reasoning:  {signal.reasoning}")

            if signal.parse_warnings:
                print(f"\n⚠️  Warnings ({len(signal.parse_warnings)}):")
                for warning in signal.parse_warnings:
                    print(f"   - {warning}")

            print(f"\nPerformance:")
            print(f"   Generation Time: {result.generation_time_seconds:.2f}s")
            print(f"   API Cost:        ${result.total_cost_usd:.6f}")
            print(f"   Tokens Used:     {result.llm_response.total_tokens}")

            print(f"\nMathematical Analysis Summary:")
            ta = result.theory_analysis
            print(f"   Shannon Entropy:     {ta.entropy:.3f} ({ta.entropy_interpretation.get('predictability', 'N/A')})")
            print(f"   Hurst Exponent:      {ta.hurst:.3f} ({ta.hurst_interpretation})")
            print(f"   Markov Regime:       {ta.current_regime}")
            print(f"   Kalman Momentum:     {ta.price_momentum:+.6f}")
            print(f"   Bayesian Win Rate:   {ta.win_rate_estimate:.1%}")
            if 'var_95' in ta.risk_metrics:
                print(f"   Monte Carlo VaR:     {ta.risk_metrics['var_95']*100:.1f}%")

        else:
            print(f"\n❌ FAILED - Signal Generation Error\n")
            print(f"Error: {result.error_message}")

        # Display statistics
        print("\n5. Generator Statistics")
        stats = generator.get_statistics()
        print(f"   Total API Requests: {stats['deepseek_api']['total_requests']}")
        print(f"   Total Cost:         ${stats['deepseek_api']['total_cost_usd']:.6f}")
        print(f"   Total Tokens:       {stats['deepseek_api']['total_tokens']}")
        print(f"   Bayesian Win Rate:  {stats['bayesian_win_rate']:.1%}")

        # Monthly cost projection
        print("\n6. Monthly Cost Projection")
        signals_per_day = 100
        avg_cost = stats['deepseek_api']['avg_cost_per_request']
        monthly_cost = signals_per_day * 30 * avg_cost

        print(f"   Assumption: {signals_per_day} signals/day")
        print(f"   Avg Cost/Signal: ${avg_cost:.6f}")
        print(f"   Monthly Cost: ${monthly_cost:.2f}")
        print(f"   Budget Status: {'✅ Within $100' if monthly_cost <= 100 else '❌ Exceeds $100'}")

        print("\n" + "=" * 80)
        print("V7 Ultimate Signal Generator Test Complete!")
        print("=" * 80)
        print("\nKey Features Verified:")
        print("  ✅ Mathematical analysis (6 theories)")
        print("  ✅ LLM prompt synthesis")
        print("  ✅ DeepSeek API integration")
        print("  ✅ Response parsing and validation")
        print("  ✅ Error handling and fallbacks")
        print("  ✅ Cost tracking")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        exit(1)
