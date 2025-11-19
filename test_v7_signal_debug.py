"""
Debug script to see exact DeepSeek LLM responses for V7 signals
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from apps.runtime.data_fetcher import get_data_fetcher
from libs.llm.signal_generator import SignalGenerator
from libs.config.config import Settings

# Configure logging
logger.remove()
logger.add(lambda msg: print(msg, end=''), level="DEBUG")

def test_v7_signal():
    """Generate one V7 signal with full debug output"""

    # Set API key
    os.environ['DEEPSEEK_API_KEY'] = 'sk-cb86184fcb974480a20615749781c198'

    config = Settings()
    fetcher = get_data_fetcher(config)

    # Fetch recent data (need 200+ for V7)
    logger.info("Fetching latest market data...")
    end = datetime.now()
    start = end - timedelta(hours=4)  # 4 hours = 240 minutes

    df = fetcher.fetch_historical_candles('BTC-USD', start, end)
    logger.info(f"Fetched {len(df)} candles")

    if len(df) < 200:
        logger.error(f"Not enough data: {len(df)} < 200")
        return

    # Take last 200 candles
    window = df.iloc[-200:].copy()

    # Prepare data
    prices = window['close'].values
    timestamps = window['timestamp'].values.view('int64') / 10**9  # nanoseconds to seconds
    current_price = float(prices[-1])
    volume = float(window['volume'].iloc[-1]) if 'volume' in window.columns else None

    logger.info(f"\nCurrent price: ${current_price:,.2f}")
    logger.info(f"Price range: ${prices.min():,.2f} - ${prices.max():,.2f}")
    logger.info(f"Price change: {((prices[-1] / prices[0]) - 1) * 100:.2f}%")

    # Initialize V7 generator
    logger.info("\nInitializing V7 Signal Generator...")
    orchestrator = SignalGenerator(
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        conservative_mode=True
    )

    # Generate signal
    logger.info("\n" + "="*80)
    logger.info("GENERATING V7 SIGNAL")
    logger.info("="*80)

    result = orchestrator.generate_signal(
        symbol='BTC-USD',
        prices=prices,
        timestamps=timestamps,
        current_price=current_price,
        timeframe='1h',
        volume=volume
    )

    # Display results
    logger.info("\n" + "="*80)
    logger.info("V7 SIGNAL RESULT")
    logger.info("="*80)

    logger.info(f"\nSignal: {result.parsed_signal.signal.value}")
    logger.info(f"Confidence: {result.parsed_signal.confidence:.2%}")
    logger.info(f"Valid: {result.parsed_signal.is_valid}")
    logger.info(f"Timestamp: {result.parsed_signal.timestamp}")

    logger.info(f"\n--- LLM REASONING ---")
    logger.info(result.parsed_signal.reasoning)

    logger.info(f"\n--- THEORY ANALYSIS ---")
    theories = result.theory_analysis
    if theories is not None:
        if theories.entropy is not None:
            logger.info(f"Shannon Entropy: {theories.entropy:.3f}")
        if theories.hurst_exponent is not None:
            logger.info(f"Hurst Exponent: {theories.hurst_exponent:.3f}")
        if theories.kolmogorov_complexity is not None:
            logger.info(f"Kolmogorov Complexity: {theories.kolmogorov_complexity:.3f}")
        if theories.regime_name:
            logger.info(f"Market Regime: {theories.regime_name} ({theories.regime_confidence*100:.0f}% conf)")
        if theories.sharpe_ratio is not None:
            logger.info(f"Sharpe Ratio: {theories.sharpe_ratio:.2f}")
        if theories.var_95 is not None:
            logger.info(f"VaR (95%): {theories.var_95*100:.2f}%")
        if theories.fractal_dimension is not None:
            logger.info(f"Fractal Dimension: {theories.fractal_dimension:.2f}")
    else:
        logger.info("No theory analysis (generation failed)")

    logger.info(f"\n--- COST ---")
    logger.info(f"Total cost: ${result.total_cost_usd:.6f}")
    logger.info(f"Input tokens: {result.input_tokens}")
    logger.info(f"Output tokens: {result.output_tokens}")

    logger.info("\n" + "="*80)

if __name__ == "__main__":
    test_v7_signal()
