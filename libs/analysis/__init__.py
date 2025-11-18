"""
Mathematical Analysis Modules for V7 Ultimate

This package contains implementation of 6 mathematical theories:
1. Shannon Entropy - Market predictability (IMPLEMENTED)
2. Hurst Exponent - Trend vs mean-reversion (IMPLEMENTED)
3. Markov Chain - Regime detection (IMPLEMENTED)
4. Kalman Filter - Price denoising (IMPLEMENTED)
5. Bayesian Inference - Online learning (TODO)
6. Monte Carlo - Risk simulation (TODO)
"""

# Import implemented modules only
from .shannon_entropy import ShannonEntropyAnalyzer, calculate_market_entropy
from .hurst_exponent import HurstExponentAnalyzer, calculate_hurst_exponent
from .markov_chain import MarkovRegimeDetector, detect_market_regime, MarketRegime
from .kalman_filter import KalmanPriceFilter, denoise_price_series

# TODO: Implement remaining theories (STEP 2.5-2.6)
# from .bayesian_inference import BayesianLearner, update_beliefs
# from .monte_carlo import MonteCarloSimulator, simulate_risk_scenarios

__all__ = [
    'ShannonEntropyAnalyzer',
    'calculate_market_entropy',
    'HurstExponentAnalyzer',
    'calculate_hurst_exponent',
    'MarkovRegimeDetector',
    'detect_market_regime',
    'MarketRegime',
    'KalmanPriceFilter',
    'denoise_price_series',
    # Remaining exports will be added as theories are implemented
]
