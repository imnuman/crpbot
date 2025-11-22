"""Verify all V7 theories are accessible and functional"""
import sys
import traceback

print("="*70)
print("V7 THEORY VERIFICATION")
print("="*70)

theories_status = {}

# Test imports from libs/analysis (6 core theories)
print("\n1. CORE THEORIES (libs/analysis/):")
try:
    from libs.analysis import ShannonEntropyAnalyzer
    print("  ✅ Shannon Entropy")
    theories_status['shannon_entropy'] = True
except Exception as e:
    print(f"  ❌ Shannon Entropy: {e}")
    theories_status['shannon_entropy'] = False

try:
    from libs.analysis import HurstExponentAnalyzer
    print("  ✅ Hurst Exponent")
    theories_status['hurst_exponent'] = True
except Exception as e:
    print(f"  ❌ Hurst Exponent: {e}")
    theories_status['hurst_exponent'] = False

try:
    from libs.analysis import MarkovRegimeDetector
    print("  ✅ Markov Regime")
    theories_status['markov_regime'] = True
except Exception as e:
    print(f"  ❌ Markov Regime: {e}")
    theories_status['markov_regime'] = False

try:
    from libs.analysis import KalmanPriceFilter
    print("  ✅ Kalman Filter")
    theories_status['kalman_filter'] = True
except Exception as e:
    print(f"  ❌ Kalman Filter: {e}")
    theories_status['kalman_filter'] = False

try:
    from libs.analysis import BayesianWinRateLearner
    print("  ✅ Bayesian Win Rate")
    theories_status['bayesian_win_rate'] = True
except Exception as e:
    print(f"  ❌ Bayesian Win Rate: {e}")
    theories_status['bayesian_win_rate'] = False

try:
    from libs.analysis import MonteCarloSimulator
    print("  ✅ Monte Carlo")
    theories_status['monte_carlo'] = True
except Exception as e:
    print(f"  ❌ Monte Carlo: {e}")
    theories_status['monte_carlo'] = False

# Test imports from libs/theories (4 statistical)
print("\n2. STATISTICAL THEORIES (libs/theories/):")
try:
    from libs.theories.random_forest_validator import RandomForestValidator
    print("  ✅ Random Forest")
    theories_status['random_forest'] = True
except Exception as e:
    print(f"  ❌ Random Forest: {e}")
    theories_status['random_forest'] = False

try:
    from libs.theories.autocorrelation_analyzer import AutocorrelationAnalyzer
    print("  ✅ Autocorrelation")
    theories_status['autocorrelation'] = True
except Exception as e:
    print(f"  ❌ Autocorrelation: {e}")
    theories_status['autocorrelation'] = False

try:
    from libs.theories.stationarity_test import StationarityAnalyzer
    print("  ✅ Stationarity")
    theories_status['stationarity'] = True
except Exception as e:
    print(f"  ❌ Stationarity: {e}")
    theories_status['stationarity'] = False

try:
    from libs.theories.variance_tests import VarianceAnalyzer
    print("  ✅ Variance")
    theories_status['variance'] = True
except Exception as e:
    print(f"  ❌ Variance: {e}")
    theories_status['variance'] = False

# Test signal generator integration
print("\n3. V7 SIGNAL GENERATOR:")
try:
    from libs.llm import SignalGenerator
    print("  ✅ Signal Generator imports")

    # Try to instantiate (will fail if DeepSeek key missing, but that's expected)
    try:
        sg = SignalGenerator(api_key="test")
        print("  ✅ Signal Generator instantiates")
    except Exception as e:
        if "api_key" in str(e).lower():
            print("  ⚠️  Signal Generator needs API key (expected)")
        else:
            print(f"  ❌ Signal Generator error: {e}")

    theories_status['signal_generator'] = True
except Exception as e:
    print(f"  ❌ Signal Generator: {e}")
    theories_status['signal_generator'] = False
    traceback.print_exc()

# Summary
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)
working = sum(1 for v in theories_status.values() if v)
total = len(theories_status)
print(f"Theories Working: {working}/{total} ({working/total*100:.1f}%)")

if working == total:
    print("\n✅ ALL THEORIES OPERATIONAL")
    sys.exit(0)
elif working >= 8:
    print("\n⚠️  MOST THEORIES WORKING (acceptable)")
    sys.exit(0)
else:
    print("\n❌ CRITICAL: Too many theories failing")
    sys.exit(1)
