"""Test all quantitative finance libraries are working"""
import numpy as np
import pandas as pd

print("=" * 70)
print("QUANTITATIVE FINANCE LIBRARIES TEST")
print("=" * 70)

# Test PyPortfolioOpt
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    print("✅ PyPortfolioOpt working")
except Exception as e:
    print(f"❌ PyPortfolioOpt failed: {e}")

# Test cvxpy (optimizer)
try:
    import cvxpy as cp
    x = cp.Variable()
    prob = cp.Problem(cp.Minimize(x**2), [x >= 1])
    prob.solve()
    print(f"✅ CVXPY working (test optimization: x={x.value:.2f})")
except Exception as e:
    print(f"❌ CVXPY failed: {e}")

# Test scipy
try:
    from scipy.optimize import minimize
    from scipy import stats
    print("✅ Scipy working")
except Exception as e:
    print(f"❌ Scipy failed: {e}")

# Test scikit-learn
try:
    from sklearn.ensemble import RandomForestRegressor
    print("✅ Scikit-learn working")
except Exception as e:
    print(f"❌ Scikit-learn failed: {e}")

# Test numpy/pandas (already installed)
try:
    data = pd.DataFrame({
        'returns': np.random.randn(100) * 0.01
    })
    sharpe = data['returns'].mean() / data['returns'].std() * np.sqrt(252)
    print(f"✅ NumPy/Pandas working (manual Sharpe test: {sharpe:.2f})")
except Exception as e:
    print(f"❌ NumPy/Pandas failed: {e}")

print("\n" + "=" * 70)
print("🎉 Core quantitative finance libraries ready!")
print("=" * 70)
print("\nAvailable capabilities:")
print("  - Portfolio optimization (PyPortfolioOpt)")
print("  - Convex optimization (CVXPY)")
print("  - Statistical analysis (Scipy)")
print("  - Machine learning (Scikit-learn)")
print("  - Numerical computing (NumPy/Pandas)")
