"""
Kalman Filter for Price Denoising and Momentum Estimation

Kalman Filter is an optimal recursive Bayesian filter for linear dynamic systems.
In trading, it's used to:
1. Denoise price series (remove market microstructure noise)
2. Estimate true momentum (price velocity)
3. Predict future price trajectories

State Vector:
  x = [price, velocity]

Process Model:
  x(t) = F * x(t-1) + w    # w ~ N(0, Q) process noise

Measurement Model:
  z(t) = H * x(t) + v      # v ~ N(0, R) measurement noise

Where:
  F = [[1, dt],   # State transition matrix
       [0,  1]]
  H = [1, 0]      # Measurement matrix (observe price only)
  Q = Process noise covariance (how much state changes)
  R = Measurement noise covariance (how noisy observations are)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KalmanState:
    """Kalman Filter state at a point in time"""
    price: float          # Estimated denoised price
    velocity: float       # Estimated momentum (price change per timestep)
    price_variance: float # Uncertainty in price estimate
    velocity_variance: float  # Uncertainty in velocity estimate
    timestamp: Optional[pd.Timestamp] = None


class KalmanPriceFilter:
    """
    2-State Kalman Filter for price denoising and momentum estimation

    State: [price, velocity]
    - Price: Denoised price estimate
    - Velocity: Momentum (rate of price change)
    """

    def __init__(
        self,
        process_noise: float = 1e-4,
        measurement_noise: float = 0.01,
        dt: float = 1.0
    ):
        """
        Initialize Kalman Filter

        Args:
            process_noise: Process noise variance (Q)
                          Higher = expect more volatility in true price
            measurement_noise: Measurement noise variance (R)
                              Higher = trust observations less
            dt: Time step between observations (default: 1.0)
        """
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # State transition matrix F
        self.F = np.array([
            [1.0, dt],   # price(t) = price(t-1) + velocity(t-1) * dt
            [0.0, 1.0]   # velocity(t) = velocity(t-1)
        ])

        # Measurement matrix H (we only observe price)
        self.H = np.array([[1.0, 0.0]])

        # Process noise covariance Q
        # Models uncertainty in state evolution
        self.Q = np.array([
            [process_noise * dt**2, process_noise * dt],
            [process_noise * dt, process_noise]
        ])

        # Measurement noise covariance R
        # Models uncertainty in observations
        self.R = np.array([[measurement_noise]])

        # State vector: [price, velocity]
        self.x = None

        # Error covariance matrix P
        self.P = None

        # History
        self.history: List[KalmanState] = []

        logger.debug(
            f"Kalman Filter initialized: "
            f"process_noise={process_noise}, "
            f"measurement_noise={measurement_noise}, dt={dt}"
        )

    def initialize(self, initial_price: float, initial_velocity: float = 0.0):
        """
        Initialize filter with first observation

        Args:
            initial_price: First price observation
            initial_velocity: Initial velocity estimate (default: 0)
        """
        self.x = np.array([[initial_price], [initial_velocity]])

        # Initialize with high uncertainty
        self.P = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        self.history = [KalmanState(
            price=initial_price,
            velocity=initial_velocity,
            price_variance=1.0,
            velocity_variance=1.0
        )]

        logger.debug(f"Initialized at price={initial_price:.2f}, velocity={initial_velocity:.4f}")

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step: Predict next state based on process model

        Returns:
            (x_pred, P_pred): Predicted state and covariance
        """
        if self.x is None:
            raise ValueError("Filter not initialized. Call initialize() first.")

        # Predict state: x_pred = F * x
        x_pred = self.F @ self.x

        # Predict covariance: P_pred = F * P * F^T + Q
        P_pred = self.F @ self.P @ self.F.T + self.Q

        return x_pred, P_pred

    def update(self, measurement: float) -> KalmanState:
        """
        Update step: Incorporate new measurement

        Args:
            measurement: Observed price

        Returns:
            KalmanState with updated estimates
        """
        # Predict
        x_pred, P_pred = self.predict()

        # Measurement residual: y = z - H * x_pred
        z = np.array([[measurement]])
        y = z - self.H @ x_pred

        # Residual covariance: S = H * P_pred * H^T + R
        S = self.H @ P_pred @ self.H.T + self.R

        # Kalman gain: K = P_pred * H^T * S^-1
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Update state: x = x_pred + K * y
        self.x = x_pred + K @ y

        # Update covariance: P = (I - K * H) * P_pred
        I = np.eye(2)
        self.P = (I - K @ self.H) @ P_pred

        # Extract state
        price = float(self.x[0, 0])
        velocity = float(self.x[1, 0])
        price_var = float(self.P[0, 0])
        velocity_var = float(self.P[1, 1])

        state = KalmanState(
            price=price,
            velocity=velocity,
            price_variance=price_var,
            velocity_variance=velocity_var
        )

        self.history.append(state)

        logger.debug(
            f"Updated: price={price:.2f}, velocity={velocity:.6f}, "
            f"innovation={float(y[0,0]):.6f}"
        )

        return state

    def filter_series(
        self,
        prices: np.ndarray,
        return_history: bool = True
    ) -> np.ndarray:
        """
        Filter entire price series

        Args:
            prices: Array of price observations
            return_history: If True, return full state history

        Returns:
            Array of denoised prices (or full history if return_history=True)
        """
        if len(prices) == 0:
            logger.warning("Empty price series")
            return np.array([])

        # Initialize with first price
        self.initialize(prices[0])

        # Process remaining prices
        denoised_prices = [prices[0]]

        for price in prices[1:]:
            state = self.update(price)
            denoised_prices.append(state.price)

        if return_history:
            return np.array(denoised_prices)
        else:
            return np.array([state.price for state in self.history])

    def get_momentum_estimate(self) -> float:
        """
        Get current momentum (velocity) estimate

        Returns:
            Estimated price velocity (momentum)
        """
        if self.x is None:
            raise ValueError("Filter not initialized")

        return float(self.x[1, 0])

    def get_denoised_price(self) -> float:
        """
        Get current denoised price estimate

        Returns:
            Denoised price
        """
        if self.x is None:
            raise ValueError("Filter not initialized")

        return float(self.x[0, 0])

    def predict_ahead(self, n_steps: int = 1) -> np.ndarray:
        """
        Predict future prices n steps ahead

        Args:
            n_steps: Number of time steps to predict

        Returns:
            Array of predicted prices [t+1, t+2, ..., t+n]
        """
        if self.x is None:
            raise ValueError("Filter not initialized")

        predictions = []
        x_pred = self.x.copy()

        for _ in range(n_steps):
            x_pred = self.F @ x_pred
            predictions.append(float(x_pred[0, 0]))

        return np.array(predictions)

    def get_prediction_confidence(self) -> Dict:
        """
        Get confidence metrics for current state estimate

        Returns:
            Dictionary with confidence metrics
        """
        if self.P is None:
            raise ValueError("Filter not initialized")

        price_std = np.sqrt(self.P[0, 0])
        velocity_std = np.sqrt(self.P[1, 1])

        return {
            'price_std': float(price_std),
            'velocity_std': float(velocity_std),
            'price_confidence': float(1.0 / (1.0 + price_std)),  # 0-1 scale
            'velocity_confidence': float(1.0 / (1.0 + velocity_std))
        }

    def analyze_signal_quality(self, prices: np.ndarray) -> Dict:
        """
        Analyze signal quality (noise reduction performance)

        Args:
            prices: Original noisy prices

        Returns:
            Dictionary with quality metrics
        """
        denoised = self.filter_series(prices)

        # Calculate noise reduction
        original_volatility = np.std(np.diff(prices))
        denoised_volatility = np.std(np.diff(denoised))
        noise_reduction = 1.0 - (denoised_volatility / original_volatility)

        # Calculate signal-to-noise ratio improvement
        signal_power = np.var(denoised)
        noise_power = np.var(prices - denoised)
        snr = signal_power / noise_power if noise_power > 0 else np.inf
        snr_db = 10 * np.log10(snr) if snr > 0 else 0.0

        # Momentum statistics
        velocities = [state.velocity for state in self.history]
        avg_momentum = np.mean(velocities)
        momentum_volatility = np.std(velocities)

        return {
            'noise_reduction_pct': float(noise_reduction * 100),
            'snr_db': float(snr_db),
            'original_volatility': float(original_volatility),
            'denoised_volatility': float(denoised_volatility),
            'avg_momentum': float(avg_momentum),
            'momentum_volatility': float(momentum_volatility),
            'num_observations': len(prices)
        }


# Convenience function for V7 runtime
def denoise_price_series(
    prices: np.ndarray,
    process_noise: float = 1e-4,
    measurement_noise: float = 0.01
) -> Dict:
    """
    Denoise price series and extract momentum

    Args:
        prices: Array of noisy price observations
        process_noise: Process noise parameter
        measurement_noise: Measurement noise parameter

    Returns:
        Dictionary with denoised prices, momentum, and quality metrics
    """
    kf = KalmanPriceFilter(
        process_noise=process_noise,
        measurement_noise=measurement_noise
    )

    denoised = kf.filter_series(prices)
    momentum = kf.get_momentum_estimate()
    confidence = kf.get_prediction_confidence()
    quality = kf.analyze_signal_quality(prices)

    return {
        'denoised_prices': denoised,
        'current_momentum': momentum,
        'confidence': confidence,
        'quality': quality,
        'predictions': kf.predict_ahead(n_steps=5).tolist()
    }


if __name__ == "__main__":
    # Test Kalman Filter implementation
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    print("=" * 80)
    print("Kalman Price Filter - Test Run")
    print("=" * 80)

    # Generate test data
    np.random.seed(42)
    n_points = 500

    # Test 1: Noisy uptrend
    print("\n1. Testing Noisy Uptrend")
    true_trend = np.linspace(100, 120, n_points)
    noise = np.random.randn(n_points) * 0.5
    noisy_uptrend = true_trend + noise

    kf1 = KalmanPriceFilter(process_noise=1e-4, measurement_noise=0.01)
    denoised1 = kf1.filter_series(noisy_uptrend)
    momentum1 = kf1.get_momentum_estimate()
    quality1 = kf1.analyze_signal_quality(noisy_uptrend)

    print(f"   Momentum:         {momentum1:.6f}")
    print(f"   Noise Reduction:  {quality1['noise_reduction_pct']:.1f}%")
    print(f"   SNR:              {quality1['snr_db']:.2f} dB")

    # Test 2: Noisy mean-reverting
    print("\n2. Testing Noisy Mean-Reverting")
    true_sine = 100 + 5 * np.sin(np.linspace(0, 4*np.pi, n_points))
    noise = np.random.randn(n_points) * 0.3
    noisy_sine = true_sine + noise

    kf2 = KalmanPriceFilter(process_noise=1e-4, measurement_noise=0.01)
    denoised2 = kf2.filter_series(noisy_sine)
    momentum2 = kf2.get_momentum_estimate()
    quality2 = kf2.analyze_signal_quality(noisy_sine)

    print(f"   Momentum:         {momentum2:.6f}")
    print(f"   Noise Reduction:  {quality2['noise_reduction_pct']:.1f}%")
    print(f"   SNR:              {quality2['snr_db']:.2f} dB")

    # Test 3: High noise scenario
    print("\n3. Testing High Noise Scenario")
    true_prices = np.ones(n_points) * 100
    high_noise = np.random.randn(n_points) * 2.0
    very_noisy = true_prices + high_noise

    kf3 = KalmanPriceFilter(process_noise=1e-4, measurement_noise=0.1)
    denoised3 = kf3.filter_series(very_noisy)
    quality3 = kf3.analyze_signal_quality(very_noisy)

    print(f"   Noise Reduction:  {quality3['noise_reduction_pct']:.1f}%")
    print(f"   SNR:              {quality3['snr_db']:.2f} dB")
    print(f"   Original Vol:     {quality3['original_volatility']:.4f}")
    print(f"   Denoised Vol:     {quality3['denoised_volatility']:.4f}")

    # Test 4: Prediction capability
    print("\n4. Testing Prediction Capability")
    kf4 = KalmanPriceFilter(process_noise=1e-4, measurement_noise=0.01)
    kf4.filter_series(noisy_uptrend)
    predictions = kf4.predict_ahead(n_steps=10)
    current_price = kf4.get_denoised_price()

    print(f"   Current Price:    {current_price:.2f}")
    print(f"   Predicted +5:     {predictions[4]:.2f}")
    print(f"   Predicted +10:    {predictions[9]:.2f}")
    print(f"   Momentum:         {kf4.get_momentum_estimate():.6f}")

    # Test 5: Confidence metrics
    print("\n5. Testing Confidence Metrics")
    confidence = kf4.get_prediction_confidence()
    print(f"   Price Std:        {confidence['price_std']:.4f}")
    print(f"   Velocity Std:     {confidence['velocity_std']:.6f}")
    print(f"   Price Confidence: {confidence['price_confidence']:.4f}")
    print(f"   Velocity Conf:    {confidence['velocity_confidence']:.4f}")

    # Visualization
    print("\n6. Creating Visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Noisy uptrend
    axes[0, 0].plot(noisy_uptrend, alpha=0.3, label='Noisy', color='gray')
    axes[0, 0].plot(denoised1, label='Denoised', color='blue', linewidth=2)
    axes[0, 0].plot(true_trend, label='True', color='green', linestyle='--')
    axes[0, 0].set_title(f"Uptrend (Noise Reduction: {quality1['noise_reduction_pct']:.1f}%)")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Price")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Noisy sine wave
    axes[0, 1].plot(noisy_sine, alpha=0.3, label='Noisy', color='gray')
    axes[0, 1].plot(denoised2, label='Denoised', color='blue', linewidth=2)
    axes[0, 1].plot(true_sine, label='True', color='green', linestyle='--')
    axes[0, 1].set_title(f"Mean-Reverting (SNR: {quality2['snr_db']:.1f} dB)")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Price")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Momentum estimate
    velocities1 = [state.velocity for state in kf1.history]
    axes[1, 0].plot(velocities1, color='red', linewidth=1.5)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1, 0].set_title("Momentum Estimate (Uptrend)")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Velocity (Momentum)")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Prediction
    last_100_noisy = noisy_uptrend[-100:]
    last_100_denoised = denoised1[-100:]
    x_axis = np.arange(len(last_100_noisy))
    pred_x = np.arange(len(last_100_noisy), len(last_100_noisy) + 10)

    axes[1, 1].plot(x_axis, last_100_noisy, alpha=0.3, label='Noisy', color='gray')
    axes[1, 1].plot(x_axis, last_100_denoised, label='Denoised', color='blue', linewidth=2)
    axes[1, 1].plot(pred_x, predictions, label='Predicted', color='orange',
                    linewidth=2, linestyle='--', marker='o')
    axes[1, 1].set_title("Prediction (10 steps ahead)")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Price")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/kalman_filter_test.png', dpi=100, bbox_inches='tight')
    print(f"   Saved visualization to /tmp/kalman_filter_test.png")

    print("\n" + "=" * 80)
    print("Kalman Filter Test Complete!")
    print("=" * 80)
    print("\nKey Insights:")
    print("  - Kalman filter effectively removes noise from price series")
    print("  - Momentum estimation captures trend direction and strength")
    print("  - SNR improvement quantifies denoising performance")
    print("  - Prediction uses estimated momentum for future extrapolation")
    print("  - Confidence metrics indicate estimate uncertainty")
    print("=" * 80)
