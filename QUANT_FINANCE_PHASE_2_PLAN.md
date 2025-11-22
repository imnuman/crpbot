# Quantitative Finance - Phase 2 Implementation Plan
## Advanced Components (Post 10-Hour Foundation)

**Date**: 2025-11-22
**Prerequisites**: Phase 1 (10-hour plan) MUST be complete
**Status**: ⏳ READY TO START (after Phase 1)

---

## 🎯 OVERVIEW

**What This Is**:
Phase 2 implements the advanced quantitative finance techniques that were excluded from the 10-hour plan due to time complexity. These components require longer implementation time but provide cutting-edge capabilities.

**Prerequisites** (Must Complete First):
- ✅ Phase 1: 10-hour plan complete (`QUANT_FINANCE_10_HOUR_PLAN.md`)
- ✅ Backtesting framework operational
- ✅ Portfolio optimization working
- ✅ Kelly Criterion + CVaR implemented
- ✅ V7 Quant Enhanced runtime functional

**Total Estimated Time**: 50-60 hours (5-6 weeks at 10 hours/week)

---

## 📋 PHASE 2 COMPONENTS

### Deferred from Phase 1 (Too Time-Consuming)

1. **Deep Learning Models** (LSTM, Transformers)
   - Reason deferred: Training takes hours/days
   - Time estimate: 12-15 hours

2. **GARCH Volatility Models**
   - Reason deferred: Complex fitting and validation
   - Time estimate: 8-10 hours

3. **Non-Ergodicity Framework**
   - Reason deferred: Theoretical complexity
   - Time estimate: 10-12 hours

4. **Pairs Trading (Cointegration)**
   - Reason deferred: Statistical analysis required
   - Time estimate: 8-10 hours

5. **Multi-Factor Models** (Fama-French adapted)
   - Reason deferred: Extensive data preparation
   - Time estimate: 10-12 hours

**Total**: 48-59 hours

---

## 🗺️ IMPLEMENTATION ROADMAP

### Module 1: Deep Learning for Price Prediction (12-15 hours)

**Goal**: Add LSTM and Transformer models for price forecasting

**Dependencies**: Phase 1 complete, historical data available

---

#### STEP 1.1: Setup Deep Learning Environment (2 hours)

**What to Install**:
```bash
# Deep learning frameworks
pip install tensorflow keras torch transformers

# GPU acceleration (optional but recommended)
pip install tensorflow-gpu  # If NVIDIA GPU available

# Time series tools
pip install sktime tslearn

# Verify installations
python -c "import tensorflow as tf; import torch; print(f'TensorFlow: {tf.__version__}, PyTorch: {torch.__version__}')"
```

**Create** (`libs/ml/dl_environment.py`):
```python
"""
Deep Learning Environment Setup

Checks GPU availability and configures TensorFlow/PyTorch
"""
import tensorflow as tf
import torch

class DLEnvironment:
    """Check and configure deep learning environment"""

    @staticmethod
    def check_gpu():
        """Check GPU availability for both TF and PyTorch"""
        print("="*70)
        print("DEEP LEARNING ENVIRONMENT CHECK")
        print("="*70)

        # TensorFlow
        tf_gpus = tf.config.list_physical_devices('GPU')
        print(f"\nTensorFlow GPUs: {len(tf_gpus)}")
        if tf_gpus:
            for gpu in tf_gpus:
                print(f"  {gpu.name}")
        else:
            print("  No GPU found - will use CPU (slower)")

        # PyTorch
        torch_gpu = torch.cuda.is_available()
        print(f"\nPyTorch CUDA available: {torch_gpu}")
        if torch_gpu:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        print("="*70)

        return len(tf_gpus) > 0 or torch_gpu

    @staticmethod
    def configure_tensorflow():
        """Configure TensorFlow for optimal performance"""
        # Limit GPU memory growth (prevent OOM)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ TensorFlow GPU memory growth enabled")
            except RuntimeError as e:
                print(f"⚠️ GPU configuration failed: {e}")

# CLI usage
if __name__ == "__main__":
    env = DLEnvironment()
    has_gpu = env.check_gpu()
    env.configure_tensorflow()

    if has_gpu:
        print("\n✅ GPU available - Deep learning will be fast")
    else:
        print("\n⚠️  No GPU - Training will be slower but will work")
```

**Run**:
```bash
python libs/ml/dl_environment.py
```

**Success Criteria**:
- [ ] TensorFlow and PyTorch installed
- [ ] GPU detected (if available)
- [ ] No import errors

---

#### STEP 1.2: LSTM Price Prediction Model (4 hours)

**What to Build**: LSTM neural network for next-hour price prediction

**Create** (`libs/ml/lstm_predictor.py`):
```python
"""
LSTM Model for Cryptocurrency Price Prediction

Uses sequence of historical prices to predict next price movement
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class LSTMPredictor:
    """LSTM model for price prediction"""

    def __init__(self, sequence_length=60, n_features=10):
        """
        Args:
            sequence_length: Number of time steps to look back (default 60 = 60 hours)
            n_features: Number of features per timestep (price, volume, indicators)
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def build_model(self):
        """
        Build LSTM architecture

        Architecture:
        - Bidirectional LSTM layer 1: 128 units
        - Dropout: 0.2
        - Bidirectional LSTM layer 2: 64 units
        - Dropout: 0.2
        - Dense layer: 32 units (ReLU)
        - Output layer: 3 units (UP, DOWN, NEUTRAL) - softmax
        """
        model = Sequential([
            # First LSTM layer (bidirectional for context from both directions)
            Bidirectional(LSTM(128, return_sequences=True),
                         input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),

            # Second LSTM layer
            Bidirectional(LSTM(64, return_sequences=False)),
            Dropout(0.2),

            # Dense layers
            Dense(32, activation='relu'),
            Dropout(0.1),

            # Output layer: 3 classes (UP, DOWN, NEUTRAL)
            Dense(3, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model

        print("LSTM Model Architecture:")
        model.summary()

        return model

    def prepare_sequences(self, df, target_col='close'):
        """
        Prepare time series sequences for LSTM

        Args:
            df: DataFrame with OHLCV and indicators
            target_col: Column to predict

        Returns:
            X: Input sequences (n_samples, sequence_length, n_features)
            y: Target labels (n_samples, 3) - one-hot encoded
        """
        # Select features (exclude timestamp)
        feature_cols = [col for col in df.columns if col not in ['timestamp']]
        data = df[feature_cols].values

        # Normalize data
        scaled_data = self.scaler.fit_transform(data)

        # Create sequences
        X, y = [], []

        for i in range(self.sequence_length, len(scaled_data) - 1):
            # Input: last sequence_length timesteps
            X.append(scaled_data[i - self.sequence_length:i])

            # Target: next price movement (UP/DOWN/NEUTRAL)
            current_price = df[target_col].iloc[i]
            next_price = df[target_col].iloc[i + 1]
            price_change_pct = (next_price - current_price) / current_price

            # Classify movement
            if price_change_pct > 0.002:  # >0.2% = UP
                label = [1, 0, 0]  # UP
            elif price_change_pct < -0.002:  # <-0.2% = DOWN
                label = [0, 1, 0]  # DOWN
            else:
                label = [0, 0, 1]  # NEUTRAL

            y.append(label)

        return np.array(X), np.array(y)

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train LSTM model

        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        # Build model if not already built
        if self.model is None:
            self.build_model()

        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        checkpoint = ModelCheckpoint(
            'models/lstm_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )

        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )

        return history

    def predict(self, X):
        """
        Predict price movement

        Args:
            X: Input sequences

        Returns:
            Predictions: (n_samples, 3) probabilities for UP/DOWN/NEUTRAL
        """
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        print(f"\nTest Results:")
        print(f"  Loss:     {loss:.4f}")
        print(f"  Accuracy: {accuracy*100:.2f}%")

        return {'loss': loss, 'accuracy': accuracy}


# CLI usage
if __name__ == "__main__":
    # Load historical data
    btc_data = pd.read_parquet('data/backtest/BTC_USD_historical.parquet')

    # Add simple features (price, volume)
    features_df = btc_data[['open', 'high', 'low', 'close', 'volume']].copy()

    # Add technical indicators (simple moving averages)
    features_df['sma_10'] = features_df['close'].rolling(10).mean()
    features_df['sma_30'] = features_df['close'].rolling(30).mean()
    features_df['rsi'] = 50  # Placeholder (implement proper RSI)
    features_df['volume_sma'] = features_df['volume'].rolling(10).mean()
    features_df['price_change'] = features_df['close'].pct_change()

    # Drop NaN
    features_df = features_df.dropna()

    # Initialize LSTM
    lstm = LSTMPredictor(sequence_length=60, n_features=len(features_df.columns))

    # Prepare sequences
    print("Preparing sequences...")
    X, y = lstm.prepare_sequences(features_df)

    print(f"Sequences created: {len(X)}")
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")

    # Split train/val/test (70/15/15)
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]

    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    # Build and train
    print("\nBuilding LSTM model...")
    lstm.build_model()

    print("\nTraining LSTM...")
    history = lstm.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

    # Evaluate
    print("\nEvaluating on test set...")
    results = lstm.evaluate(X_test, y_test)

    print(f"\n{'='*70}")
    print("LSTM TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Final Test Accuracy: {results['accuracy']*100:.2f}%")

    if results['accuracy'] > 0.55:
        print("✅ Model has predictive power (>55% accuracy)")
    else:
        print("⚠️  Model accuracy low, may need more features or tuning")
```

**Run Training**:
```bash
python libs/ml/lstm_predictor.py
```

**Success Criteria**:
- [ ] LSTM model trains without errors
- [ ] Training completes in <2 hours (with GPU) or <4 hours (CPU)
- [ ] Test accuracy >55% (better than random)
- [ ] Model saved to `models/lstm_best.h5`

**Time Breakdown**:
- 2 hours: Write LSTM code
- 1.5 hours: Train model (with GPU) or 3 hours (CPU)
- 0.5 hours: Evaluate and tune

---

#### STEP 1.3: Transformer Model (Advanced) (4 hours)

**What to Build**: Transformer (attention-based) model for time series

**Create** (`libs/ml/transformer_predictor.py`):
```python
"""
Transformer Model for Time Series Prediction

Uses attention mechanism to capture long-range dependencies
More advanced than LSTM
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TimeSeriesTransformer:
    """Transformer model for crypto price prediction"""

    def __init__(self, sequence_length=60, n_features=10):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """Transformer encoder block"""
        # Multi-head attention
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed forward network
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

    def build_model(self, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], dropout=0.25, mlp_dropout=0.4):
        """
        Build Transformer model

        Args:
            head_size: Size of attention heads
            num_heads: Number of attention heads
            ff_dim: Feedforward dimension multiplier
            num_transformer_blocks: Number of transformer blocks
            mlp_units: Dense layer sizes after transformer
            dropout: Dropout rate
            mlp_dropout: Dropout for MLP layers
        """
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        x = inputs

        # Transformer encoder blocks
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        # Global average pooling
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)

        # MLP head
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)

        # Output layer (3 classes: UP, DOWN, NEUTRAL)
        outputs = layers.Dense(3, activation="softmax")(x)

        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.model = model

        print("Transformer Model Architecture:")
        model.summary()

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train transformer model"""
        if self.model is None:
            self.build_model()

        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint('models/transformer_best.h5', save_best_only=True)
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history


# CLI usage
if __name__ == "__main__":
    # Same data prep as LSTM
    print("Training Transformer model...")
    print("Note: Transformers require more data and compute than LSTM")

    # Load sequences (reuse from LSTM)
    from libs.ml.lstm_predictor import LSTMPredictor
    import pandas as pd

    btc_data = pd.read_parquet('data/backtest/BTC_USD_historical.parquet')
    features_df = btc_data[['open', 'high', 'low', 'close', 'volume']].copy()
    features_df['sma_10'] = features_df['close'].rolling(10).mean()
    features_df['sma_30'] = features_df['close'].rolling(30).mean()
    features_df['rsi'] = 50
    features_df['volume_sma'] = features_df['volume'].rolling(10).mean()
    features_df['price_change'] = features_df['close'].pct_change()
    features_df = features_df.dropna()

    lstm = LSTMPredictor(sequence_length=60, n_features=len(features_df.columns))
    X, y = lstm.prepare_sequences(features_df)

    # Split
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    # Train Transformer
    transformer = TimeSeriesTransformer(sequence_length=60, n_features=len(features_df.columns))
    transformer.build_model()

    print("Training...")
    history = transformer.train(X_train, y_train, X_val, y_val, epochs=50)

    # Evaluate
    loss, accuracy = transformer.model.evaluate(X_test, y_test)
    print(f"\nTransformer Test Accuracy: {accuracy*100:.2f}%")
```

**Success Criteria**:
- [ ] Transformer trains successfully
- [ ] Accuracy comparable to or better than LSTM
- [ ] Attention mechanism captures long-range patterns

**Time Breakdown**:
- 2 hours: Implement transformer
- 1.5 hours: Train (GPU) or 3 hours (CPU)
- 0.5 hours: Compare to LSTM

---

#### STEP 1.4: Model Ensemble (2 hours)

**What to Build**: Combine LSTM + Transformer + Random Forest predictions

**Create** (`libs/ml/ensemble.py`):
```python
"""
ML Model Ensemble

Combines predictions from:
- LSTM
- Transformer
- Random Forest (existing)
- XGBoost
"""
import numpy as np
from libs.ml.lstm_predictor import LSTMPredictor
from libs.ml.transformer_predictor import TimeSeriesTransformer
from libs.theories.random_forest_validator import RandomForestValidator

class MLEnsemble:
    """Ensemble of ML models for robust predictions"""

    def __init__(self):
        self.lstm = None
        self.transformer = None
        self.rf = RandomForestValidator()
        self.weights = {
            'lstm': 0.35,
            'transformer': 0.35,
            'rf': 0.30
        }

    def load_models(self):
        """Load pre-trained models"""
        # Load LSTM
        self.lstm = LSTMPredictor()
        self.lstm.model = tf.keras.models.load_model('models/lstm_best.h5')

        # Load Transformer
        self.transformer = TimeSeriesTransformer()
        self.transformer.model = tf.keras.models.load_model('models/transformer_best.h5')

        print("✅ Ensemble models loaded")

    def predict(self, X):
        """
        Ensemble prediction

        Args:
            X: Input sequences

        Returns:
            Combined probabilities (n_samples, 3)
        """
        # Get predictions from each model
        lstm_pred = self.lstm.predict(X)
        transformer_pred = self.transformer.predict(X)
        rf_pred = self.rf.predict_proba(X.reshape(X.shape[0], -1))  # Flatten for RF

        # Weighted average
        ensemble_pred = (
            lstm_pred * self.weights['lstm'] +
            transformer_pred * self.weights['transformer'] +
            rf_pred * self.weights['rf']
        )

        return ensemble_pred

    def predict_class(self, X):
        """Get class predictions (UP/DOWN/NEUTRAL)"""
        probs = self.predict(X)
        classes = np.argmax(probs, axis=1)

        class_names = ['UP', 'DOWN', 'NEUTRAL']
        return [class_names[c] for c in classes]


# CLI usage
if __name__ == "__main__":
    ensemble = MLEnsemble()
    ensemble.load_models()

    # Test on sample data
    print("Ensemble ready for predictions")
```

**Success Criteria**:
- [ ] Ensemble combines all models
- [ ] Accuracy >60% (improvement over individual models)

**Time Breakdown**:
- 1 hour: Implement ensemble
- 1 hour: Tune weights, validate

---

#### STEP 1.5: Integration with V7 (2 hours)

**What to Build**: Add ML ensemble to V7 signal generation

**Update** (`apps/runtime/v7_quant_enhanced.py`):
```python
# Add at top
from libs.ml.ensemble import MLEnsemble

class V7QuantEnhancedML(V7QuantEnhanced):
    """V7 with ML Ensemble"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load ML ensemble
        self.ml_ensemble = MLEnsemble()
        self.ml_ensemble.load_models()

        print("✅ ML Ensemble loaded")

    def generate_signal_with_ml(self, symbol):
        """Generate signal using theories + DeepSeek + ML"""
        # Get base signal (theories + DeepSeek)
        base_signal = self.generate_enhanced_signal(symbol)

        # Get ML prediction
        # (prepare sequence from recent prices)
        recent_prices = self.get_recent_prices(symbol, lookback=60)
        X = self.prepare_ml_input(recent_prices)

        ml_prediction = self.ml_ensemble.predict_class(X)[0]
        ml_confidence = self.ml_ensemble.predict(X).max()

        # Combine with base signal
        if ml_prediction == base_signal['direction'].upper():
            # ML agrees - increase confidence
            base_signal['confidence'] = min(base_signal['confidence'] * 1.1, 0.95)
            base_signal['ml_agreement'] = True
        else:
            # ML disagrees - decrease confidence
            base_signal['confidence'] *= 0.9
            base_signal['ml_agreement'] = False

        base_signal['ml_prediction'] = ml_prediction
        base_signal['ml_confidence'] = ml_confidence

        return base_signal
```

**Success Criteria**:
- [ ] ML ensemble integrated into V7
- [ ] Signals include ML predictions
- [ ] Confidence adjusted based on ML agreement

**Time Breakdown**:
- 1 hour: Integration code
- 1 hour: Testing

---

### Module 2: GARCH Volatility Models (8-10 hours)

**Goal**: Model volatility clustering for better risk estimates

---

#### STEP 2.1: GARCH(1,1) Implementation (4 hours)

**Create** (`libs/econometrics/garch_model.py`):
```python
"""
GARCH(1,1) Model for Volatility Forecasting

GARCH = Generalized Autoregressive Conditional Heteroskedasticity
Models volatility clustering (high volatility follows high volatility)
"""
from arch import arch_model
import pandas as pd
import numpy as np

class GARCHVolatility:
    """GARCH model for volatility forecasting"""

    def __init__(self, p=1, q=1):
        """
        Args:
            p: GARCH lag order
            q: ARCH lag order
        """
        self.p = p
        self.q = q
        self.model = None
        self.results = None

    def fit(self, returns, vol='Garch'):
        """
        Fit GARCH model to returns

        Args:
            returns: Series of returns
            vol: Volatility process ('Garch', 'EGARCH', 'FIGARCH')

        Returns:
            Fitted model results
        """
        # Remove mean (GARCH models residuals)
        returns_clean = returns.dropna() * 100  # Scale to percentages

        # Fit GARCH(p,q)
        self.model = arch_model(returns_clean, vol=vol, p=self.p, q=self.q)
        self.results = self.model.fit(disp='off')

        print(f"\nGARCH({self.p},{self.q}) Model Fitted")
        print(f"AIC: {self.results.aic:.2f}")
        print(f"BIC: {self.results.bic:.2f}")

        return self.results

    def forecast(self, horizon=24):
        """
        Forecast future volatility

        Args:
            horizon: Number of periods ahead

        Returns:
            Volatility forecast (annualized std dev)
        """
        forecast = self.results.forecast(horizon=horizon)

        # Extract variance forecast
        variance_forecast = forecast.variance.values[-1, :]

        # Convert to annualized volatility
        volatility_forecast = np.sqrt(variance_forecast) / 100  # Unscale

        return volatility_forecast

    def conditional_volatility(self):
        """Get conditional volatility (fitted values)"""
        return self.results.conditional_volatility / 100  # Unscale


# CLI usage
if __name__ == "__main__":
    # Load BTC returns
    btc_prices = pd.read_parquet('data/backtest/BTC_USD_historical.parquet')['close']
    returns = btc_prices.pct_change().dropna()

    # Fit GARCH
    garch = GARCHVolatility(p=1, q=1)
    garch.fit(returns)

    # Forecast next 24 hours
    vol_forecast = garch.forecast(horizon=24)

    print(f"\nVolatility Forecast (next 24 hours):")
    for i, vol in enumerate(vol_forecast[:5]):
        print(f"  Hour {i+1}: {vol*100:.2f}%")

    print(f"\nAverage forecasted volatility: {vol_forecast.mean()*100:.2f}%")
```

**Success Criteria**:
- [ ] GARCH fits to returns
- [ ] Volatility forecast generated
- [ ] AIC/BIC reasonable (lower = better fit)

**Time Breakdown**:
- 2 hours: Implement GARCH
- 1.5 hours: Fit to data
- 0.5 hours: Validate forecasts

---

#### STEP 2.2: Integrate GARCH into Risk Management (2 hours)

**Update** (`libs/risk/cvar_calculator.py`):
```python
from libs/econometrics.garch_model import GARCHVolatility

class CVaRCalculatorGARCH(CVaRCalculator):
    """CVaR with GARCH volatility forecasts"""

    def __init__(self, confidence=0.95):
        super().__init__(confidence)
        self.garch = GARCHVolatility()

    def calculate_cvar_with_garch(self, returns):
        """CVaR using GARCH volatility forecast"""
        # Fit GARCH
        self.garch.fit(returns)

        # Forecast volatility for next period
        vol_forecast = self.garch.forecast(horizon=1)[0]

        # Parametric CVaR using forecasted vol
        from scipy import stats
        mu = returns.mean()
        sigma = vol_forecast

        z = stats.norm.ppf(1 - self.confidence)
        cvar_garch = mu - sigma * stats.norm.pdf(z) / (1 - self.confidence)

        return cvar_garch
```

**Success Criteria**:
- [ ] GARCH volatility used in CVaR
- [ ] Risk estimates more accurate during volatile periods

---

### Module 3: Non-Ergodicity Framework (10-12 hours)

**Goal**: Implement Ole Peters' non-ergodicity economics for crypto

---

#### STEP 3.1: Ergodicity Testing (4 hours)

**Create** (`libs/econometrics/ergodicity.py`):
```python
"""
Non-Ergodicity Analysis (Ole Peters, 2011)

Tests if crypto markets are ergodic:
- Ergodic: Time average = Ensemble average
- Non-ergodic: Time average ≠ Ensemble average

If non-ergodic, use Kelly Criterion for optimal growth
"""
import numpy as np
import pandas as pd

class ErgodicityAnalyzer:
    """Test and analyze market ergodicity"""

    def test_ergodicity(self, returns):
        """
        Test ergodicity hypothesis

        Args:
            returns: Series of returns

        Returns:
            Dict with ergodicity test results
        """
        # Time average (geometric mean)
        time_avg = np.prod(1 + returns) ** (1/len(returns)) - 1

        # Ensemble average (arithmetic mean)
        ensemble_avg = returns.mean()

        # Ergodicity ratio
        ergodicity_ratio = time_avg / ensemble_avg if ensemble_avg != 0 else 0

        # Interpretation
        if ergodicity_ratio < 0.8:
            interpretation = "NON-ERGODIC (time average << ensemble average)"
            recommendation = "Use Kelly Criterion for growth-optimal strategy"
        elif ergodicity_ratio < 0.95:
            interpretation = "WEAKLY NON-ERGODIC"
            recommendation = "Consider fractional Kelly"
        else:
            interpretation = "ERGODIC (time ≈ ensemble)"
            recommendation = "Standard optimization techniques apply"

        return {
            'time_average': time_avg,
            'ensemble_average': ensemble_avg,
            'ergodicity_ratio': ergodicity_ratio,
            'interpretation': interpretation,
            'recommendation': recommendation
        }

    def time_average_growth_rate(self, returns):
        """
        Calculate time-average growth rate (TAGR)

        For non-ergodic systems, maximize TAGR instead of expected return
        """
        # TAGR = geometric mean of (1 + return)
        tagr = np.prod(1 + returns) ** (1/len(returns)) - 1

        return tagr

    def ensemble_average_growth_rate(self, returns):
        """
        Calculate ensemble-average growth rate (EAGR)

        For ergodic systems, maximize EAGR
        """
        # EAGR = arithmetic mean
        eagr = returns.mean()

        return eagr


# CLI usage
if __name__ == "__main__":
    btc_prices = pd.read_parquet('data/backtest/BTC_USD_historical.parquet')['close']
    returns = btc_prices.pct_change().dropna()

    analyzer = ErgodicityAnalyzer()
    results = analyzer.test_ergodicity(returns)

    print("\nERGODICITY TEST RESULTS")
    print("="*70)
    print(f"Time Average (Geometric):    {results['time_average']*100:.3f}%")
    print(f"Ensemble Average (Arithmetic): {results['ensemble_average']*100:.3f}%")
    print(f"Ergodicity Ratio:            {results['ergodicity_ratio']:.3f}")
    print(f"\nInterpretation: {results['interpretation']}")
    print(f"Recommendation: {results['recommendation']}")
```

**Success Criteria**:
- [ ] Determines if BTC market is ergodic
- [ ] Recommends strategy based on ergodicity

---

### Module 4: Pairs Trading (8-10 hours)

**Goal**: Implement cointegration-based pairs trading

---

#### STEP 4.1: Cointegration Analysis (4 hours)

**Create** (`libs/strategies/pairs_trading.py`):
```python
"""
Pairs Trading using Cointegration

Finds pairs of cryptocurrencies that move together
Trades the spread when it deviates
"""
from statsmodels.tsa.stattools import coint
import pandas as pd
import numpy as np

class PairsTradingStrategy:
    """Cointegration-based pairs trading"""

    def find_cointegrated_pairs(self, prices_df, significance=0.05):
        """
        Find all cointegrated pairs

        Args:
            prices_df: DataFrame with prices for multiple symbols
            significance: P-value threshold (0.05 = 95% confidence)

        Returns:
            List of (symbol1, symbol2, p-value, hedge_ratio) tuples
        """
        n = prices_df.shape[1]
        pairs = []

        for i in range(n):
            for j in range(i+1, n):
                S1 = prices_df.iloc[:, i]
                S2 = prices_df.iloc[:, j]

                # Test cointegration
                score, pvalue, _ = coint(S1, S2)

                if pvalue < significance:
                    # Calculate hedge ratio (OLS regression)
                    hedge_ratio = np.polyfit(S2, S1, 1)[0]

                    pairs.append({
                        'symbol1': prices_df.columns[i],
                        'symbol2': prices_df.columns[j],
                        'pvalue': pvalue,
                        'hedge_ratio': hedge_ratio,
                        'score': score
                    })

        # Sort by p-value (lower = stronger cointegration)
        pairs_df = pd.DataFrame(pairs).sort_values('pvalue')

        return pairs_df

    def calculate_spread(self, price1, price2, hedge_ratio):
        """Calculate spread between two cointegrated assets"""
        spread = price1 - hedge_ratio * price2
        return spread

    def generate_signals(self, spread, entry_z=2.0, exit_z=0.5):
        """
        Generate trading signals from spread

        Args:
            spread: Time series of spread values
            entry_z: Z-score threshold to enter (default 2.0)
            exit_z: Z-score threshold to exit (default 0.5)

        Returns:
            DataFrame with trading signals
        """
        # Calculate rolling mean and std
        spread_mean = spread.rolling(window=60).mean()
        spread_std = spread.rolling(window=60).std()

        # Z-score
        z_score = (spread - spread_mean) / spread_std

        # Generate signals
        signals = pd.Series(index=spread.index, data='HOLD')

        signals[z_score < -entry_z] = 'LONG_SPREAD'   # Long asset1, Short asset2
        signals[z_score > entry_z] = 'SHORT_SPREAD'  # Short asset1, Long asset2
        signals[abs(z_score) < exit_z] = 'CLOSE'      # Exit position

        return signals, z_score


# CLI usage
if __name__ == "__main__":
    # Load prices for multiple symbols
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    prices = {}

    for symbol in symbols:
        df = pd.read_parquet(f"data/backtest/{symbol.replace('-', '_')}_historical.parquet")
        prices[symbol] = df['close']

    prices_df = pd.DataFrame(prices)

    # Find cointegrated pairs
    pairs_trader = PairsTradingStrategy()
    pairs = pairs_trader.find_cointegrated_pairs(prices_df)

    print("\nCOINTEGRATED PAIRS FOUND:")
    print("="*70)
    print(pairs.to_string(index=False))

    if len(pairs) > 0:
        # Trade the best pair
        best_pair = pairs.iloc[0]
        print(f"\nTrading pair: {best_pair['symbol1']} vs {best_pair['symbol2']}")
        print(f"Hedge ratio: {best_pair['hedge_ratio']:.4f}")

        # Calculate spread
        spread = pairs_trader.calculate_spread(
            prices_df[best_pair['symbol1']],
            prices_df[best_pair['symbol2']],
            best_pair['hedge_ratio']
        )

        # Generate signals
        signals, z_score = pairs_trader.generate_signals(spread)

        print(f"\nSignals generated: {len(signals[signals != 'HOLD'])}")
```

**Success Criteria**:
- [ ] Identifies cointegrated pairs (p-value <0.05)
- [ ] Generates spread trading signals
- [ ] Backtest shows profit potential

---

### Module 5: Multi-Factor Models (10-12 hours)

**Goal**: Implement Fama-French-style factor models adapted for crypto

---

#### STEP 5.1: Factor Extraction with PCA (4 hours)

**Create** (`libs/factors/factor_models.py`):
```python
"""
Multi-Factor Models for Cryptocurrency

Adapted from Fama-French (1993) for crypto markets

Factors:
- Market factor (overall crypto market)
- Size factor (large cap vs small cap)
- Momentum factor (winners vs losers)
- Volatility factor (low vol vs high vol)
"""
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

class CryptoFactorModel:
    """Extract and analyze crypto market factors"""

    def extract_pca_factors(self, returns, n_factors=5):
        """
        Extract principal components as factors

        Args:
            returns: DataFrame of returns for multiple assets
            n_factors: Number of factors to extract

        Returns:
            Factor loadings and scores
        """
        pca = PCA(n_components=n_factors)
        factor_scores = pca.fit_transform(returns.fillna(0))

        factor_df = pd.DataFrame(
            factor_scores,
            index=returns.index,
            columns=[f'Factor_{i+1}' for i in range(n_factors)]
        )

        explained_var = pca.explained_variance_ratio_

        print(f"\nPCA Factor Analysis:")
        for i, var in enumerate(explained_var):
            print(f"  Factor {i+1}: {var*100:.1f}% variance explained")

        return factor_df, pca

    def construct_crypto_factors(self, prices_df, market_caps_df):
        """
        Construct Fama-French-style factors for crypto

        Factors:
        - MKT: Market return (equal-weighted portfolio)
        - SMB: Small Minus Big (small cap - large cap)
        - MOM: Momentum (high momentum - low momentum)
        """
        returns = prices_df.pct_change()

        # MKT: Market factor (equal-weighted)
        mkt = returns.mean(axis=1)

        # SMB: Size factor
        median_mcap = market_caps_df.median(axis=1)
        small_stocks = returns.where(market_caps_df.lt(median_mcap, axis=0))
        big_stocks = returns.where(market_caps_df.ge(median_mcap, axis=0))
        smb = small_stocks.mean(axis=1) - big_stocks.mean(axis=1)

        # MOM: Momentum factor (past 30 days)
        past_returns = returns.rolling(30).sum()
        median_mom = past_returns.median(axis=1)
        winners = returns.where(past_returns.gt(median_mom, axis=0))
        losers = returns.where(past_returns.le(median_mom, axis=0))
        mom = winners.mean(axis=1) - losers.mean(axis=1)

        factors = pd.DataFrame({
            'MKT': mkt,
            'SMB': smb,
            'MOM': mom
        })

        return factors


# CLI usage
if __name__ == "__main__":
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    prices = {}

    for symbol in symbols:
        df = pd.read_parquet(f"data/backtest/{symbol.replace('-', '_')}_historical.parquet")
        prices[symbol] = df['close']

    prices_df = pd.DataFrame(prices)
    returns = prices_df.pct_change()

    # Extract PCA factors
    factor_model = CryptoFactorModel()
    factors, pca = factor_model.extract_pca_factors(returns, n_factors=3)

    print("\nFactor scores (last 5 periods):")
    print(factors.tail())
```

**Success Criteria**:
- [ ] Extracts 3-5 factors explaining >70% variance
- [ ] Constructs SMB, MOM factors
- [ ] Factor returns calculated

---

## 📅 IMPLEMENTATION TIMELINE

**Phase 2 Breakdown** (Total: 50-60 hours):

| Module | Component | Hours |
|--------|-----------|-------|
| **Module 1** | Deep Learning | 12-15 |
| - Step 1.1 | Setup DL Environment | 2 |
| - Step 1.2 | LSTM Model | 4 |
| - Step 1.3 | Transformer Model | 4 |
| - Step 1.4 | Model Ensemble | 2 |
| - Step 1.5 | V7 Integration | 2 |
| **Module 2** | GARCH Models | 8-10 |
| - Step 2.1 | GARCH Implementation | 4 |
| - Step 2.2 | GARCH + CVaR | 2 |
| - Step 2.3 | Volatility Forecasting | 2-3 |
| **Module 3** | Non-Ergodicity | 10-12 |
| - Step 3.1 | Ergodicity Testing | 4 |
| - Step 3.2 | Time-Average Growth | 3-4 |
| - Step 3.3 | Strategy Adaptation | 3-4 |
| **Module 4** | Pairs Trading | 8-10 |
| - Step 4.1 | Cointegration Analysis | 4 |
| - Step 4.2 | Spread Trading | 2-3 |
| - Step 4.3 | Backtesting | 2-3 |
| **Module 5** | Factor Models | 10-12 |
| - Step 5.1 | PCA Factors | 4 |
| - Step 5.2 | Crypto Factors | 3-4 |
| - Step 5.3 | Factor Portfolios | 3-4 |

**Recommended Schedule** (10 hours/week):
- Week 1-2: Module 1 (Deep Learning)
- Week 3: Module 2 (GARCH)
- Week 4-5: Module 3 (Non-Ergodicity)
- Week 6: Module 4 (Pairs Trading)
- Week 7-8: Module 5 (Factor Models)

---

## ✅ SUCCESS CRITERIA (Phase 2)

**Minimum** (Must Achieve):
- [ ] LSTM model trains and achieves >55% accuracy
- [ ] GARCH volatility forecasts generated
- [ ] Ergodicity test completed (know if market is ergodic)
- [ ] At least 1 cointegrated pair found
- [ ] 3+ factors extracted with PCA

**Good** (Target):
- [ ] Ensemble model >60% accuracy
- [ ] GARCH improves CVaR accuracy by >10%
- [ ] Non-ergodic strategy outperforms standard approach
- [ ] Pairs trading strategy profitable in backtest
- [ ] Factors explain >70% of variance

**Excellent** (Stretch):
- [ ] Transformer outperforms LSTM
- [ ] ML ensemble >65% accuracy
- [ ] All 5 modules integrated into V7
- [ ] Combined Sharpe ratio >2.0

---

## 🔄 HANDOFF TO BUILDER CLAUDE

**When Phase 1 is Complete**:

1. ✅ **Verify Phase 1 Prerequisites**:
   ```bash
   # Check Phase 1 completion
   python libs/backtest/simple_backtest.py       # Should work
   python libs/portfolio/optimizer.py            # Should work
   python libs/risk/kelly_criterion.py           # Should work
   python libs/risk/cvar_calculator.py           # Should work
   python apps/runtime/v7_quant_enhanced.py --iterations 1  # Should work
   ```

2. ✅ **Start Phase 2 - Module 1**:
   ```bash
   # Begin with Deep Learning
   cd /root/crpbot  # Builder Claude working directory

   # Step 1.1: Setup
   pip install tensorflow keras torch transformers
   python libs/ml/dl_environment.py

   # Step 1.2: Train LSTM
   python libs/ml/lstm_predictor.py

   # Continue with Steps 1.3-1.5...
   ```

3. ✅ **Track Progress**:
   - Create `PHASE_2_PROGRESS.md` to track completion
   - Update after each module
   - Report issues/blockers

4. ✅ **Commit Strategy**:
   ```bash
   # After each major step
   git add libs/ml/
   git commit -m "feat(ml): implement LSTM price predictor (Phase 2, Step 1.2)"
   git push origin feature/v7-ultimate
   ```

---

## 📊 EXPECTED IMPROVEMENTS

**After Phase 2 Complete**:

| Metric | Phase 1 | Phase 2 Target | Improvement |
|--------|---------|----------------|-------------|
| Sharpe Ratio | 1.5 | 2.0+ | +33% |
| Prediction Accuracy | 55% | 65% | +10% |
| Max Drawdown | 18% | 12% | -33% |
| Signal Quality (IC) | 0.05 | 0.10 | +100% |
| Risk Management | CVaR | CVaR + GARCH | Better |

---

## 💡 IMPORTANT NOTES

1. **GPU Highly Recommended**:
   - LSTM/Transformer training: 10x faster with GPU
   - Without GPU: Budget 2-4x more time for Module 1

2. **Data Requirements**:
   - Deep learning needs >1 year of data (minimum)
   - 2+ years preferred for better generalization

3. **Sequential Dependencies**:
   - Module 1 (LSTM) → Module 2 (GARCH) → Module 3 (Integration)
   - Don't skip modules (each builds on previous)

4. **Resource Management**:
   - Deep learning: 4-8GB RAM during training
   - Monitor with `htop` and `nvidia-smi` (if GPU)

5. **Validation Critical**:
   - Backtest EVERY new component
   - Only keep if improves Sharpe ratio
   - Remove if decreases performance

---

**Status**: ⏳ READY TO START (after Phase 1 complete)
**Est. Duration**: 5-6 weeks (10 hours/week) or 50-60 hours total
**Expected Outcome**: Institutional-grade quant system with cutting-edge ML and econometrics
