"""Ensemble model inference for production runtime.

Loads V5 models and generates real-time predictions from live market data.
"""
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

from apps.trainer.models.lstm import LSTMDirectionModel


class EnsemblePredictor:
    """Ensemble predictor for runtime inference using V5 models."""

    def __init__(self, symbol: str, model_dir: str = "models/promoted", device: Optional[str] = None):
        self.symbol = symbol
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        
        logger.info(f"Initializing EnsemblePredictor for {symbol} on {self.device}")
        
        self.lstm_model = None
        self._load_models()

    def _load_models(self):
        """Load LSTM model from promoted directory."""
        # Prioritize V6 enhanced models, then V6 real, fallback to V5 FIXED models
        patterns = [
            f"lstm_{self.symbol}_v6_enhanced.pt",                  # V6 enhanced models (HIGHEST priority)
            f"lstm_{self.symbol.replace('-', '-')}_*_v6_real.pt",  # V6 real models
            f"lstm_{self.symbol.replace('-', '_')}_*_FIXED.pt",    # V5 FIXED models
            f"lstm_{self.symbol.replace('-', '-')}_*_FIXED.pt",    # V5 FIXED models (alt format)
        ]

        lstm_path = None
        for pattern in patterns:
            files = list(self.model_dir.glob(pattern))
            if files:
                lstm_path = files[0]
                break

        if not lstm_path:
            logger.warning(f"⚠️  No LSTM model found for {self.symbol}")
            return

        logger.info(f"Loading LSTM: {lstm_path.name}")

        checkpoint = torch.load(lstm_path, map_location=self.device)
        input_size = checkpoint.get('input_size', 80)

        # Store input size for feature selection in predict()
        self._checkpoint_input_size = input_size

        # Detect model architecture from checkpoint
        state_dict = checkpoint['model_state_dict']
        first_weight_key = list(state_dict.keys())[0]

        import torch.nn as nn

        # Check if V6 Enhanced FNN (feedforward network) - has fc1, fc2, fc3, fc4
        is_v6_enhanced_fnn = ('fc1.weight' in state_dict and 'fc4.weight' in state_dict)

        # Check if V6 LSTM model (2-layer, bidirectional, hidden_size=64)
        # V6: weight_ih_l0 shape is [256, 31] = 4 * 64 (bidirectional)
        # V5: weight_ih_l0 shape is [512, 31] = 4 * 128 (non-bidirectional)
        is_v6_lstm = (not is_v6_enhanced_fnn and
                      first_weight_key.startswith('lstm.') and
                      state_dict[first_weight_key].shape[0] == 256 and
                      'lstm.weight_ih_l2' not in state_dict)

        if is_v6_enhanced_fnn:
            # V6 Enhanced FNN: 4-layer feedforward network (72 -> 256 -> 128 -> 64 -> 3)
            class V6EnhancedFNN(nn.Module):
                def __init__(self, input_size=72):
                    super().__init__()
                    self.fc1 = nn.Linear(input_size, 256)
                    self.fc2 = nn.Linear(256, 128)
                    self.fc3 = nn.Linear(128, 64)
                    self.fc4 = nn.Linear(64, 3)
                    self.relu = nn.ReLU()

                def forward(self, x):
                    # Input: (batch, sequence, features) - take last timestep
                    if len(x.shape) == 3:
                        x = x[:, -1, :]  # (batch, features)
                    x = self.relu(self.fc1(x))
                    x = self.relu(self.fc2(x))
                    x = self.relu(self.fc3(x))
                    return self.fc4(x)  # (batch, 3) - logits for 3 classes

            self.lstm_model = V6EnhancedFNN(input_size=input_size)
            logger.info(f"Using V6 Enhanced FNN architecture: 4-layer feedforward (72→256→128→64→3)")
        elif is_v6_lstm:
            # V6 model: 2-layer non-bidirectional LSTM, hidden_size=64, no dropout
            class V6LSTM(nn.Module):
                def __init__(self, input_size, hidden_size=64, num_layers=2):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                       batch_first=True, bidirectional=False)
                    self.fc = nn.Linear(hidden_size, 1)

                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    return self.fc(lstm_out[:, -1, :])

            self.lstm_model = V6LSTM(input_size=input_size)
            logger.info(f"Using V6 architecture: 2-layer non-bidirectional, hidden_size=64")
        else:
            # V5 model: 3-layer non-bidirectional LSTM, hidden_size=128
            class SimpleV5LSTM(nn.Module):
                def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
                    self.fc = nn.Linear(hidden_size, 1)

                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    return self.fc(lstm_out[:, -1, :])

            self.lstm_model = SimpleV5LSTM(input_size=input_size)
            logger.info(f"Using V5 architecture: 3-layer non-bidirectional, hidden_size=128")

        self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
        self.lstm_model.to(self.device)
        self.lstm_model.eval()

        logger.info(f"✅ LSTM loaded: {input_size} features")

    def predict(self, df):
        """Generate prediction from market data."""
        if len(df) < 60:
            raise ValueError(f"Need ≥60 rows, got {len(df)}")

        # V6 models use specific 31 features - select only those
        v6_feature_list = [
            'open', 'high', 'low', 'close', 'volume',
            '5m_open', '5m_high', '5m_low', '5m_close', '5m_volume',
            '15m_open', '15m_high', '15m_low', '15m_close', '15m_volume',
            '1h_open', '1h_high', '1h_low', '1h_close', '1h_volume',
            'tf_alignment_score', 'tf_alignment_direction', 'tf_alignment_strength',
            'atr', 'atr_percentile',
            'volatility_regime', 'volatility_low', 'volatility_medium', 'volatility_high',
            'session', 'session_tokyo'
        ]

        # Check if this is a V6 model by inspecting loaded checkpoint
        # V6 Enhanced FNN models have input_size=72
        # V6 LSTM models have input_size=31 in checkpoint metadata
        checkpoint_input_size = getattr(self, '_checkpoint_input_size', None)

        if checkpoint_input_size == 72:
            # V6 Enhanced FNN - use Amazon Q's exact 72 features
            from apps.trainer.amazon_q_features import engineer_amazon_q_features

            # Check if Amazon Q features are already engineered
            amazon_q_features = [
                'atr_14', 'bb_lower_20', 'bb_position_20', 'ema_10', 'rsi_14',
                'macd_12_26', 'momentum_10', 'roc_10', 'stoch_k_14', 'williams_r_14'
            ]

            if not all(f in df.columns for f in amazon_q_features[:5]):
                # Need to engineer Amazon Q features from OHLCV
                logger.info("Engineering Amazon Q's 72 features...")
                df = engineer_amazon_q_features(df)

            # Get exact 72 features in correct order
            features = [
                'atr_14',
                'bb_lower_20', 'bb_lower_50',
                'bb_position_20', 'bb_position_50',
                'bb_upper_20', 'bb_upper_50',
                'close_open_ratio',
                'ema_10', 'ema_20', 'ema_200', 'ema_5', 'ema_50',
                'high_low_ratio',
                'log_returns',
                'macd_12_26', 'macd_5_35',
                'macd_histogram_12_26', 'macd_histogram_5_35',
                'macd_signal_12_26', 'macd_signal_5_35',
                'momentum_10', 'momentum_20', 'momentum_5', 'momentum_50',
                'price_channel_high_20', 'price_channel_high_50',
                'price_channel_low_20', 'price_channel_low_50',
                'price_channel_position_20', 'price_channel_position_50',
                'price_to_ema_10', 'price_to_ema_20', 'price_to_ema_200', 'price_to_ema_5', 'price_to_ema_50',
                'price_to_sma_10', 'price_to_sma_20', 'price_to_sma_200', 'price_to_sma_5', 'price_to_sma_50',
                'returns',
                'returns_lag_1', 'returns_lag_2', 'returns_lag_3', 'returns_lag_5',
                'roc_10', 'roc_20', 'roc_5', 'roc_50',
                'rsi_14', 'rsi_21', 'rsi_30',
                'sma_10', 'sma_20', 'sma_200', 'sma_5', 'sma_50',
                'stoch_d_14', 'stoch_d_21',
                'stoch_k_14', 'stoch_k_21',
                'volatility_20', 'volatility_50',
                'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5',
                'volume_price_trend',
                'volume_ratio',
                'williams_r_14', 'williams_r_21'
            ]

            logger.debug(f"V6 Enhanced FNN detected - using Amazon Q's 72 features")
        elif checkpoint_input_size == 31:
            # V6 LSTM model - use exact 31-feature set
            features = [f for f in v6_feature_list if f in df.columns]
            missing_features = [f for f in v6_feature_list if f not in df.columns]

            if missing_features:
                logger.warning(f"Missing V6 features: {missing_features}")

            logger.debug(f"V6 LSTM detected - using {len(features)}/31 features")
        else:
            # V5 model or unknown - use all numeric features except excluded
            exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'session', 'volatility_regime']
            all_features = [c for c in df.columns if c not in exclude]
            features = [c for c in all_features if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

            logger.debug(f"V5 model - using {len(features)} numeric features")

        # Debug logging
        logger.debug(f"Total columns in DF: {len(df.columns)}")
        logger.debug(f"Features selected for inference: {len(features)}")
        if len(features) < 31:
            logger.warning(f"Feature count mismatch: expected 31, got {len(features)}")

        lstm_pred = 0.5
        if self.lstm_model:
            try:
                seq = torch.FloatTensor(df[features].iloc[-60:].values).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.lstm_model(seq)

                    # Check if V6 Enhanced FNN (3-class output)
                    if output.shape[-1] == 3:
                        # V6 Enhanced FNN: 3 classes (Down, Neutral, Up)
                        # Log raw logits for debugging
                        raw_logits = output.squeeze()
                        logger.debug(f"Raw logits: Down={raw_logits[0].item():.3f}, Neutral={raw_logits[1].item():.3f}, Up={raw_logits[2].item():.3f}")

                        # Clamp logits to prevent extreme values
                        # This forces more balanced probabilities
                        # ±2.0 with T=2.0 gives ~79% max confidence (instead of 99%)
                        clamped_logits = torch.clamp(output, min=-2.0, max=2.0)
                        temperature = 2.0
                        probs = torch.softmax(clamped_logits / temperature, dim=-1).squeeze()
                        down_prob = probs[0].item()
                        neutral_prob = probs[1].item()
                        up_prob = probs[2].item()

                        # Convert to binary: combine down+neutral vs up
                        # or: up_prob > down_prob for long signal
                        lstm_pred = up_prob  # Use up probability as confidence

                        logger.debug(f"V6 Enhanced FNN output (clamped ±2, T={temperature}): Down={down_prob:.3f}, Neutral={neutral_prob:.3f}, Up={up_prob:.3f}")
                    else:
                        # Binary output (V5/V6 LSTM)
                        lstm_pred = torch.sigmoid(output).item()
            except Exception as e:
                logger.warning(f"Model inference failed: {e}")
        
        ensemble = lstm_pred
        direction = "long" if ensemble >= 0.5 else "short"
        
        return {
            'lstm_prediction': lstm_pred,
            'transformer_prediction': 0.5,
            'rl_prediction': 0.5,
            'ensemble_prediction': ensemble,
            'direction': direction,
            'confidence': ensemble
        }


def load_ensemble(symbol: str, model_dir: str = "models/promoted"):
    return EnsemblePredictor(symbol=symbol, model_dir=model_dir)

# V6 Statistical Model Support
import json

class V6StatisticalEnsemble:
    """V6 Statistical model ensemble"""
    
    def __init__(self, model_dir="models/v6_statistical"):
        self.model_dir = model_dir
        self.models = {}
        self.symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        self._load_v6_models()
    
    def _load_v6_models(self):
        """Load V6 statistical models"""
        for symbol in self.symbols:
            model_path = f"{self.model_dir}/lstm_{symbol}_1m_v6_stat.json"
            
            if os.path.exists(model_path):
                with open(model_path, 'r') as f:
                    model_data = json.load(f)
                
                self.models[symbol] = {
                    'accuracy': model_data['accuracy'],
                    'params': model_data['statistical_params'],
                    'input_size': model_data['input_size']
                }
                
                logger.info(f"✅ Loaded V6 {symbol}: {model_data['accuracy']:.1%} accuracy")
    
    def predict_v6(self, features):
        """Generate V6 statistical predictions"""
        predictions = {}
        
        for symbol in self.symbols:
            if symbol not in self.models:
                continue
                
            if symbol not in features:
                continue
            
            # Get model params
            model = self.models[symbol]
            params = model['params']
            
            # Extract features (assume last row of sequence)
            feature_dict = {}
            if len(features[symbol]) > 0:
                last_features = features[symbol][-1]  # Last time step
                
                # Map to feature names (simplified)
                feature_names = ['returns', 'rsi', 'macd', 'bb_position', 'volume_ratio']
                for i, name in enumerate(feature_names):
                    if i < len(last_features):
                        feature_dict[name] = last_features[i]
            
            # Statistical prediction
            returns = feature_dict.get('returns', 0)
            rsi = feature_dict.get('rsi', 50)
            bb_pos = feature_dict.get('bb_position', 0.5)
            vol_ratio = feature_dict.get('volume_ratio', 1.0)
            
            # Weighted prediction
            trend_signal = 0.5 + (returns * 5)
            momentum_signal = (rsi - 50) / 100 if rsi != 0 else 0
            volatility_signal = max(0, min(1, bb_pos))
            volume_signal = min(2, vol_ratio) / 2
            
            prediction = (
                trend_signal * params['trend_weight'] +
                (0.5 + momentum_signal) * params['momentum_weight'] +
                volatility_signal * params['volatility_weight'] +
                volume_signal * params['volume_weight']
            )
            
            # Ensure bounds and add slight randomness
            import random
            noise = random.uniform(-0.02, 0.02)
            prediction = max(0.1, min(0.9, prediction + noise))
            
            # Generate signal
            signal = "BUY" if prediction > 0.5 else "SELL"
            confidence = abs(prediction - 0.5) * 2
            
            predictions[symbol] = {
                'signal': signal,
                'probability': prediction,
                'confidence': confidence,
                'model_accuracy': model['accuracy']
            }
        
        return predictions

# Global V6 ensemble
_v6_ensemble = None

def get_v6_ensemble():
    """Get V6 statistical ensemble"""
    global _v6_ensemble
    if _v6_ensemble is None:
        _v6_ensemble = V6StatisticalEnsemble()
    return _v6_ensemble
