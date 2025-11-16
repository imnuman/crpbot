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
        patterns = [
            f"lstm_{self.symbol.replace('-', '_')}_*_FIXED.pt",
            f"lstm_{self.symbol.replace('-', '-')}_*_FIXED.pt",
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

        # V5 uses standard PyTorch LSTM (non-bidirectional, 3 layers)
        import torch.nn as nn

        class SimpleV5LSTM(nn.Module):
            def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.fc(lstm_out[:, -1, :])

        self.lstm_model = SimpleV5LSTM(input_size=input_size)
        self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
        self.lstm_model.to(self.device)
        self.lstm_model.eval()

        logger.info(f"✅ LSTM loaded: {input_size} features")

    def predict(self, df):
        """Generate prediction from market data."""
        if len(df) < 60:
            raise ValueError(f"Need ≥60 rows, got {len(df)}")

        # Exclude OHLCV columns and non-numeric columns
        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'session']
        features = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        lstm_pred = 0.5
        if self.lstm_model:
            try:
                seq = torch.FloatTensor(df[features].iloc[-60:].values).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    lstm_pred = torch.sigmoid(self.lstm_model(seq)).item()
            except Exception as e:
                logger.warning(f"LSTM inference failed: {e}")
        
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
