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
        
        self.lstm_model = LSTMDirectionModel(input_size=input_size, hidden_size=128, num_layers=3, dropout=0.3)
        self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
        self.lstm_model.to(self.device)
        self.lstm_model.eval()
        
        logger.info(f"✅ LSTM loaded: {input_size} features")

    def predict(self, df):
        """Generate prediction from market data."""
        if len(df) < 60:
            raise ValueError(f"Need ≥60 rows, got {len(df)}")

        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        features = [c for c in df.columns if c not in exclude]
        
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
