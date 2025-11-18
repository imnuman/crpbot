#!/usr/bin/env python3
"""
V6 Fixed Models Runtime
Simple runtime using V6 Fixed models with temperature scaling
"""

import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import time
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from apps.trainer.amazon_q_features import engineer_amazon_q_features
from libs.data.kraken_client import KrakenClient
from libs.database.operations import create_signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class V6EnhancedFNN(nn.Module):
    """Original V6 model architecture."""
    def __init__(self, input_size=72):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        if len(x.shape) == 3:
            x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)


class V6FixedWrapper(nn.Module):
    """Wrapper that adds normalization and temperature scaling to V6 models."""

    def __init__(self, base_model, scaler, temperature=1.0, logit_clip=15.0):
        super().__init__()
        self.base_model = base_model
        self.scaler = scaler
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.logit_clip = logit_clip

    def forward(self, x):
        # Get raw logits from base model
        logits = self.base_model(x)

        # Clamp logits to prevent numerical overflow
        logits = torch.clamp(logits, -self.logit_clip, self.logit_clip)

        # Apply temperature scaling
        logits = logits / self.temperature

        return logits


class V6FixedRuntime:
    """Runtime for V6 Fixed models"""

    def __init__(self):
        self.symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        self.models = {}
        self.scalers = {}
        self.kraken_client = KrakenClient()
        self.feature_cols = [
            'atr_14', 'bb_lower_20', 'bb_lower_50', 'bb_position_20', 'bb_position_50',
            'bb_upper_20', 'bb_upper_50', 'close_open_ratio', 'ema_10', 'ema_20',
            'ema_200', 'ema_5', 'ema_50', 'high_low_ratio', 'log_returns',
            'macd_12_26', 'macd_5_35', 'macd_histogram_12_26', 'macd_histogram_5_35',
            'macd_signal_12_26', 'macd_signal_5_35', 'momentum_10', 'momentum_20',
            'momentum_5', 'momentum_50', 'price_channel_high_20', 'price_channel_high_50',
            'price_channel_low_20', 'price_channel_low_50', 'price_channel_position_20',
            'price_channel_position_50', 'price_to_ema_10', 'price_to_ema_20',
            'price_to_ema_200', 'price_to_ema_5', 'price_to_ema_50', 'price_to_sma_10',
            'price_to_sma_20', 'price_to_sma_200', 'price_to_sma_5', 'price_to_sma_50',
            'returns', 'returns_lag_1', 'returns_lag_2', 'returns_lag_3', 'returns_lag_5',
            'roc_10', 'roc_20', 'roc_5', 'roc_50', 'rsi_14', 'rsi_21', 'rsi_30',
            'sma_10', 'sma_20', 'sma_200', 'sma_5', 'sma_50', 'stoch_d_14', 'stoch_d_21',
            'stoch_k_14', 'stoch_k_21', 'volatility_20', 'volatility_50', 'volume_lag_1',
            'volume_lag_2', 'volume_lag_3', 'volume_lag_5', 'volume_price_trend',
            'volume_ratio', 'williams_r_14', 'williams_r_21'
        ]

    def load_models(self):
        """Load all V6 Fixed models"""
        model_dir = Path("models/v6_fixed")

        for symbol in self.symbols:
            try:
                model_path = model_dir / f"lstm_{symbol}_v6_FIXED.pt"
                scaler_path = model_dir / f"scaler_{symbol}_v6_fixed.pkl"

                # Load scaler
                with open(scaler_path, 'rb') as f:
                    self.scalers[symbol] = pickle.load(f)

                # Load model
                checkpoint = torch.load(model_path, map_location='cpu')

                # Recreate base model
                base_model = V6EnhancedFNN(input_size=72)
                base_model.load_state_dict(checkpoint['base_model_state_dict'])

                # Recreate wrapper
                model = V6FixedWrapper(
                    base_model=base_model,
                    scaler=self.scalers[symbol],
                    temperature=checkpoint['temperature'],
                    logit_clip=checkpoint['logit_clip']
                )
                model.eval()

                self.models[symbol] = model
                logger.info(f"✅ Loaded {symbol}: T={checkpoint['temperature']:.1f}, clip=±{checkpoint['logit_clip']}")

            except Exception as e:
                logger.error(f"❌ Failed to load {symbol}: {e}")

    async def get_market_data(self, symbol: str) -> pd.DataFrame:
        """Get recent market data from Kraken"""
        kraken_pairs = {
            'BTC-USD': 'XBTUSD',
            'ETH-USD': 'ETHUSD',
            'SOL-USD': 'SOLUSD'
        }

        pair = kraken_pairs[symbol]
        ohlc_data = await self.kraken_client.get_ohlc(pair, interval=60, count=300)

        df = pd.DataFrame(ohlc_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('timestamp').sort_index()

        return df

    def predict(self, symbol: str, df: pd.DataFrame):
        """Make prediction using V6 Fixed model"""
        if symbol not in self.models:
            logger.error(f"Model not loaded for {symbol}")
            return None

        try:
            # Engineer features
            df_features = engineer_amazon_q_features(df)

            if len(df_features) < 60:
                logger.warning(f"Insufficient data for {symbol}")
                return None

            # Get last 60 rows of features
            features = df_features[self.feature_cols].iloc[-60:].values
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

            # Normalize
            scaler = self.scalers[symbol]
            features_normalized = scaler.transform(features)

            # Convert to tensor
            sample = torch.FloatTensor(features_normalized).unsqueeze(0)

            # Get prediction
            model = self.models[symbol]
            with torch.no_grad():
                logits = model(sample)
                probs = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probs, dim=-1).item()
                confidence = torch.max(probs).item()

            # Map to signal
            signal_map = {0: 'DOWN', 1: 'NEUTRAL', 2: 'UP'}
            signal = signal_map[predicted_class]

            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'logits': logits.squeeze().numpy().tolist(),
                'probabilities': probs.squeeze().numpy().tolist()
            }

        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return None

    async def run(self, sleep_seconds=60, iterations=-1):
        """Run trading loop"""
        logger.info("🚀 V6 Fixed Runtime Starting...")

        # Load models
        self.load_models()

        if not self.models:
            logger.error("❌ No models loaded!")
            return

        logger.info(f"✅ Loaded {len(self.models)}/3 V6 Fixed models")
        logger.info(f"📊 Scanning every {sleep_seconds}s")

        iteration = 0
        while iterations < 0 or iteration < iterations:
            iteration += 1
            logger.info(f"\n{'='*80}")
            logger.info(f"Iteration {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*80}")

            for symbol in self.symbols:
                try:
                    # Get market data
                    df = await self.get_market_data(symbol)

                    # Make prediction
                    result = self.predict(symbol, df)

                    if result:
                        logger.info(f"{symbol}: {result['signal']} ({result['confidence']:.1%} confidence)")
                        logger.info(f"  Probs: DOWN={result['probabilities'][0]:.2%}, "
                                   f"NEUTRAL={result['probabilities'][1]:.2%}, "
                                   f"UP={result['probabilities'][2]:.2%}")

                        # Save to database
                        create_signal(
                            symbol=symbol,
                            signal=result['signal'],
                            confidence=result['confidence'],
                            model_version='v6_fixed_t1.0',
                            metadata={
                                'logits': result['logits'],
                                'probabilities': result['probabilities']
                            }
                        )

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

            if iterations < 0 or iteration < iterations:
                logger.info(f"\n💤 Sleeping {sleep_seconds}s...")
                time.sleep(sleep_seconds)


async def main():
    runtime = V6FixedRuntime()
    await runtime.run(sleep_seconds=60, iterations=-1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
