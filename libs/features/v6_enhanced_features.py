"""
V6 Enhanced Feature Engineering
Matches Amazon Q's exact 72 features from GPU training
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


class V6EnhancedFeatures:
    """Feature engineering matching Amazon Q's V6 Enhanced model (72 features)"""
    
    def __init__(self):
        self.feature_columns = [
            # Basic features (6)
            'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio', 
            'volume_ratio', 'volume_price_trend',
            
            # Momentum indicators (8)
            'momentum_5', 'momentum_10', 'momentum_20', 'momentum_50',
            'roc_5', 'roc_10', 'roc_20', 'roc_50',
            
            # Moving averages (20)
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
            'price_to_sma_5', 'price_to_sma_10', 'price_to_sma_20', 'price_to_sma_50', 'price_to_sma_200',
            'price_to_ema_5', 'price_to_ema_10', 'price_to_ema_20', 'price_to_ema_50', 'price_to_ema_200',
            
            # Volatility (3)
            'volatility_20', 'volatility_50', 'atr_14',
            
            # RSI (3)
            'rsi_14', 'rsi_21', 'rsi_30',
            
            # Bollinger Bands (6)
            'bb_upper_20', 'bb_lower_20', 'bb_position_20',
            'bb_upper_50', 'bb_lower_50', 'bb_position_50',
            
            # MACD (6)
            'macd_12_26', 'macd_signal_12_26', 'macd_histogram_12_26',
            'macd_5_35', 'macd_signal_5_35', 'macd_histogram_5_35',
            
            # Stochastic (4)
            'stoch_k_14', 'stoch_d_14', 'stoch_k_21', 'stoch_d_21',
            
            # Williams %R (2)
            'williams_r_14', 'williams_r_21',
            
            # Price channels (6)
            'price_channel_high_20', 'price_channel_low_20', 'price_channel_position_20',
            'price_channel_high_50', 'price_channel_low_50', 'price_channel_position_50',
            
            # Lagged features (8)
            'returns_lag_1', 'returns_lag_2', 'returns_lag_3', 'returns_lag_5',
            'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5'
        ]
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create exact 72 features matching V6 Enhanced training"""
        features = df.copy()
        
        # Ensure we have the right columns
        close_col = 'close' if 'close' in df.columns else 'Close'
        high_col = 'high' if 'high' in df.columns else 'High'
        low_col = 'low' if 'low' in df.columns else 'Low'
        open_col = 'open' if 'open' in df.columns else 'Open'
        volume_col = 'volume' if 'volume' in df.columns else 'Volume'
        
        # Basic price features (6)
        features['returns'] = df[close_col].pct_change()
        features['log_returns'] = np.log(df[close_col] / df[close_col].shift(1))
        features['high_low_ratio'] = df[high_col] / df[low_col]
        features['close_open_ratio'] = df[close_col] / df[open_col]
        
        # Volume features (2)
        features['volume_ratio'] = df[volume_col] / df[volume_col].rolling(20).mean()
        features['volume_price_trend'] = df[volume_col] * features['returns']
        
        # Momentum indicators (8)
        for period in [5, 10, 20, 50]:
            features[f'momentum_{period}'] = df[close_col] / df[close_col].shift(period) - 1
            features[f'roc_{period}'] = ((df[close_col] - df[close_col].shift(period)) / df[close_col].shift(period)) * 100
        
        # Moving averages and ratios (20)
        for period in [5, 10, 20, 50, 200]:
            features[f'sma_{period}'] = df[close_col].rolling(period).mean()
            features[f'ema_{period}'] = df[close_col].ewm(span=period).mean()
            features[f'price_to_sma_{period}'] = df[close_col] / features[f'sma_{period}']
            features[f'price_to_ema_{period}'] = df[close_col] / features[f'ema_{period}']
        
        # Volatility indicators (3)
        features['volatility_20'] = features['returns'].rolling(20).std()
        features['volatility_50'] = features['returns'].rolling(50).std()
        features['atr_14'] = (df[high_col] - df[low_col]).rolling(14).mean()
        
        # RSI (3)
        for period in [14, 21, 30]:
            delta = df[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands (6)
        for period in [20, 50]:
            sma = df[close_col].rolling(period).mean()
            std = df[close_col].rolling(period).std()
            features[f'bb_upper_{period}'] = sma + (2 * std)
            features[f'bb_lower_{period}'] = sma - (2 * std)
            features[f'bb_position_{period}'] = (df[close_col] - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'])
        
        # MACD variations (6)
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5)]:
            ema_fast = df[close_col].ewm(span=fast).mean()
            ema_slow = df[close_col].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            features[f'macd_{fast}_{slow}'] = macd
            features[f'macd_signal_{fast}_{slow}'] = macd_signal
            features[f'macd_histogram_{fast}_{slow}'] = macd - macd_signal
        
        # Stochastic oscillator (4)
        for period in [14, 21]:
            low_min = df[low_col].rolling(window=period).min()
            high_max = df[high_col].rolling(window=period).max()
            features[f'stoch_k_{period}'] = 100 * ((df[close_col] - low_min) / (high_max - low_min))
            features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(3).mean()
        
        # Williams %R (2)
        for period in [14, 21]:
            high_max = df[high_col].rolling(window=period).max()
            low_min = df[low_col].rolling(window=period).min()
            features[f'williams_r_{period}'] = -100 * ((high_max - df[close_col]) / (high_max - low_min))
        
        # Price channels (6)
        for period in [20, 50]:
            features[f'price_channel_high_{period}'] = df[high_col].rolling(period).max()
            features[f'price_channel_low_{period}'] = df[low_col].rolling(period).min()
            features[f'price_channel_position_{period}'] = (df[close_col] - features[f'price_channel_low_{period}']) / (features[f'price_channel_high_{period}'] - features[f'price_channel_low_{period}'])
        
        # Lagged features (8)
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'volume_lag_{lag}'] = features['volume_ratio'].shift(lag)
        
        return features
    
    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Get exactly 72 features as numpy array for model prediction"""
        features_df = self.create_features(df)
        
        # Clean data and get feature matrix
        features_df = features_df.dropna()
        
        # Extract only the 72 feature columns in exact order
        feature_matrix = features_df[self.feature_columns].values
        
        return feature_matrix
    
    def get_feature_count(self) -> int:
        """Return exact feature count (should be 72)"""
        return len(self.feature_columns)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        return self.feature_columns.copy()


def create_v6_features(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create V6 Enhanced features matching Amazon Q's training
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        Dict with features array and metadata
    """
    v6_features = V6EnhancedFeatures()
    
    # Create feature matrix
    feature_matrix = v6_features.get_feature_matrix(df)
    
    return {
        'features': feature_matrix,
        'feature_names': v6_features.get_feature_names(),
        'feature_count': v6_features.get_feature_count(),
        'version': 'v6_enhanced',
        'compatible_models': ['lstm_BTC-USD_v6_enhanced.pt', 'lstm_ETH-USD_v6_enhanced.pt', 'lstm_SOL-USD_v6_enhanced.pt']
    }
