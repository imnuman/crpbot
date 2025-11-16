"""
V6 Enhanced Model Loader
Loads Amazon Q's V6 models with proper feature compatibility
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import logging

from .v6_enhanced_features import V6EnhancedFeatures

logger = logging.getLogger(__name__)


class V6TradingNet(nn.Module):
    """V6 Enhanced Neural Network Architecture (matches Amazon Q's training)"""
    
    def __init__(self, input_size: int, hidden_size: int = 256, num_classes: int = 3):
        super(V6TradingNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class V6ModelLoader:
    """Load and use V6 Enhanced models with proper feature engineering"""
    
    def __init__(self, model_dir: str = "models/v6_enhanced"):
        self.model_dir = Path(model_dir)
        self.feature_engine = V6EnhancedFeatures()
        self.models = {}
        self.metadata = {}
        
    def load_model(self, symbol: str) -> bool:
        """Load V6 model for specific symbol"""
        model_file = self.model_dir / f"lstm_{symbol}_v6_enhanced.pt"
        
        if not model_file.exists():
            logger.error(f"Model file not found: {model_file}")
            return False
            
        try:
            # Load model data
            model_data = torch.load(model_file, map_location='cpu')
            
            # Extract model info
            input_size = model_data['input_size']
            accuracy = model_data['accuracy']
            
            # Verify feature compatibility
            if input_size != 72:
                logger.error(f"Model expects {input_size} features, V6 provides 72")
                return False
                
            # Create model architecture
            model = V6TradingNet(input_size)
            model.load_state_dict(model_data['model_state_dict'])
            model.eval()
            
            # Store model and metadata
            self.models[symbol] = {
                'model': model,
                'accuracy': accuracy,
                'normalization': model_data['normalization_params'],
                'feature_columns': model_data['feature_columns'],
                'training_date': model_data['training_date']
            }
            
            logger.info(f"✅ Loaded {symbol} model: {accuracy:.1%} accuracy, {input_size} features")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {symbol}: {e}")
            return False
    
    def load_all_models(self) -> Dict[str, bool]:
        """Load all available V6 models"""
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        results = {}
        
        for symbol in symbols:
            results[symbol] = self.load_model(symbol)
            
        logger.info(f"Loaded {sum(results.values())}/{len(results)} V6 models")
        return results
    
    def predict(self, symbol: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Make prediction using V6 model"""
        if symbol not in self.models:
            logger.error(f"Model not loaded for {symbol}")
            return None
            
        try:
            # Get model and metadata
            model_info = self.models[symbol]
            model = model_info['model']
            norm_params = model_info['normalization']
            
            # Create V6 features
            feature_matrix = self.feature_engine.get_feature_matrix(df)
            
            if len(feature_matrix) == 0:
                logger.warning("No valid features generated")
                return None
                
            # Get latest features (last row)
            latest_features = feature_matrix[-1:].astype(np.float32)
            
            # Apply normalization
            mean = np.array(norm_params['mean'])
            std = np.array(norm_params['std'])
            normalized_features = (latest_features - mean) / std
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(normalized_features)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
            
            # Convert class to signal
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            signal = signal_map[predicted_class]
            
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'probabilities': {
                    'sell': probabilities[0][0].item(),
                    'hold': probabilities[0][1].item(),
                    'buy': probabilities[0][2].item()
                },
                'model_accuracy': model_info['accuracy'],
                'features_used': 72,
                'version': 'v6_enhanced'
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return None
    
    def get_model_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get model information"""
        if symbol not in self.models:
            return None
            
        model_info = self.models[symbol]
        return {
            'symbol': symbol,
            'accuracy': model_info['accuracy'],
            'training_date': model_info['training_date'],
            'features': 72,
            'version': 'v6_enhanced',
            'loaded': True
        }
    
    def get_all_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information for all loaded models"""
        return {symbol: self.get_model_info(symbol) for symbol in self.models.keys()}


def load_v6_models(model_dir: str = "models/v6_enhanced") -> V6ModelLoader:
    """
    Load V6 Enhanced models
    
    Args:
        model_dir: Directory containing V6 model files
        
    Returns:
        V6ModelLoader instance with loaded models
    """
    loader = V6ModelLoader(model_dir)
    loader.load_all_models()
    return loader


def test_v6_compatibility():
    """Test V6 feature compatibility"""
    print("=== V6 Enhanced Feature Compatibility Test ===")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=300, freq='1H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(40000, 50000, 300),
        'high': np.random.uniform(50000, 55000, 300),
        'low': np.random.uniform(35000, 45000, 300),
        'close': np.random.uniform(40000, 50000, 300),
        'volume': np.random.uniform(1000, 10000, 300)
    })
    
    # Test feature generation
    v6_features = V6EnhancedFeatures()
    feature_matrix = v6_features.get_feature_matrix(sample_data)
    
    print(f"✅ Generated {feature_matrix.shape[1]} features (expected: 72)")
    print(f"✅ Data points: {feature_matrix.shape[0]}")
    print(f"✅ Feature names: {len(v6_features.get_feature_names())}")
    
    # Test model loading (if available)
    try:
        loader = V6ModelLoader()
        results = loader.load_all_models()
        print(f"✅ Model loading test: {sum(results.values())}/{len(results)} models loaded")
    except Exception as e:
        print(f"⚠️  Model loading test: {e}")
    
    return feature_matrix.shape[1] == 72


if __name__ == "__main__":
    test_v6_compatibility()
