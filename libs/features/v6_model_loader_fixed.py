"""
V6 Enhanced Model Loader with V6 Fixed Support
Loads both original V6 models and V6 Fixed models with temperature scaling
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import logging
import pickle

from .v6_enhanced_features import V6EnhancedFeatures

logger = logging.getLogger(__name__)


class V6TradingNet(nn.Module):
    """V6 Enhanced Neural Network Architecture (original)"""

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


class V6EnhancedFNN(nn.Module):
    """V6 Enhanced FNN (base architecture for V6 Fixed)"""

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
    """Wrapper that adds normalization and temperature scaling to V6 models"""

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


class V6ModelLoader:
    """Load and use V6 Enhanced models with proper feature engineering"""

    def __init__(self, model_dir: str = "models/v6_enhanced"):
        self.model_dir = Path(model_dir)
        self.feature_engine = V6EnhancedFeatures()
        self.models = {}
        self.metadata = {}

    def load_model(self, symbol: str) -> bool:
        """Load V6 model for specific symbol (supports both original and fixed versions)"""

        # Try V6 Fixed first
        fixed_model_file = self.model_dir / f"lstm_{symbol}_v6_FIXED.pt"

        if fixed_model_file.exists():
            return self._load_v6_fixed_model(symbol, fixed_model_file)

        # Fall back to original V6 model
        original_model_file = self.model_dir / f"lstm_{symbol}_v6_enhanced.pt"

        if original_model_file.exists():
            return self._load_original_v6_model(symbol, original_model_file)

        logger.error(f"No V6 model found for {symbol} in {self.model_dir}")
        return False

    def _load_v6_fixed_model(self, symbol: str, model_file: Path) -> bool:
        """Load V6 Fixed model with temperature scaling"""
        try:
            logger.info(f"Loading V6 Fixed model: {model_file.name}")

            # Load model checkpoint
            checkpoint = torch.load(model_file, map_location='cpu')

            # Verify it's a v6_fixed model
            if checkpoint.get('version') != 'v6_fixed':
                logger.warning(f"Model version mismatch: {checkpoint.get('version')}")

            # Load scaler
            scaler_file = model_file.parent / f"scaler_{symbol}_v6_fixed.pkl"
            if not scaler_file.exists():
                logger.error(f"Scaler not found: {scaler_file}")
                return False

            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)

            # Recreate base model
            input_size = checkpoint.get('input_size', 72)
            base_model = V6EnhancedFNN(input_size=input_size)
            base_model.load_state_dict(checkpoint['base_model_state_dict'])
            base_model.eval()

            # Recreate wrapper
            model = V6FixedWrapper(
                base_model=base_model,
                scaler=scaler,
                temperature=checkpoint['temperature'],
                logit_clip=checkpoint['logit_clip']
            )
            model.eval()

            # Store model and metadata
            self.models[symbol] = {
                'model': model,
                'scaler': scaler,
                'temperature': checkpoint['temperature'],
                'logit_clip': checkpoint['logit_clip'],
                'version': 'v6_fixed',
                'input_size': input_size,
                'model_type': 'fixed_wrapper'
            }

            logger.info(f"✅ Loaded {symbol} V6 Fixed: T={checkpoint['temperature']:.1f}, clip=±{checkpoint['logit_clip']}")
            return True

        except Exception as e:
            logger.error(f"Error loading V6 Fixed model {symbol}: {e}")
            return False

    def _load_original_v6_model(self, symbol: str, model_file: Path) -> bool:
        """Load original V6 model"""
        try:
            logger.info(f"Loading original V6 model: {model_file.name}")

            # Load model data
            model_data = torch.load(model_file, map_location='cpu')

            # Extract model info
            input_size = model_data['input_size']
            accuracy = model_data.get('accuracy', 0.0)

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
                'training_date': model_data.get('training_date', 'unknown'),
                'version': 'v6_original',
                'model_type': 'original'
            }

            logger.info(f"✅ Loaded {symbol} V6 original: {accuracy:.1%} accuracy, {input_size} features")
            return True

        except Exception as e:
            logger.error(f"Error loading original V6 model {symbol}: {e}")
            return False

    def load_all_models(self) -> Dict[str, bool]:
        """Load all available V6 models"""
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        results = {}

        for symbol in symbols:
            results[symbol] = self.load_model(symbol)

        loaded_count = sum(results.values())
        logger.info(f"Loaded {loaded_count}/{len(results)} V6 models")

        # Log which versions were loaded
        for symbol in symbols:
            if symbol in self.models:
                version = self.models[symbol]['version']
                logger.info(f"  {symbol}: {version}")

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
            model_type = model_info['model_type']

            # Create V6 features
            feature_matrix = self.feature_engine.get_feature_matrix(df)

            if len(feature_matrix) == 0:
                logger.warning("No valid features generated")
                return None

            # Get latest features (last row)
            latest_features = feature_matrix[-1:].astype(np.float32)

            # Apply normalization based on model type
            if model_type == 'fixed_wrapper':
                # V6 Fixed uses StandardScaler (already applied in model forward pass)
                # Just convert to tensor
                input_tensor = torch.FloatTensor(latest_features)

                # Apply scaler normalization
                scaler = model_info['scaler']
                normalized_features = scaler.transform(latest_features)
                input_tensor = torch.FloatTensor(normalized_features)

            else:
                # Original V6 uses mean/std normalization
                norm_params = model_info['normalization']
                mean = np.array(norm_params['mean'])
                std = np.array(norm_params['std'])
                normalized_features = (latest_features - mean) / std
                input_tensor = torch.FloatTensor(normalized_features)

            # Make prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()

            # Convert class to signal
            signal_map = {0: 'DOWN', 1: 'NEUTRAL', 2: 'UP'}
            signal = signal_map[predicted_class]

            result = {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'probabilities': {
                    'down': probabilities[0][0].item(),
                    'neutral': probabilities[0][1].item(),
                    'up': probabilities[0][2].item()
                },
                'features_used': 72,
                'version': model_info['version']
            }

            # Add version-specific metadata
            if model_type == 'fixed_wrapper':
                result['temperature'] = model_info['temperature']
                result['logit_clip'] = model_info['logit_clip']
            else:
                result['model_accuracy'] = model_info.get('accuracy', 0.0)

            return result

        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def get_model_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get model information"""
        if symbol not in self.models:
            return None

        model_info = self.models[symbol]
        info = {
            'symbol': symbol,
            'version': model_info['version'],
            'features': model_info.get('input_size', 72),
            'loaded': True
        }

        if model_info['model_type'] == 'fixed_wrapper':
            info['temperature'] = model_info['temperature']
            info['logit_clip'] = model_info['logit_clip']
        else:
            info['accuracy'] = model_info.get('accuracy', 0.0)
            info['training_date'] = model_info.get('training_date', 'unknown')

        return info

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

        # Show which versions were loaded
        for symbol, loaded in results.items():
            if loaded:
                info = loader.get_model_info(symbol)
                print(f"   {symbol}: {info['version']}")
    except Exception as e:
        print(f"⚠️  Model loading test: {e}")

    return feature_matrix.shape[1] == 72


if __name__ == "__main__":
    test_v6_compatibility()
