#!/usr/bin/env python3
"""Test GPU models in runtime context."""

import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime

class SimpleGPULSTM(nn.Module):
    """Simple LSTM matching GPU-trained architecture."""
    
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use last timestep
        return out

def generate_mock_features(sequence_length=60):
    """Generate mock OHLCV features."""
    # Mock 5 features: open, high, low, close, volume (normalized)
    features = np.random.randn(sequence_length, 5)
    return torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension

def test_signal_generation():
    """Test signal generation with GPU models."""
    print("🚀 Testing GPU Runtime Signal Generation")
    print("=" * 50)
    
    models = {
        "BTC-USD": "models/gpu_trained/BTC_lstm_model.pt",
        "ETH-USD": "models/gpu_trained/ETH_lstm_model.pt", 
        "SOL-USD": "models/gpu_trained/SOL_lstm_model.pt",
        "ADA-USD": "models/gpu_trained/ADA_lstm_model.pt"
    }
    
    signals = []
    
    for pair, model_path in models.items():
        print(f"\n📊 Generating signal for {pair}")
        
        try:
            # Load model
            state_dict = torch.load(model_path, map_location='cpu')
            model = SimpleGPULSTM()
            model.load_state_dict(state_dict)
            model.eval()
            
            # Generate mock features
            features = generate_mock_features()
            
            # Get prediction
            with torch.no_grad():
                output = model(features)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities, dim=1).values.item()
            
            # Map prediction to signal
            signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            direction = signal_map[prediction]
            
            # Create signal
            signal = {
                "pair": pair,
                "direction": direction,
                "confidence": round(confidence, 3),
                "probabilities": {
                    "sell": round(probabilities[0][0].item(), 3),
                    "hold": round(probabilities[0][1].item(), 3), 
                    "buy": round(probabilities[0][2].item(), 3)
                },
                "timestamp": datetime.now().isoformat(),
                "model": "GPU_LSTM",
                "status": "SUCCESS"
            }
            
            signals.append(signal)
            
            print(f"   Direction: {direction}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Probabilities: SELL={probabilities[0][0]:.1%}, HOLD={probabilities[0][1]:.1%}, BUY={probabilities[0][2]:.1%}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            signals.append({
                "pair": pair,
                "status": "ERROR",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    # Summary
    print(f"\n📈 Signal Generation Summary:")
    successful = sum(1 for s in signals if s.get('status') == 'SUCCESS')
    print(f"   Successful: {successful}/{len(models)}")
    
    # Show signal distribution
    directions = [s.get('direction') for s in signals if s.get('direction')]
    if directions:
        from collections import Counter
        dist = Counter(directions)
        print(f"   Signal distribution: {dict(dist)}")
    
    # Save results
    with open('gpu_runtime_test_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test_type': 'gpu_runtime_signal_generation',
            'signals': signals,
            'summary': {
                'total_models': len(models),
                'successful': successful,
                'failed': len(models) - successful
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to gpu_runtime_test_results.json")
    
    if successful == len(models):
        print("\n🎉 ALL MODELS GENERATING SIGNALS SUCCESSFULLY!")
        print("   Ready for production deployment")
        return True
    else:
        print(f"\n⚠️  {len(models) - successful} models failed")
        return False

if __name__ == "__main__":
    success = test_signal_generation()
    exit(0 if success else 1)
