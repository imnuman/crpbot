#!/usr/bin/env python3
"""Quick validation of GPU-trained models."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

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

def validate_model(model_path: str, model_name: str):
    """Validate a single GPU model."""
    print(f"\n=== Validating {model_name} ===")
    
    # Load model
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Create model with correct architecture
        model = SimpleGPULSTM(input_size=5, hidden_size=64, num_layers=2, num_classes=3)
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"   Input size: 5 features")
        print(f"   Hidden size: 64")
        print(f"   Output classes: 3 (sell/hold/buy)")
        
        # Test inference
        batch_size = 32
        sequence_length = 60
        test_input = torch.randn(batch_size, sequence_length, 5)
        
        with torch.no_grad():
            output = model(test_input)
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        print(f"✅ Inference test passed")
        print(f"   Output shape: {output.shape}")
        print(f"   Sample predictions: {predictions[:5].tolist()}")
        print(f"   Sample probabilities: {probabilities[0].tolist()}")
        
        # Basic performance metrics
        unique_preds, counts = torch.unique(predictions, return_counts=True)
        pred_distribution = {int(pred): int(count) for pred, count in zip(unique_preds, counts)}
        print(f"   Prediction distribution: {pred_distribution}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def main():
    """Validate all GPU models."""
    models_dir = Path("models/gpu_trained")
    
    models = {
        "BTC": models_dir / "BTC_lstm_model.pt",
        "ETH": models_dir / "ETH_lstm_model.pt", 
        "SOL": models_dir / "SOL_lstm_model.pt",
        "ADA": models_dir / "ADA_lstm_model.pt"
    }
    
    print("🔍 GPU Model Validation Report")
    print("=" * 50)
    
    results = {}
    for name, path in models.items():
        if path.exists():
            results[name] = validate_model(str(path), name)
        else:
            print(f"\n❌ {name}: Model file not found at {path}")
            results[name] = False
    
    print(f"\n📊 Summary:")
    print(f"   Total models: {len(models)}")
    print(f"   Validated: {sum(results.values())}")
    print(f"   Failed: {len(models) - sum(results.values())}")
    
    if all(results.values()):
        print("\n🎉 All GPU models validated successfully!")
        print("   Ready for runtime testing")
    else:
        print(f"\n⚠️  Some models failed validation")
        
    return all(results.values())

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
