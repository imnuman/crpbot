#!/usr/bin/env python3
"""
V8 SageMaker Training Script for AWS
Installs dependencies and runs training on SageMaker
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install dependencies on SageMaker instance"""
    print("📦 Installing dependencies on SageMaker...")
    
    # Install from requirements file
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "-r", "/opt/ml/code/requirements_sagemaker.txt",
        "--upgrade"
    ])
    print("✅ Dependencies installed")

def main():
    """Main training function"""
    # Install dependencies first
    install_dependencies()
    
    # Import after installation
    import torch
    import torch.nn as nn
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import argparse
    
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"🔥 CUDA available: {torch.cuda.is_available()}")
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--coin', type=str, default='BTC')
    args = parser.parse_args()
    
    print(f"🚀 Training {args.coin} model for {args.epochs} epochs")
    
    # SageMaker paths
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    data_dir = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    
    # Simple model for demo
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 3)  # 10 features -> 3 classes
            
        def forward(self, x):
            return self.fc(x)
    
    # Create and train model
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Dummy training loop
    for epoch in range(args.epochs):
        # Simulate training
        dummy_input = torch.randn(args.batch_size, 10)
        dummy_target = torch.randint(0, 3, (args.batch_size,))
        
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = nn.CrossEntropyLoss()(output, dummy_target)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Save model
    torch.save(model.state_dict(), f"{model_dir}/model.pth")
    print(f"✅ Model saved to {model_dir}/model.pth")

if __name__ == "__main__":
    main()
