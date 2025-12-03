#!/usr/bin/env python3
"""
Multi-GPU SageMaker Training Script
Supports distributed training across multiple GPUs
"""

import subprocess
import sys
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import argparse

def install_dependencies():
    """Install dependencies on SageMaker instance"""
    print("📦 Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "-r", "/opt/ml/code/requirements_sagemaker.txt",
        "--upgrade"
    ])

def setup_distributed():
    """Setup distributed training"""
    if 'WORLD_SIZE' in os.environ:
        # Distributed training
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return True, local_rank
    else:
        # Single GPU or CPU
        return False, 0

class TradingModel(nn.Module):
    """Enhanced trading model for multi-GPU training"""
    def __init__(self, input_size=72, hidden_size=256, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])  # Last timestep
        return self.fc(out)

def main():
    install_dependencies()
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--coin', type=str, default='BTC')
    args = parser.parse_args()
    
    # Setup distributed training
    is_distributed, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    print(f"🔥 Device: {device}")
    print(f"🔥 Distributed: {is_distributed}")
    print(f"🔥 Available GPUs: {torch.cuda.device_count()}")
    
    # Create model
    model = TradingModel().to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
        print(f"🔥 Using DDP on rank {local_rank}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Simulate training data
    batch_size = args.batch_size
    seq_len = 60  # 60 timesteps
    input_size = 72  # 72 features
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        
        # Simulate batch
        dummy_input = torch.randn(batch_size, seq_len, input_size).to(device)
        dummy_target = torch.randint(0, 3, (batch_size,)).to(device)
        
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 and (not is_distributed or local_rank == 0):
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Save model (only on rank 0 for distributed)
    if not is_distributed or local_rank == 0:
        model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
        model_to_save = model.module if is_distributed else model
        torch.save(model_to_save.state_dict(), f"{model_dir}/model.pth")
        print(f"✅ Model saved to {model_dir}/model.pth")

if __name__ == "__main__":
    main()
