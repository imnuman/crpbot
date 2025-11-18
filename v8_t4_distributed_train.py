#!/usr/bin/env python3
"""
4x T4 GPU Distributed Training Script
Installs all dependencies and runs distributed training
"""

import subprocess
import sys
import os
import json

def install_dependencies():
    """Install all dependencies for distributed training"""
    print("📦 Installing distributed training dependencies...")
    
    # Install requirements
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "-r", "/opt/ml/code/requirements_t4_distributed.txt",
        "--upgrade", "--no-cache-dir"
    ])
    
    # Install additional distributed packages
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch-distributed-gpu", "--upgrade"
    ])
    
    print("✅ All dependencies installed")

def setup_distributed_environment():
    """Setup distributed training environment"""
    import torch
    import torch.distributed as dist
    
    # Get distributed info from SageMaker
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    print(f"🌍 World size: {world_size}")
    print(f"🏆 Rank: {rank}")
    print(f"📍 Local rank: {local_rank}")
    
    if world_size > 1:
        # Initialize distributed training
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(local_rank)
        print(f"✅ Distributed training initialized")
    
    return world_size, rank, local_rank

def main():
    """Main distributed training function"""
    # Install dependencies first
    install_dependencies()
    
    # Import after installation
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)  # Total batch size
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--coin', type=str, default='BTC')
    args = parser.parse_args()
    
    # Setup distributed environment
    world_size, rank, local_rank = setup_distributed_environment()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    print(f"🔥 Device: {device}")
    print(f"🔥 Available GPUs: {torch.cuda.device_count()}")
    print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Enhanced trading model
    class AdvancedTradingModel(nn.Module):
        def __init__(self, input_size=72, hidden_size=512, num_classes=3):
            super().__init__()
            self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.3)
            self.lstm2 = nn.LSTM(hidden_size, hidden_size//2, batch_first=True, num_layers=2, dropout=0.3)
            self.attention = nn.MultiheadAttention(hidden_size//2, num_heads=8, batch_first=True)
            self.dropout = nn.Dropout(0.4)
            self.fc1 = nn.Linear(hidden_size//2, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            # LSTM layers
            lstm_out1, _ = self.lstm1(x)
            lstm_out2, _ = self.lstm2(lstm_out1)
            
            # Attention mechanism
            attn_out, _ = self.attention(lstm_out2, lstm_out2, lstm_out2)
            
            # Final layers
            out = self.dropout(attn_out[:, -1, :])  # Last timestep
            out = self.relu(self.fc1(out))
            out = self.dropout(out)
            return self.fc2(out)
    
    # Create model
    model = AdvancedTradingModel().to(device)
    
    # Wrap with DDP for distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print(f"✅ Model wrapped with DDP on rank {rank}")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Create synthetic dataset
    seq_len = 60
    input_size = 72
    dataset_size = 10000
    
    # Generate synthetic data
    X = torch.randn(dataset_size, seq_len, input_size)
    y = torch.randint(0, 3, (dataset_size,))
    dataset = TensorDataset(X, y)
    
    # Distributed sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    
    # DataLoader with distributed sampler
    batch_size_per_gpu = args.batch_size // world_size
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size_per_gpu, 
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True
    )
    
    print(f"🎯 Training setup:")
    print(f"   Total batch size: {args.batch_size}")
    print(f"   Batch size per GPU: {batch_size_per_gpu}")
    print(f"   Dataset size: {dataset_size}")
    print(f"   Steps per epoch: {len(dataloader)}")
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)  # Shuffle data differently each epoch
        
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Log progress (only on rank 0)
            if rank == 0 and batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        # Average loss across all GPUs
        if world_size > 1:
            avg_loss = torch.tensor(epoch_loss / len(dataloader)).to(device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss /= world_size
            epoch_loss = avg_loss.item()
        else:
            epoch_loss /= len(dataloader)
        
        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}, Avg Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save model (only on rank 0)
    if rank == 0:
        model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
        model_to_save = model.module if hasattr(model, 'module') else model
        
        # Save model state
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': args.epochs,
            'loss': epoch_loss,
            'config': {
                'input_size': 72,
                'hidden_size': 512,
                'num_classes': 3,
                'coin': args.coin
            }
        }, f"{model_dir}/model.pth")
        
        # Save training info
        training_info = {
            'final_loss': epoch_loss,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'world_size': world_size,
            'coin': args.coin,
            'model_type': 'AdvancedTradingModel'
        }
        
        with open(f"{model_dir}/training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"✅ Model saved to {model_dir}/model.pth")
        print(f"✅ Training info saved to {model_dir}/training_info.json")
    
    # Cleanup distributed training
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
