"""Training utilities and loops."""
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from loguru import logger


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float):
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class ModelTrainer:
    """Base trainer for model training."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        save_dir: Path | str = "models",
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            device: Device to train on (default: auto-detect)
            save_dir: Directory to save models
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training on device: {self.device}")

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: Data loader for training data
            optimizer: Optimizer
            criterion: Loss function

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            features = batch["features"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """
        Validate model.

        Args:
            dataloader: Data loader for validation data
            criterion: Loss function

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating", leave=False):
                features = batch["features"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(features)

                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Calculate accuracy (for binary classification)
                predictions = (outputs >= 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def save_model(
        self,
        model_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save model with versioning.

        Args:
            model_name: Base name for model file
            metadata: Additional metadata to save

        Returns:
            Path to saved model file
        """
        # Create model hash from architecture params
        model_state = self.model.state_dict()
        model_str = json.dumps(
            {
                "state_dict_keys": list(model_state.keys()),
                "num_params": sum(p.numel() for p in self.model.parameters()),
            },
            sort_keys=True,
        )
        model_hash = hashlib.md5(model_str.encode()).hexdigest()[:8]

        # Create filename with hash
        filename = f"{model_name}_{model_hash}.pt"
        filepath = self.save_dir / filename

        # Save model
        torch.save(
            {
                "model_state_dict": model_state,
                "model_class": self.model.__class__.__name__,
                "metadata": metadata or {},
            },
            filepath,
        )

        logger.info(f"Saved model to {filepath}")
        return filepath

    def load_model(self, filepath: Path | str) -> None:
        """
        Load model from file.

        Args:
            filepath: Path to model file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model from {filepath}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: torch.device | None = None,
    save_dir: Path | str = "models",
    model_name: str = "model",
) -> dict[str, Any]:
    """
    Train a model with validation, weighted loss, LR scheduler, and early stopping.

    Args:
        model: PyTorch model to train
        train_loader: Data loader for training data
        val_loader: Data loader for validation data
        num_epochs: Number of training epochs (improved: 50)
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save models
        model_name: Base name for saved model

    Returns:
        Dictionary with training history and best metrics
    """
    trainer = ModelTrainer(model, device=device, save_dir=save_dir)

    # Calculate class weights for imbalanced data
    logger.info("Calculating class weights from training data...")
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch["label"].cpu().numpy().flatten())
    all_labels = np.array(all_labels)

    pos_count = np.sum(all_labels)
    neg_count = len(all_labels) - pos_count
    pos_weight = neg_count / (pos_count + 1e-7)  # Avoid division by zero

    logger.info(f"Class distribution: pos={pos_count}, neg={neg_count}")
    logger.info(f"Positive class weight: {pos_weight:.3f}")

    # Setup optimizer and weighted loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(trainer.device))

    # Learning rate scheduler with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double period after each restart
        eta_min=1e-6
    )

    # Early stopping with increased patience
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    best_val_accuracy = 0.0
    best_model_path = None

    logger.info(f"Starting training for {num_epochs} epochs with improved configuration...")
    logger.info(f"  - LR scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")
    logger.info(f"  - Early stopping: patience=7")
    logger.info(f"  - Weighted loss: pos_weight={pos_weight:.3f}")

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs} (LR: {current_lr:.6f})")

        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer, criterion)
        history["train_loss"].append(train_loss)

        # Validate
        val_loss, val_accuracy = trainer.validate(val_loader, criterion)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rate"].append(current_lr)

        logger.info(
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        # Learning rate scheduler step
        scheduler.step()

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            metadata = {
                "epoch": epoch + 1,
                "val_accuracy": val_accuracy,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            }
            best_model_path = trainer.save_model(model_name, metadata=metadata)
            logger.info(f"✅ New best model saved! Val Accuracy: {val_accuracy:.4f}")

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.should_stop:
            logger.info(f"🛑 Early stopping triggered at epoch {epoch + 1}")
            logger.info(f"No improvement for {early_stopping.patience} epochs")
            break

    logger.info(f"\n✅ Training complete! Best validation accuracy: {best_val_accuracy:.4f}")
    logger.info(f"Best model saved to: {best_model_path}")

    return {
        "history": history,
        "best_val_accuracy": best_val_accuracy,
        "best_model_path": str(best_model_path) if best_model_path else None,
        "epochs_trained": epoch + 1,
        "early_stopped": early_stopping.should_stop,
    }

