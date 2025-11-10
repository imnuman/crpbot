"""Model versioning utilities."""
import hashlib
from pathlib import Path
from typing import Any

from loguru import logger

from apps.trainer.eval.tracking import ExperimentTracker


def create_model_version(
    model_path: Path | str,
    semantic_version: str,
    registry_path: Path | str = "models/registry.json",
) -> Path:
    """
    Create a versioned copy of a model file.

    Args:
        model_path: Path to model file
        semantic_version: Semantic version (e.g., 'v1.0.0')
        registry_path: Path to model registry

    Returns:
        Path to versioned model file
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Create versioned filename
    model_dir = model_path.parent
    model_stem = model_path.stem
    versioned_filename = f"{model_stem}_{semantic_version}.pt"
    versioned_path = model_dir / versioned_filename

    # Copy file
    import shutil

    shutil.copy2(model_path, versioned_path)
    logger.info(f"Created versioned model: {versioned_path}")

    return versioned_path


def rollback_model(
    version: str,
    model_type: str,
    symbol: str | None = None,
    registry_path: Path | str = "models/registry.json",
    promoted_dir: Path | str = "models/promoted",
) -> None:
    """
    Rollback to a previous model version.

    Args:
        version: Model version to rollback to (e.g., 'v1.0.0')
        model_type: Type of model (LSTM, Transformer)
        symbol: Trading pair symbol (optional)
        registry_path: Path to model registry
        promoted_dir: Directory for promoted models
    """
    tracker = ExperimentTracker(registry_path)

    # Find model with specified version
    models = tracker.list_models(model_type=model_type, symbol=symbol)
    target_model = None

    for model in models:
        if model["version"].startswith(version) or model["version"] == version:
            target_model = model
            break

    if not target_model:
        raise ValueError(f"Model not found: {model_type} {symbol or ''} version {version}")

    # Verify model file exists
    model_path = Path(target_model["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Promote the old version
    tracker.promote_model(target_model["model_id"], promoted_dir=promoted_dir)

    logger.info(f"Rolled back to version {version}: {target_model['model_path']}")


def get_model_info(model_path: Path | str) -> dict[str, Any]:
    """
    Get information about a model file.

    Args:
        model_path: Path to model file

    Returns:
        Dictionary with model information
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load model checkpoint
    import torch

    checkpoint = torch.load(model_path, map_location="cpu")
    model_class = checkpoint.get("model_class", "Unknown")
    metadata = checkpoint.get("metadata", {})

    # Calculate file hash
    file_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    # Get file size
    file_size = model_path.stat().st_size

    return {
        "model_path": str(model_path),
        "model_class": model_class,
        "file_hash": file_hash,
        "file_size_bytes": file_size,
        "metadata": metadata,
        "modified_at": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(),
    }


# Import datetime here to avoid circular imports
from datetime import datetime
