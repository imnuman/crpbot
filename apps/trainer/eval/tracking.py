"""Experiment tracking and model versioning."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


class ExperimentTracker:
    """
    Simple experiment tracker using CSV and JSON.

    Tracks:
    - Model runs with hyperparameters
    - Metrics and results
    - Model versions and deployment status
    """

    def __init__(self, registry_path: Path | str = "models/registry.json"):
        """
        Initialize experiment tracker.

        Args:
            registry_path: Path to model registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()

    def _load_registry(self) -> dict[str, Any]:
        """Load model registry from JSON file."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}. Creating new registry.")
        return {"models": {}, "experiments": []}

    def _save_registry(self) -> None:
        """Save model registry to JSON file."""
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def register_model(
        self,
        model_path: str,
        model_type: str,
        symbol: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        version: str | None = None,
    ) -> str:
        """
        Register a model in the registry.

        Args:
            model_path: Path to model file
            symbol: Trading pair symbol (if applicable)
            model_type: Type of model (LSTM, Transformer, etc.)
            hyperparameters: Model hyperparameters
            metrics: Training/validation metrics
            version: Semantic version (e.g., 'v1.0.0'). If None, auto-generates

        Returns:
            Model version tag
        """
        model_file = Path(model_path)

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Generate version if not provided
        if version is None:
            # Generate from timestamp and hash
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
            file_hash = hashlib.md5(model_file.read_bytes()).hexdigest()[:8]
            version = f"v1.0.0-{timestamp}-{file_hash}"

        # Calculate file hash
        file_hash = hashlib.sha256(model_file.read_bytes()).hexdigest()

        # Register model
        model_id = (
            f"{model_type}_{symbol or 'multi'}_{version}"
            if symbol
            else f"{model_type}_multi_{version}"
        )

        model_entry = {
            "model_id": model_id,
            "version": version,
            "model_path": str(model_path),
            "file_hash": file_hash,
            "model_type": model_type,
            "symbol": symbol,
            "hyperparameters": hyperparameters or {},
            "metrics": metrics or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "deployed_at": None,
            "promoted": False,
        }

        if "models" not in self.registry:
            self.registry["models"] = {}

        self.registry["models"][model_id] = model_entry
        self._save_registry()

        logger.info(f"Registered model: {model_id} (version: {version})")
        return version

    def promote_model(self, model_id: str, promoted_dir: Path | str = "models/promoted") -> None:
        """
        Promote a model (create symlink in promoted directory).

        Args:
            model_id: Model ID to promote
            promoted_dir: Directory for promoted models
        """
        if model_id not in self.registry["models"]:
            raise ValueError(f"Model not found in registry: {model_id}")

        model_entry = self.registry["models"][model_id]
        model_path = Path(model_entry["model_path"])

        promoted_dir = Path(promoted_dir)
        promoted_dir.mkdir(parents=True, exist_ok=True)

        # Create symlink
        symlink_path = (
            promoted_dir
            / f"{model_entry['model_type']}_{model_entry.get('symbol', 'multi')}_latest.pt"
        )

        if symlink_path.exists():
            symlink_path.unlink()

        symlink_path.symlink_to(model_path.absolute())
        logger.info(f"Promoted model: {model_id} -> {symlink_path}")

        # Update registry
        model_entry["promoted"] = True
        model_entry["deployed_at"] = datetime.now(timezone.utc).isoformat()
        self._save_registry()

    def log_experiment(
        self,
        experiment_name: str,
        hyperparameters: dict[str, Any],
        metrics: dict[str, Any],
        model_id: str | None = None,
    ) -> None:
        """
        Log an experiment run.

        Args:
            experiment_name: Name of the experiment
            hyperparameters: Experiment hyperparameters
            metrics: Experiment metrics
            model_id: Associated model ID (if any)
        """
        experiment = {
            "name": experiment_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "model_id": model_id,
        }

        if "experiments" not in self.registry:
            self.registry["experiments"] = []

        self.registry["experiments"].append(experiment)
        self._save_registry()

        logger.info(f"Logged experiment: {experiment_name}")

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        """
        Get model entry from registry.

        Args:
            model_id: Model ID

        Returns:
            Model entry or None if not found
        """
        return self.registry.get("models", {}).get(model_id)

    def list_models(
        self, model_type: str | None = None, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        """
        List models in registry.

        Args:
            model_type: Filter by model type
            symbol: Filter by symbol

        Returns:
            List of model entries
        """
        models = list(self.registry.get("models", {}).values())

        if model_type:
            models = [m for m in models if m["model_type"] == model_type]
        if symbol:
            models = [m for m in models if m.get("symbol") == symbol]

        return sorted(models, key=lambda x: x["created_at"], reverse=True)

    def get_promoted_model(
        self, model_type: str, symbol: str | None = None
    ) -> dict[str, Any] | None:
        """
        Get the currently promoted model.

        Args:
            model_type: Type of model
            symbol: Trading pair symbol

        Returns:
            Model entry or None if not found
        """
        models = self.list_models(model_type=model_type, symbol=symbol)
        promoted = [m for m in models if m.get("promoted", False)]

        if promoted:
            return promoted[0]  # Return most recently promoted
        return None
