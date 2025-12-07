"""
HYDRA State Checkpoint System

Provides atomic state checkpointing and recovery for HYDRA runtime.
Ensures no data loss on crashes or restarts.

Features:
- Atomic writes (temp file + rename)
- Automatic backup before overwrite
- Recovery from backup if main file corrupted
- Thread-safe operations
- Periodic auto-checkpoint
"""

import json
import os
import tempfile
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from loguru import logger


@dataclass
class RuntimeCheckpoint:
    """
    Complete HYDRA runtime state for checkpoint/resume.

    Captures all critical state needed to resume trading
    without losing position or portfolio data.
    """
    # Version for forward compatibility
    version: str = "1.0"

    # Timestamp of checkpoint
    checkpoint_time: str = ""

    # Engine portfolios (critical for independent trading mode)
    engine_portfolios: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "A": {"balance": 25000.0, "trades": 0, "wins": 0, "pnl": 0.0},
        "B": {"balance": 25000.0, "trades": 0, "wins": 0, "pnl": 0.0},
        "C": {"balance": 25000.0, "trades": 0, "wins": 0, "pnl": 0.0},
        "D": {"balance": 25000.0, "trades": 0, "wins": 0, "pnl": 0.0},
    })

    # Open positions (critical for risk management)
    open_positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Runtime counters
    iteration: int = 0
    total_signals_generated: int = 0
    total_trades_executed: int = 0

    # Session info
    session_start: str = ""
    last_signal_time: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuntimeCheckpoint':
        """Create from dictionary."""
        # Handle version migration if needed
        version = data.get("version", "1.0")

        return cls(
            version=version,
            checkpoint_time=data.get("checkpoint_time", ""),
            engine_portfolios=data.get("engine_portfolios", {
                "A": {"balance": 25000.0, "trades": 0, "wins": 0, "pnl": 0.0},
                "B": {"balance": 25000.0, "trades": 0, "wins": 0, "pnl": 0.0},
                "C": {"balance": 25000.0, "trades": 0, "wins": 0, "pnl": 0.0},
                "D": {"balance": 25000.0, "trades": 0, "wins": 0, "pnl": 0.0},
            }),
            open_positions=data.get("open_positions", {}),
            iteration=data.get("iteration", 0),
            total_signals_generated=data.get("total_signals_generated", 0),
            total_trades_executed=data.get("total_trades_executed", 0),
            session_start=data.get("session_start", ""),
            last_signal_time=data.get("last_signal_time", ""),
        )


class StateCheckpointManager:
    """
    Manages atomic state checkpointing for HYDRA runtime.

    Uses write-to-temp-then-rename pattern for crash safety.
    Maintains backup for corruption recovery.
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        if checkpoint_path is None:
            from .config import RUNTIME_CHECKPOINT_FILE
            checkpoint_path = RUNTIME_CHECKPOINT_FILE

        self.checkpoint_path = Path(checkpoint_path)
        self.backup_path = Path(str(checkpoint_path) + ".backup")
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._lock = threading.Lock()

        # Auto-checkpoint settings
        self._auto_checkpoint_interval = 60  # seconds
        self._auto_checkpoint_thread: Optional[threading.Thread] = None
        self._auto_checkpoint_running = False
        self._runtime_ref: Any = None  # Weak reference to runtime

        logger.info(f"StateCheckpointManager initialized: {self.checkpoint_path}")

    def save_checkpoint(self, checkpoint: RuntimeCheckpoint) -> bool:
        """
        Save checkpoint atomically.

        Uses temp file + rename pattern to ensure no corruption
        even if process crashes mid-write.

        Args:
            checkpoint: RuntimeCheckpoint to save

        Returns:
            True if saved successfully, False otherwise
        """
        with self._lock:
            try:
                # Update checkpoint time
                checkpoint.checkpoint_time = datetime.now(timezone.utc).isoformat()

                # Get directory for temp file (must be same filesystem for atomic rename)
                checkpoint_dir = self.checkpoint_path.parent

                # Create temp file
                fd, temp_path = tempfile.mkstemp(
                    dir=checkpoint_dir,
                    prefix=".checkpoint_",
                    suffix=".tmp"
                )

                try:
                    # Write checkpoint to temp file
                    with os.fdopen(fd, 'w') as f:
                        json.dump(checkpoint.to_dict(), f, indent=2, default=str)
                        f.flush()
                        os.fsync(f.fileno())

                    # Backup current checkpoint before replacing
                    if self.checkpoint_path.exists():
                        shutil.copy2(self.checkpoint_path, self.backup_path)

                    # Atomic rename (POSIX guarantees)
                    shutil.move(temp_path, self.checkpoint_path)

                    logger.debug(
                        f"Checkpoint saved: iteration={checkpoint.iteration}, "
                        f"positions={len(checkpoint.open_positions)}"
                    )
                    return True

                except Exception as e:
                    # Clean up temp file on error
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise

            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
                return False

    def load_checkpoint(self) -> Optional[RuntimeCheckpoint]:
        """
        Load checkpoint with backup recovery.

        Attempts to load from main file first, falls back to
        backup if main is corrupted.

        Returns:
            RuntimeCheckpoint if found, None otherwise
        """
        with self._lock:
            checkpoint = None
            loaded_from_backup = False

            # Try main file first
            if self.checkpoint_path.exists():
                try:
                    checkpoint = self._load_from_file(self.checkpoint_path)
                    if checkpoint:
                        logger.success(
                            f"Checkpoint loaded: iteration={checkpoint.iteration}, "
                            f"portfolios={list(checkpoint.engine_portfolios.keys())}"
                        )
                        return checkpoint
                except Exception as e:
                    logger.error(f"Failed to load main checkpoint: {e}")

            # Try backup if main failed
            if self.backup_path.exists():
                try:
                    logger.warning("Attempting recovery from backup checkpoint...")
                    checkpoint = self._load_from_file(self.backup_path)
                    if checkpoint:
                        loaded_from_backup = True
                        logger.success("Recovered from backup checkpoint!")
                except Exception as e:
                    logger.error(f"Backup recovery failed: {e}")

            # Restore main file from backup if needed
            if loaded_from_backup and checkpoint:
                try:
                    shutil.copy2(self.backup_path, self.checkpoint_path)
                    logger.info("Restored main checkpoint from backup")
                except Exception as e:
                    logger.error(f"Failed to restore main checkpoint: {e}")

            return checkpoint

    def _load_from_file(self, file_path: Path) -> Optional[RuntimeCheckpoint]:
        """Load checkpoint from specific file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
            return RuntimeCheckpoint.from_dict(data)

    def start_auto_checkpoint(self, runtime: Any, interval_seconds: int = 60):
        """
        Start automatic periodic checkpointing.

        Args:
            runtime: HydraRuntime instance to checkpoint
            interval_seconds: How often to checkpoint
        """
        self._runtime_ref = runtime
        self._auto_checkpoint_interval = interval_seconds
        self._auto_checkpoint_running = True

        self._auto_checkpoint_thread = threading.Thread(
            target=self._auto_checkpoint_loop,
            daemon=True,
            name="StateCheckpointAutoSave"
        )
        self._auto_checkpoint_thread.start()
        logger.info(f"Auto-checkpoint started: every {interval_seconds}s")

    def stop_auto_checkpoint(self):
        """Stop automatic checkpointing."""
        self._auto_checkpoint_running = False
        if self._auto_checkpoint_thread:
            self._auto_checkpoint_thread.join(timeout=5)
        logger.info("Auto-checkpoint stopped")

    def _auto_checkpoint_loop(self):
        """Background thread for auto-checkpointing."""
        import time

        while self._auto_checkpoint_running:
            try:
                time.sleep(self._auto_checkpoint_interval)

                if not self._auto_checkpoint_running:
                    break

                if self._runtime_ref:
                    checkpoint = self.create_checkpoint_from_runtime(self._runtime_ref)
                    self.save_checkpoint(checkpoint)

            except Exception as e:
                logger.error(f"Auto-checkpoint error: {e}")

    def create_checkpoint_from_runtime(self, runtime: Any) -> RuntimeCheckpoint:
        """
        Create checkpoint from HydraRuntime instance.

        Args:
            runtime: HydraRuntime instance

        Returns:
            RuntimeCheckpoint with current state
        """
        checkpoint = RuntimeCheckpoint(
            checkpoint_time=datetime.now(timezone.utc).isoformat(),
            engine_portfolios=getattr(runtime, 'engine_portfolios', {}),
            open_positions=getattr(runtime, 'open_positions', {}),
            iteration=getattr(runtime, 'iteration', 0),
            total_signals_generated=getattr(runtime, 'total_signals_generated', 0),
            total_trades_executed=getattr(runtime, 'total_trades_executed', 0),
            session_start=getattr(runtime, 'session_start', ''),
            last_signal_time=getattr(runtime, 'last_signal_time', ''),
        )
        return checkpoint

    def apply_checkpoint_to_runtime(self, runtime: Any, checkpoint: RuntimeCheckpoint):
        """
        Apply checkpoint state to runtime instance.

        Args:
            runtime: HydraRuntime instance to update
            checkpoint: Checkpoint to apply
        """
        if hasattr(runtime, 'engine_portfolios') and checkpoint.engine_portfolios:
            runtime.engine_portfolios = checkpoint.engine_portfolios
            logger.info(f"Restored engine portfolios from checkpoint")

            # Log portfolio states
            for engine, portfolio in runtime.engine_portfolios.items():
                logger.info(
                    f"  Engine {engine}: ${portfolio['balance']:.2f} "
                    f"({portfolio['trades']} trades, {portfolio['wins']} wins)"
                )

        if hasattr(runtime, 'open_positions') and checkpoint.open_positions:
            runtime.open_positions = checkpoint.open_positions
            logger.info(f"Restored {len(checkpoint.open_positions)} open positions")

        if hasattr(runtime, 'iteration'):
            runtime.iteration = checkpoint.iteration

        if hasattr(runtime, 'total_signals_generated'):
            runtime.total_signals_generated = checkpoint.total_signals_generated

        if hasattr(runtime, 'total_trades_executed'):
            runtime.total_trades_executed = checkpoint.total_trades_executed

        logger.success(
            f"Checkpoint applied: iteration={checkpoint.iteration}, "
            f"signals={checkpoint.total_signals_generated}, "
            f"trades={checkpoint.total_trades_executed}"
        )


# Global singleton
_checkpoint_manager: Optional[StateCheckpointManager] = None


def get_checkpoint_manager() -> StateCheckpointManager:
    """Get or create global checkpoint manager singleton."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = StateCheckpointManager()
    return _checkpoint_manager
