"""
MT5 Remote Executor - Sends signals to Windows VPS for execution.

This module allows the HYDRA Linux server to send trade signals
to the MT5 Executor running on a Windows VPS (ForexVPS).

Usage:
    # Set MT5_API_SECRET env var first
    executor = MT5RemoteExecutor(
        windows_vps_url="http://YOUR_FOREXVPS_IP:5000"
    )

    result = executor.execute_signal({
        "symbol": "BTC-USD",
        "direction": "BUY",
        "volume": 0.01,
        "stop_loss": 99000,
        "take_profit": 102000,
        "engine": "A",
        "trade_id": "BTC-USD_A_123456"
    })

Environment Variables:
    MT5_EXECUTOR_URL: URL of Windows VPS executor (e.g., http://10.0.0.2:5000)
    MT5_API_SECRET: Shared authentication secret
"""

import os
import time
import requests
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class RemoteExecutorConfig:
    """Configuration for remote MT5 executor."""
    executor_url: str = os.getenv("MT5_EXECUTOR_URL", "http://localhost:5000")
    api_secret: str = os.getenv("MT5_API_SECRET", "")  # REQUIRED: Set MT5_API_SECRET env var
    timeout: int = 30  # Request timeout in seconds
    retry_attempts: int = 3
    retry_delay: float = 1.0

    def __post_init__(self):
        if not self.api_secret:
            raise ValueError("MT5_API_SECRET environment variable is required for remote execution")


class MT5RemoteExecutor:
    """
    Remote MT5 Executor Client.

    Sends trade signals to the Windows VPS running the MT5 Executor Service.
    """

    def __init__(
        self,
        windows_vps_url: Optional[str] = None,
        api_secret: Optional[str] = None,
        config: Optional[RemoteExecutorConfig] = None
    ):
        """
        Initialize remote executor.

        Args:
            windows_vps_url: URL of Windows VPS (e.g., http://10.0.0.2:5000)
            api_secret: Shared authentication secret
            config: Optional RemoteExecutorConfig
        """
        if config:
            self.config = config
        else:
            secret = api_secret or os.getenv("MT5_API_SECRET", "")
            if not secret:
                raise ValueError("MT5_API_SECRET environment variable or api_secret parameter is required")
            self.config = RemoteExecutorConfig(
                executor_url=windows_vps_url or os.getenv("MT5_EXECUTOR_URL", "http://localhost:5000"),
                api_secret=secret
            )

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.config.api_secret}",
            "Content-Type": "application/json"
        })

        logger.info(f"[MT5Remote] Initialized - Target: {self.config.executor_url}")

    def _request(self, method: str, endpoint: str, json_data: Optional[Dict] = None) -> Dict:
        """Make HTTP request to executor."""
        url = f"{self.config.executor_url}{endpoint}"

        for attempt in range(self.config.retry_attempts):
            try:
                if method == "GET":
                    response = self._session.get(url, timeout=self.config.timeout)
                elif method == "POST":
                    response = self._session.post(url, json=json_data, timeout=self.config.timeout)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                logger.warning(f"[MT5Remote] Timeout (attempt {attempt + 1}/{self.config.retry_attempts})")
                time.sleep(self.config.retry_delay)

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"[MT5Remote] Connection error: {e}")
                time.sleep(self.config.retry_delay)

            except requests.exceptions.HTTPError as e:
                logger.error(f"[MT5Remote] HTTP error: {e}")
                return {"success": False, "error": str(e)}

            except Exception as e:
                logger.error(f"[MT5Remote] Request error: {e}")
                return {"success": False, "error": str(e)}

        return {"success": False, "error": "Max retries exceeded"}

    def health_check(self) -> Dict:
        """Check if executor is healthy and reachable."""
        return self._request("GET", "/health")

    def get_account_status(self) -> Dict:
        """Get MT5 account status from Windows VPS."""
        return self._request("GET", "/account")

    def get_positions(self) -> Dict:
        """Get open positions from MT5."""
        return self._request("GET", "/positions")

    def execute_signal(self, signal: Dict) -> Dict:
        """
        Execute a trade signal on MT5 via Windows VPS.

        Args:
            signal: Trade signal with keys:
                - symbol: Trading symbol (e.g., "BTC-USD")
                - direction: "BUY" or "SELL"
                - volume: Lot size (e.g., 0.01)
                - stop_loss: SL price
                - take_profit: TP price
                - engine: Engine identifier (A/B/C/D)
                - trade_id: Unique trade ID
                - confidence: Confidence level (optional)

        Returns:
            Execution result dict
        """
        logger.info(
            f"[MT5Remote] Sending signal: {signal.get('direction')} "
            f"{signal.get('symbol')} x{signal.get('volume', 0.01)}"
        )

        start_time = time.time()
        result = self._request("POST", "/execute", signal)
        execution_time = (time.time() - start_time) * 1000  # ms

        if result.get("success"):
            logger.success(
                f"[MT5Remote] Executed: Ticket {result.get('ticket')} "
                f"@ {result.get('price')} ({execution_time:.0f}ms)"
            )
        else:
            logger.error(f"[MT5Remote] Failed: {result.get('error')}")

        result["execution_time_ms"] = execution_time
        return result

    def close_position(self, ticket: int) -> Dict:
        """Close a position by ticket number."""
        logger.info(f"[MT5Remote] Closing position: {ticket}")
        return self._request("POST", f"/close/{ticket}")

    def is_connected(self) -> bool:
        """Check if executor is connected and MT5 is working."""
        try:
            health = self.health_check()
            return health.get("mt5_connected", False)
        except Exception:
            return False


# Singleton instance
_remote_executor: Optional[MT5RemoteExecutor] = None


def get_remote_executor() -> MT5RemoteExecutor:
    """Get or create singleton remote executor instance."""
    global _remote_executor

    if _remote_executor is None:
        _remote_executor = MT5RemoteExecutor()

    return _remote_executor


# ============================================================================
# Integration with LiveExecutor
# ============================================================================

class HybridExecutor:
    """
    Hybrid executor that routes trades to remote MT5 executor.

    Integrates with HYDRA's existing Guardian and risk management.
    """

    def __init__(self, remote_executor: Optional[MT5RemoteExecutor] = None):
        self.remote = remote_executor or get_remote_executor()

        # Import HYDRA components
        from libs.hydra.guardian import get_guardian
        from libs.hydra.duplicate_order_guard import get_duplicate_guard
        from libs.notifications.alert_manager import get_alert_manager, AlertSeverity

        self.guardian = get_guardian()
        self.duplicate_guard = get_duplicate_guard()
        self.alert_manager = get_alert_manager()
        self.AlertSeverity = AlertSeverity

    def execute_trade(
        self,
        symbol: str,
        direction: str,
        volume: float,
        stop_loss: float,
        take_profit: float,
        engine: str,
        trade_id: str,
        confidence: float = 0.5
    ) -> Dict:
        """
        Execute trade with full HYDRA safety checks.

        Args:
            symbol: Trading symbol
            direction: BUY or SELL
            volume: Lot size
            stop_loss: SL price
            take_profit: TP price
            engine: Engine identifier
            trade_id: Unique trade ID
            confidence: Confidence level

        Returns:
            Execution result
        """
        # 1. Check Guardian limits
        guardian_ok, guardian_msg = self.guardian.can_trade()
        if not guardian_ok:
            logger.warning(f"[HybridExecutor] Guardian blocked: {guardian_msg}")
            self.alert_manager.send_alert(
                title="Trade Blocked by Guardian",
                message=f"{direction} {symbol} blocked: {guardian_msg}",
                severity=self.AlertSeverity.WARNING
            )
            return {"success": False, "error": f"Guardian: {guardian_msg}"}

        # 2. Check duplicate order guard
        dup_ok, dup_msg = self.duplicate_guard.can_place_order(
            symbol=symbol,
            direction=direction,
            engine=engine
        )
        if not dup_ok:
            logger.warning(f"[HybridExecutor] Duplicate blocked: {dup_msg}")
            return {"success": False, "error": f"Duplicate: {dup_msg}"}

        # 3. Execute on remote MT5
        signal = {
            "symbol": symbol,
            "direction": direction,
            "volume": volume,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "engine": engine,
            "trade_id": trade_id,
            "confidence": confidence
        }

        result = self.remote.execute_signal(signal)

        # 4. Record execution
        if result.get("success"):
            self.duplicate_guard.record_order(
                symbol=symbol,
                direction=direction,
                engine=engine,
                order_id=trade_id
            )

            # Send success alert
            self.alert_manager.send_alert(
                title=f"Trade Executed: {direction} {symbol}",
                message=(
                    f"Engine {engine} | Ticket: {result.get('ticket')}\n"
                    f"Price: {result.get('price')} | Volume: {volume}\n"
                    f"SL: {stop_loss} | TP: {take_profit}"
                ),
                severity=self.AlertSeverity.INFO
            )
        else:
            # Send failure alert
            self.alert_manager.send_alert(
                title=f"Trade Failed: {direction} {symbol}",
                message=f"Engine {engine} | Error: {result.get('error')}",
                severity=self.AlertSeverity.WARNING
            )

        return result


# Export
__all__ = [
    "MT5RemoteExecutor",
    "RemoteExecutorConfig",
    "HybridExecutor",
    "get_remote_executor"
]
