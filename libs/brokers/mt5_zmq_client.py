"""
HYDRA MT5 ZeroMQ Client
Runs on Linux - Connects to Windows MT5 Executor via SSH tunnel

Usage:
    from libs.brokers.mt5_zmq_client import MT5ZMQClient
    
    client = MT5ZMQClient()
    client.connect()
    
    # Get account info
    account = client.get_account()
    
    # Get price
    price = client.get_price("XAUUSD")
    
    # Execute trade
    result = client.trade("XAUUSD", "BUY", 0.01, sl=2650.0, tp=2700.0)
    
    # Close position
    client.close(ticket=12345)
"""

import os
import json
import time
import subprocess
import threading
from typing import Dict, Optional, Any
from dataclasses import dataclass
from loguru import logger

try:
    import zmq
except ImportError:
    zmq = None
    logger.warning("pyzmq not installed. Run: pip install pyzmq")


@dataclass
class MT5ZMQConfig:
    """ZMQ client configuration."""
    # Windows VPS - use WireGuard IP (10.10.0.2) by default for stable connection
    # Falls back to public IP if ZMQ_HOST env var is set
    windows_host: str = os.getenv("ZMQ_HOST", os.getenv("WINDOWS_VPS_IP", "10.10.0.2"))
    windows_user: str = os.getenv("WINDOWS_VPS_USER", "trader")
    windows_pass: str = os.getenv("WINDOWS_VPS_PASS", "80B#^yOr2b5s")

    # ZMQ Settings
    zmq_port: int = int(os.getenv("ZMQ_PORT", "5555"))
    local_port: int = int(os.getenv("ZMQ_LOCAL_PORT", "15555"))

    # Timeouts
    connect_timeout: int = 10000  # ms
    request_timeout: int = 30000  # ms

    # Auto SSH tunnel - disabled by default now that WireGuard is used
    auto_tunnel: bool = os.getenv("ZMQ_AUTO_TUNNEL", "false").lower() == "true"

    # Direct connection - enabled by default for WireGuard VPN
    # Set ZMQ_DIRECT=false to use SSH tunnel instead
    direct_connect: bool = os.getenv("ZMQ_DIRECT", "true").lower() == "true"


class SSHTunnel:
    """Manages SSH tunnel to Windows VPS."""
    
    def __init__(self, config: MT5ZMQConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
    
    def start(self) -> bool:
        """Start SSH tunnel."""
        with self._lock:
            if self.is_running():
                return True
            
            # Kill any existing tunnel on this port
            subprocess.run(
                f"pkill -f 'ssh.*-L {self.config.local_port}:'",
                shell=True,
                capture_output=True
            )
            time.sleep(0.5)
            
            # Start new tunnel
            cmd = [
                "sshpass", "-p", self.config.windows_pass,
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ServerAliveInterval=30",
                "-o", "ServerAliveCountMax=3",
                "-N",  # No remote command
                "-L", f"{self.config.local_port}:127.0.0.1:{self.config.zmq_port}",
                f"{self.config.windows_user}@{self.config.windows_host}"
            ]
            
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                time.sleep(2)  # Wait for tunnel to establish
                
                if self.process.poll() is None:
                    logger.info(f"[SSH] Tunnel started: localhost:{self.config.local_port} -> Windows:{self.config.zmq_port}")
                    return True
                else:
                    stderr = self.process.stderr.read().decode() if self.process.stderr else ""
                    logger.error(f"[SSH] Tunnel failed: {stderr}")
                    return False
                    
            except Exception as e:
                logger.error(f"[SSH] Failed to start tunnel: {e}")
                return False
    
    def stop(self):
        """Stop SSH tunnel."""
        with self._lock:
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("[SSH] Tunnel didn't terminate gracefully, killing...")
                    self.process.kill()
                    self.process.wait()
                finally:
                    self.process = None
                logger.info("[SSH] Tunnel stopped")
    
    def is_running(self) -> bool:
        """Check if tunnel is running."""
        return self.process is not None and self.process.poll() is None
    
    def ensure_running(self) -> bool:
        """Ensure tunnel is running, restart if needed."""
        if not self.is_running():
            return self.start()
        return True


class MT5ZMQClient:
    """ZeroMQ client for MT5 executor."""
    
    def __init__(self, config: Optional[MT5ZMQConfig] = None):
        if zmq is None:
            raise ImportError("pyzmq not installed. Run: pip install pyzmq")
        
        self.config = config or MT5ZMQConfig()
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None
        self.tunnel: Optional[SSHTunnel] = None
        self._connected = False
        self._lock = threading.Lock()
        
        if self.config.auto_tunnel:
            self.tunnel = SSHTunnel(self.config)
    
    def connect(self) -> bool:
        """Connect to MT5 executor."""
        with self._lock:
            try:
                # Start SSH tunnel if configured (skip if using direct connection)
                if not self.config.direct_connect and self.tunnel and not self.tunnel.ensure_running():
                    logger.error("[ZMQ] Failed to establish SSH tunnel")
                    return False
                
                # Create ZMQ context and socket
                if self.context is None:
                    self.context = zmq.Context()
                
                if self.socket:
                    self.socket.close()
                
                self.socket = self.context.socket(zmq.REQ)
                self.socket.setsockopt(zmq.RCVTIMEO, self.config.request_timeout)
                self.socket.setsockopt(zmq.SNDTIMEO, self.config.request_timeout)
                self.socket.setsockopt(zmq.LINGER, 0)
                
                # Connect to endpoint (direct or via tunnel)
                if self.config.direct_connect:
                    endpoint = f"tcp://{self.config.windows_host}:{self.config.zmq_port}"
                else:
                    endpoint = f"tcp://127.0.0.1:{self.config.local_port}"
                self.socket.connect(endpoint)

                logger.info(f"[ZMQ] Connected to {endpoint} (direct={self.config.direct_connect})")
                self._connected = True
                return True
                
            except Exception as e:
                logger.error(f"[ZMQ] Connection failed: {e}")
                self._connected = False
                return False
    
    def disconnect(self):
        """Disconnect from MT5 executor."""
        with self._lock:
            if self.socket:
                try:
                    self.socket.setsockopt(zmq.LINGER, 1000)  # Wait 1s for pending msgs
                    self.socket.close()
                except Exception:
                    pass
                finally:
                    self.socket = None

            if self.context:
                try:
                    self.context.term()
                except Exception:
                    pass
                finally:
                    self.context = None

            if self.tunnel:
                self.tunnel.stop()

            self._connected = False
            logger.info("[ZMQ] Disconnected")

    def _recreate_socket(self):
        """Recreate socket after error (called with lock held)."""
        try:
            if self.socket:
                self.socket.setsockopt(zmq.LINGER, 0)
                self.socket.close()
                self.socket = None

            if self.context:
                self.socket = self.context.socket(zmq.REQ)
                self.socket.setsockopt(zmq.RCVTIMEO, self.config.request_timeout)
                self.socket.setsockopt(zmq.SNDTIMEO, self.config.request_timeout)
                self.socket.setsockopt(zmq.LINGER, 0)
                if self.config.direct_connect:
                    endpoint = f"tcp://{self.config.windows_host}:{self.config.zmq_port}"
                else:
                    endpoint = f"tcp://127.0.0.1:{self.config.local_port}"
                self.socket.connect(endpoint)
                self._connected = True
                logger.debug(f"[ZMQ] Socket recreated after error (direct={self.config.direct_connect})")
        except Exception as e:
            logger.error(f"[ZMQ] Failed to recreate socket: {e}")
            self._connected = False

    def _send_command(self, cmd: Dict) -> Dict:
        """Send command and get response (thread-safe)."""
        with self._lock:
            if not self._connected:
                # Release lock for connect() which also acquires it
                pass

        # Try to connect if not connected (connect() has its own lock)
        if not self._connected:
            if not self.connect():
                return {"success": False, "error": "Not connected"}

        with self._lock:
            try:
                if not self.socket:
                    return {"success": False, "error": "Socket not available"}
                self.socket.send_json(cmd)
                response = self.socket.recv_json()
                return response
            except zmq.Again:
                logger.error("[ZMQ] Request timeout - recreating socket")
                self._recreate_socket()  # Socket is in bad state after timeout
                return {"success": False, "error": "Request timeout"}
            except zmq.ZMQError as e:
                logger.error(f"[ZMQ] Error: {e} - recreating socket")
                self._recreate_socket()  # Clean up bad socket
                return {"success": False, "error": str(e)}
    
    def ping(self) -> Dict:
        """Check connection and MT5 status."""
        return self._send_command({"cmd": "PING"})
    
    def get_account(self) -> Optional[Dict]:
        """Get account information."""
        result = self._send_command({"cmd": "ACCOUNT"})
        if result.get("success"):
            return result
        logger.error(f"[ZMQ] get_account failed: {result.get('error')}")
        return None
    
    def get_price(self, symbol: str) -> Optional[Dict]:
        """Get current price for symbol."""
        result = self._send_command({"cmd": "PRICE", "symbol": symbol})
        if result.get("success"):
            return result
        logger.error(f"[ZMQ] get_price failed: {result.get('error')}")
        return None
    
    def trade(
        self,
        symbol: str,
        direction: str,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "HYDRA"
    ) -> Dict:
        """Execute a trade."""
        cmd = {
            "cmd": "TRADE",
            "symbol": symbol,
            "direction": direction.upper(),
            "volume": volume,
            "comment": comment
        }
        if sl:
            cmd["sl"] = sl
        if tp:
            cmd["tp"] = tp
        
        result = self._send_command(cmd)
        if result.get("success"):
            logger.info(f"[ZMQ] Trade executed: {direction} {volume} {symbol}")
        else:
            logger.error(f"[ZMQ] Trade failed: {result.get('error')}")
        return result
    
    def close(self, ticket: int) -> Dict:
        """Close a position by ticket."""
        result = self._send_command({"cmd": "CLOSE", "ticket": ticket})
        if result.get("success"):
            logger.info(f"[ZMQ] Position {ticket} closed")
        else:
            logger.error(f"[ZMQ] Close failed: {result.get('error')}")
        return result
    
    def get_positions(self) -> list:
        """Get all open positions."""
        result = self._send_command({"cmd": "POSITIONS"})
        if result.get("success"):
            return result.get("positions", [])
        logger.error(f"[ZMQ] get_positions failed: {result.get('error')}")
        return []

    def get_candles(self, symbol: str, timeframe: str = "M1", count: int = 100) -> list:
        """Get historical candle data via ZMQ."""
        result = self._send_command({
            "cmd": "CANDLES",
            "symbol": symbol,
            "timeframe": timeframe,
            "count": count
        })
        if result.get("success"):
            return result.get("candles", [])
        logger.error(f"[ZMQ] get_candles failed: {result.get('error')}")
        return []

    def reconnect_mt5(self) -> bool:
        """Force MT5 reconnection on Windows side."""
        result = self._send_command({"cmd": "RECONNECT"})
        return result.get("success", False)

    def get_history(self, days: int = 7, ticket: Optional[int] = None) -> list:
        """
        Get closed trade history from MT5.

        Args:
            days: Number of days to look back (default 7)
            ticket: Optional specific ticket to find

        Returns:
            List of closed trades with entry/exit details
        """
        cmd = {"cmd": "HISTORY", "days": days}
        if ticket:
            cmd["ticket"] = ticket

        result = self._send_command(cmd)
        if result.get("success"):
            return result.get("trades", [])
        logger.error(f"[ZMQ] get_history failed: {result.get('error')}")
        return []

    def get_deal_info(self, ticket: int) -> Optional[Dict]:
        """
        Get details of a specific closed trade by ticket.

        Args:
            ticket: MT5 position ticket number

        Returns:
            Trade details dict or None if not found
        """
        result = self._send_command({"cmd": "DEAL_INFO", "ticket": ticket})
        if result.get("success"):
            return result
        logger.error(f"[ZMQ] get_deal_info failed: {result.get('error')}")
        return None

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


# Singleton instance
_client: Optional[MT5ZMQClient] = None
_client_lock = threading.Lock()


def get_mt5_client() -> MT5ZMQClient:
    """Get or create MT5 ZMQ client singleton (thread-safe)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = MT5ZMQClient()
    return _client


# CLI test
if __name__ == "__main__":
    import sys
    
    print("MT5 ZMQ Client Test")
    print("=" * 40)
    
    client = MT5ZMQClient()
    
    if not client.connect():
        print("Failed to connect!")
        sys.exit(1)
    
    # Test ping
    print("\n[PING]")
    result = client.ping()
    print(f"  Result: {result}")
    
    # Test account
    print("\n[ACCOUNT]")
    account = client.get_account()
    if account:
        print(f"  Balance: ${account.get('balance', 0):.2f}")
        print(f"  Equity: ${account.get('equity', 0):.2f}")
    
    # Test price
    print("\n[PRICE XAUUSD]")
    price = client.get_price("XAUUSD")
    if price:
        print(f"  Bid: {price.get('bid')}")
        print(f"  Ask: {price.get('ask')}")
    
    # Test positions
    print("\n[POSITIONS]")
    positions = client.get_positions()
    print(f"  Open positions: {len(positions)}")
    
    client.disconnect()
    print("\nDone!")
