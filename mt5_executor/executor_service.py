"""
HYDRA MT5 Executor Service - Runs on Windows VPS (ForexVPS)

This service:
1. Receives trade signals from HYDRA Linux server via HTTP API
2. Validates signals against risk limits
3. Executes trades on MT5 (FTMO)
4. Reports execution results back

Run on Windows VPS:
    python executor_service.py

Environment Variables:
    FTMO_LOGIN: MT5 account login
    FTMO_PASS: MT5 account password
    FTMO_SERVER: MT5 server (e.g., FTMO-Server3)
    API_SECRET: Shared secret for authentication
    HYDRA_SERVER: Linux server IP for callbacks (optional)
"""

import os
import sys
import time
import json
import threading
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from flask import Flask, request, jsonify
import MetaTrader5 as mt5

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # MT5 Settings - With FTMO fallback credentials
    mt5_login: int = int(os.getenv("FTMO_LOGIN", "531025383"))
    mt5_password: str = os.getenv("FTMO_PASS", "c*B@lWp41b784c")
    mt5_server: str = os.getenv("FTMO_SERVER", "FTMO-Server3")

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 5000
    api_secret: str = os.getenv("API_SECRET", "hydra_secret_2024")

    # Risk Settings (FTMO Limits)
    max_daily_loss_pct: float = 4.5  # 4.5% daily max
    max_total_loss_pct: float = 9.0  # 9% total max
    max_position_size: float = 0.5   # Max lot size
    max_open_positions: int = 5

    # HYDRA Magic Number
    magic_number: int = 20241207

config = Config()

# ============================================================================
# MT5 Connection Manager
# ============================================================================

class MT5Manager:
    """Manages MT5 connection and trading operations."""

    def __init__(self, config: Config):
        self.config = config
        self.connected = False
        self.account_info = None
        self.initial_balance = 0
        self._lock = threading.Lock()

    def connect(self, timeout_seconds: int = 15) -> bool:
        """Connect to MT5 terminal with timeout."""
        import concurrent.futures

        with self._lock:
            if self.connected:
                return True

            print(f"[MT5] Connecting to {self.config.mt5_server}... (timeout: {timeout_seconds}s)")

            def do_connect():
                # Initialize with timeout
                if not mt5.initialize(timeout=timeout_seconds * 1000):  # ms
                    return False, f"Initialize failed: {mt5.last_error()}"

                # Login
                authorized = mt5.login(
                    login=self.config.mt5_login,
                    password=self.config.mt5_password,
                    server=self.config.mt5_server,
                    timeout=timeout_seconds * 1000  # ms
                )

                if not authorized:
                    mt5.shutdown()
                    return False, f"Login failed: {mt5.last_error()}"

                # Get account info
                account = mt5.account_info()
                if account is None:
                    mt5.shutdown()
                    return False, "Failed to get account info"

                return True, account

            # Run with timeout using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(do_connect)
                try:
                    success, result = future.result(timeout=timeout_seconds + 5)
                except concurrent.futures.TimeoutError:
                    print(f"[MT5] Connection timed out after {timeout_seconds}s")
                    try:
                        mt5.shutdown()
                    except:
                        pass
                    return False
                except Exception as e:
                    print(f"[MT5] Connection error: {e}")
                    return False

            if not success:
                print(f"[MT5] {result}")
                return False

            self.account_info = result
            self.initial_balance = self.account_info.balance
            self.connected = True

            print(f"[MT5] Connected successfully!")
            print(f"[MT5] Account: {self.account_info.login}")
            print(f"[MT5] Balance: ${self.account_info.balance:,.2f}")
            print(f"[MT5] Equity: ${self.account_info.equity:,.2f}")
            print(f"[MT5] Server: {self.account_info.server}")

            return True

    def disconnect(self):
        """Disconnect from MT5."""
        with self._lock:
            if self.connected:
                mt5.shutdown()
                self.connected = False
                print("[MT5] Disconnected")

    def ensure_connected(self) -> bool:
        """Ensure connection is active."""
        if not self.connected:
            return self.connect()

        # Verify connection
        info = mt5.account_info()
        if info is None:
            self.connected = False
            return self.connect()

        return True

    def get_account_status(self) -> Dict:
        """Get current account status."""
        if not self.ensure_connected():
            return {"error": "Not connected"}

        info = mt5.account_info()
        if info is None:
            return {"error": "Failed to get account info"}

        # Calculate drawdown
        daily_pnl = info.equity - self.initial_balance
        daily_pnl_pct = (daily_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0

        return {
            "login": info.login,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "profit": info.profit,
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": daily_pnl_pct,
            "leverage": info.leverage,
            "server": info.server,
            "connected": True
        }

    def check_risk_limits(self) -> tuple[bool, str]:
        """Check if trading is allowed based on risk limits."""
        status = self.get_account_status()

        if "error" in status:
            return False, status["error"]

        # Check daily loss limit
        if status["daily_pnl_pct"] < -self.config.max_daily_loss_pct:
            return False, f"Daily loss limit exceeded: {status['daily_pnl_pct']:.2f}%"

        # Check open positions
        positions = mt5.positions_total()
        if positions >= self.config.max_open_positions:
            return False, f"Max positions reached: {positions}"

        return True, "OK"

    def execute_trade(self, signal: Dict) -> Dict:
        """Execute a trade signal on MT5."""
        if not self.ensure_connected():
            return {"success": False, "error": "Not connected to MT5"}

        # Check risk limits
        allowed, reason = self.check_risk_limits()
        if not allowed:
            return {"success": False, "error": f"Risk limit: {reason}"}

        # Parse signal
        symbol = signal.get("symbol", "").replace("-", "")  # BTC-USD -> BTCUSD
        direction = signal.get("direction", "").upper()
        volume = min(float(signal.get("volume", 0.01)), self.config.max_position_size)
        sl_price = float(signal.get("stop_loss", 0))
        tp_price = float(signal.get("take_profit", 0))

        # Validate
        if not symbol or direction not in ["BUY", "SELL"]:
            return {"success": False, "error": f"Invalid signal: {symbol} {direction}"}

        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            # Try with suffix
            for suffix in ["", ".raw", ".pro", "m"]:
                test_symbol = f"{symbol}{suffix}"
                symbol_info = mt5.symbol_info(test_symbol)
                if symbol_info is not None:
                    symbol = test_symbol
                    break

        if symbol_info is None:
            return {"success": False, "error": f"Symbol not found: {symbol}"}

        # Ensure symbol is visible
        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"success": False, "error": f"No tick data for {symbol}"}

        # Determine order type and price
        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        # Prepare request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,  # Max slippage in points
            "magic": self.config.magic_number,
            "comment": f"HYDRA_{signal.get('engine', 'X')}_{signal.get('trade_id', '')}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        print(f"[MT5] Executing: {direction} {volume} {symbol} @ {price}")
        print(f"[MT5] SL: {sl_price}, TP: {tp_price}")

        # Send order
        result = mt5.order_send(request)

        if result is None:
            return {"success": False, "error": f"Order send failed: {mt5.last_error()}"}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "success": False,
                "error": f"Order rejected: {result.retcode} - {result.comment}",
                "retcode": result.retcode
            }

        # Success!
        execution_result = {
            "success": True,
            "ticket": result.order,
            "symbol": symbol,
            "direction": direction,
            "volume": result.volume,
            "price": result.price,
            "sl": sl_price,
            "tp": tp_price,
            "comment": request["comment"],
            "execution_time": datetime.now(timezone.utc).isoformat()
        }

        print(f"[MT5] SUCCESS! Ticket: {result.order}, Price: {result.price}")

        return execution_result

    def get_positions(self) -> list:
        """Get all open positions."""
        if not self.ensure_connected():
            return []

        positions = mt5.positions_get()
        if positions is None:
            return []

        return [
            {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.type == 0 else "SELL",
                "volume": p.volume,
                "price_open": p.price_open,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "magic": p.magic,
                "comment": p.comment
            }
            for p in positions
            if p.magic == self.config.magic_number  # Only HYDRA positions
        ]

    def close_position(self, ticket: int) -> Dict:
        """Close a specific position."""
        if not self.ensure_connected():
            return {"success": False, "error": "Not connected"}

        position = mt5.positions_get(ticket=ticket)
        if not position:
            return {"success": False, "error": f"Position {ticket} not found"}

        position = position[0]

        # Determine close direction
        if position.type == 0:  # BUY position
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(position.symbol).bid
        else:  # SELL position
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(position.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": self.config.magic_number,
            "comment": "HYDRA_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "error": f"Close failed: {result.retcode if result else 'None'}"}

        return {"success": True, "ticket": ticket, "close_price": result.price}

    def get_price(self, symbol: str) -> Dict:
        """Get current bid/ask price for a symbol."""
        if not self.ensure_connected():
            return {"success": False, "error": "Not connected"}

        # Try symbol as-is first, then with common suffixes
        symbol_info = mt5.symbol_info(symbol)
        actual_symbol = symbol

        if symbol_info is None:
            for suffix in ["", ".raw", ".pro", "m", "_"]:
                test_symbol = f"{symbol}{suffix}"
                symbol_info = mt5.symbol_info(test_symbol)
                if symbol_info is not None:
                    actual_symbol = test_symbol
                    break

        if symbol_info is None:
            return {"success": False, "error": f"Symbol not found: {symbol}"}

        # Ensure symbol is visible
        if not symbol_info.visible:
            mt5.symbol_select(actual_symbol, True)
            time.sleep(0.1)  # Brief pause for selection

        tick = mt5.symbol_info_tick(actual_symbol)
        if tick is None:
            return {"success": False, "error": f"No tick data for {actual_symbol}"}

        return {
            "success": True,
            "symbol": actual_symbol,
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "volume": tick.volume,
            "time": datetime.fromtimestamp(tick.time, tz=timezone.utc).isoformat(),
            "spread": round((tick.ask - tick.bid) / symbol_info.point, 1) if symbol_info.point > 0 else 0
        }

    def get_candles(self, symbol: str, timeframe: str = "M5", count: int = 100) -> Dict:
        """Get historical OHLC candles for a symbol."""
        if not self.ensure_connected():
            return {"success": False, "error": "Not connected"}

        # Map timeframe string to MT5 constant
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        mt5_tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_M5)

        # Try symbol as-is first, then with common suffixes
        symbol_info = mt5.symbol_info(symbol)
        actual_symbol = symbol

        if symbol_info is None:
            for suffix in ["", ".raw", ".pro", "m", "_"]:
                test_symbol = f"{symbol}{suffix}"
                symbol_info = mt5.symbol_info(test_symbol)
                if symbol_info is not None:
                    actual_symbol = test_symbol
                    break

        if symbol_info is None:
            return {"success": False, "error": f"Symbol not found: {symbol}"}

        # Ensure symbol is visible
        if not symbol_info.visible:
            mt5.symbol_select(actual_symbol, True)
            time.sleep(0.1)

        # Fetch candles
        rates = mt5.copy_rates_from_pos(actual_symbol, mt5_tf, 0, min(count, 1000))

        if rates is None or len(rates) == 0:
            return {"success": False, "error": f"No candle data for {actual_symbol}"}

        candles = []
        for rate in rates:
            candles.append({
                "time": datetime.fromtimestamp(rate['time'], tz=timezone.utc).isoformat(),
                "open": float(rate['open']),
                "high": float(rate['high']),
                "low": float(rate['low']),
                "close": float(rate['close']),
                "volume": int(rate['tick_volume']),
            })

        return {
            "success": True,
            "symbol": actual_symbol,
            "timeframe": timeframe,
            "count": len(candles),
            "candles": candles
        }


# ============================================================================
# Flask API Server
# ============================================================================

app = Flask(__name__)
mt5_manager: Optional[MT5Manager] = None


def verify_auth(req) -> bool:
    """Verify API authentication."""
    auth_header = req.headers.get("Authorization", "")
    expected = f"Bearer {config.api_secret}"
    return auth_header == expected


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "HYDRA MT5 Executor",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mt5_connected": mt5_manager.connected if mt5_manager else False
    })


@app.route("/account", methods=["GET"])
def account():
    """Get account status."""
    if not verify_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if mt5_manager is None:
        return jsonify({"error": "MT5 not initialized"}), 500

    return jsonify(mt5_manager.get_account_status())


@app.route("/positions", methods=["GET"])
def positions():
    """Get open positions."""
    if not verify_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if mt5_manager is None:
        return jsonify({"error": "MT5 not initialized"}), 500

    return jsonify({"positions": mt5_manager.get_positions()})


@app.route("/execute", methods=["POST"])
def execute():
    """Execute a trade signal."""
    if not verify_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if mt5_manager is None:
        return jsonify({"error": "MT5 not initialized"}), 500

    signal = request.json
    if not signal:
        return jsonify({"error": "No signal data"}), 400

    print(f"\n[API] Received signal: {json.dumps(signal, indent=2)}")

    result = mt5_manager.execute_trade(signal)

    return jsonify(result)


@app.route("/close/<int:ticket>", methods=["POST"])
def close(ticket: int):
    """Close a position."""
    if not verify_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if mt5_manager is None:
        return jsonify({"error": "MT5 not initialized"}), 500

    result = mt5_manager.close_position(ticket)
    return jsonify(result)


@app.route("/price/<symbol>", methods=["GET"])
def price(symbol: str):
    """Get current bid/ask price for a symbol."""
    if not verify_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if mt5_manager is None:
        return jsonify({"error": "MT5 not initialized"}), 500

    result = mt5_manager.get_price(symbol)
    return jsonify(result)


@app.route("/candles/<symbol>", methods=["GET"])
def candles(symbol: str):
    """Get historical OHLC candles for a symbol."""
    if not verify_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if mt5_manager is None:
        return jsonify({"error": "MT5 not initialized"}), 500

    # Get optional parameters
    timeframe = request.args.get("timeframe", "M5")
    count = int(request.args.get("count", 100))

    result = mt5_manager.get_candles(symbol, timeframe, count)
    return jsonify(result)


@app.route("/symbols", methods=["GET"])
def symbols():
    """List available symbols."""
    if not verify_auth(request):
        return jsonify({"error": "Unauthorized"}), 401

    if mt5_manager is None:
        return jsonify({"error": "MT5 not initialized"}), 500

    if not mt5_manager.ensure_connected():
        return jsonify({"error": "Not connected"}), 500

    # Get all visible symbols
    all_symbols = mt5.symbols_get()
    if all_symbols is None:
        return jsonify({"symbols": []})

    # Filter for common trading symbols
    result = []
    for s in all_symbols:
        if s.visible and s.trade_mode != 0:  # Only tradeable symbols
            result.append({
                "symbol": s.name,
                "description": s.description,
                "digits": s.digits,
                "point": s.point,
                "spread": s.spread,
                "trade_mode": s.trade_mode
            })

    return jsonify({"count": len(result), "symbols": result[:100]})  # Limit to 100


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    global mt5_manager

    print("=" * 60)
    print("    HYDRA MT5 Executor Service")
    print("    ForexVPS Edition - Ultra Low Latency")
    print("=" * 60)
    print()

    # Validate config
    if not config.mt5_login or not config.mt5_password:
        print("[ERROR] MT5 credentials not set!")
        print("Set environment variables: FTMO_LOGIN, FTMO_PASS, FTMO_SERVER")
        sys.exit(1)

    print(f"[CONFIG] MT5 Login: {config.mt5_login}")
    print(f"[CONFIG] MT5 Server: {config.mt5_server}")
    print(f"[CONFIG] API Port: {config.api_port}")
    print(f"[CONFIG] Max Daily Loss: {config.max_daily_loss_pct}%")
    print(f"[CONFIG] Max Total Loss: {config.max_total_loss_pct}%")
    print()

    # Initialize MT5 (don't exit on failure - start Flask anyway)
    mt5_manager = MT5Manager(config)
    if not mt5_manager.connect():
        print("[WARNING] Failed to connect to MT5!")
        print("Make sure MT5 terminal is running and logged in.")
        print("API will start but MT5 operations will fail until connected.")
        print()

    print()
    print(f"[API] Starting server on http://0.0.0.0:{config.api_port}")
    print(f"[API] Health check: http://localhost:{config.api_port}/health")
    print()
    print("Ready to receive signals from HYDRA!")
    print("=" * 60)

    # Run Flask
    app.run(
        host=config.api_host,
        port=config.api_port,
        threaded=True,
        debug=False
    )


if __name__ == "__main__":
    main()
