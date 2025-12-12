"""
HYDRA MT5 ZeroMQ Executor Server
Runs on Windows VPS - More robust than Flask

Architecture:
- ZMQ REP (Reply) socket
- Receives JSON commands
- Executes on MT5
- Returns JSON responses

Commands:
- PING: Health check
- ACCOUNT: Get account info
- PRICE: Get symbol price
- TRADE: Execute trade
- CLOSE: Close position
- POSITIONS: List open positions
- MODIFY: Modify SL/TP
- CANDLES: Get historical candles (symbol, timeframe, count)
"""

import os
import sys
import json
import time
import signal
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import zmq
import MetaTrader5 as mt5


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # MT5 Settings
    mt5_login: int = int(os.getenv("FTMO_LOGIN", "531025383"))
    mt5_password: str = os.getenv("FTMO_PASS", "c*B@lWp41b784c")
    mt5_server: str = os.getenv("FTMO_SERVER", "FTMO-Server3")
    
    # ZMQ Settings
    zmq_port: int = int(os.getenv("ZMQ_PORT", "5555"))
    zmq_bind: str = "tcp://0.0.0.0"
    
    # Risk Settings (FTMO Limits)
    max_daily_loss_pct: float = 4.5
    max_total_loss_pct: float = 9.0
    max_position_size: float = 10.0  # Max lots per trade


config = Config()


# ============================================================================
# MT5 Manager
# ============================================================================

class MT5Manager:
    """Manages MT5 connection and operations."""
    
    def __init__(self, cfg: Config):
        self.config = cfg
        self.connected = False
        self._lock = threading.Lock()
    
    def connect(self, timeout: int = 30) -> bool:
        """Connect to MT5 with timeout."""
        with self._lock:
            try:
                # Use ThreadPoolExecutor for timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self._do_connect)
                    try:
                        result = future.result(timeout=timeout)
                        self.connected = result
                        return result
                    except FuturesTimeoutError:
                        print(f"[MT5] Connection timed out after {timeout}s")
                        self.connected = False
                        return False
            except Exception as e:
                print(f"[MT5] Connection error: {e}")
                self.connected = False
                return False
    
    def _do_connect(self) -> bool:
        """Internal connect logic - simplified like price_streamer."""
        # Initialize MT5 (don't shutdown first - let MT5 handle reconnection)
        if not mt5.initialize():
            error = mt5.last_error()
            print(f"[MT5] Initialize failed: {error}")
            return False

        # Login only if credentials are provided
        if self.config.mt5_login and self.config.mt5_password:
            if not mt5.login(
                login=self.config.mt5_login,
                password=self.config.mt5_password,
                server=self.config.mt5_server
            ):
                error = mt5.last_error()
                print(f"[MT5] Login failed: {error}")
                # Don't shutdown - terminal might still be usable
                return False

        account = mt5.account_info()
        if account:
            print(f"[MT5] Connected: {account.login} @ {account.server}")
            print(f"[MT5] Balance: ${account.balance:.2f}")
            return True

        return False
    
    def ensure_connected(self) -> bool:
        """Ensure MT5 is connected, reconnect if needed."""
        if self.connected:
            # Quick check
            info = mt5.terminal_info()
            if info and info.connected:
                return True
        
        print("[MT5] Reconnecting...")
        return self.connect()
    
    def get_account(self) -> Optional[Dict]:
        """Get account information."""
        if not self.ensure_connected():
            return None
        
        account = mt5.account_info()
        if not account:
            return None
        
        return {
            "login": account.login,
            "server": account.server,
            "balance": account.balance,
            "equity": account.equity,
            "margin": account.margin,
            "free_margin": account.margin_free,
            "profit": account.profit,
            "leverage": account.leverage,
            "currency": account.currency
        }
    
    def get_price(self, symbol: str) -> Optional[Dict]:
        """Get current price for symbol."""
        if not self.ensure_connected():
            return None

        # Try to select symbol first (in case it's not in Market Watch)
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return None

        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                return None

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return None

        return {
            "symbol": symbol,
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": round((tick.ask - tick.bid) / symbol_info.point, 1) if symbol_info.point else 0,
            "time": datetime.fromtimestamp(tick.time, tz=timezone.utc).isoformat()
        }

    def get_symbols(self, pattern: str = "*") -> list:
        """Get available symbols matching pattern."""
        if not self.ensure_connected():
            return []

        symbols = mt5.symbols_get(group=pattern)
        if not symbols:
            return []

        result = []
        for s in symbols[:100]:  # Limit to 100
            result.append({
                "name": s.name,
                "description": s.description,
                "visible": s.visible
            })
        return result

    def get_candles(self, symbol: str, timeframe: str = "H1", count: int = 100) -> Optional[List[Dict]]:
        """Get historical candles for a symbol."""
        if not self.ensure_connected():
            return None

        # Map timeframe string to MT5 constant
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1,
        }
        tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_H1)

        # Ensure symbol is visible
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return None
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                return None

        # Get candles
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            return None

        candles = []
        for r in rates:
            candles.append({
                "time": int(r["time"]),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": int(r["tick_volume"]),
            })
        return candles

    def execute_trade(
        self,
        symbol: str,
        direction: str,  # BUY or SELL
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "HYDRA"
    ) -> Dict:
        """Execute a trade."""
        if not self.ensure_connected():
            return {"success": False, "error": "Not connected to MT5"}
        
        # Validate
        if volume > self.config.max_position_size:
            return {"success": False, "error": f"Volume {volume} exceeds max {self.config.max_position_size}"}
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return {"success": False, "error": f"Symbol {symbol} not found"}
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                return {"success": False, "error": f"Failed to select {symbol}"}
        
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return {"success": False, "error": "Failed to get price"}
        
        # Determine order type and price
        if direction.upper() == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        elif direction.upper() == "SELL":
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            return {"success": False, "error": f"Invalid direction: {direction}"}
        
        # Build request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 202412,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        if sl:
            request["sl"] = float(sl)
        if tp:
            request["tp"] = float(tp)
        
        # Send order
        result = mt5.order_send(request)
        
        if result is None:
            error = mt5.last_error()
            return {"success": False, "error": f"Order send failed: {error}"}
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "success": False,
                "error": f"Order rejected: {result.comment}",
                "retcode": result.retcode
            }
        
        return {
            "success": True,
            "ticket": result.order,
            "volume": result.volume,
            "price": result.price,
            "symbol": symbol,
            "direction": direction
        }
    
    def close_position(self, ticket: int) -> Dict:
        """Close a position by ticket."""
        if not self.ensure_connected():
            return {"success": False, "error": "Not connected to MT5"}
        
        # Get position
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return {"success": False, "error": f"Position {ticket} not found"}
        
        position = position[0]
        
        # Determine close direction
        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(position.symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(position.symbol).ask
        
        # Build close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 202412,
            "comment": "HYDRA_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = result.comment if result else mt5.last_error()
            return {"success": False, "error": f"Close failed: {error}"}
        
        return {
            "success": True,
            "ticket": ticket,
            "closed_volume": position.volume,
            "profit": position.profit
        }
    
    def get_positions(self) -> list:
        """Get all open positions."""
        if not self.ensure_connected():
            return []

        positions = mt5.positions_get()
        if not positions:
            return []

        result = []
        for p in positions:
            result.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                "volume": p.volume,
                "price_open": p.price_open,
                "price_current": p.price_current,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "comment": p.comment
            })

        return result

    def get_history(self, days: int = 7, ticket: Optional[int] = None) -> List[Dict]:
        """
        Get trade history (closed positions) from MT5.

        Args:
            days: Number of days to look back
            ticket: Optional specific ticket to find

        Returns:
            List of closed trade records with entry/exit details
        """
        if not self.ensure_connected():
            return []

        from_date = datetime.now() - timedelta(days=days)
        to_date = datetime.now() + timedelta(days=1)

        # Get deals from history
        deals = mt5.history_deals_get(from_date, to_date)
        if not deals:
            return []

        # Group deals by position ticket (entry + exit deals)
        positions = {}
        for d in deals:
            # Skip balance operations, deposits, etc.
            if d.type > 1:  # 0=BUY, 1=SELL, 2+=other
                continue

            pos_id = d.position_id
            if pos_id not in positions:
                positions[pos_id] = {
                    "ticket": pos_id,
                    "symbol": d.symbol,
                    "deals": []
                }
            positions[pos_id]["deals"].append({
                "deal_id": d.ticket,
                "order": d.order,
                "time": datetime.fromtimestamp(d.time, tz=timezone.utc).isoformat(),
                "type": "BUY" if d.type == 0 else "SELL",
                "entry": d.entry,  # 0=IN, 1=OUT, 2=INOUT, 3=OUT_BY
                "volume": d.volume,
                "price": d.price,
                "profit": d.profit,
                "commission": d.commission,
                "swap": d.swap,
                "comment": d.comment
            })

        # Build complete trade records
        result = []
        for pos_id, pos in positions.items():
            # If looking for specific ticket
            if ticket and pos_id != ticket:
                continue

            deals_list = pos["deals"]
            if len(deals_list) < 2:
                continue  # Need at least entry + exit

            # Find entry deal (entry=0) and exit deal (entry=1)
            entry_deal = None
            exit_deal = None
            for d in deals_list:
                if d["entry"] == 0:  # IN
                    entry_deal = d
                elif d["entry"] == 1:  # OUT
                    exit_deal = d

            if not entry_deal or not exit_deal:
                continue

            # Determine exit reason
            exit_reason = "MANUAL"
            comment = exit_deal.get("comment", "")
            if "tp" in comment.lower() or exit_deal["profit"] > 0:
                exit_reason = "TP_HIT"
            elif "sl" in comment.lower() or exit_deal["profit"] < 0:
                exit_reason = "SL_HIT"

            trade = {
                "ticket": pos_id,
                "symbol": pos["symbol"],
                "direction": entry_deal["type"],
                "volume": entry_deal["volume"],
                "entry_price": entry_deal["price"],
                "entry_time": entry_deal["time"],
                "exit_price": exit_deal["price"],
                "exit_time": exit_deal["time"],
                "exit_reason": exit_reason,
                "profit": exit_deal["profit"],
                "commission": entry_deal["commission"] + exit_deal["commission"],
                "swap": exit_deal["swap"],
                "comment": entry_deal.get("comment", "")
            }
            result.append(trade)

        return result

    def get_deal_by_ticket(self, ticket: int) -> Optional[Dict]:
        """Get details of a specific closed trade by ticket."""
        trades = self.get_history(days=30, ticket=ticket)
        return trades[0] if trades else None


# ============================================================================
# ZMQ Server
# ============================================================================

class ZMQServer:
    """ZeroMQ server for MT5 commands."""
    
    def __init__(self, mt5_mgr: MT5Manager):
        self.mt5 = mt5_mgr
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.running = False
    
    def start(self, port: int = 5555):
        """Start the ZMQ server."""
        bind_addr = f"tcp://0.0.0.0:{port}"
        self.socket.bind(bind_addr)
        self.running = True
        
        print(f"[ZMQ] Server listening on {bind_addr}")
        print("[ZMQ] Commands: PING, ACCOUNT, PRICE, TRADE, CLOSE, POSITIONS, HISTORY, DEAL_INFO")
        
        while self.running:
            try:
                # Wait for message with timeout
                if self.socket.poll(timeout=1000):  # 1 second timeout
                    message = self.socket.recv_json()
                    response = self.handle_command(message)
                    self.socket.send_json(response)
            except zmq.ZMQError as e:
                if self.running:
                    print(f"[ZMQ] Error: {e}")
            except Exception as e:
                print(f"[ZMQ] Handler error: {e}")
                try:
                    self.socket.send_json({"success": False, "error": str(e)})
                except:
                    pass
    
    def stop(self):
        """Stop the server."""
        self.running = False
        self.socket.close()
        self.context.term()
    
    def handle_command(self, msg: Dict) -> Dict:
        """Handle incoming command."""
        cmd = msg.get("cmd", "").upper()
        
        print(f"[ZMQ] Received: {cmd}")
        
        if cmd == "PING":
            return {
                "success": True,
                "mt5_connected": self.mt5.connected,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        elif cmd == "ACCOUNT":
            account = self.mt5.get_account()
            if account:
                return {"success": True, **account}
            return {"success": False, "error": "Failed to get account"}
        
        elif cmd == "PRICE":
            symbol = msg.get("symbol")
            if not symbol:
                return {"success": False, "error": "Symbol required"}
            price = self.mt5.get_price(symbol)
            if price:
                return {"success": True, **price}
            return {"success": False, "error": f"Failed to get price for {symbol}"}
        
        elif cmd == "TRADE":
            return self.mt5.execute_trade(
                symbol=msg.get("symbol"),
                direction=msg.get("direction"),
                volume=msg.get("volume", 0.01),
                sl=msg.get("sl"),
                tp=msg.get("tp"),
                comment=msg.get("comment", "HYDRA")
            )
        
        elif cmd == "CLOSE":
            ticket = msg.get("ticket")
            if not ticket:
                return {"success": False, "error": "Ticket required"}
            return self.mt5.close_position(int(ticket))
        
        elif cmd == "POSITIONS":
            positions = self.mt5.get_positions()
            return {"success": True, "count": len(positions), "positions": positions}
        
        elif cmd == "RECONNECT":
            success = self.mt5.connect()
            return {"success": success, "mt5_connected": self.mt5.connected}

        elif cmd == "SYMBOLS":
            pattern = msg.get("pattern", "*XAU*,*GOLD*,*EUR*,*US30*,*NAS*")
            symbols = self.mt5.get_symbols(pattern)
            return {"success": True, "count": len(symbols), "symbols": symbols}

        elif cmd == "HISTORY":
            # Get closed trade history
            days = msg.get("days", 7)
            ticket = msg.get("ticket")  # Optional specific ticket
            trades = self.mt5.get_history(days=days, ticket=ticket)
            return {"success": True, "count": len(trades), "trades": trades}

        elif cmd == "DEAL_INFO":
            # Get specific trade details
            ticket = msg.get("ticket")
            if not ticket:
                return {"success": False, "error": "Ticket required"}
            trade = self.mt5.get_deal_by_ticket(int(ticket))
            if trade:
                return {"success": True, **trade}
            return {"success": False, "error": f"Trade {ticket} not found in history"}

        elif cmd == "CANDLES":
            # Get historical candles
            symbol = msg.get("symbol")
            if not symbol:
                return {"success": False, "error": "Symbol required"}
            timeframe = msg.get("timeframe", "H1")
            count = min(msg.get("count", 100), 500)  # Limit to 500 candles
            candles = self.mt5.get_candles(symbol, timeframe, count)
            if candles:
                return {"success": True, "symbol": symbol, "timeframe": timeframe, "count": len(candles), "candles": candles}
            return {"success": False, "error": f"Failed to get candles for {symbol}"}

        else:
            return {"success": False, "error": f"Unknown command: {cmd}"}


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("    HYDRA MT5 ZeroMQ Executor")
    print("    Production-Grade Trading Bridge")
    print("=" * 60)
    print()
    
    print(f"[CONFIG] MT5 Login: {config.mt5_login}")
    print(f"[CONFIG] MT5 Server: {config.mt5_server}")
    print(f"[CONFIG] ZMQ Port: {config.zmq_port}")
    print()
    
    # Initialize MT5
    mt5_mgr = MT5Manager(config)
    print("[MT5] Connecting...")
    
    if mt5_mgr.connect():
        print("[MT5] Connected successfully!")
    else:
        print("[MT5] WARNING: Failed to connect to MT5!")
        print("[MT5] Server will start but trades will fail until MT5 connects.")
    
    print()
    
    # Start ZMQ server
    server = ZMQServer(mt5_mgr)
    
    # Handle shutdown
    def shutdown(signum, frame):
        print("\n[SERVER] Shutting down...")
        server.stop()
        mt5.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    print("=" * 60)
    print("Ready to receive commands!")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()
    
    server.start(config.zmq_port)


if __name__ == "__main__":
    main()
