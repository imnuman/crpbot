#!/usr/bin/env python3
"""
Manual Trading Script with Risk Controls
Max loss: $50/day
"""

import zmq
import sys
from datetime import datetime, timezone

# Configuration
MAX_DAILY_LOSS = 50.0  # CAD
LOT_SIZE = 0.05  # Conservative
DEFAULT_SL_PIPS = 50
DEFAULT_TP_PIPS = 100

class ManualTrader:
    def __init__(self):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, 10000)
        self.sock.connect('tcp://127.0.0.1:15555')
        self.starting_balance = None
        self.trades = []

    def send_cmd(self, cmd: dict) -> dict:
        self.sock.send_json(cmd)
        return self.sock.recv_json()

    def get_account(self) -> dict:
        return self.send_cmd({'cmd': 'ACCOUNT'})

    def get_price(self, symbol: str) -> dict:
        return self.send_cmd({'cmd': 'PRICE', 'symbol': symbol})

    def get_positions(self) -> dict:
        return self.send_cmd({'cmd': 'POSITIONS'})

    def trade(self, symbol: str, direction: str, lots: float, sl_pips: float, tp_pips: float) -> dict:
        """Place a trade with SL/TP in pips."""
        price_data = self.get_price(symbol)
        if not price_data.get('success'):
            return {'success': False, 'error': 'Failed to get price'}

        bid = price_data['bid']
        ask = price_data['ask']

        # Calculate pip value based on symbol
        if 'XAU' in symbol:
            pip_size = 0.01  # Gold
        elif 'JPY' in symbol:
            pip_size = 0.01
        else:
            pip_size = 0.0001  # Forex pairs

        if direction.upper() == 'BUY':
            entry = ask
            sl = entry - (sl_pips * pip_size)
            tp = entry + (tp_pips * pip_size)
        else:  # SELL
            entry = bid
            sl = entry + (sl_pips * pip_size)
            tp = entry - (tp_pips * pip_size)

        return self.send_cmd({
            'cmd': 'TRADE',
            'symbol': symbol,
            'direction': direction.upper(),
            'volume': lots,
            'sl': round(sl, 5),
            'tp': round(tp, 5),
            'comment': f'HYDRA_MANUAL_{datetime.now().strftime("%H%M")}'
        })

    def close_position(self, ticket: int) -> dict:
        return self.send_cmd({'cmd': 'CLOSE', 'ticket': ticket})

    def close_all(self) -> list:
        """Close all open positions."""
        positions = self.get_positions()
        results = []
        if positions.get('success') and positions.get('positions'):
            for pos in positions['positions']:
                result = self.close_position(pos['ticket'])
                results.append(result)
        return results

    def check_daily_pnl(self) -> float:
        """Check current daily P&L."""
        account = self.get_account()
        if not account.get('success'):
            return 0.0

        if self.starting_balance is None:
            self.starting_balance = account['balance']

        current_equity = account['equity']
        return current_equity - self.starting_balance

    def can_trade(self) -> tuple:
        """Check if we can place more trades based on daily loss limit."""
        pnl = self.check_daily_pnl()
        if pnl <= -MAX_DAILY_LOSS:
            return False, f"Daily loss limit reached: ${abs(pnl):.2f}"

        remaining_risk = MAX_DAILY_LOSS + pnl
        return True, f"Can trade. Current P&L: ${pnl:.2f}, Remaining risk budget: ${remaining_risk:.2f}"

    def status(self):
        """Print current status."""
        account = self.get_account()
        positions = self.get_positions()
        can, reason = self.can_trade()

        print("\n" + "="*60)
        print("FTMO TRADING STATUS")
        print("="*60)

        if account.get('success'):
            print(f"Balance:     ${account['balance']:.2f} {account['currency']}")
            print(f"Equity:      ${account['equity']:.2f}")
            print(f"Free Margin: ${account['free_margin']:.2f}")
            print(f"Profit:      ${account['profit']:.2f}")

        print(f"\nDaily P&L:   ${self.check_daily_pnl():.2f}")
        print(f"Max Loss:    ${MAX_DAILY_LOSS:.2f}")
        print(f"Can Trade:   {reason}")

        if positions.get('success'):
            print(f"\nOpen Positions: {positions['count']}")
            for pos in positions.get('positions', []):
                print(f"  #{pos['ticket']}: {pos['symbol']} {pos['type']} "
                      f"{pos['volume']} @ {pos['price_open']:.2f} "
                      f"P&L: ${pos['profit']:.2f}")

        print("="*60)

    def cleanup(self):
        self.sock.close()
        self.ctx.term()


def main():
    trader = ManualTrader()

    try:
        # Show initial status
        trader.status()

        print("\nCommands:")
        print("  status        - Show current status")
        print("  price SYMBOL  - Get current price")
        print("  buy SYMBOL    - Buy with default lot/SL/TP")
        print("  sell SYMBOL   - Sell with default lot/SL/TP")
        print("  positions     - List open positions")
        print("  close TICKET  - Close specific position")
        print("  closeall      - Close all positions")
        print("  quit          - Exit")

        while True:
            try:
                cmd = input("\n> ").strip().lower().split()
                if not cmd:
                    continue

                action = cmd[0]

                if action == 'quit' or action == 'exit':
                    break

                elif action == 'status':
                    trader.status()

                elif action == 'price':
                    symbol = cmd[1].upper() if len(cmd) > 1 else 'XAUUSD'
                    result = trader.get_price(symbol)
                    if result.get('success'):
                        print(f"{symbol}: bid={result['bid']}, ask={result['ask']}, spread={result.get('spread')}")
                    else:
                        print(f"Error: {result.get('error')}")

                elif action == 'buy':
                    symbol = cmd[1].upper() if len(cmd) > 1 else 'XAUUSD'
                    can, reason = trader.can_trade()
                    if not can:
                        print(f"Cannot trade: {reason}")
                        continue

                    result = trader.trade(symbol, 'BUY', LOT_SIZE, DEFAULT_SL_PIPS, DEFAULT_TP_PIPS)
                    if result.get('success'):
                        print(f"BUY order placed! Ticket: {result.get('ticket')}")
                        print(f"  Entry: {result.get('price_open')}, SL: {result.get('sl')}, TP: {result.get('tp')}")
                    else:
                        print(f"Error: {result.get('error')}")

                elif action == 'sell':
                    symbol = cmd[1].upper() if len(cmd) > 1 else 'XAUUSD'
                    can, reason = trader.can_trade()
                    if not can:
                        print(f"Cannot trade: {reason}")
                        continue

                    result = trader.trade(symbol, 'SELL', LOT_SIZE, DEFAULT_SL_PIPS, DEFAULT_TP_PIPS)
                    if result.get('success'):
                        print(f"SELL order placed! Ticket: {result.get('ticket')}")
                        print(f"  Entry: {result.get('price_open')}, SL: {result.get('sl')}, TP: {result.get('tp')}")
                    else:
                        print(f"Error: {result.get('error')}")

                elif action == 'positions':
                    result = trader.get_positions()
                    if result.get('success'):
                        if result['count'] == 0:
                            print("No open positions")
                        else:
                            for pos in result['positions']:
                                print(f"#{pos['ticket']}: {pos['symbol']} {pos['type']} "
                                      f"{pos['volume']} @ {pos['price_open']:.2f} "
                                      f"SL={pos.get('sl', 'N/A')} TP={pos.get('tp', 'N/A')} "
                                      f"P&L: ${pos['profit']:.2f}")
                    else:
                        print(f"Error: {result.get('error')}")

                elif action == 'close':
                    if len(cmd) < 2:
                        print("Usage: close TICKET")
                        continue
                    ticket = int(cmd[1])
                    result = trader.close_position(ticket)
                    if result.get('success'):
                        print(f"Position {ticket} closed! Profit: ${result.get('profit', 0):.2f}")
                    else:
                        print(f"Error: {result.get('error')}")

                elif action == 'closeall':
                    results = trader.close_all()
                    if results:
                        print(f"Closed {len(results)} positions")
                    else:
                        print("No positions to close")

                else:
                    print(f"Unknown command: {action}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

    finally:
        trader.cleanup()
        print("\nTrader closed.")


if __name__ == '__main__':
    main()
