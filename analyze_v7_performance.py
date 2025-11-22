"""Analyze V7 performance from database"""
import sqlite3
import pandas as pd
import numpy as np

print("="*70)
print("V7 PERFORMANCE ANALYSIS")
print("="*70)

conn = sqlite3.connect('tradingai.db')

# 1. Signal Statistics
print("\n1. SIGNAL GENERATION STATISTICS (Last 7 Days)")
print("-"*70)

query = """
SELECT
  DATE(timestamp) as date,
  COUNT(*) as total_signals,
  SUM(CASE WHEN direction IN ('buy', 'long') THEN 1 ELSE 0 END) as buy_signals,
  SUM(CASE WHEN direction IN ('sell', 'short') THEN 1 ELSE 0 END) as sell_signals,
  SUM(CASE WHEN direction = 'hold' THEN 1 ELSE 0 END) as hold_signals,
  ROUND(AVG(confidence), 4) as avg_confidence
FROM signals
WHERE timestamp > datetime('now', '-7 days')
GROUP BY DATE(timestamp)
ORDER BY date DESC
"""

df_signals = pd.read_sql(query, conn)
if len(df_signals) > 0:
    print(df_signals.to_string(index=False))
else:
    print("No signals in last 7 days")

# 2. Paper Trading Performance
print("\n2. PAPER TRADING RESULTS")
print("-"*70)

query = """
SELECT
  COUNT(*) as total_trades,
  SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
  SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
  ROUND(AVG(pnl_percent), 4) as avg_pnl_pct,
  ROUND(SUM(pnl_percent), 4) as total_pnl_pct,
  ROUND(MIN(pnl_percent), 4) as worst_trade,
  ROUND(MAX(pnl_percent), 4) as best_trade
FROM signal_results
WHERE pnl_percent IS NOT NULL
"""

df_trades = pd.read_sql(query, conn)
if len(df_trades) > 0 and df_trades['total_trades'].iloc[0] > 0:
    print(df_trades.to_string(index=False))

    wins = df_trades['wins'].iloc[0] or 0
    total = df_trades['total_trades'].iloc[0]
    if total > 0:
        win_rate = wins / total
        print(f"\nWin Rate: {win_rate*100:.1f}%")
else:
    print("No paper trading data yet")

# 3. A/B Test Results
print("\n3. A/B TEST RESULTS")
print("-"*70)

query = """
SELECT
  strategy,
  COUNT(*) as signals,
  ROUND(AVG(confidence), 4) as avg_confidence,
  COUNT(DISTINCT symbol) as symbols_traded
FROM signals
WHERE strategy IS NOT NULL
  AND timestamp > datetime('now', '-7 days')
GROUP BY strategy
"""

df_ab = pd.read_sql(query, conn)
if len(df_ab) > 0:
    print(df_ab.to_string(index=False))
else:
    print("No A/B test data yet")

# 4. Per-Symbol Performance
print("\n4. PER-SYMBOL BREAKDOWN")
print("-"*70)

query = """
SELECT
  symbol,
  COUNT(*) as signals,
  ROUND(AVG(confidence), 4) as avg_conf,
  SUM(CASE WHEN direction IN ('buy', 'long', 'sell', 'short') THEN 1 ELSE 0 END) as actionable,
  SUM(CASE WHEN direction = 'hold' THEN 1 ELSE 0 END) as holds
FROM signals
WHERE timestamp > datetime('now', '-7 days')
GROUP BY symbol
ORDER BY signals DESC
"""

df_symbols = pd.read_sql(query, conn)
if len(df_symbols) > 0:
    print(df_symbols.to_string(index=False))
else:
    print("No symbol data in last 7 days")

# 5. Overall Stats
print("\n5. OVERALL STATISTICS")
print("-"*70)

total_signals_query = "SELECT COUNT(*) FROM signals"
total_signals = pd.read_sql(total_signals_query, conn).iloc[0, 0]

signals_24h_query = "SELECT COUNT(*) FROM signals WHERE timestamp > datetime('now', '-24 hours')"
signals_24h = pd.read_sql(signals_24h_query, conn).iloc[0, 0]

print(f"Total Signals (all time): {total_signals}")
print(f"Signals (last 24h): {signals_24h}")

conn.close()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
