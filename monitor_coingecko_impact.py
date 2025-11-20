#!/usr/bin/env python3
"""
Monitor CoinGecko integration impact on V7 signals
Analyzes recent signals to see signal quality and DeepSeek reasoning
"""

import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict

# Connect to database
conn = sqlite3.connect('/root/crpbot/tradingai.db')
cursor = conn.cursor()

# Get signals from the last 24 hours (since CoinGecko integration)
twenty_four_hours_ago = (datetime.now() - timedelta(hours=24)).isoformat()

cursor.execute("""
    SELECT
        id,
        timestamp,
        symbol,
        signal_type,
        confidence,
        reasoning,
        current_price,
        entry_price,
        stop_loss,
        take_profit
    FROM signals
    WHERE timestamp >= ?
    ORDER BY timestamp DESC
""", (twenty_four_hours_ago,))

signals = cursor.fetchall()

print("="*80)
print("V7 SIGNALS WITH COINGECKO INTEGRATION - MONITORING REPORT")
print("="*80)
print(f"Time Range: Last 24 hours (since {twenty_four_hours_ago[:19]})")
print(f"Total Signals: {len(signals)}")
print()

# Statistics
signal_counts = defaultdict(int)
confidence_by_type = defaultdict(list)
symbols = set()

for sig in signals:
    signal_type = sig[3]
    confidence = sig[4]
    symbol = sig[2]

    signal_counts[signal_type] += 1
    confidence_by_type[signal_type].append(confidence)
    symbols.add(symbol)

print("SIGNAL STATISTICS:")
print(f"  Symbols Tracked: {', '.join(sorted(symbols))}")
print()
print("  Signal Distribution:")
for sig_type, count in sorted(signal_counts.items()):
    pct = (count / len(signals) * 100) if signals else 0
    avg_conf = sum(confidence_by_type[sig_type]) / len(confidence_by_type[sig_type]) if confidence_by_type[sig_type] else 0
    print(f"    {sig_type:4}: {count:3} ({pct:5.1f}%) | Avg Confidence: {avg_conf:5.1f}%")

print()
print("="*80)
print("RECENT SIGNALS (Last 10):")
print("="*80)

for i, sig in enumerate(signals[:10], 1):
    timestamp = sig[1][:19]
    symbol = sig[2]
    signal_type = sig[3]
    confidence = sig[4]
    reasoning = sig[5][:120] + "..." if len(sig[5]) > 120 else sig[5]

    print(f"\n{i}. {timestamp} | {symbol} | {signal_type} @ {confidence:.0f}%")
    print(f"   Reasoning: {reasoning}")

print()
print("="*80)
print("ANALYSIS: COINGECKO IMPACT")
print("="*80)

# Check if reasoning mentions market conditions
market_keywords = ['market', 'volume', 'liquidity', 'sentiment', 'cap', 'macro']
reasoning_with_market_refs = 0

for sig in signals:
    reasoning = sig[5].lower()
    if any(keyword in reasoning for keyword in market_keywords):
        reasoning_with_market_refs += 1

pct_market_refs = (reasoning_with_market_refs / len(signals) * 100) if signals else 0
print(f"Signals mentioning market context: {reasoning_with_market_refs}/{len(signals)} ({pct_market_refs:.1f}%)")
print()
print("Note: DeepSeek may synthesize CoinGecko data into its analysis without")
print("explicitly mentioning market conditions. The context influences the model's")
print("decision-making even if not explicitly stated in the reasoning.")
print()

# Average confidence trends
print("CONFIDENCE TRENDS:")
if signals:
    recent_10 = signals[:10]
    older_10 = signals[-10:] if len(signals) > 10 else []

    recent_avg = sum(s[4] for s in recent_10) / len(recent_10)
    if older_10:
        older_avg = sum(s[4] for s in older_10) / len(older_10)
        print(f"  Recent 10 signals avg: {recent_avg:.1f}%")
        print(f"  Older 10 signals avg: {older_avg:.1f}%")
        print(f"  Trend: {'📈 Increasing' if recent_avg > older_avg else '📉 Decreasing' if recent_avg < older_avg else '➡️  Stable'}")
    else:
        print(f"  Average confidence (all): {recent_avg:.1f}%")

print()
print("="*80)
print("DASHBOARD ACCESS")
print("="*80)
print("View live signals at: http://localhost:5000")
print("="*80)

conn.close()
