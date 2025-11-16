#!/usr/bin/env python3
"""
V6 Rebuild: Multi-Source Data Collection
Replaces Binance with Canada-compatible sources
"""

import os
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import requests
import time

class MultiSourceDataCollector:
    """Collect data from multiple Canada-compatible sources"""
    
    def __init__(self):
        self.symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        self.start_date = "2023-11-01"
        self.end_date = "2025-11-16"
        
    def collect_coinbase_historical(self, symbol: str, days: int = 730):
        """Collect historical data from Coinbase Pro (replaces Binance)"""
        print(f"📊 Collecting Coinbase historical data for {symbol}...")
        
        # Coinbase Pro API for historical candles
        url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
        
        all_data = []
        end_time = datetime.now()
        
        # Collect in chunks (300 candles per request)
        for chunk in range(0, days * 24 * 60 // 300):
            start_time = end_time - timedelta(minutes=300)
            
            params = {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'granularity': 60  # 1-minute candles
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    all_data.extend(data)
                    print(f"  Collected {len(data)} candles ending {end_time}")
                else:
                    print(f"  Error: {response.status_code}")
                
                end_time = start_time
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"  Error: {e}")
                break
        
        # Convert to DataFrame
        if all_data:
            df = pd.DataFrame(all_data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Save raw data
            output_dir = f"data/raw/coinbase/{symbol}"
            os.makedirs(output_dir, exist_ok=True)
            df.to_parquet(f"{output_dir}/historical_1m.parquet", index=False)
            
            print(f"✅ Saved {len(df)} candles for {symbol}")
            return df
        
        return pd.DataFrame()
    
    def collect_coingecko_fundamentals(self, symbol: str):
        """Collect fundamental data from CoinGecko"""
        print(f"📈 Collecting CoinGecko fundamentals for {symbol}...")
        
        # Map symbols to CoinGecko IDs
        coin_map = {
            "BTC-USD": "bitcoin",
            "ETH-USD": "ethereum", 
            "SOL-USD": "solana"
        }
        
        coin_id = coin_map.get(symbol)
        if not coin_id:
            return pd.DataFrame()
        
        # Get market data
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': '730',  # 2 years
            'interval': 'hourly'
        }
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Extract market cap and volume
                market_caps = data.get('market_caps', [])
                volumes = data.get('total_volumes', [])
                
                if market_caps and volumes:
                    df = pd.DataFrame({
                        'timestamp': [pd.to_datetime(mc[0], unit='ms') for mc in market_caps],
                        'market_cap': [mc[1] for mc in market_caps],
                        'volume_24h': [vol[1] for vol in volumes]
                    })
                    
                    # Save fundamental data
                    output_dir = f"data/raw/coingecko/{symbol}"
                    os.makedirs(output_dir, exist_ok=True)
                    df.to_parquet(f"{output_dir}/fundamentals.parquet", index=False)
                    
                    print(f"✅ Saved {len(df)} fundamental records for {symbol}")
                    return df
                    
        except Exception as e:
            print(f"  Error: {e}")
        
        return pd.DataFrame()
    
    def collect_global_metrics(self):
        """Collect global crypto metrics"""
        print("🌍 Collecting global crypto metrics...")
        
        try:
            # Global market data
            url = "https://api.coingecko.com/api/v3/global"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()['data']
                
                global_data = {
                    'timestamp': datetime.now(),
                    'total_market_cap_usd': data.get('total_market_cap', {}).get('usd', 0),
                    'total_volume_24h_usd': data.get('total_volume', {}).get('usd', 0),
                    'btc_dominance': data.get('market_cap_percentage', {}).get('btc', 0),
                    'eth_dominance': data.get('market_cap_percentage', {}).get('eth', 0),
                    'active_cryptocurrencies': data.get('active_cryptocurrencies', 0)
                }
                
                # Save global metrics
                os.makedirs("data/raw/global", exist_ok=True)
                df = pd.DataFrame([global_data])
                df.to_parquet("data/raw/global/metrics.parquet", index=False)
                
                print("✅ Saved global crypto metrics")
                return df
                
        except Exception as e:
            print(f"  Error: {e}")
        
        return pd.DataFrame()
    
    def run_collection(self):
        """Run complete multi-source data collection"""
        print("🚀 Starting V6 Rebuild Data Collection")
        print("=" * 50)
        
        results = {}
        
        # Collect for each symbol
        for symbol in self.symbols:
            print(f"\n📊 Processing {symbol}...")
            
            # Coinbase historical (replaces Binance)
            coinbase_data = self.collect_coinbase_historical(symbol, days=730)
            
            # CoinGecko fundamentals
            coingecko_data = self.collect_coingecko_fundamentals(symbol)
            
            results[symbol] = {
                'coinbase_rows': len(coinbase_data),
                'coingecko_rows': len(coingecko_data)
            }
        
        # Global metrics
        global_data = self.collect_global_metrics()
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 COLLECTION SUMMARY:")
        for symbol, data in results.items():
            print(f"  {symbol}:")
            print(f"    Coinbase: {data['coinbase_rows']:,} candles")
            print(f"    CoinGecko: {data['coingecko_rows']:,} records")
        
        print(f"  Global metrics: {len(global_data)} records")
        
        print("\n✅ Multi-source data collection complete!")
        print("🔄 Next: Feature engineering pipeline")

if __name__ == "__main__":
    collector = MultiSourceDataCollector()
    collector.run_collection()
