"""
V6 Enhanced Runtime
Uses Amazon Q's V6 models with proper 72-feature compatibility
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from libs.features.v6_model_loader import V6ModelLoader
from libs.data.kraken_client import KrakenClient
from libs.data.coingecko_client import CoinGeckoClient

logger = logging.getLogger(__name__)


class V6Runtime:
    """V6 Enhanced Runtime with proper feature compatibility"""
    
    def __init__(self):
        self.model_loader = V6ModelLoader()
        self.kraken_client = KrakenClient()
        self.coingecko_client = CoinGeckoClient()
        self.symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        self.running = False
        
    async def initialize(self) -> bool:
        """Initialize V6 runtime"""
        try:
            logger.info("🚀 Initializing V6 Enhanced Runtime...")
            
            # Load V6 models
            model_results = self.model_loader.load_all_models()
            loaded_count = sum(model_results.values())
            
            if loaded_count == 0:
                logger.error("❌ No V6 models loaded")
                return False
                
            logger.info(f"✅ Loaded {loaded_count}/3 V6 models")
            
            # Display model info
            for symbol in self.symbols:
                info = self.model_loader.get_model_info(symbol)
                if info:
                    logger.info(f"  {symbol}: {info['accuracy']:.1%} accuracy")
            
            # Test data connections
            await self._test_connections()
            
            logger.info("✅ V6 Runtime initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ V6 Runtime initialization failed: {e}")
            return False
    
    async def _test_connections(self):
        """Test data source connections"""
        try:
            # Test Kraken
            btc_data = await self.kraken_client.get_ohlc('XBTUSD', interval=60, count=10)
            logger.info(f"✅ Kraken: {len(btc_data)} BTC candles")
            
            # Test CoinGecko
            cg_data = await self.coingecko_client.get_price(['bitcoin'], vs_currencies=['usd'])
            logger.info(f"✅ CoinGecko: BTC price ${cg_data['bitcoin']['usd']:,.0f}")
            
        except Exception as e:
            logger.warning(f"⚠️  Connection test: {e}")
    
    async def get_market_data(self, symbol: str, hours: int = 300) -> Optional[pd.DataFrame]:
        """Get market data for V6 feature generation"""
        try:
            # Map symbol to Kraken pair
            kraken_pairs = {
                'BTC-USD': 'XBTUSD',
                'ETH-USD': 'ETHUSD', 
                'SOL-USD': 'SOLUSD'
            }
            
            pair = kraken_pairs.get(symbol)
            if not pair:
                logger.error(f"Unknown symbol: {symbol}")
                return None
            
            # Get OHLCV data
            ohlc_data = await self.kraken_client.get_ohlc(pair, interval=60, count=hours)
            
            if len(ohlc_data) < 200:  # Need enough data for 200-period EMA
                logger.warning(f"Insufficient data for {symbol}: {len(ohlc_data)} candles")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlc_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp')
            
            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"📊 {symbol}: {len(df)} candles, latest: {df.index[-1]}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    async def generate_signal(self, symbol: str) -> Optional[Dict]:
        """Generate V6 enhanced signal"""
        try:
            # Get market data
            df = await self.get_market_data(symbol)
            if df is None:
                return None
            
            # Generate V6 prediction
            prediction = self.model_loader.predict(symbol, df)
            if prediction is None:
                return None
            
            # Add market context
            latest_price = df['close'].iloc[-1]
            price_change_24h = ((latest_price - df['close'].iloc[-25]) / df['close'].iloc[-25]) * 100
            
            signal = {
                **prediction,
                'timestamp': datetime.now().isoformat(),
                'price': latest_price,
                'price_change_24h': price_change_24h,
                'data_points': len(df),
                'runtime_version': 'v6_enhanced'
            }
            
            logger.info(f"🎯 {symbol}: {signal['signal']} ({signal['confidence']:.1%} confidence)")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    async def run_scan(self) -> List[Dict]:
        """Run V6 enhanced scan for all symbols"""
        logger.info("🔍 Running V6 Enhanced Scan...")
        signals = []
        
        for symbol in self.symbols:
            signal = await self.generate_signal(symbol)
            if signal:
                signals.append(signal)
                
        logger.info(f"✅ Generated {len(signals)} V6 signals")
        return signals
    
    async def run_continuous(self, interval_minutes: int = 15):
        """Run continuous V6 scanning"""
        logger.info(f"🔄 Starting V6 continuous scanning (every {interval_minutes}m)")
        self.running = True
        
        while self.running:
            try:
                # Run scan
                signals = await self.run_scan()
                
                # Log results
                for signal in signals:
                    logger.info(
                        f"📈 {signal['symbol']}: {signal['signal']} "
                        f"({signal['confidence']:.1%}) - ${signal['price']:,.2f}"
                    )
                
                # Wait for next interval
                await asyncio.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopping V6 runtime...")
                break
            except Exception as e:
                logger.error(f"Runtime error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    def stop(self):
        """Stop continuous scanning"""
        self.running = False
        logger.info("🛑 V6 Runtime stop requested")


async def main():
    """Main V6 runtime entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    runtime = V6Runtime()
    
    # Initialize
    if not await runtime.initialize():
        logger.error("Failed to initialize V6 runtime")
        return
    
    try:
        # Run single scan
        logger.info("Running single V6 scan...")
        signals = await runtime.run_scan()
        
        print("\n" + "="*60)
        print("V6 ENHANCED SIGNALS")
        print("="*60)
        
        for signal in signals:
            print(f"\n{signal['symbol']}:")
            print(f"  Signal: {signal['signal']}")
            print(f"  Confidence: {signal['confidence']:.1%}")
            print(f"  Price: ${signal['price']:,.2f}")
            print(f"  24h Change: {signal['price_change_24h']:+.1f}%")
            print(f"  Model Accuracy: {signal['model_accuracy']:.1%}")
        
        print(f"\n✅ V6 Enhanced Runtime completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Runtime interrupted by user")
    except Exception as e:
        logger.error(f"Runtime error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
