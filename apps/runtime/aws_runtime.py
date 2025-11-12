#!/usr/bin/env python3
"""
AWS-integrated runtime for CRPBot
Works with RDS, S3, and Secrets Manager
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any

import boto3
import psycopg2
from psycopg2.extras import RealDictCursor

from .signal import SignalGenerator
from .telegram_bot import TelegramBot
from .ftmo_rules import FTMORules
from .confidence import ConfidenceCalibrator

class AWSRuntime:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_aws_clients()
        self.db_conn = None
        self.telegram_bot = None
        self.signal_generator = None
        
    def setup_aws_clients(self):
        """Initialize AWS clients"""
        self.s3 = boto3.client('s3')
        self.secrets = boto3.client('secretsmanager')
        
        # Get secrets
        self.secrets_cache = {}
        self.load_secrets()
        
    def load_secrets(self):
        """Load all secrets from AWS Secrets Manager"""
        try:
            # Telegram bot token
            telegram_secret = self.secrets.get_secret_value(
                SecretId='crpbot/telegram-bot/dev'
            )
            self.secrets_cache['telegram'] = json.loads(telegram_secret['SecretString'])
            
            # FTMO credentials  
            ftmo_secret = self.secrets.get_secret_value(
                SecretId='crpbot/ftmo-account/dev'
            )
            self.secrets_cache['ftmo'] = json.loads(ftmo_secret['SecretString'])
            
            # Coinbase API
            coinbase_secret = self.secrets.get_secret_value(
                SecretId='crpbot/coinbase-api/dev'
            )
            self.secrets_cache['coinbase'] = json.loads(coinbase_secret['SecretString'])
            
            self.logger.info("✅ Loaded secrets from AWS Secrets Manager")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load secrets: {e}")
            raise
            
    def connect_database(self):
        """Connect to RDS PostgreSQL"""
        try:
            # Read password from local file (temporary)
            with open('/home/numan/crpbot/.db_password', 'r') as f:
                password = f.read().strip()
                
            self.db_conn = psycopg2.connect(
                host='crpbot-dev.cyjcoys82evx.us-east-1.rds.amazonaws.com',
                port=5432,
                database='postgres',  # Change to crpbot_dev after schema creation
                user='crpbot_admin',
                password=password,
                cursor_factory=RealDictCursor
            )
            self.logger.info("✅ Connected to RDS PostgreSQL")
            
        except Exception as e:
            self.logger.error(f"❌ Database connection failed: {e}")
            raise
            
    def load_models_from_s3(self):
        """Load trained models from S3"""
        try:
            # List available models
            response = self.s3.list_objects_v2(
                Bucket='crpbot-market-data-dev',
                Prefix='models/'
            )
            
            if 'Contents' not in response:
                self.logger.warning("⚠️ No models found in S3, using CPU training models")
                return self.load_local_models()
                
            # Download latest models
            model_files = []
            for obj in response['Contents']:
                if obj['Key'].endswith('.pkl') or obj['Key'].endswith('.pt'):
                    model_files.append(obj['Key'])
                    
            self.logger.info(f"📥 Found {len(model_files)} models in S3")
            return model_files
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load models from S3: {e}")
            return self.load_local_models()
            
    def load_local_models(self):
        """Fallback to local CPU-trained models"""
        local_models = []
        models_dir = '/home/numan/crpbot/models'
        
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.pkl') or file.endswith('.pt'):
                    local_models.append(os.path.join(models_dir, file))
                    
        self.logger.info(f"📁 Using {len(local_models)} local models")
        return local_models
        
    def initialize_components(self):
        """Initialize trading components"""
        try:
            # Initialize Telegram bot
            telegram_token = self.secrets_cache['telegram']['bot_token']
            self.telegram_bot = TelegramBot(telegram_token)
            
            # Initialize signal generator with models
            models = self.load_models_from_s3()
            self.signal_generator = SignalGenerator(models)
            
            # Initialize FTMO rules
            self.ftmo_rules = FTMORules(
                account_id=self.secrets_cache['ftmo']['account_id'],
                max_daily_loss=self.secrets_cache['ftmo']['max_daily_loss']
            )
            
            self.logger.info("✅ All components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
            raise
            
    async def runtime_loop(self):
        """Main runtime loop"""
        self.logger.info("🚀 Starting CRPBot AWS Runtime")
        
        while True:
            try:
                # Generate signals
                signals = await self.signal_generator.generate_signals()
                
                # Apply FTMO rules
                filtered_signals = self.ftmo_rules.filter_signals(signals)
                
                # Store in database
                if filtered_signals:
                    self.store_signals(filtered_signals)
                    
                    # Send via Telegram
                    await self.telegram_bot.send_signals(filtered_signals)
                    
                # Health check
                await self.health_check()
                
                # Wait 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"❌ Runtime loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
                
    def store_signals(self, signals):
        """Store signals in RDS"""
        try:
            with self.db_conn.cursor() as cursor:
                for signal in signals:
                    cursor.execute("""
                        INSERT INTO trading.signals 
                        (symbol, signal_type, confidence, price, timestamp)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        signal['symbol'],
                        signal['type'],
                        signal['confidence'],
                        signal['price'],
                        datetime.now()
                    ))
                self.db_conn.commit()
                self.logger.info(f"💾 Stored {len(signals)} signals in database")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store signals: {e}")
            
    async def health_check(self):
        """Health check and metrics"""
        try:
            # Check database connection
            with self.db_conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                
            # Log to CloudWatch (via S3 for now)
            health_data = {
                'timestamp': datetime.now().isoformat(),
                'status': 'healthy',
                'components': {
                    'database': 'ok',
                    'telegram': 'ok',
                    'models': 'ok'
                }
            }
            
            # Upload health check to S3
            self.s3.put_object(
                Bucket='crpbot-logs-dev',
                Key=f'health/{datetime.now().strftime("%Y/%m/%d")}/health.json',
                Body=json.dumps(health_data)
            )
            
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")

async def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO)
    
    runtime = AWSRuntime()
    runtime.connect_database()
    runtime.initialize_components()
    
    await runtime.runtime_loop()

if __name__ == "__main__":
    asyncio.run(main())
