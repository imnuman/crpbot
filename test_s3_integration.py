#!/usr/bin/env python3
"""Test S3 integration with created buckets."""

import os
import sys
from datetime import datetime
import pandas as pd

# Add libs to path
sys.path.append('libs')

from aws.s3_client import S3Client


def test_s3_upload():
    """Test uploading sample data to S3."""
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv('.env.aws')
    
    # Create S3 client
    s3_client = S3Client()
    
    # Create sample market data
    sample_data = pd.DataFrame({
        'timestamp': [datetime.now()],
        'open': [50000.0],
        'high': [51000.0],
        'low': [49000.0],
        'close': [50500.0],
        'volume': [1000.0]
    })
    
    # Test upload
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    try:
        # Test market data upload
        result = s3_client.upload_market_data('BTC-USD', sample_data, timestamp)
        print(f"✅ Market data uploaded: {result}")
        
        # Test log upload
        log_result = s3_client.upload_logs(f'test_{timestamp}.log', 'Test log entry')
        print(f"✅ Log uploaded: {log_result}")
        
        # Test backup upload
        backup_result = s3_client.upload_backup(f'test_backup_{timestamp}.sql', b'-- Test backup')
        print(f"✅ Backup uploaded: {backup_result}")
        
        print("\n🎉 All S3 integrations working!")
        
    except Exception as e:
        print(f"❌ S3 test failed: {e}")
        return False
    
    return True


if __name__ == '__main__':
    test_s3_upload()