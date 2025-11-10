#!/usr/bin/env python3
"""Simple S3 test without pandas dependency."""

from datetime import datetime

import boto3


def test_s3_buckets():
    """Test S3 bucket access."""

    s3 = boto3.client("s3")

    buckets = ["crpbot-market-data-dev", "crpbot-backups-dev", "crpbot-logs-dev"]

    for bucket in buckets:
        try:
            # Test upload
            key = f'test/{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            s3.put_object(
                Bucket=bucket, Key=key, Body=f"Test upload to {bucket}", ContentType="text/plain"
            )
            print(f"✅ {bucket}: Upload successful")

            # Test list
            response = s3.list_objects_v2(Bucket=bucket, Prefix="test/", MaxKeys=1)
            if "Contents" in response:
                print(f"✅ {bucket}: List successful")

        except Exception as e:
            print(f"❌ {bucket}: {e}")

    print("\n🎉 S3 integration test complete!")


if __name__ == "__main__":
    test_s3_buckets()
