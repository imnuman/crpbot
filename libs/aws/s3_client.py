"""S3 client for CRPBot data storage."""

import os
from io import StringIO

import boto3
import pandas as pd


class S3Client:
    """S3 client for market data, backups, and logs."""

    def __init__(self):
        self.s3 = boto3.client("s3")
        self.market_data_bucket = os.getenv("S3_MARKET_DATA_BUCKET", "crpbot-market-data-dev")
        self.backups_bucket = os.getenv("S3_BACKUPS_BUCKET", "crpbot-backups-dev")
        self.logs_bucket = os.getenv("S3_LOGS_BUCKET", "crpbot-logs-dev")

    def upload_market_data(self, symbol: str, data: pd.DataFrame, timestamp: str) -> str:
        """Upload market data to S3."""
        key = f"raw/coinbase/{timestamp[:10]}/{symbol}_{timestamp}.parquet"

        # Convert to parquet bytes
        buffer = StringIO()
        data.to_parquet(buffer)

        self.s3.put_object(
            Bucket=self.market_data_bucket,
            Key=key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )

        return f"s3://{self.market_data_bucket}/{key}"

    def upload_backup(self, filename: str, data: bytes) -> str:
        """Upload backup file to S3."""
        key = f"database/{filename}"

        self.s3.put_object(
            Bucket=self.backups_bucket, Key=key, Body=data, ContentType="application/octet-stream"
        )

        return f"s3://{self.backups_bucket}/{key}"

    def upload_logs(self, log_file: str, content: str) -> str:
        """Upload log file to S3."""
        key = f"application/{log_file}"

        self.s3.put_object(Bucket=self.logs_bucket, Key=key, Body=content, ContentType="text/plain")

        return f"s3://{self.logs_bucket}/{key}"
