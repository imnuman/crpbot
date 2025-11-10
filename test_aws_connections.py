#!/usr/bin/env python3
"""Test AWS connections: RDS, Secrets Manager, and API integrations."""

import json
import sys

import boto3

# Add libs to path
sys.path.append("libs")


def test_secrets_manager():
    """Test Secrets Manager connections."""
    print("🔐 Testing Secrets Manager...")

    secrets_client = boto3.client("secretsmanager")

    secrets = [
        (
            "Coinbase API",
            "arn:aws:secretsmanager:us-east-1:980104576869:secret:crpbot/coinbase-api/dev-dHLD4h",
        ),
        (
            "Telegram Bot",
            "arn:aws:secretsmanager:us-east-1:980104576869:secret:crpbot/telegram-bot/dev-mIN8RP",
        ),
        (
            "FTMO Account",
            "arn:aws:secretsmanager:us-east-1:980104576869:secret:crpbot/ftmo-account/dev-QEkZgM",
        ),
    ]

    for name, arn in secrets:
        try:
            response = secrets_client.get_secret_value(SecretId=arn)
            secret_data = json.loads(response["SecretString"])

            # Check if credentials are populated (not placeholder values)
            if name == "Coinbase API":
                has_real_data = not secret_data.get("api_key", "").startswith("REPLACE_")
                print(
                    f"  ✅ {name}: {'Real credentials' if has_real_data else 'Placeholder values'}"
                )
                if has_real_data:
                    print(f"    - API Key: {secret_data.get('api_key', '')[:20]}...")
                    print(
                        f"    - Private Key: {'Present' if secret_data.get('private_key') else 'Missing'}"
                    )

            elif name == "Telegram Bot":
                has_real_data = not secret_data.get("bot_token", "").startswith("REPLACE_")
                print(
                    f"  ✅ {name}: {'Real credentials' if has_real_data else 'Placeholder values'}"
                )

            elif name == "FTMO Account":
                has_real_data = not secret_data.get("login", "").startswith("REPLACE_")
                print(
                    f"  ✅ {name}: {'Real credentials' if has_real_data else 'Placeholder values'}"
                )

        except Exception as e:
            print(f"  ❌ {name}: {e}")


def test_rds_connection():
    """Test RDS PostgreSQL connection."""
    print("\n🗄️ Testing RDS Connection...")

    try:
        import psycopg

        # Connection details
        host = "crpbot-dev.cyjcoys82evx.us-east-1.rds.amazonaws.com"
        port = "5432"
        database = "postgres"
        username = "crpbot_admin"
        password = "TempPassword123!"

        # Test connection
        conn_string = (
            f"host={host} port={port} dbname={database} user={username} password={password}"
        )

        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                print(f"  ✅ RDS Connected: {version}")

                # Test creating a simple table
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS connection_test (
                        id SERIAL PRIMARY KEY,
                        test_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """
                )

                cur.execute("INSERT INTO connection_test DEFAULT VALUES;")
                conn.commit()

                cur.execute("SELECT COUNT(*) FROM connection_test;")
                count = cur.fetchone()[0]
                print(f"  ✅ Database operations working: {count} test records")

    except ImportError:
        print("  ⚠️ psycopg not installed - install with: pip install psycopg[binary]")
    except Exception as e:
        print(f"  ❌ RDS Connection failed: {e}")


def test_coinbase_api():
    """Test Coinbase API connection."""
    print("\n💰 Testing Coinbase API...")

    try:
        from aws.secrets import SecretsManager

        secrets_mgr = SecretsManager()
        creds = secrets_mgr.get_coinbase_credentials()

        if not creds.get("api_key") or creds.get("api_key").startswith("REPLACE_"):
            print("  ⚠️ Coinbase credentials not updated in Secrets Manager")
            return

        # Basic validation of credentials format
        api_key = creds.get("api_key", "")
        private_key = creds.get("private_key", "")

        if api_key.startswith("organizations/") and "BEGIN EC PRIVATE KEY" in private_key:
            print("  ✅ Coinbase credentials format looks correct")
            print(f"    - API Key: {api_key[:30]}...")
            print("    - Private Key: PEM format detected")
        else:
            print("  ❌ Coinbase credentials format incorrect")
            print("    - API Key should start with 'organizations/'")
            print("    - Private Key should be PEM format")

    except Exception as e:
        print(f"  ❌ Coinbase test failed: {e}")


if __name__ == "__main__":
    print("🧪 Testing AWS Connections\n")

    test_secrets_manager()
    test_rds_connection()
    test_coinbase_api()

    print("\n✅ Connection tests complete!")
