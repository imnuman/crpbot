"""Real-time multi-timeframe data fetcher for runtime."""
import pandas as pd
from loguru import logger

from apps.runtime.data_fetcher import MarketDataFetcher


def fetch_multi_tf_data(
    data_fetcher: MarketDataFetcher,
    symbol: str,
    intervals: list[str] = ["1m", "5m", "15m", "1h"],
    num_candles: int = 150
) -> dict[str, pd.DataFrame]:
    """
    Fetch data from multiple timeframes in real-time.

    Args:
        data_fetcher: Market data fetcher instance
        symbol: Trading pair (e.g., 'BTC-USD')
        intervals: List of timeframe intervals
        num_candles: Number of candles to fetch per interval

    Returns:
        Dictionary mapping interval -> DataFrame
    """
    multi_tf_data = {}

    # Map our intervals to Coinbase granularity
    granularity_map = {
        "1m": "ONE_MINUTE",
        "5m": "FIVE_MINUTE",
        "15m": "FIFTEEN_MINUTE",
        "1h": "ONE_HOUR"
    }

    for interval in intervals:
        try:
            if interval not in granularity_map:
                logger.warning(f"Unsupported interval: {interval}")
                continue

            # For non-1m intervals, fetch from Coinbase with different granularity
            if interval == "1m":
                # We already have 1m data
                continue

            logger.debug(f"Fetching {interval} data for {symbol}")

            # Use Coinbase SDK directly for different granularities
            response = data_fetcher.client.get_candles(
                product_id=symbol,
                start=None,
                end=None,
                granularity=granularity_map[interval]
            )

            if not response or not response.candles:
                logger.warning(f"No {interval} data returned for {symbol}")
                continue

            # Convert to DataFrame
            candles_data = []
            for candle in response.candles[:num_candles]:
                candles_data.append({
                    'timestamp': pd.to_datetime(int(candle['start']), unit='s', utc=True),
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close']),
                    'volume': float(candle['volume'])
                })

            df = pd.DataFrame(candles_data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            multi_tf_data[interval] = df

            logger.debug(f"✅ Fetched {len(df)} {interval} candles for {symbol}")

        except Exception as e:
            logger.error(f"Failed to fetch {interval} data for {symbol}: {e}")
            continue

    return multi_tf_data


def align_multi_tf_to_base(
    base_df: pd.DataFrame,
    multi_tf_data: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Align multi-timeframe data to base 1m timeframe.

    Args:
        base_df: Base 1m DataFrame
        multi_tf_data: Dict of higher TF DataFrames

    Returns:
        Base DataFrame with multi-TF features added
    """
    df = base_df.copy()

    # Ensure timestamp is datetime with UTC timezone
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    elif df["timestamp"].dt.tz is None:
        # Convert timezone-naive to timezone-aware (UTC)
        df["timestamp"] = df["timestamp"].dt.tz_localize('UTC')

    for interval, tf_df in multi_tf_data.items():
        if interval == "1m":
            continue

        try:
            # Ensure timestamp is datetime with UTC timezone
            if not pd.api.types.is_datetime64_any_dtype(tf_df["timestamp"]):
                tf_df["timestamp"] = pd.to_datetime(tf_df["timestamp"], utc=True)
            elif tf_df["timestamp"].dt.tz is None:
                # Convert timezone-naive to timezone-aware (UTC)
                tf_df["timestamp"] = tf_df["timestamp"].dt.tz_localize('UTC')

            # Merge using asof (forward-fill higher TF values)
            df = df.sort_values('timestamp')
            tf_df = tf_df.sort_values('timestamp')

            # Prefix for this timeframe
            prefix = interval.replace('m', 'm_').replace('h', 'h_')

            # Merge OHLCV
            for col in ['open', 'high', 'low', 'close', 'volume']:
                tf_col = f"{prefix}{col}"
                df[tf_col] = pd.merge_asof(
                    df[['timestamp']],
                    tf_df[['timestamp', col]].rename(columns={col: tf_col}),
                    on='timestamp',
                    direction='backward'
                )[tf_col]

            logger.debug(f"Aligned {interval} features to base timeframe")

        except Exception as e:
            logger.error(f"Failed to align {interval} data: {e}")
            continue

    return df


def add_tf_alignment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add timeframe alignment features.

    Args:
        df: DataFrame with multi-TF OHLCV

    Returns:
        DataFrame with alignment features added
    """
    # Simple implementation: check if trend direction agrees across timeframes
    try:
        # Calculate simple trend for each TF (positive if close > open)
        trend_1m = (df['close'] > df['open']).astype(int) * 2 - 1  # +1 or -1

        # For higher TFs, use the TF close vs open if available
        trends = [trend_1m]

        for tf in ['5m', '15m', '1h']:
            tf_close = f"{tf}_close"
            tf_open = f"{tf}_open"
            if tf_close in df.columns and tf_open in df.columns:
                trend = (df[tf_close] > df[tf_open]).astype(int) * 2 - 1
                trends.append(trend)

        # Alignment score: how many TFs agree with 1m trend
        if len(trends) > 1:
            trend_matrix = pd.concat(trends, axis=1)
            df['tf_alignment_score'] = (trend_matrix == trend_matrix.iloc[:, 0:1].values).sum(axis=1) / len(trends)

            # Overall direction: majority vote
            df['tf_alignment_direction'] = trend_matrix.mode(axis=1)[0]

            # Strength: normalized alignment score
            df['tf_alignment_strength'] = df['tf_alignment_score']
        else:
            # Fallback
            df['tf_alignment_score'] = 0.5
            df['tf_alignment_direction'] = 0
            df['tf_alignment_strength'] = 0.5

    except Exception as e:
        logger.error(f"Failed to add TF alignment features: {e}")
        # Add fallback values
        df['tf_alignment_score'] = 0.5
        df['tf_alignment_direction'] = 0
        df['tf_alignment_strength'] = 0.5

    return df
