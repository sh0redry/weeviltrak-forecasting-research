"""
Diagnose griddedweather S3 cache behavior.

Checks:
1. WeatherConfig — bucket, prefix, AWS profile
2. Cache index — exists, entries, referenced parquet files accessible
3. Test pull — does a small weather pull and checks if write-back happens

Usage:
    poetry run python scripts/weather/diagnose_weather_cache.py
"""

import os
import sys
import logging
from pathlib import Path

import pandas as pd
import dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
dotenv.load_dotenv(PROJECT_ROOT / ".env")

from griddedweather import GriddedWeatherClient, WeatherConfig
from griddedweather.s3_store import S3Manager

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def check_config():
    """Print resolved WeatherConfig values."""
    print("=" * 60)
    print("1. WeatherConfig (resolved from environment)")
    print("=" * 60)
    cfg = WeatherConfig.from_env(env=os.environ)
    print(f"  s3_bucket:     {cfg.s3_bucket}")
    print(f"  cache_prefix:  {cfg.cache_prefix!r}")
    print(f"  aws_profile:   {cfg.aws_profile}")
    print(f"  redshift_host: {cfg.redshift.host}")
    print(f"  redshift_port: {cfg.redshift.port}")
    print(f"  redshift_db:   {cfg.redshift.database}")
    print(f"  redshift_user: {cfg.redshift.user}")
    print()

    # Also check env vars directly
    print("  Relevant env vars:")
    for var in ["CACHE_S3_BUCKET", "S3_BUCKET", "AWS_S3_BUCKET",
                "CACHE_S3_PREFIX", "AWS_PROFILE", "AWS_DEFAULT_PROFILE"]:
        val = os.environ.get(var)
        print(f"    {var} = {val!r}")
    print()
    return cfg


def check_cache_index(client: GriddedWeatherClient):
    """Read and summarize the cache index on S3."""
    print("=" * 60)
    print("2. Cache Index")
    print("=" * 60)

    idx_key = client._cache_index_key()
    print(f"  Cache index S3 key: {idx_key}")

    if not client.s3_manager:
        print("  ERROR: No S3 manager configured — caching is disabled!")
        return pd.DataFrame()

    if not client.s3_manager.file_exists(idx_key):
        print("  Cache index does NOT exist on S3.")
        print("  This means no weather data has been cached yet.")
        return pd.DataFrame()

    idx = client.s3_manager.read_file(idx_key, index_col=None)
    print(f"  Cache index rows: {len(idx)}")
    if idx.empty:
        print("  Cache index is empty.")
        return idx

    print(f"  Columns: {idx.columns.tolist()}")
    print(f"  Unique place_ids: {idx['place_id'].nunique()}")
    print(f"  Date range: {idx['start_compact'].min()} -> {idx['end_compact'].max()}")
    print()
    print("  First 10 entries:")
    print(idx.head(10).to_string(index=False))
    print()
    return idx


def check_cache_file_access(client: GriddedWeatherClient, idx: pd.DataFrame, n_tests: int = 3):
    """Try reading a few cached parquet files to verify they're accessible."""
    print("=" * 60)
    print("3. Cache File Accessibility")
    print("=" * 60)

    if idx.empty or "key" not in idx.columns:
        print("  No cache entries to test.")
        return

    test_keys = idx["key"].head(n_tests).tolist()
    success = 0
    fail = 0

    for key in test_keys:
        try:
            df = client.s3_manager.read_file(key)
            if df is not None and not df.empty:
                print(f"  OK: {key} -> {len(df)} rows, dates: {df['date'].min()} to {df['date'].max()}")
                success += 1
            else:
                print(f"  EMPTY: {key} -> returned None/empty")
                fail += 1
        except Exception as e:
            print(f"  FAIL: {key} -> {type(e).__name__}: {e}")
            fail += 1

    print()
    print(f"  Result: {success}/{len(test_keys)} files readable, {fail} failed")
    print()


def check_s3_listing(client: GriddedWeatherClient):
    """List actual parquet files under the cache prefix to see what's on S3."""
    print("=" * 60)
    print("4. S3 Listing (actual files under cache prefix)")
    print("=" * 60)

    prefix = client.cache_prefix
    print(f"  Listing prefix: {prefix}")

    if not client.s3_manager:
        print("  No S3 manager — skipping.")
        return

    try:
        files = client.s3_manager.list_files(prefix=prefix)
        parquet_files = [f for f in files if f.endswith(".parquet")]
        csv_files = [f for f in files if f.endswith(".csv")]
        print(f"  Total objects: {len(files)}")
        print(f"  Parquet files: {len(parquet_files)}")
        print(f"  CSV files: {len(csv_files)}")
        if parquet_files:
            print(f"  First 5 parquet files:")
            for f in parquet_files[:5]:
                print(f"    {f}")
        if csv_files:
            print(f"  CSV files:")
            for f in csv_files[:5]:
                print(f"    {f}")
    except Exception as e:
        print(f"  ERROR listing: {type(e).__name__}: {e}")
    print()


def test_small_pull(client: GriddedWeatherClient):
    """Do a tiny weather pull (1 place, 3 days) and check if it caches."""
    print("=" * 60)
    print("5. Test Pull (1 place, 3 days)")
    print("=" * 60)

    # Read cache index before
    idx_before = client._load_cache_index()
    print(f"  Cache index entries BEFORE pull: {len(idx_before)}")

    # Use a known place_id from the cache index if available, otherwise pick a test one
    test_place_id = None
    if not idx_before.empty:
        test_place_id = str(idx_before.iloc[0]["place_id"])
    else:
        # Try to find a place_id from a quick centroid lookup
        print("  No cache index entries — using a test place_id from centroid table")
        test_place_id = "US_A10_100050"  # fallback; may not exist

    test_start = "20250301"
    test_end = "20250303"

    print(f"  Pulling: place_id={test_place_id}, {test_start}-{test_end}")

    requests = pd.DataFrame({
        "place_id": [test_place_id],
        "start_date": [test_start],
        "end_date": [test_end],
    }).set_index("place_id")

    try:
        result = client.pull_weather_data(requests)
        print(f"  Result shape: {result.shape}")
        if not result.empty:
            print(f"  Date range: {result['date'].min()} to {result['date'].max()}")
            print(f"  Columns: {result.columns.tolist()[:8]}...")
    except Exception as e:
        print(f"  Pull FAILED: {type(e).__name__}: {e}")
        return

    # Read cache index after
    idx_after = client._load_cache_index()
    print(f"  Cache index entries AFTER pull: {len(idx_after)}")

    new_entries = len(idx_after) - len(idx_before)
    if new_entries > 0:
        print(f"  +{new_entries} new cache index entries (write-back is WORKING)")
    elif not result.empty:
        print("  No new entries despite successful pull — cache was already populated or write-back failed")
    print()


def main():
    print()
    print("griddedweather Cache Diagnostic")
    print("=" * 60)
    print()

    cfg = check_config()

    client = GriddedWeatherClient(cfg)
    print(f"  Client cache_prefix: {client.cache_prefix!r}")
    print(f"  Client s3_manager:   {'configured' if client.s3_manager else 'NONE'}")
    print()

    idx = check_cache_index(client)
    check_cache_file_access(client, idx)
    check_s3_listing(client)
    test_small_pull(client)

    print("=" * 60)
    print("Diagnostic complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
