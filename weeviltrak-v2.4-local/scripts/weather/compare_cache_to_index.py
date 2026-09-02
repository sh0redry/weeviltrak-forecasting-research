import argparse
import io
import re
from datetime import datetime
from typing import Dict, Iterable, List, Set

import boto3
import pandas as pd
import pyarrow.parquet as pq

DEFAULT_BUCKET = "sps-ds-bucket"
DEFAULT_PREFIX = "gridded-weather/cache/10by10/"
DEFAULT_INDEX_KEY = DEFAULT_PREFIX + "cache_index.csv"


DATE_RE = re.compile(r"_(\d{8})\.parquet$")


def list_parquet_keys(s3_client, bucket: str, prefix: str) -> List[str]:
    keys: List[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if key.endswith(".parquet") and "cache_index" not in key:
                keys.append(key)
    return keys


def read_parquet_object(s3_client, bucket: str, key: str, columns: Iterable[str] | None = None) -> pd.DataFrame:
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    body = io.BytesIO(resp["Body"].read())
    table = pq.read_table(body, columns=columns)
    return table.to_pandas()


def load_cache_index(s3_client, bucket: str, key: str) -> pd.DataFrame:
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(resp["Body"])


def build_actual_dates(s3_client, bucket: str, keys: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for key in keys:
        m = DATE_RE.search(key)
        if not m:
            continue
        date_str = m.group(1)
        date_val = datetime.strptime(date_str, "%Y%m%d").date()

        df = read_parquet_object(s3_client, bucket, key, columns=["place_id"])
        if df.empty or "place_id" not in df.columns:
            continue

        df = df[["place_id"]].dropna().copy()
        df["date"] = date_val
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["place_id", "date"])

    actual = pd.concat(frames, ignore_index=True)
    actual = actual.drop_duplicates()
    return actual


def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.lower() for c in df.columns})

    # cache_index has: place_id, start_compact (YYYYMMDD), end_compact, key
    if "place_id" not in df.columns or "start_compact" not in df.columns or "end_compact" not in df.columns:
        raise ValueError("cache_index.csv missing expected columns: place_id, start_compact, end_compact")

    df["start_date"] = pd.to_datetime(df["start_compact"].astype(str)).dt.date
    df["end_date"] = pd.to_datetime(df["end_compact"].astype(str)).dt.date
    df["place_id"] = df["place_id"].astype(str)

    return df[["place_id", "start_date", "end_date"]].drop_duplicates()


def expected_date_set(start_date, end_date) -> Set:
    return set(pd.date_range(start_date, end_date, freq="D").date)


def compare(actual: pd.DataFrame, index_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    actual_groups = actual.groupby("place_id")

    for _, row in index_df.iterrows():
        pid = row["place_id"]
        expected = expected_date_set(row["start_date"], row["end_date"])
        actual_dates = set(actual_groups.get_group(pid)["date"]) if pid in actual_groups.groups else set()
        missing = sorted(expected - actual_dates)
        results[pid] = {
            "expected_dates": len(expected),
            "actual_dates": len(actual_dates),
            "missing_count": len(missing),
            "missing_dates": missing,
        }

    extra_keys = set(actual["place_id"].unique()) - set(index_df["place_id"].unique())
    if extra_keys:
        results["__extra_keys__"] = {"keys": sorted(extra_keys)}
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare cached weather parquet files against cache_index.csv")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--index-key", default=DEFAULT_INDEX_KEY, dest="index_key")
    args = parser.parse_args()

    s3 = boto3.client("s3")

    print(f"Listing parquet files under s3://{args.bucket}/{args.prefix} ...")
    parquet_keys = list_parquet_keys(s3, args.bucket, args.prefix)
    print(f"Found {len(parquet_keys)} parquet files")

    print(f"Loading index: s3://{args.bucket}/{args.index_key}")
    idx_raw = load_cache_index(s3, args.bucket, args.index_key)
    idx = normalize_index(idx_raw)
    print(f"Index rows: {len(idx)}")

    print("Loading parquet data (date + place_id)...")
    actual = build_actual_dates(s3, args.bucket, parquet_keys)
    print(f"Rows loaded: {len(actual)} across {actual['place_id'].nunique()} place_id entries")

    print("Comparing expected vs actual dates...")
    results = compare(actual, idx)

    missing_summary = [
        (k, v["missing_count"], v["expected_dates"], v["actual_dates"])
        for k, v in results.items()
        if k != "__extra_keys__"
    ]
    missing_summary.sort(key=lambda t: t[1], reverse=True)

    print("\nTop missing-date keys (location_year_key | missing/expected | actual):")
    for k, miss, expected, actual_dates in missing_summary[:20]:
        print(f"  {k}: missing {miss} / expected {expected} (actual {actual_dates})")

    if "__extra_keys__" in results:
        print("\nKeys present in parquet but not in index:")
        for k in results["__extra_keys__"]["keys"]:
            print(f"  {k}")

    # Optionally dump detailed missing dates for debugging
    print("\nDetailed missing dates (only keys with gaps):")
    for k, v in results.items():
        if k == "__extra_keys__" or v["missing_count"] == 0:
            continue
        print(f"\n{k}: missing {v['missing_count']} dates")
        print(", ".join(d.strftime("%Y-%m-%d") for d in v["missing_dates"]))


if __name__ == "__main__":
    main()
