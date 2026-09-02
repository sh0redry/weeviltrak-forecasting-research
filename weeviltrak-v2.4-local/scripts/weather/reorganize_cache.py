"""Reorganize S3 cache: split gapped range files into continuous segments, rebuild index.

Existing files are named  place_id-YYYYMMDD-YYYYMMDD.parquet  and may contain
internal date gaps.  This script:
  1. Reads each range file, detects continuous date segments.
  2. Files that are already continuous are left untouched.
  3. Files with gaps are split into one file per continuous segment.
  4. Old gapped files are deleted; new segment files are uploaded.
  5. cache_index.csv is rewritten to match the final file set.
"""

import argparse
import io
import re
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import pandas as pd
import pyarrow.parquet as pq

DEFAULT_BUCKET = "sps-ds-bucket"
DEFAULT_PREFIX = "gridded-weather/cache/10by10/"
DEFAULT_INDEX_KEY = DEFAULT_PREFIX + "cache_index.csv"
RANGE_RE = re.compile(r"(.+)-(\d{8})-(\d{8})\.parquet$")


# ── helpers ──────────────────────────────────────────────────────────────────


def list_parquet_keys(s3_client, bucket: str, prefix: str) -> List[str]:
    keys: List[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet") and "cache_index" not in key:
                keys.append(key)
    return sorted(keys)


def read_parquet_bytes(s3_client, bucket: str, key: str) -> pd.DataFrame:
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(resp["Body"].read()))


def find_continuous_segments(dates: pd.Series) -> List[Tuple[date, date]]:
    """Return (start, end) pairs for each run of consecutive dates."""
    dates_sorted = dates.drop_duplicates().sort_values().reset_index(drop=True)
    dates_dt = pd.to_datetime(dates_sorted)
    if dates_dt.empty:
        return []
    segments: List[Tuple[date, date]] = []
    start = dates_dt.iloc[0]
    prev = start
    for d in dates_dt.iloc[1:]:
        if (d - prev).days == 1:
            prev = d
        else:
            segments.append((start.date(), prev.date()))
            start = d
            prev = d
    segments.append((start.date(), prev.date()))
    return segments


def segment_key(prefix: str, place_id: str, start: date, end: date) -> str:
    fname = f"{place_id}-{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}.parquet"
    return f"{prefix}{fname}"


def delete_keys(s3_client, bucket: str, keys: List[str]):
    if not keys:
        return
    batch = []
    for k in keys:
        batch.append({"Key": k})
        if len(batch) == 1000:
            s3_client.delete_objects(Bucket=bucket, Delete={"Objects": batch})
            batch = []
    if batch:
        s3_client.delete_objects(Bucket=bucket, Delete={"Objects": batch})


def write_index(s3_client, bucket: str, index_key: str, rows: List[dict]):
    df = pd.DataFrame(rows)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    s3_client.put_object(Bucket=bucket, Key=index_key, Body=csv_bytes)


# ── main logic ───────────────────────────────────────────────────────────────


def reorganize(bucket: str, prefix: str, index_key: str, apply: bool, local_dir: Optional[str] = None):
    s3 = boto3.client("s3")
    tmp = Path(local_dir) if local_dir else Path(tempfile.mkdtemp(prefix="cache_reorg_"))

    print(f"Listing parquet files under s3://{bucket}/{prefix} ...")
    all_keys = list_parquet_keys(s3, bucket, prefix)
    print(f"Found {len(all_keys)} parquet files")

    # Parse range files vs non-range files
    range_files: List[Tuple[str, str]] = []  # (key, place_id)
    other_files: List[str] = []
    for key in all_keys:
        fname = key.rsplit("/", 1)[-1]
        m = RANGE_RE.match(fname)
        if m:
            range_files.append((key, m.group(1)))
        else:
            other_files.append(key)

    if other_files:
        print(f"Non-range files (will be deleted): {len(other_files)}")
        for o in other_files:
            print(f"  {o}")

    # Analyse each range file
    files_ok = 0
    files_with_gaps = 0
    keys_to_delete: List[str] = list(other_files)  # always remove non-range files
    keys_to_upload: List[Tuple[str, Path]] = []  # (s3_key, local_path)
    index_rows: List[dict] = []

    print(f"\nAnalysing {len(range_files)} range files for continuity...")
    for key, place_id in range_files:
        df = read_parquet_bytes(s3, bucket, key)
        if "date" not in df.columns:
            print(f"  SKIP (no date column): {key}")
            keys_to_delete.append(key)
            continue

        segments = find_continuous_segments(df["date"])
        if not segments:
            print(f"  SKIP (empty): {key}")
            keys_to_delete.append(key)
            continue

        if len(segments) == 1:
            # Already continuous — keep as-is, ensure key matches expected name
            seg_start, seg_end = segments[0]
            expected_key = segment_key(prefix, place_id, seg_start, seg_end)
            if key == expected_key:
                files_ok += 1
                index_rows.append({
                    "place_id": place_id,
                    "start_compact": seg_start.strftime("%Y%m%d"),
                    "end_compact": seg_end.strftime("%Y%m%d"),
                    "key": key,
                })
            else:
                # Continuous but mis-named — rewrite with correct name
                files_with_gaps += 1
                local_path = tmp / expected_key.rsplit("/", 1)[-1]
                df.to_parquet(local_path, index=False)
                keys_to_upload.append((expected_key, local_path))
                keys_to_delete.append(key)
                index_rows.append({
                    "place_id": place_id,
                    "start_compact": seg_start.strftime("%Y%m%d"),
                    "end_compact": seg_end.strftime("%Y%m%d"),
                    "key": expected_key,
                })
        else:
            # Has gaps — split into per-segment files
            files_with_gaps += 1
            keys_to_delete.append(key)
            df["_date_dt"] = pd.to_datetime(df["date"])
            for seg_start, seg_end in segments:
                mask = (df["_date_dt"].dt.date >= seg_start) & (df["_date_dt"].dt.date <= seg_end)
                seg_df = df.loc[mask].drop(columns=["_date_dt"]).copy()
                new_key = segment_key(prefix, place_id, seg_start, seg_end)
                local_path = tmp / new_key.rsplit("/", 1)[-1]
                seg_df.to_parquet(local_path, index=False)
                keys_to_upload.append((new_key, local_path))
                index_rows.append({
                    "place_id": place_id,
                    "start_compact": seg_start.strftime("%Y%m%d"),
                    "end_compact": seg_end.strftime("%Y%m%d"),
                    "key": new_key,
                })
            df.drop(columns=["_date_dt"], inplace=True)

    # Don't delete keys that are also being uploaded (already-correct files)
    upload_keys = {k for k, _ in keys_to_upload}
    keep_keys = {r["key"] for r in index_rows} - upload_keys
    keys_to_delete = [k for k in keys_to_delete if k not in keep_keys]

    print(f"\n{'='*60}")
    print(f"Files already continuous:   {files_ok}")
    print(f"Files needing split/rename: {files_with_gaps}")
    print(f"New segment files to upload: {len(keys_to_upload)}")
    print(f"Old files to delete:         {len(keys_to_delete)}")
    print(f"Total index rows:            {len(index_rows)}")

    if not apply:
        print("\nFiles to upload:")
        for s3_key, _ in sorted(keys_to_upload):
            print(f"  + {s3_key}")
        print("\nFiles to delete:")
        for k in sorted(keys_to_delete):
            print(f"  - {k}")
        print("\nDry run only. Use --apply to perform changes.")
        return

    # Upload new segment files
    print(f"\nUploading {len(keys_to_upload)} segment files...")
    for i, (s3_key, local_path) in enumerate(keys_to_upload, 1):
        s3.upload_file(str(local_path), bucket, s3_key)
        if i % 50 == 0 or i == len(keys_to_upload):
            print(f"  uploaded {i}/{len(keys_to_upload)}")

    # Delete old files
    print(f"Deleting {len(keys_to_delete)} old files...")
    delete_keys(s3, bucket, keys_to_delete)

    # Rewrite index
    print("Writing updated cache_index.csv...")
    write_index(s3, bucket, index_key, index_rows)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize cached weather parquet files: split gapped files into continuous segments, rebuild index."
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--index-key", default=DEFAULT_INDEX_KEY, dest="index_key")
    parser.add_argument("--apply", action="store_true", help="Perform writes/deletes; otherwise dry run")
    parser.add_argument("--local-dir", default=None, help="Local directory for temp parquet output")
    args = parser.parse_args()

    reorganize(
        bucket=args.bucket,
        prefix=args.prefix,
        index_key=args.index_key,
        apply=args.apply,
        local_dir=args.local_dir,
    )


if __name__ == "__main__":
    main()
