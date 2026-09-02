"""Quick inspection of existing cache files to understand structure and continuity."""
import io
import re
import boto3
import pandas as pd
import pyarrow.parquet as pq

BUCKET = "sps-ds-bucket"
PREFIX = "gridded-weather/cache/10by10/"
RANGE_RE = re.compile(r"(.+)-(\d{8})-(\d{8})\.parquet$")

s3 = boto3.client("s3")

# List all parquet keys
paginator = s3.get_paginator("list_objects_v2")
keys = []
for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
    for obj in page.get("Contents", []):
        k = obj["Key"]
        if k.endswith(".parquet") and "cache_index" not in k:
            keys.append(k)
keys.sort()
print(f"Total parquet files: {len(keys)}")

# Categorize
range_files = []
other_files = []
for k in keys:
    fname = k.split("/")[-1]
    m = RANGE_RE.match(fname)
    if m:
        range_files.append((k, m.group(1), m.group(2), m.group(3)))
    else:
        other_files.append(k)

print(f"Range files (place-start-end): {len(range_files)}")
print(f"Other files: {len(other_files)}")
for o in other_files:
    print(f"  {o}")

# Inspect a sample range file for structure
if range_files:
    sample_key = range_files[0][0]
    print(f"\n--- Sample file: {sample_key} ---")
    resp = s3.get_object(Bucket=BUCKET, Key=sample_key)
    df = pq.read_table(io.BytesIO(resp["Body"].read())).to_pandas()
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Dtypes:\n{df.dtypes}")
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"]).drop_duplicates().sort_values().reset_index(drop=True)
        print(f"\nDate range: {dates.min().date()} to {dates.max().date()}")
        print(f"Unique dates: {len(dates)}")
        diffs = dates.diff().dropna()
        gaps = diffs[diffs > pd.Timedelta(days=1)]
        print(f"Internal gaps: {len(gaps)}")
        for i, g in gaps.items():
            prev = dates[i - 1]
            curr = dates[i]
            print(f"  {prev.date()} -> {curr.date()} ({(curr - prev).days} days)")
    if "place_id" in df.columns:
        print(f"Unique place_ids: {df['place_id'].unique().tolist()}")

# Spot-check continuity across ALL range files (summary only)
print("\n--- Continuity check across all range files ---")
files_with_gaps = 0
files_ok = 0
for key, pid, start, end in range_files:
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    df = pq.read_table(io.BytesIO(resp["Body"].read())).to_pandas()
    if "date" not in df.columns:
        print(f"  SKIP (no date col): {key}")
        continue
    dates = pd.to_datetime(df["date"]).drop_duplicates().sort_values().reset_index(drop=True)
    diffs = dates.diff().dropna()
    gaps = diffs[diffs > pd.Timedelta(days=1)]
    if len(gaps) > 0:
        files_with_gaps += 1
        print(f"  GAPS in {key}: {len(gaps)} gap(s)")
        for i, g in gaps.items():
            prev = dates[i - 1]
            curr = dates[i]
            print(f"    {prev.date()} -> {curr.date()} ({(curr - prev).days} days)")
    else:
        files_ok += 1

print(f"\nSummary: {files_ok} files OK, {files_with_gaps} files with gaps")
