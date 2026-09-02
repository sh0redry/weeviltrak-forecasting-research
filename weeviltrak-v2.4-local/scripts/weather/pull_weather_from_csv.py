"""
# usage example:
mkdir -p outputs/logs outputs/weather outputs/cache
nohup python -u scripts/weather/pull_weather_from_csv.py \
  --csv time_and_location_data.csv \
  --out_parquet outputs/weather_strict.parquet \
  --resolution 10by10 \
  --chunk_months 2 \
  --grid_date 20250308 \
  --grid_cache outputs/cache/grid_points_10by10_20250308.parquet \
  --save_loc_mapping_csv outputs/loc_place_mapping.csv \
  --summary_csv outputs/weather_strict_summary.csv \

> outputs/logs/pull_weather_strict.log 2>&1 &
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.weevilltrak_config import WeevillTrakConfig
from app.services.database_service import DatabaseManager
from app.services.data_preparation_service import DataPreparationService
from griddedweather.s3_store import S3Manager
from app.settings import setup_logger

# ---------------------------
# Distance helpers
# ---------------------------
def haversine_km_point_to_many(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance (km) from single point to many points."""
    R = 6371.0088
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c


def map_locs_to_nearest_place_id(
    req: pd.DataFrame,
    grid_points: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scheme 1: one mapping per unique loc.
    Returns:
      - loc_map: loc -> place_id (+ centroid + distance_km)
      - req_mapped: req rows with place_id attached
    """
    gp = grid_points[["place_id", "centroid_lat", "centroid_lon"]].drop_duplicates().copy()
    gp["centroid_lat"] = pd.to_numeric(gp["centroid_lat"], errors="coerce")
    gp["centroid_lon"] = pd.to_numeric(gp["centroid_lon"], errors="coerce")
    gp = gp.dropna(subset=["place_id", "centroid_lat", "centroid_lon"]).reset_index(drop=True)

    # Use float32 to reduce memory footprint; compute distances in float64.
    gp_lat = gp["centroid_lat"].to_numpy(dtype=np.float32)
    gp_lon = gp["centroid_lon"].to_numpy(dtype=np.float32)
    gp_pid = gp["place_id"].astype(str).to_numpy()

    locs = req[["loc", "latitude", "longitude"]].drop_duplicates(subset=["loc"]).copy()
    locs["latitude"] = pd.to_numeric(locs["latitude"], errors="coerce")
    locs["longitude"] = pd.to_numeric(locs["longitude"], errors="coerce")
    locs = locs.dropna(subset=["loc", "latitude", "longitude"]).reset_index(drop=True)

    out_rows = []
    for r in locs.itertuples(index=False):
        d = haversine_km_point_to_many(float(r.latitude), float(r.longitude), gp_lat.astype(np.float64), gp_lon.astype(np.float64))
        j = int(np.argmin(d))
        out_rows.append(
            {
                "loc": r.loc,
                "input_lat": float(r.latitude),
                "input_lon": float(r.longitude),
                "place_id": str(gp_pid[j]),
                "centroid_lat": float(gp_lat[j]),
                "centroid_lon": float(gp_lon[j]),
                "distance_km": float(d[j]),
            }
        )

    loc_map = pd.DataFrame(out_rows)
    req_mapped = req.merge(loc_map[["loc", "place_id"]], on="loc", how="left")
    return loc_map, req_mapped


# ---------------------------
# Phase 0: build/cached grid points from Redshift/Spectrum
# ---------------------------
def _grid_one_shot_query(grid_date_yyyymmdd: str) -> str:
    return f"""
    SELECT
        place_id,
        lat AS centroid_lat,
        lon AS centroid_lon
    FROM spectrum_schema.mio171_weather_10x10_pivoted_archive
    WHERE date = '{grid_date_yyyymmdd}'
    """


def _grid_prefix_query(grid_date_yyyymmdd: str, prefix_like: str) -> str:
    return f"""
    SELECT
        place_id,
        lat AS centroid_lat,
        lon AS centroid_lon
    FROM spectrum_schema.mio171_weather_10x10_pivoted_archive
    WHERE date = '{grid_date_yyyymmdd}'
      AND place_id LIKE '{prefix_like}'
    """


def build_grid_points_cache(
    svc,
    grid_date_yyyymmdd: str,
    cache_path: Path,
    prefix_degree_chunking: bool = True,
) -> pd.DataFrame:
    """
    Try one-shot (date-only) grid pull. If it fails (e.g., WLM/QMR abort), fallback to prefix-chunked pulls.
    Cache result to parquet.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        gp = pd.read_parquet(cache_path)
        # Basic sanity
        if {"place_id", "centroid_lat", "centroid_lon"}.issubset(gp.columns) and len(gp) > 0:
            print(f"[INFO] Loaded cached grid points: {cache_path} | rows={len(gp)}")
            return gp
        else:
            print(f"[WARN] Cache exists but invalid; rebuilding: {cache_path}")

    print(f"[INFO] Building grid points from Redshift for date={grid_date_yyyymmdd} ...")

    # 0A: one-shot
    try:
        with svc.db_manager as db:
            conn = db.get_connection()
            gp = pd.read_sql(_grid_one_shot_query(grid_date_yyyymmdd), conn)
        gp = gp.dropna(subset=["place_id", "centroid_lat", "centroid_lon"]).drop_duplicates(subset=["place_id"]).reset_index(drop=True)
        gp.to_parquet(cache_path, index=False)
        print(f"[INFO] Grid points cached (one-shot): {cache_path} | rows={len(gp)}")
        return gp
    except Exception as e:
        print(f"[WARN] One-shot grid query failed; falling back to prefix-chunking. Error: {e}")

    if not prefix_degree_chunking:
        raise RuntimeError("Grid one-shot failed and prefix chunking is disabled; cannot proceed.")

    # 0B: fallback — chunk by place_id prefix using 4-digit latitude integer degree (0000..0090 and -0000..-0090)
    # This reduces per-query scope; we dedupe locally.
    prefixes = []
    for deg in range(0, 91):
        prefixes.append(f"gridpoint_{deg:04d}%")   # e.g. gridpoint_0034%
    for deg in range(0, 91):
        prefixes.append(f"gridpoint_-{deg:04d}%")  # e.g. gridpoint_-0034% (if exists)

    parts = []
    with svc.db_manager as db:
        conn = db.get_connection()
        for i, pref in enumerate(prefixes, start=1):
            try:
                df = pd.read_sql(_grid_prefix_query(grid_date_yyyymmdd, pref), conn)
                if df is not None and not df.empty:
                    parts.append(df)
                print(f"[INFO] Prefix {i}/{len(prefixes)} done: {pref} | rows={0 if df is None else len(df)}")
            except Exception as e:
                # Continue; some prefixes may not exist or may fail independently
                print(f"[WARN] Prefix query failed: {pref} | {e}")
                continue

    if not parts:
        raise RuntimeError("Prefix-chunking produced no grid points; cannot proceed.")

    gp = pd.concat(parts, ignore_index=True)
    gp = gp.dropna(subset=["place_id", "centroid_lat", "centroid_lon"]).drop_duplicates(subset=["place_id"]).reset_index(drop=True)
    gp.to_parquet(cache_path, index=False)
    print(f"[INFO] Grid points cached (prefix-chunked): {cache_path} | rows={len(gp)}")
    return gp


# ---------------------------
# Main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=False, help="Path to time_and_location_data.csv")
    p.add_argument("--out_parquet", required=True, help="Output Parquet file path")
    p.add_argument("--summary_csv", default=None, help="Optional summary CSV path")
    p.add_argument("--save_loc_mapping_csv", default=None, help="Optional loc->place_id mapping CSV path")
    p.add_argument("--resolution", default="10by10", choices=["10by10", "30by30"])
    p.add_argument("--chunk_months", type=int, default=2)
    p.add_argument("--grid_date", default="20250308", help="YYYYMMDD used to build the 10x10 grid list")
    p.add_argument("--grid_cache", default="outputs/cache/grid_points_10by10_20250308.parquet", help="Parquet path for cached grid points")
    return p.parse_args()


def main():
    CONFIG_PATH = PROJECT_ROOT / "app" / "config" / "weeviltrak_v2.3.yml"

    args = parse_args()

    csv_path = "./time_and_location_data.csv"
    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    req = pd.read_csv(csv_path)
    # req = pd.DataFrame(
    # [
    #     {"loc": "pt1", "latitude": 40.46946638846145, "longitude": -74.4237558160298, "start_date": "2020-05-19", "end_date": "2020-07-08"},
    #     {"loc": "pt1", "latitude": 40.46946638846145, "longitude": -74.4237558160298, "start_date": "2021-05-24", "end_date": "2021-09-04"},
    #     {"loc": "pt1", "latitude": 40.46946638846145, "longitude": -74.4237558160298, "start_date": "2022-05-14", "end_date": "2022-08-15"},
    #     {"loc": "pt1", "latitude": 40.46946638846145, "longitude": -74.4237558160298, "start_date": "2023-05-17", "end_date": "2023-07-21"},
    # ]
    # )
    required = {"loc", "latitude", "longitude", "start_date", "end_date"}
    missing = required - set(req.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Parse dates (inclusive end)
    req["start_date"] = pd.to_datetime(req["start_date"], utc=True).dt.date
    req["end_date"] = pd.to_datetime(req["end_date"], utc=True).dt.date

    global_start = req["start_date"].min()
    global_end = req["end_date"].max()

    print(f"[INFO] Requests={len(req)} | unique locs={req['loc'].nunique()}")
    print(f"[INFO] Global date range: {global_start} .. {global_end} (inclusive)")


    # Build service + load spatial coverage grid
    config = WeevillTrakConfig(config_path=CONFIG_PATH)  # if needed
    db_manager = DatabaseManager(config=config)  #
    bucket_name = config.config.get("data_source", {}).get("s3_bucket", "sps-ds-bucket")

    s3_manager = S3Manager(bucket_name)
    svc = DataPreparationService(db_manager=db_manager,s3_manager=s3_manager)

    # Phase 0: build/load grid points cache (place_id + centroid)
    grid_cache = Path(args.grid_cache)
    grid_points = build_grid_points_cache(
        svc=svc,
        grid_date_yyyymmdd=args.grid_date,
        cache_path=grid_cache,
        prefix_degree_chunking=True,
    )

    # Phase 1: loc -> nearest place_id (distance; no rounding of input lat/lon)
    loc_map, req_mapped = map_locs_to_nearest_place_id(req=req, grid_points=grid_points)

    if args.save_loc_mapping_csv:
        p = Path(args.save_loc_mapping_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        loc_map.to_csv(p, index=False)
        print(f"[INFO] Wrote loc mapping CSV: {p}")

    unique_place_ids = sorted(loc_map["place_id"].unique().tolist())
    print(f"[INFO] Unique place_ids needed: {len(unique_place_ids)}")

    # Phase 2: pull weather by place_id (fast path)
    pull_start = pd.to_datetime(global_start).strftime("%Y%m%d")
    pull_end = pd.to_datetime(global_end).strftime("%Y%m%d")

    # Parallel pull
    weather = svc._parallel_pull_weather_data_by_place_id_streaming(
    place_ids=unique_place_ids,
    start_date=pull_start,
    end_date=pull_end,
    resolution="10by10",
    chunk_months=2,
    tmp_dir="outputs/weather_tmp",
    n_processes=4,
    maxtasksperchild=25,
    cleanup_tmp=False,
)
    # Basic check
    if weather is None or weather.empty:
        print("[WARN] Weather pull returned empty.")
        pd.DataFrame().to_parquet(out_parquet, index=False)
        return
    # Parse date
    weather["date"] = pd.to_datetime(weather["date"]).dt.date

    # Phase 3: strict output (per-row window inclusive)
    # Scheme 1: each loc maps to one place_id, but multiple windows per loc can exist in CSV.
    windows = req_mapped[["loc", "place_id", "start_date", "end_date"]].copy()
    merged = weather.merge(windows, on="place_id", how="inner")
    strict = merged[(merged["date"] >= merged["start_date"]) & (merged["date"] <= merged["end_date"])].copy()

    # Attach loc->distance/inputs for audit (optional but useful)
    strict = strict.merge(
        loc_map[["loc", "input_lat", "input_lon", "centroid_lat", "centroid_lon", "distance_km", "place_id"]],
        on=["loc", "place_id"],
        how="left",
        suffixes=("", "_mapped"),
    )

    strict.to_parquet(out_parquet, index=False)
    print(f"[INFO] Wrote strict output: {out_parquet} | rows={len(strict)}")

    if args.summary_csv:
        s = Path(args.summary_csv)
        s.parent.mkdir(parents=True, exist_ok=True)
        summary = (
            strict.groupby("loc", as_index=False)
            .agg(
                n_rows=("date", "size"),
                min_date=("date", "min"),
                max_date=("date", "max"),
                n_place_ids=("place_id", "nunique"),
            )
            .merge(loc_map[["loc", "place_id", "centroid_lat", "centroid_lon", "distance_km"]], on="loc", how="left")
        )
        summary.to_csv(s, index=False)
        print(f"[INFO] Wrote summary CSV: {s} | rows={len(summary)}")


if __name__ == "__main__":
    main()
