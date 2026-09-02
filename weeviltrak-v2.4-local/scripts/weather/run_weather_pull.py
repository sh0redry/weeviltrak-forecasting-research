"""
Run a weather-only pull with logging, mirroring the notebook defaults.
- Loads .env
- Pulls weevil data up to --date
- Samples every Nth row (default 10, same as notebook slice weevil_data[::10])
- Calls query_weather_data_for_weeviltrak_locs
- Prints summary and optionally writes parquet
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.train_predict import WeevilTrakPipeline  # noqa: E402
from app.settings import setup_logger  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Weather pull for WeevilTrak locations")
    parser.add_argument(
        "--date",
        default=datetime.today().strftime("%Y-%m-%d"),
        help="Cutoff date for weevil + weather pulls (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=10,
        help="Row stride for weevil_data sampling (default 10 matches notebook weevil_data[::10])",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on sampled weevil rows (e.g., 10 to test first 10 rows)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write weather_data as parquet",
    )
    return parser.parse_args()


def load_env():
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
        print(f"Loaded .env from {dotenv_path}")
    else:
        print(f"No .env found at {dotenv_path}")


if __name__ == "__main__":
    args = parse_args()
    load_env()

    logger = setup_logger()

    config_path = PROJECT_ROOT / "app" / "config" / "weeviltrak_v2.3.yml"
    logger.info("Config path: %s", config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    pipeline = WeevilTrakPipeline(config_path=str(config_path))

    t0 = time.time()
    weevil_data = pipeline.data_service.pull_weevil_data(today=args.date)
    logger.info(
        "Weevil data pulled: rows=%d, cols=%d, years=%s-%s, locations=%d",
        len(weevil_data),
        len(weevil_data.columns),
        weevil_data["year"].min() if not weevil_data.empty else None,
        weevil_data["year"].max() if not weevil_data.empty else None,
        weevil_data["location_id"].nunique() if not weevil_data.empty else 0,
    )

    if args.sample_step and args.sample_step > 1:
        weevil_data = weevil_data.iloc[:: args.sample_step].copy()
        logger.info("Sampled weevil_data every %d rows -> %d rows", args.sample_step, len(weevil_data))

    if args.max_rows is not None and args.max_rows > 0:
        weevil_data = weevil_data.head(args.max_rows).copy()
        logger.info("Capped weevil_data to first %d rows", len(weevil_data))

    t1 = time.time()
    weather_data = pipeline.data_service.query_weather_data_for_weeviltrak_locs(
        pipeline.config,
        args.date,
        weevil_data=weevil_data,
    )
    t2 = time.time()
    logger.info(
        "Weather data pulled: rows=%d, cols=%d, date_range=%s -> %s",
        len(weather_data),
        len(weather_data.columns),
        weather_data["date"].min() if not weather_data.empty else None,
        weather_data["date"].max() if not weather_data.empty else None,
    )

    if args.output and not weather_data.empty:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        weather_data.to_parquet(out_path, index=False)
        logger.info("Wrote weather_data to %s", out_path)

    logger.info("Timing: weevil pull %.1fs | weather pull %.1fs | total %.1fs", t1 - t0, t2 - t1, t2 - t0)
    logger.info("Done")
