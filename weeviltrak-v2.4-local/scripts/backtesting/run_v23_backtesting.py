"""
Run strict walk-forward backtesting for the WeevilTrak v2.3 model.

This runner intentionally reuses ``BacktestingFramework`` instead of the
hindcast pipeline. Backtesting retrains one v2.3 LightGBM model per prediction
year using only data before Jan 1 of that year, then evaluates historical
predictions against actual stage dates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.backtesting import BacktestingFramework
from app.settings import setup_logger

logger = setup_logger()

DEFAULT_CONFIG_PATH = "app/config/weeviltrak_v2.3.yml"
DEFAULT_OUTPUT_DIR = "outputs/backtesting/v2.3"
DEFAULT_YEARS = [2023, 2024, 2025]
DEFAULT_STAGES = [1, 2, 3]
DEFAULT_LEAD_DAYS = 7


def _parse_int_list(raw_values: Iterable[str] | None, default: list[int]) -> list[int]:
    if not raw_values:
        return default

    parsed: list[int] = []
    for raw in raw_values:
        for part in raw.split(","):
            part = part.strip()
            if part:
                parsed.append(int(part))

    if not parsed:
        return default
    return list(dict.fromkeys(parsed))


def run_v23_backtesting(
    config_path: str = DEFAULT_CONFIG_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    years: list[int] | None = None,
    stages: list[int] | None = None,
    lead_days: int = DEFAULT_LEAD_DAYS,
    training_window_years: int | None = None,
    upload_to_s3: bool = True,
    delete_local: bool = False,
) -> str:
    """
    Run v2.3 strict backtesting and save both prediction and evaluation outputs.

    Args:
        config_path: v2.3 YAML config path.
        output_dir: Local directory for backtesting artifacts.
        years: Prediction years to evaluate.
        stages: Stage IDs to evaluate.
        lead_days: Forecast lead time in days.
        training_window_years: None for expanding window; int for rolling window.
        upload_to_s3: Whether to upload final prediction/evaluation artifacts.
        delete_local: Whether to delete the final local prediction file after upload.

    Returns:
        S3 path if predictions were uploaded, otherwise the local prediction path.
    """
    years = years or DEFAULT_YEARS
    stages = stages or DEFAULT_STAGES

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Starting strict v2.3 backtesting")
    logger.info("Config: %s", config_path)
    logger.info("Output dir: %s", output_dir)
    logger.info(
        "years=%s stages=%s lead_days=%s training_window_years=%s",
        years,
        stages,
        lead_days,
        training_window_years,
    )

    backtester = BacktestingFramework(
        config_path=config_path,
        output_dir=output_dir,
    )
    predictions_path = backtester.run_backtest(
        years=years,
        stages=stages,
        lead_days=lead_days,
        training_window_years=training_window_years,
        upload_to_s3=upload_to_s3,
        delete_local=delete_local,
    )

    if predictions_path is None:
        raise RuntimeError("Backtesting produced no predictions path.")

    backtester.evaluate_7day_forecast_error(
        predictions_file=predictions_path,
        years=years,
        stages=stages,
        lead_days=lead_days,
    )

    logger.info("v2.3 backtesting complete. Predictions: %s", predictions_path)
    return predictions_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run strict walk-forward backtesting for WeevilTrak v2.3."
    )
    parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to v2.3 YAML config. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Local output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--year",
        dest="years",
        action="append",
        help="Prediction year to evaluate. Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--stage",
        dest="stages",
        action="append",
        help="Stage ID to evaluate. Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--lead-days",
        type=int,
        default=DEFAULT_LEAD_DAYS,
        help=f"Forecast lead time in days. Default: {DEFAULT_LEAD_DAYS}",
    )
    parser.add_argument(
        "--training-window-years",
        type=int,
        help="Use a rolling last-N-years training window. Omit for expanding window.",
    )
    parser.add_argument(
        "--no-upload-to-s3",
        action="store_true",
        help="Keep outputs local and skip S3 upload.",
    )
    parser.add_argument(
        "--delete-local",
        action="store_true",
        help="Delete the final local prediction file after successful S3 upload.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    predictions_path = run_v23_backtesting(
        config_path=args.config_path,
        output_dir=args.output_dir,
        years=_parse_int_list(args.years, DEFAULT_YEARS),
        stages=_parse_int_list(args.stages, DEFAULT_STAGES),
        lead_days=args.lead_days,
        training_window_years=args.training_window_years,
        upload_to_s3=not args.no_upload_to_s3,
        delete_local=args.delete_local,
    )

    print("DONE")
    print(f"PREDICTIONS={predictions_path}")
    print(f"OUTPUT_DIR={args.output_dir}")


if __name__ == "__main__":
    main()
