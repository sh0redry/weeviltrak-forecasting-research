"""
Export Redshift-backed WeevilTrak label data for offline backtesting.

This script is intentionally narrow: it only pulls processed weevil/stage-date
data through one or more cutoff dates and writes local parquet/csv files. It
does not pull weather, train models, run predictions, or upload anything.

Use this on a machine that can connect to Redshift, then copy the output
directory back to an offline analysis machine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.base import PipelineBase
from app.settings import setup_logger

logger = setup_logger()

DEFAULT_CONFIG_PATH = "app/config/weeviltrak_v2.4.yml"
DEFAULT_OUTPUT_DIR = (
    "backtesting_results/trajectory_v23_v24_3day/weevil_data_cache"
)
DEFAULT_CUTOFFS = [
    "2022-01-01",
    "2023-01-01",
    "2024-01-01",
    "2025-01-01",
]


def _parse_csv_values(values: Iterable[str] | None) -> list[str]:
    parsed: list[str] = []
    for value in values or []:
        for part in value.split(","):
            part = part.strip()
            if part:
                parsed.append(part)
    return list(dict.fromkeys(parsed))


def export_weevil_data_for_cutoffs(
    config_path: str = DEFAULT_CONFIG_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    cutoffs: list[str] | None = None,
    write_csv: bool = True,
    overwrite: bool = False,
) -> list[Path]:
    """
    Pull and save processed weevil data through each cutoff date.

    The output file names match the offline trajectory notebook cache convention:
    ``weevil_data_through_YYYY-MM-DD.parquet``.
    """
    cutoffs = cutoffs or DEFAULT_CUTOFFS
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline = PipelineBase(config_path=config_path)
    written: list[Path] = []

    for cutoff in cutoffs:
        parquet_path = output_path / f"weevil_data_through_{cutoff}.parquet"
        csv_path = output_path / f"weevil_data_through_{cutoff}.csv"
        if parquet_path.exists() and not overwrite:
            logger.info("Skipping existing cache: %s", parquet_path)
            written.append(parquet_path)
            continue

        logger.info("Pulling weevil data through cutoff=%s", cutoff)
        data = pipeline.data_service.pull_weevil_data(today=cutoff)
        data.to_parquet(parquet_path, index=False)
        if write_csv:
            data.to_csv(csv_path, index=False)
        logger.info(
            "Saved cutoff=%s rows=%s parquet=%s",
            cutoff,
            len(data),
            parquet_path,
        )
        written.append(parquet_path)

    manifest_path = output_path / "weevil_data_cache_manifest.csv"
    manifest_rows = [
        {
            "cutoff": path.stem.replace("weevil_data_through_", ""),
            "parquet_path": str(path),
            "csv_path": str(path.with_suffix(".csv")) if write_csv else "",
        }
        for path in written
    ]
    if manifest_rows:
        import pandas as pd

        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        logger.info("Saved manifest: %s", manifest_path)

    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export processed WeevilTrak Redshift label data for offline "
            "trajectory/v2.4 termination testing."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cutoff",
        action="append",
        help=(
            "Cutoff date(s), e.g. --cutoff 2022-01-01 --cutoff 2023-01-01. "
            "Comma-separated values are also accepted. Defaults to 2022-2025 "
            "walk-forward cutoffs."
        ),
    )
    parser.add_argument(
        "--include-production-cutoff",
        default=None,
        help=(
            "Optional extra cutoff for full training, e.g. 2026-06-11. This is "
            "not needed for the current offline v2.4 notebook."
        ),
    )
    parser.add_argument("--no-csv", action="store_true", help="Only write parquet.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cutoffs = _parse_csv_values(args.cutoff) or DEFAULT_CUTOFFS.copy()
    if args.include_production_cutoff:
        cutoffs.extend(_parse_csv_values([args.include_production_cutoff]))
        cutoffs = list(dict.fromkeys(cutoffs))

    written = export_weevil_data_for_cutoffs(
        config_path=args.config,
        output_dir=args.output_dir,
        cutoffs=cutoffs,
        write_csv=not args.no_csv,
        overwrite=args.overwrite,
    )
    print("Export complete. Files:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
