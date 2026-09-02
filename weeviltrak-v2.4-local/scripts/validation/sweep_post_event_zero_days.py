#!/usr/bin/env python3
"""
Sweep ``post_event_zero_days`` values and save one Markdown comparison report.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.monotonic_business_check import MonotonicBusinessCheck


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"
    return df.to_markdown(index=False)


def _summarize_run(
    summary_csv: Path,
    post_event_zero_days: int,
) -> pd.DataFrame:
    summary = pd.read_csv(summary_csv).dropna(subset=["model_version"]).copy()
    summary["post_event_zero_days"] = post_event_zero_days
    cols = [
        "post_event_zero_days",
        "model_version",
        "n_event_windows",
        "n_prediction_rows",
        "bounce_back_count",
        "bounce_back_rate",
        "post_event_positive_count",
        "post_event_positive_rate",
    ]
    return summary[cols]


def build_report(
    runs_df: pd.DataFrame,
    report_path: Path,
    today: str,
    cutoff_date: str,
    rf_config_path: str,
    lgbm_config_path: str,
    values: List[int],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    v21 = runs_df[runs_df["model_version"] == "v2.1"].copy()
    v22 = runs_df[runs_df["model_version"] == "v2.2"].copy()

    best_post = (
        v22.sort_values(
            ["post_event_positive_count", "post_event_positive_rate", "bounce_back_count"]
        ).head(1)
    )
    best_bounce = (
        v22.sort_values(
            ["bounce_back_count", "bounce_back_rate", "post_event_positive_count"]
        ).head(1)
    )

    lines = [
        "# Post-Event Zero Days Sweep Report",
        "",
        "## Run Setup",
        "",
        f"- Generated at: `{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC`",
        f"- `today`: `{today}`",
        f"- `cutoff_date`: `{cutoff_date}`",
        f"- RF config: `{rf_config_path}`",
        f"- LGBM config base: `{lgbm_config_path}`",
        f"- Swept `post_event_zero_days`: `{values}`",
        "",
        "## Full Results",
        "",
        _markdown_table(runs_df),
        "",
        "## v2.2 Only",
        "",
        _markdown_table(v22),
        "",
        "## Quick Read",
        "",
    ]

    if not best_post.empty:
        row = best_post.iloc[0]
        lines.extend(
            [
                (
                    f"- Best `post_event_positive_count`: "
                    f"`post_event_zero_days={int(row['post_event_zero_days'])}` "
                    f"with count `{int(row['post_event_positive_count'])}` "
                    f"and rate `{row['post_event_positive_rate']:.6f}`."
                )
            ]
        )

    if not best_bounce.empty:
        row = best_bounce.iloc[0]
        lines.extend(
            [
                (
                    f"- Best `bounce_back_count`: "
                    f"`post_event_zero_days={int(row['post_event_zero_days'])}` "
                    f"with count `{int(row['bounce_back_count'])}` "
                    f"and rate `{row['bounce_back_rate']:.6f}`."
                )
            ]
        )

    if not v21.empty:
        lines.extend(
            [
                "",
                "## Notes",
                "",
                "- `v2.1` is shown for reference because the training-data augmentation affects both model families.",
                "- The best practical setting should usually prioritize lower `post_event_positive_count` first, then lower `bounce_back_count`.",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep post_event_zero_days and save one Markdown comparison report."
    )
    parser.add_argument(
        "--values",
        nargs="+",
        type=int,
        required=True,
        help="Values to test, e.g. --values 3 5 7 10",
    )
    parser.add_argument(
        "--today",
        required=True,
        help="Upper bound passed to pull_weevil_data(), e.g. 2025-06-01",
    )
    parser.add_argument(
        "--cutoff-date",
        required=True,
        help="Fixed training cutoff date for all runs.",
    )
    parser.add_argument(
        "--rf-config-path",
        default="app/config/weeviltrak_v2.1.yml",
        help="RF baseline config path.",
    )
    parser.add_argument(
        "--lgbm-config-path",
        default="app/config/weeviltrak_v2.3.yml",
        help="Base LightGBM config path.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/business_checks/sweep",
        help="Root directory for per-run business check outputs.",
    )
    parser.add_argument(
        "--report-path",
        default="docs/implemention_results/post_event_zero_days_sweep_report.md",
        help="Single Markdown report path.",
    )
    parser.add_argument(
        "--stage-id",
        type=int,
        default=2,
        help="Stage ID to inspect. Default is 2.",
    )
    args = parser.parse_args()

    base_lgbm_config = _load_yaml(Path(args.lgbm_config_path))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[pd.DataFrame] = []

    with tempfile.TemporaryDirectory(prefix="post_event_zero_days_sweep_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        for value in args.values:
            config_data = dict(base_lgbm_config)
            config_data["post_event_zero_days"] = int(value)

            tmp_config_path = tmpdir_path / f"weeviltrak_v2_2_post_event_{value}.yml"
            _write_yaml(tmp_config_path, config_data)

            run_output_dir = output_root / f"post_event_zero_days_{value}"
            checker = MonotonicBusinessCheck(
                rf_config_path=args.rf_config_path,
                lgbm_config_path=str(tmp_config_path),
                output_dir=str(run_output_dir),
            )
            result = checker.run(
                today=args.today,
                cutoff_date=args.cutoff_date,
                stage_id=args.stage_id,
            )

            all_rows.append(
                _summarize_run(
                    summary_csv=Path(result["summary_csv"]),
                    post_event_zero_days=int(value),
                )
            )

    runs_df = pd.concat(all_rows, ignore_index=True)
    build_report(
        runs_df=runs_df,
        report_path=Path(args.report_path),
        today=args.today,
        cutoff_date=args.cutoff_date,
        rf_config_path=args.rf_config_path,
        lgbm_config_path=args.lgbm_config_path,
        values=args.values,
    )

    print(f"Saved report: {args.report_path}")


if __name__ == "__main__":
    main()
