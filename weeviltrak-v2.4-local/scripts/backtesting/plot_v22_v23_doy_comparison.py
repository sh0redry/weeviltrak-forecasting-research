"""
Plot v2.2 vs v2.3 backtesting predicted-vs-actual DOY comparisons.

The figure is a faceted scatter-regression view:
  - rows: prediction year
  - columns: stage ID
  - each facet overlays v2.2 and v2.3 points
  - each version gets a least-squares regression line with a 95% confidence band
  - x/y axes are fixed to DOY 90..150 for direct visual comparison
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "outputs/cache/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.settings import setup_logger

logger = setup_logger()

DEFAULT_V22_DIR = PROJECT_ROOT / "outputs" / "backtesting" / "runs" / "v2.2"
DEFAULT_V23_DIR = PROJECT_ROOT / "outputs" / "backtesting" / "runs"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "backtesting" / "runs" / "v22_v23_doy_comparison"
)
AXIS_LIMITS = (90, 150)
VERSION_STYLES = {
    "v2.2": {"color": "#2F6BFF", "marker": "o"},
    "v2.3": {"color": "#F28E2B", "marker": "^"},
}


def _latest_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")
    return matches[-1]


def _ensure_sequence(raw_values: Iterable[int] | None, default: list[int]) -> list[int]:
    if raw_values is None:
        return default
    values = [int(value) for value in raw_values]
    return list(dict.fromkeys(values)) or default


def load_evaluation_detail(path: Path, model_version: str) -> pd.DataFrame:
    """Load one backtesting evaluation CSV and add predicted/actual DOY columns."""
    required = [
        "location_id",
        "stage_id",
        "prediction_year",
        "predicted_stage_date",
        "actual_stage_date",
    ]
    df = pd.read_csv(path)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    out = df.copy()
    out["model_version"] = model_version
    out["location_id"] = out["location_id"].astype(str)
    out["stage_id"] = pd.to_numeric(out["stage_id"], errors="coerce").astype("Int64")
    out["prediction_year"] = pd.to_numeric(
        out["prediction_year"], errors="coerce"
    ).astype("Int64")
    out["predicted_stage_date"] = pd.to_datetime(
        out["predicted_stage_date"], errors="coerce"
    )
    out["actual_stage_date"] = pd.to_datetime(out["actual_stage_date"], errors="coerce")
    out["predicted_doy"] = out["predicted_stage_date"].dt.dayofyear
    out["actual_doy"] = out["actual_stage_date"].dt.dayofyear
    out["doy_error"] = out["predicted_doy"] - out["actual_doy"]

    return out.dropna(
        subset=["stage_id", "prediction_year", "predicted_doy", "actual_doy"]
    ).copy()


def build_comparison_frame(v22_path: Path, v23_path: Path) -> pd.DataFrame:
    """Combine the v2.2 and v2.3 evaluation details into one plotting frame."""
    combined = pd.concat(
        [
            load_evaluation_detail(v22_path, "v2.2"),
            load_evaluation_detail(v23_path, "v2.3"),
        ],
        ignore_index=True,
    )
    combined["stage_id"] = combined["stage_id"].astype(int)
    combined["prediction_year"] = combined["prediction_year"].astype(int)
    return combined.sort_values(
        ["prediction_year", "stage_id", "location_id", "model_version"]
    ).reset_index(drop=True)


def _regression_ci(
    x: pd.Series,
    y: pd.Series,
    x_grid: np.ndarray,
    confidence_z: float = 1.96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Return OLS mean-prediction line and an approximate 95% confidence interval.

    The z approximation is intentionally used to keep this plotting utility
    independent from statsmodels while still producing stable confidence bands.
    """
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(data) < 3:
        return None

    x_arr = data["x"].astype(float).to_numpy()
    y_arr = data["y"].astype(float).to_numpy()
    x_mean = float(x_arr.mean())
    sxx = float(np.sum((x_arr - x_mean) ** 2))
    if sxx <= 0:
        return None

    slope, intercept = np.polyfit(x_arr, y_arr, deg=1)
    y_fit = intercept + slope * x_grid
    residuals = y_arr - (intercept + slope * x_arr)
    dof = len(x_arr) - 2
    if dof <= 0:
        return None

    mse = float(np.sum(residuals**2) / dof)
    se_mean = np.sqrt(mse * (1 / len(x_arr) + ((x_grid - x_mean) ** 2 / sxx)))
    ci = confidence_z * se_mean
    return y_fit, y_fit - ci, y_fit + ci


def plot_doy_comparison(
    comparison_df: pd.DataFrame,
    output_png: Path,
    output_pdf: Path | None = None,
    years: list[int] | None = None,
    stages: list[int] | None = None,
    axis_limits: tuple[int, int] = AXIS_LIMITS,
) -> None:
    """Create and save the faceted v2.2/v2.3 DOY comparison figure."""
    df = comparison_df.copy()
    years = years or sorted(
        df["prediction_year"].dropna().astype(int).unique().tolist()
    )
    stages = stages or sorted(df["stage_id"].dropna().astype(int).unique().tolist())

    n_rows = len(years)
    n_cols = len(stages)
    if n_rows == 0 or n_cols == 0:
        raise ValueError("No years or stages available to plot.")

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.9 * n_cols, 4.1 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    x_grid = np.linspace(axis_limits[0], axis_limits[1], 200)

    for row_idx, year in enumerate(years):
        for col_idx, stage in enumerate(stages):
            ax = axes[row_idx][col_idx]
            facet = df[
                (df["prediction_year"] == year) & (df["stage_id"] == stage)
            ].copy()

            ax.plot(
                axis_limits,
                axis_limits,
                color="#222222",
                linewidth=1.1,
                linestyle="--",
                alpha=0.7,
                zorder=1,
            )

            for version, style in VERSION_STYLES.items():
                sub = facet[facet["model_version"] == version]
                if sub.empty:
                    continue

                ax.scatter(
                    sub["predicted_doy"],
                    sub["actual_doy"],
                    s=42,
                    marker=style["marker"],
                    color=style["color"],
                    alpha=0.72,
                    edgecolors="white",
                    linewidths=0.55,
                    label=version,
                    zorder=3,
                )

                fit = _regression_ci(sub["predicted_doy"], sub["actual_doy"], x_grid)
                if fit is not None:
                    y_fit, y_low, y_high = fit
                    ax.plot(
                        x_grid,
                        y_fit,
                        color=style["color"],
                        linewidth=2.2,
                        alpha=0.95,
                        zorder=4,
                    )
                    ax.fill_between(
                        x_grid,
                        y_low,
                        y_high,
                        color=style["color"],
                        alpha=0.16,
                        linewidth=0,
                        zorder=2,
                    )

            if facet.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#777777",
                    fontsize=11,
                )

            ax.set_xlim(axis_limits)
            ax.set_ylim(axis_limits)
            ax.set_xticks(np.arange(axis_limits[0], axis_limits[1] + 1, 10))
            ax.set_yticks(np.arange(axis_limits[0], axis_limits[1] + 1, 10))
            ax.grid(True, color="#E6E8EF", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_title(f"{year} · Stage {stage}", fontsize=12, pad=8)

            if col_idx == 0:
                ax.set_ylabel("Actual DOY", fontsize=11)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Predicted DOY", fontsize=11)

    handles, labels = axes[0][0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=max(1, len(by_label)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
        fontsize=12,
    )
    fig.suptitle(
        "Predicted vs Actual Day of Year: v2.2 vs v2.3 Backtesting",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.925,
        "Points are event-level predictions; lines are per-version OLS fits with 95% confidence bands; dashed line is perfect agreement.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.03, 0.035, 0.995, 0.885))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a faceted v2.2/v2.3 predicted-vs-actual DOY plot."
    )
    parser.add_argument(
        "--v22-evaluation",
        type=Path,
        help="Path to v2.2 evaluation CSV. Defaults to latest evaluation_lead*.csv in outputs/backtesting/runs/v2.2.",
    )
    parser.add_argument(
        "--v23-evaluation",
        type=Path,
        help="Path to v2.3 evaluation CSV. Defaults to latest evaluation_lead*.csv in outputs/backtesting/runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--year",
        dest="years",
        action="append",
        type=int,
        help="Prediction year to include. Repeat for multiple years.",
    )
    parser.add_argument(
        "--stage",
        dest="stages",
        action="append",
        type=int,
        help="Stage ID to include. Repeat for multiple stages.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    v22_path = args.v22_evaluation or _latest_file(
        DEFAULT_V22_DIR, "evaluation_lead*.csv"
    )
    v23_path = args.v23_evaluation or _latest_file(
        DEFAULT_V23_DIR, "evaluation_lead*.csv"
    )
    years = _ensure_sequence(args.years, [])
    stages = _ensure_sequence(args.stages, [])

    logger.info("Loading v2.2 evaluation: %s", v22_path)
    logger.info("Loading v2.3 evaluation: %s", v23_path)
    comparison = build_comparison_frame(v22_path=v22_path, v23_path=v23_path)
    if years:
        comparison = comparison[comparison["prediction_year"].isin(years)].copy()
    if stages:
        comparison = comparison[comparison["stage_id"].isin(stages)].copy()
    if comparison.empty:
        raise RuntimeError("No rows remain after applying year/stage filters.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "v22_v23_doy_comparison_detail.csv"
    png_path = args.output_dir / "v22_v23_predicted_vs_actual_doy_facets.png"
    pdf_path = args.output_dir / "v22_v23_predicted_vs_actual_doy_facets.pdf"

    comparison.to_csv(detail_path, index=False)
    plot_doy_comparison(
        comparison_df=comparison,
        output_png=png_path,
        output_pdf=pdf_path,
        years=years or None,
        stages=stages or None,
    )

    print("DONE")
    print(f"V22_EVALUATION={v22_path}")
    print(f"V23_EVALUATION={v23_path}")
    print(f"DETAIL_CSV={detail_path}")
    print(f"FIGURE_PNG={png_path}")
    print(f"FIGURE_PDF={pdf_path}")


if __name__ == "__main__":
    main()
