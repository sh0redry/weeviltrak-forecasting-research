"""Export trajectory inspection plots as PNG files.

Each PNG represents one historical trajectory identified by
``(test_year, location_id, stage_id)`` and overlays all available model versions
for that trajectory. The title, text panel, and filename include the trajectory
identifier and key detail fields so the image can be inspected outside the
notebook.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "outputs/cache/matplotlib")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DETAIL_PATH = (
    DEFAULT_PROJECT_ROOT
    / "backtesting_results"
    / "trajectory_v22_v23_3day"
    / "trajectory_detail_v22_v23.parquet"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT
    / "backtesting_results"
    / "trajectory_v22_v23_3day"
    / "trajectory_pngs"
)

VERSION_COLORS = {
    "v2.2": "#4E79A7",
    "v2.3": "#E15759",
    "v2.4": "#59A14F",
}

VERSION_LEARNED_COLORS = {
    "v2.4": "#B07AA1",
}

LIGHT_BLUE_GRID = "#dbeafe"
LIGHT_AXIS_SPINE = "#bfdbfe"


def _safe_filename_part(value: object, max_len: int = 42) -> str:
    text = "NA" if pd.isna(value) else str(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return (text or "NA")[:max_len]


def _ensure_detail_types(detail: pd.DataFrame) -> pd.DataFrame:
    df = detail.copy()
    df["location_id"] = df["location_id"].astype(str)
    df["stage_id"] = pd.to_numeric(df["stage_id"], errors="coerce").astype(int)
    df["test_year"] = pd.to_numeric(df["test_year"], errors="coerce").astype(int)
    for col in ["prediction_date", "predicted_stage_date", "actual_stage_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    for col in [
        "predicted_days",
        "learned_days_remaining",
        "stage_date_error_days",
        "abs_stage_date_error_days",
        "predicted_days_delta",
        "abs_predicted_stage_date_drift_days",
        "termination_threshold",
        "base_stage_date_error_days",
        "base_abs_stage_date_error_days",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [
        "base_predicted_stage_date",
        "learned_estimated_stage_date",
        "termination_decision_date",
        "termination_estimated_stage_date",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    if "bounce_up" in df.columns:
        df["bounce_up"] = df["bounce_up"].astype("boolean").fillna(False).astype(bool)
    for col in ["learned_hit", "termination_first_hit", "termination_reached"]:
        if col in df.columns:
            df[col] = df[col].astype("boolean").fillna(False).astype(bool)
    return df


def load_trajectory_detail(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Trajectory detail file not found: {path}")
    if path.suffix.lower() == ".parquet":
        detail = pd.read_parquet(path)
    else:
        detail = pd.read_csv(path)
    required = {
        "model_version",
        "test_year",
        "location_id",
        "stage_id",
        "prediction_date",
        "predicted_days",
        "predicted_stage_date",
        "actual_stage_date",
    }
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return _ensure_detail_types(detail)


def summarize_version(group: pd.DataFrame) -> dict[str, object]:
    error = pd.to_numeric(group.get("abs_stage_date_error_days"), errors="coerce")
    drift = pd.to_numeric(
        group.get("abs_predicted_stage_date_drift_days"), errors="coerce"
    )
    bounce = group.get("bounce_up")
    summary = {
        "n_points": int(len(group)),
        "bounce_up_count": int(bounce.sum()) if bounce is not None else np.nan,
        "mae_stage_date_error": float(error.mean()) if error.notna().any() else np.nan,
        "median_abs_error": float(error.median()) if error.notna().any() else np.nan,
        "mean_abs_drift_days": float(drift.mean()) if drift.notna().any() else np.nan,
        "first_prediction_date": group["prediction_date"].min(),
        "last_prediction_date": group["prediction_date"].max(),
    }
    if "termination_reached" in group.columns:
        reached = group["termination_reached"].dropna()
        summary["termination_reached"] = bool(reached.any()) if not reached.empty else np.nan
    if "termination_first_hit" in group.columns:
        hits = group[group["termination_first_hit"].fillna(False)]
        summary["termination_first_hit_count"] = int(len(hits))
        summary["first_hit_prediction_date"] = (
            hits["prediction_date"].min() if not hits.empty else pd.NaT
        )
    if "termination_decision_date" in group.columns:
        values = group["termination_decision_date"].dropna()
        summary["termination_decision_date"] = values.min() if not values.empty else pd.NaT
    if "termination_estimated_stage_date" in group.columns:
        values = group["termination_estimated_stage_date"].dropna()
        summary["termination_estimated_stage_date"] = (
            values.min() if not values.empty else pd.NaT
        )
    if "termination_threshold" in group.columns:
        values = group["termination_threshold"].dropna()
        summary["termination_threshold"] = values.iloc[0] if not values.empty else np.nan
    return summary


def build_trajectory_manifest(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["test_year", "location_id", "stage_id"]
    optional_cols = [
        "location_name",
        "location_state",
        "actual_latitude",
        "actual_longitude",
        "latitude",
        "longitude",
    ]
    present_optional = [col for col in optional_cols if col in detail.columns]

    for trajectory_id, (key, g) in enumerate(
        detail.groupby(group_cols, sort=True), start=1
    ):
        test_year, location_id, stage_id = key
        actual_stage_date = g["actual_stage_date"].dropna().iloc[0]
        row: dict[str, object] = {
            "trajectory_id": trajectory_id,
            "test_year": int(test_year),
            "location_id": str(location_id),
            "stage_id": int(stage_id),
            "actual_stage_date": actual_stage_date,
            "model_versions": ",".join(sorted(g["model_version"].astype(str).unique())),
            "n_rows": int(len(g)),
        }
        for col in present_optional:
            values = g[col].dropna()
            row[col] = values.iloc[0] if not values.empty else np.nan
        for version, vg in g.groupby("model_version", sort=True):
            stats = summarize_version(vg)
            for metric, value in stats.items():
                row[f"{version}_{metric}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def trajectory_filename(row: pd.Series) -> str:
    name = row.get("location_name", "")
    state = row.get("location_state", "")
    actual = pd.to_datetime(row["actual_stage_date"]).strftime("%Y%m%d")
    return (
        f"trajectory_{int(row['trajectory_id']):04d}"
        f"_year{int(row['test_year'])}"
        f"_loc{_safe_filename_part(row['location_id'])}"
        f"_stage{int(row['stage_id'])}"
        f"_actual{actual}"
        f"_{_safe_filename_part(state, 12)}"
        f"_{_safe_filename_part(name, 36)}.png"
    )


def _fmt_date(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _fmt_num(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _build_info_text(row: pd.Series, versions: Iterable[str]) -> str:
    lines = [
        f"trajectory_id: {int(row['trajectory_id']):04d}",
        f"year/location/stage: {int(row['test_year'])} / {row['location_id']} / {int(row['stage_id'])}",
        f"actual_stage_date: {_fmt_date(row['actual_stage_date'])}",
    ]
    if "location_name" in row and not pd.isna(row.get("location_name")):
        lines.append(f"location_name: {row['location_name']}")
    if "location_state" in row and not pd.isna(row.get("location_state")):
        lines.append(f"location_state: {row['location_state']}")
    lat = row.get("actual_latitude", row.get("latitude", np.nan))
    lon = row.get("actual_longitude", row.get("longitude", np.nan))
    if not pd.isna(lat) and not pd.isna(lon):
        lines.append(f"lat/lon: {_fmt_num(lat, 4)}, {_fmt_num(lon, 4)}")

    lines.append("")
    lines.append("Per-model details:")
    for version in versions:
        prefix = f"{version}_"
        lines.extend(
            [
                (
                    f"{version}: n={row.get(prefix + 'n_points', 'NA')}, "
                    f"bounce={row.get(prefix + 'bounce_up_count', 'NA')}, "
                    f"MAE={_fmt_num(row.get(prefix + 'mae_stage_date_error'))}d, "
                    f"median_abs={_fmt_num(row.get(prefix + 'median_abs_error'))}d"
                ),
                (
                    f"    mean_abs_drift={_fmt_num(row.get(prefix + 'mean_abs_drift_days'))}d, "
                    f"first={_fmt_date(row.get(prefix + 'first_prediction_date'))}, "
                    f"last={_fmt_date(row.get(prefix + 'last_prediction_date'))}"
                ),
            ]
        )
        if (
            prefix + "termination_decision_date" in row
            or prefix + "termination_estimated_stage_date" in row
            or prefix + "termination_threshold" in row
        ):
            lines.append(
                (
                    f"    terminate: reached={row.get(prefix + 'termination_reached', 'NA')}, "
                    f"theta={_fmt_num(row.get(prefix + 'termination_threshold'))}, "
                    f"first_hit={_fmt_date(row.get(prefix + 'first_hit_prediction_date'))}, "
                    f"decision={_fmt_date(row.get(prefix + 'termination_decision_date'))}, "
                    f"estimated={_fmt_date(row.get(prefix + 'termination_estimated_stage_date'))}"
                )
            )
    return "\n".join(lines)


def plot_trajectory_png(
    trajectory_detail: pd.DataFrame,
    manifest_row: pd.Series,
    output_path: Path,
    dpi: int = 160,
) -> None:
    versions = sorted(trajectory_detail["model_version"].astype(str).unique().tolist())
    actual_stage_date = pd.to_datetime(manifest_row["actual_stage_date"])

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(13.5, 9.0),
        sharex=True,
        constrained_layout=True,
    )
    ax_date, ax_days = axes
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(LIGHT_AXIS_SPINE)

    for version in versions:
        g = (
            trajectory_detail[trajectory_detail["model_version"].astype(str) == version]
            .sort_values("prediction_date")
            .copy()
        )
        color = VERSION_COLORS.get(version, None)
        ax_date.plot(
            g["prediction_date"],
            g["predicted_stage_date"],
            marker="o",
            linewidth=1.7,
            markersize=4.0,
            label=version,
            color=color,
        )
        ax_days.plot(
            g["prediction_date"],
            g["predicted_days"],
            marker="o",
            linewidth=1.7,
            markersize=4.0,
            label=version,
            color=color,
        )
        if "learned_days_remaining" in g.columns and g["learned_days_remaining"].notna().any():
            learned_color = VERSION_LEARNED_COLORS.get(version, color)
            ax_days.plot(
                g["prediction_date"],
                g["learned_days_remaining"],
                marker="s",
                linewidth=1.45,
                markersize=3.4,
                linestyle="--",
                label=f"{version} learned_days_remaining",
                color=learned_color,
            )
        if (
            "termination_first_hit" in g.columns
            and g["termination_first_hit"].fillna(False).any()
        ):
            hit_rows = g[g["termination_first_hit"].fillna(False)]
            ax_days.scatter(
                hit_rows["prediction_date"],
                hit_rows.get("learned_days_remaining", hit_rows["predicted_days"]),
                s=72,
                marker="*",
                color=VERSION_LEARNED_COLORS.get(version, color),
                edgecolor="black",
                linewidth=0.6,
                zorder=6,
                label=f"{version} termination first hit",
            )
        if (
            "termination_decision_date" in g.columns
            and g["termination_decision_date"].notna().any()
        ):
            decision_date = g["termination_decision_date"].dropna().min()
            if not pd.isna(decision_date):
                ax_days.axvline(
                    decision_date,
                    color=VERSION_LEARNED_COLORS.get(version, color),
                    linestyle="-.",
                    linewidth=1.1,
                    alpha=0.9,
                    label=f"{version} decision date",
                )
        if "bounce_up" in g.columns and g["bounce_up"].any():
            bounce_rows = g[g["bounce_up"]]
            ax_days.scatter(
                bounce_rows["prediction_date"],
                bounce_rows["predicted_days"],
                s=54,
                marker="^",
                color=color,
                edgecolor="black",
                linewidth=0.6,
                zorder=5,
            )

    ax_date.axhline(
        actual_stage_date,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="actual stage date",
    )
    ax_days.axhline(0, color="black", linestyle="--", linewidth=1.2, label="0 days")
    ax_days.axvline(actual_stage_date, color="black", linestyle=":", linewidth=1.1)

    title = (
        f"Trajectory {int(manifest_row['trajectory_id']):04d}: "
        f"year={int(manifest_row['test_year'])}, "
        f"location={manifest_row['location_id']}, "
        f"stage={int(manifest_row['stage_id'])}, "
        f"actual={_fmt_date(actual_stage_date)}"
    )
    if "location_name" in manifest_row and not pd.isna(manifest_row.get("location_name")):
        title += f"\n{manifest_row['location_name']}"

    fig.suptitle(title, fontsize=14, fontweight="bold")
    ax_date.set_ylabel("Predicted stage date")
    ax_days.set_ylabel("Predicted remaining days")
    ax_days.set_xlabel("Prediction date")

    for ax in axes:
        ax.grid(True, linestyle="-", color=LIGHT_BLUE_GRID, linewidth=0.8, alpha=1.0)
        ax.legend(loc="upper right", fontsize=8.5)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    ax_date.yaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax_date.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax_days.get_xticklabels(), rotation=35, ha="right")

    info_text = _build_info_text(manifest_row, versions)
    ax_date.text(
        0.01,
        0.99,
        info_text,
        transform=ax_date.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_trajectory_pngs(
    detail_path: Path = DEFAULT_DETAIL_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    overwrite: bool = False,
    dpi: int = 160,
    make_zip: bool = False,
) -> pd.DataFrame:
    detail = load_trajectory_detail(detail_path)
    manifest = build_trajectory_manifest(detail)

    if limit is not None:
        manifest_to_plot = manifest.head(int(limit)).copy()
    else:
        manifest_to_plot = manifest.copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    group_cols = ["test_year", "location_id", "stage_id"]
    exported_paths: list[str] = []

    for _, row in manifest_to_plot.iterrows():
        filename = trajectory_filename(row)
        output_path = output_dir / filename
        exported_paths.append(str(output_path))
        if output_path.exists() and not overwrite:
            continue

        g = detail[
            (detail["test_year"] == int(row["test_year"]))
            & (detail["location_id"].astype(str) == str(row["location_id"]))
            & (detail["stage_id"] == int(row["stage_id"]))
        ].copy()
        plot_trajectory_png(g, row, output_path, dpi=dpi)

    manifest = manifest.copy()
    manifest["png_path"] = manifest.apply(
        lambda row: str(output_dir / trajectory_filename(row)), axis=1
    )
    manifest_path = output_dir / "trajectory_png_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    if make_zip:
        shutil.make_archive(str(output_dir), "zip", root_dir=output_dir)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export all trajectory inspection plots as PNG files."
    )
    parser.add_argument("--detail-path", type=Path, default=DEFAULT_DETAIL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, help="Optional number of trajectories to export.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--make-zip", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = export_trajectory_pngs(
        detail_path=args.detail_path,
        output_dir=args.output_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        dpi=args.dpi,
        make_zip=args.make_zip,
    )
    exported = len(manifest) if args.limit is None else min(args.limit, len(manifest))
    print(f"PNG_OUTPUT_DIR={args.output_dir}")
    print(f"MANIFEST={args.output_dir / 'trajectory_png_manifest.csv'}")
    if args.make_zip:
        print(f"ZIP={args.output_dir.with_suffix('.zip')}")
    print(f"TRAJECTORIES_IN_MANIFEST={len(manifest)}")
    print(f"TRAJECTORIES_REQUESTED={exported}")


if __name__ == "__main__":
    main()
