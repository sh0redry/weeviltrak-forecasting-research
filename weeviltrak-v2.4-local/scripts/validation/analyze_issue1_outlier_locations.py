from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "outputs/cache/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return [int(v) for v in values]


def plot_predicted_days(detail: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for (loc, year), group in detail.groupby(["location_id", "event_year"], sort=False):
        ax.plot(
            group["days_from_event"],
            group["predicted_days"],
            marker="o",
            linewidth=1.6,
            label=f"loc {loc} / {year}",
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Issue 1 Outlier Check: Predicted Days Around Actual Stage Date")
    ax.set_xlabel("Days From Actual Stage Date")
    ax.set_ylabel("Predicted Days")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_error_days(detail: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for (loc, year), group in detail.groupby(["location_id", "event_year"], sort=False):
        ax.plot(
            group["days_from_event"],
            group["error_days"],
            marker="o",
            linewidth=1.6,
            label=f"loc {loc} / {year}",
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Issue 1 Outlier Check: Stage Date Error Around Actual Stage Date")
    ax.set_xlabel("Days From Actual Stage Date")
    ax.set_ylabel("Error Days")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    summary = (
        detail.groupby(
            ["location_id", "event_year", "stage_date", "latitude", "longitude"],
            as_index=False,
        )
        .agg(
            min_predicted_days=("predicted_days", "min"),
            max_predicted_days=("predicted_days", "max"),
            median_abs_error=("abs_error_days", "median"),
            mean_abs_error=("abs_error_days", "mean"),
            first_predicted_stage_date=("predicted_stage_date", "min"),
            last_predicted_stage_date=("predicted_stage_date", "max"),
            any_terminated=("terminated", "max"),
        )
        .sort_values(["event_year", "location_id"])
        .reset_index(drop=True)
    )
    return summary


def write_markdown(summary: pd.DataFrame, detail: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Issue 1 Outlier Location Analysis",
        "",
        "## Summary",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.extend(
            [
                f"### location_id {row.location_id} / {row.event_year}",
                "",
                f"- actual_stage_date: `{str(row.stage_date)[:10]}`",
                f"- latitude/longitude: `{row.latitude:.6f}, {row.longitude:.6f}`",
                f"- min_predicted_days: `{row.min_predicted_days:.3f}`",
                f"- max_predicted_days: `{row.max_predicted_days:.3f}`",
                f"- median_abs_error_days: `{row.median_abs_error:.3f}`",
                f"- mean_abs_error_days: `{row.mean_abs_error:.3f}`",
                f"- terminated_within_window: `{bool(row.any_terminated)}`",
                "",
            ]
        )

        sub = detail[
            (detail["location_id"] == row.location_id)
            & (detail["event_year"] == row.event_year)
        ].copy()
        if not sub.empty:
            worst = sub.loc[sub["abs_error_days"].idxmax()]
            best = sub.loc[sub["abs_error_days"].idxmin()]
            lines.extend(
                [
                    f"- worst_error_point: `days_from_event={int(worst['days_from_event'])}`, "
                    f"`predicted_days={pd.to_numeric(worst['predicted_days'], errors='coerce'):.3f}`, "
                    f"`error_days={pd.to_numeric(worst['error_days'], errors='coerce'):.1f}`",
                    f"- best_error_point: `days_from_event={int(best['days_from_event'])}`, "
                    f"`predicted_days={pd.to_numeric(best['predicted_days'], errors='coerce'):.3f}`, "
                    f"`error_days={pd.to_numeric(best['error_days'], errors='coerce'):.1f}`",
                    "",
                ]
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Issue 1 outlier locations from existing validation CSV.")
    parser.add_argument(
        "--csv",
        default="docs/implemention_results/issue1_signed_target_validation.csv",
        help="Validation CSV path.",
    )
    parser.add_argument(
        "--location-ids",
        default="103,104",
        help="Comma-separated location IDs to inspect.",
    )
    parser.add_argument(
        "--years",
        default="2024",
        help="Comma-separated event years to inspect.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/implemention_results",
        help="Directory for outputs.",
    )
    args = parser.parse_args()

    location_ids = {str(v) for v in _parse_int_list(args.location_ids)}
    years = set(_parse_int_list(args.years))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df["location_id"] = df["location_id"].astype(str)
    df["event_year"] = pd.to_numeric(df["event_year"], errors="coerce").astype(int)
    df["stage_date"] = pd.to_datetime(df["stage_date"], errors="coerce")
    df["predicted_stage_date"] = pd.to_datetime(df["predicted_stage_date"], errors="coerce")

    detail = df[
        df["location_id"].isin(location_ids) & df["event_year"].isin(years)
    ].copy()
    if detail.empty:
        raise RuntimeError("No rows matched the requested location_ids/years.")

    detail = detail.sort_values(["event_year", "location_id", "days_from_event"]).reset_index(drop=True)
    summary = build_summary(detail)

    detail_csv = output_dir / "issue1_outlier_locations_detail.csv"
    summary_csv = output_dir / "issue1_outlier_locations_summary.csv"
    summary_md = output_dir / "issue1_outlier_locations_summary.md"
    pred_plot = output_dir / "issue1_outlier_locations_predicted_days.png"
    err_plot = output_dir / "issue1_outlier_locations_error_days.png"

    detail.to_csv(detail_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    write_markdown(summary, detail, summary_md)
    plot_predicted_days(detail, pred_plot)
    plot_error_days(detail, err_plot)


if __name__ == "__main__":
    main()
