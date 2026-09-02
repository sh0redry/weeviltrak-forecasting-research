from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "outputs/cache/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def plot_location_map(df: pd.DataFrame, year: int, output_path: Path) -> None:
    work = (
        df[df["event_year"] == year][["location_id", "latitude", "longitude"]]
        .drop_duplicates()
        .copy()
    )
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work = work.dropna(subset=["latitude", "longitude"])

    half_size = 0.28
    features = []
    locations = []
    customdata = []
    for row in work.itertuples(index=False):
        location_key = f"{int(row.location_id)}"
        lon = float(row.longitude)
        lat = float(row.latitude)
        features.append(
            {
                "type": "Feature",
                "properties": {"location_key": location_key},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon - half_size, lat - half_size],
                        [lon + half_size, lat - half_size],
                        [lon + half_size, lat + half_size],
                        [lon - half_size, lat + half_size],
                        [lon - half_size, lat - half_size],
                    ]],
                },
            }
        )
        locations.append(location_key)
        customdata.append([str(row.location_id), lat, lon])

    geojson = {"type": "FeatureCollection", "features": features}
    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            locations=locations,
            z=[1] * len(locations),
            featureidkey="properties.location_key",
            colorscale=[[0, "#1b9e77"], [1, "#1b9e77"]],
            showscale=False,
            marker_line_color="white",
            marker_line_width=0.8,
            customdata=customdata,
            hovertemplate=(
                "location_id=%{customdata[0]}<br>"
                "latitude=%{customdata[1]:.3f}<br>"
                "longitude=%{customdata[2]:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="mercator",
    )
    fig.update_layout(
        title=f"Issue 1 Validation {year}: Historical Location Distribution",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


def plot_lines(df: pd.DataFrame, year: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    work = df[df["event_year"] == year].copy()

    for _, group in work.groupby(["location_id", "stage_date"], sort=False):
        ax.plot(
            group["days_from_event"],
            group["predicted_days"],
            color="#2f5bea",
            alpha=0.14,
            linewidth=0.9,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_title(f"Issue 1 Validation {year}: Stage 1 PredictedDays Around Actual Stage Date")
    ax.set_xlabel("Days From Actual Stage 1 Date")
    ax.set_ylabel("Predicted Days")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _bin_predicted_days(values: pd.Series) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="coerce")
    labels = [
        "0",
        "(0,1]",
        "(1,3]",
        "(3,7]",
        "(7,14]",
        "(14,30]",
        ">30",
    ]

    def classify(value: float) -> str | pd._libs.missing.NAType:
        if pd.isna(value):
            return pd.NA
        if value <= 0:
            return "0"
        if value <= 1:
            return "(0,1]"
        if value <= 3:
            return "(1,3]"
        if value <= 7:
            return "(3,7]"
        if value <= 14:
            return "(7,14]"
        if value <= 30:
            return "(14,30]"
        return ">30"

    out = numeric.map(classify)
    return pd.Categorical(out, categories=labels, ordered=True)


def plot_predicted_days_heatmap(df: pd.DataFrame, year: int, output_path: Path) -> None:
    work = df[df["event_year"] == year].copy()
    work["predicted_days_bin"] = _bin_predicted_days(work["predicted_days"])
    work = work.dropna(subset=["predicted_days_bin", "days_from_event"])

    heat = (
        work.groupby(["predicted_days_bin", "days_from_event"], observed=False)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    if heat.empty:
        return

    ordered_x = sorted(pd.to_numeric(work["days_from_event"], errors="coerce").dropna().astype(int).unique())
    heat = heat.reindex(columns=ordered_x, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    im = ax.imshow(heat.values, aspect="auto", cmap="YlOrRd", origin="upper")

    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns.tolist())
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels([str(v) for v in heat.index.tolist()])
    ax.set_xlabel("Days From Actual Stage 1 Date")
    ax.set_ylabel("Predicted Days Bin")
    ax.set_title(f"Issue 1 Validation {year}: Stage 1 PredictedDays Heatmap")

    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            value = int(heat.iat[i, j])
            if value:
                ax.text(j, i, str(value), ha="center", va="center", color="black", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_predicted_days_boxplot(df: pd.DataFrame, year: int, output_path: Path) -> None:
    work = df[df["event_year"] == year].copy()
    ordered_x = sorted(pd.to_numeric(work["days_from_event"], errors="coerce").dropna().astype(int).unique())
    series = [
        pd.to_numeric(
            work.loc[work["days_from_event"] == day, "predicted_days"],
            errors="coerce",
        ).dropna()
        for day in ordered_x
    ]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bp = ax.boxplot(
        series,
        positions=np.arange(len(ordered_x)),
        widths=0.65,
        patch_artist=True,
        showfliers=True,
    )
    for box in bp["boxes"]:
        box.set(facecolor="#9ecae1", edgecolor="#2b6c9c")
    for median in bp["medians"]:
        median.set(color="#b30000", linewidth=1.5)

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(ordered_x.index(0), color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(np.arange(len(ordered_x)))
    ax.set_xticklabels(ordered_x)
    ax.set_xlabel("Days From Actual Stage 1 Date")
    ax.set_ylabel("Predicted Days")
    ax.set_title(f"Issue 1 Validation {year}: Stage 1 PredictedDays Boxplot")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_error_distribution(df: pd.DataFrame, year: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    work = df[df["event_year"] == year].copy()
    error_days = pd.to_numeric(work["error_days"], errors="coerce").dropna()

    ax.hist(error_days, bins=25, color="#d95f02", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Issue 1 Validation {year}: Stage Date Error Distribution")
    ax.set_xlabel("Error Days (Predicted Stage Date - Actual Stage Date)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot Issue 1 validation outputs by year.")
    parser.add_argument(
        "--csv",
        default="docs/implemention_results/issue1_signed_target_validation.csv",
        help="Input validation CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/implemention_results",
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    years = sorted(pd.to_numeric(df["event_year"], errors="coerce").dropna().astype(int).unique().tolist())
    for year in years:
        plot_location_map(
            df,
            year,
            output_dir / f"issue1_signed_target_validation_{year}_location_map.html",
        )
        plot_lines(
            df,
            year,
            output_dir / f"issue1_signed_target_validation_{year}_lines.png",
        )
        plot_predicted_days_heatmap(
            df,
            year,
            output_dir / f"issue1_signed_target_validation_{year}_heatmap.png",
        )
        plot_predicted_days_boxplot(
            df,
            year,
            output_dir / f"issue1_signed_target_validation_{year}_boxplot.png",
        )
        plot_error_distribution(
            df,
            year,
            output_dir / f"issue1_signed_target_validation_{year}_error_distribution.png",
        )


if __name__ == "__main__":
    main()
