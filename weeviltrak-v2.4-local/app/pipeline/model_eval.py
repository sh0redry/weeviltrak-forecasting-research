"""
Model Evaluation Script
=======================
Loads backtest predictions + evaluation results from S3, computes summary
metrics, generates diagnostic plots, and uploads everything to
``s3://sps-ds-bucket/model_testing/weeviltrak/``.

Outputs (written to S3):
    - metrics_summary.csv          – overall + per-stage MAE & RMSE
    - predicted_vs_actual_doy.html – scatter of predicted vs actual DOY (±7 d)
    - mae_map.html                 – MAE scatter-map grid (Year × Stage)

Usage::

    python -m app.pipeline.model_eval                                # defaults
    python -m app.pipeline.model_eval --eval-key <s3_key>            # custom eval
    python -m app.pipeline.model_eval --pred-key <key> --eval-key <key>
"""

from __future__ import annotations

import argparse
import collections
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.config.weevilltrak_config import WeevillTrakConfig
from app.services.database_service import DatabaseManager
from griddedweather.s3_store import S3Manager

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────
DEFAULT_PRED_KEY = (
    "weeviltrak_data/api-testing/"
    "backtest_predictions_lead7_20260109_mean_sliding_window.csv"
)
DEFAULT_EVAL_KEY = (
    "weeviltrak_data/api-testing/"
    "evaluation_lead7_20260109_mean_sliding_window.csv"
)
OUTPUT_PREFIX = "model_testing/weeviltrak"

LEAD_DAYS = 7
YEARS = [2023, 2024, 2025]
STAGES = [1, 2, 3]
WITHIN_DAYS = 7


# ────────────────────────────────────────────────────────────
# Helper utilities
# ────────────────────────────────────────────────────────────
def _ensure_datetime(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


def _ensure_str(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = df[col].astype(str)


def _ensure_int(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")


def _safe_dayofyear(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.dayofyear


def _metrics_from_error(err: pd.Series) -> dict:
    err = pd.to_numeric(err, errors="coerce").dropna()
    if err.empty:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "median_abs_error": np.nan,
            f"within_{WITHIN_DAYS}d": np.nan,
        }
    return {
        "n": int(err.shape[0]),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "median_abs_error": float(np.median(np.abs(err))),
        f"within_{WITHIN_DAYS}d": float(np.mean(np.abs(err) <= WITHIN_DAYS)),
    }


# ────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────
def load_predictions(s3: S3Manager, key: str) -> pd.DataFrame:
    """Load and normalise the predictions CSV from S3."""
    logger.info("Loading predictions from %s", key)
    df = s3.read_file(f"{s3.bucket_name}/{key}")
    _ensure_datetime(df, "prediction_date")
    _ensure_datetime(df, "predicted_stage_date")
    _ensure_str(df, "location_id")
    df["year"] = df["prediction_date"].dt.year
    return df


def load_evaluation(s3: S3Manager, key: str) -> pd.DataFrame:
    """Load and normalise the evaluation CSV from S3."""
    logger.info("Loading evaluation from %s", key)
    df = s3.read_file(f"{s3.bucket_name}/{key}")
    for col in ("prediction_date", "predicted_stage_date", "actual_stage_date"):
        _ensure_datetime(df, col)
    _ensure_str(df, "location_id")
    _ensure_int(df, "stage_id")
    _ensure_int(df, "year")
    return df


# ────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────
def compute_metrics(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with overall + per-stage metrics."""
    rows = [{"scope": "overall", **_metrics_from_error(evaluation_df["error_days"])}]

    by_stage_flat = (
        evaluation_df.groupby("stage_id")["error_days"]
        .apply(_metrics_from_error)
        .to_dict()
    )
    by_stage: dict[int, dict] = collections.defaultdict(dict)
    for (sid, metric), val in by_stage_flat.items():
        by_stage[sid][metric] = val
    for sid, m in by_stage.items():
        rows.append({"scope": f"stage_id={sid}", **m})

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────
# Plots
# ────────────────────────────────────────────────────────────
def _axis_ref(i: int):
    xref = "x" if i == 1 else f"x{i}"
    yref = "y" if i == 1 else f"y{i}"
    return xref, yref


def _axis_domain_ref(i: int):
    xd = "x domain" if i == 1 else f"x{i} domain"
    yd = "y domain" if i == 1 else f"y{i} domain"
    return xd, yd


def plot_predicted_vs_actual(
    evaluation_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> go.Figure:
    """Scatter: predicted vs actual DOY, per stage, with ±7-day bands."""
    stage_order = STAGES

    mae_by_stage = (
        metrics_df.loc[
            metrics_df["scope"].str.startswith("stage_id="), ["scope", "mae"]
        ]
        .assign(
            stage_id=lambda d: d["scope"].str.extract(r"stage_id=(\d+)").astype(int)
        )
        .set_index("stage_id")["mae"]
        .round(2)
        .to_dict()
    )

    hover_cols = [
        c
        for c in [
            "location_id",
            "year",
            "prediction_date",
            "predicted_stage_date",
            "actual_stage_date",
            "error_days",
        ]
        if c in evaluation_df.columns
    ]

    tmp = (
        evaluation_df.assign(
            pred_DOY=lambda d: _safe_dayofyear(d["predicted_stage_date"]),
            actual_DOY=lambda d: _safe_dayofyear(d["actual_stage_date"]),
            stage_id=lambda d: pd.Categorical(
                d["stage_id"], categories=stage_order, ordered=True
            ).astype(str),
        ).dropna(subset=["pred_DOY", "actual_DOY", "stage_id"])
    )

    fig = px.scatter(
        tmp,
        x="pred_DOY",
        y="actual_DOY",
        facet_col="stage_id",
        facet_col_wrap=3,
        category_orders={"stage_id": [str(s) for s in stage_order]},
        color="stage_id",
        hover_data=hover_cols,
        labels={
            "pred_DOY": "Predicted DOY",
            "actual_DOY": "Actual DOY",
            "stage_id": "Stage",
        },
        title=f"Predicted vs Actual DOY (±{WITHIN_DAYS}d)",
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(x=0.5, xanchor="center"),
        legend_title_text="Stage",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1
        ),
        margin=dict(l=50, r=20, t=80, b=50),
    )
    fig.update_traces(marker=dict(size=8, opacity=0.75))

    gmin = int(min(tmp["pred_DOY"].min(), tmp["actual_DOY"].min()))
    gmax = int(max(tmp["pred_DOY"].max(), tmp["actual_DOY"].max()))
    pad = max(3, int((gmax - gmin) * 0.03))
    x0, x1 = gmin - pad, gmax + pad

    fig.update_xaxes(range=[x0, x1], showgrid=True, zeroline=False)
    fig.update_yaxes(range=[x0, x1], showgrid=True, zeroline=False)

    for col_idx, stage in enumerate(stage_order, start=1):
        xref, yref = _axis_ref(col_idx)
        # y = x
        fig.add_shape(
            type="line",
            x0=x0, y0=x0, x1=x1, y1=x1,
            xref=xref, yref=yref,
            line=dict(color="black", width=2),
        )
        # ± WITHIN_DAYS bands
        for sgn in (-1, +1):
            fig.add_shape(
                type="line",
                x0=x0, y0=x0 + sgn * WITHIN_DAYS,
                x1=x1, y1=x1 + sgn * WITHIN_DAYS,
                xref=xref, yref=yref,
                line=dict(color="gray", dash="dash", width=1),
            )
        # MAE annotation
        xd, yd = _axis_domain_ref(col_idx)
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref=xd,
            yref=yd,
            text=f"MAE: {mae_by_stage.get(stage, 0):.2f} days",
            showarrow=False,
            align="left",
            font=dict(size=13, color="crimson", family="Arial"),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
        )

    fig.for_each_annotation(
        lambda a: a.update(text=a.text.replace("stage_id=", "Stage "))
    )
    return fig


def plot_mae_scattermap_grid(
    plot_df: pd.DataFrame,
    years: tuple = (2023, 2024, 2025),
    stages: tuple = (1, 2, 3),
    map_style: str = "carto-positron",
    color_col: str = "mae_state",
    size_col: str = "mae",
    zoom: float = 3.6,
) -> go.Figure:
    """MAE scatter-map grid (Year × Stage)."""
    df = plot_df.copy()
    df["year"] = df["year"].astype(int)
    df["stage_id"] = df["stage_id"].astype(int)

    cmin = float(df[color_col].quantile(0.05))
    cmax = float(df[color_col].quantile(0.95))
    if cmin == cmax:
        cmin, cmax = float(df[color_col].min()), float(df[color_col].max())

    center = dict(
        lat=float(df["latitude"].mean()), lon=float(df["longitude"].mean())
    )

    subplot_titles = [f"Year {y} · Stage {s}" for s in stages for y in years]
    fig = make_subplots(
        rows=len(stages),
        cols=len(years),
        specs=[[{"type": "map"} for _ in years] for _ in stages],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.06,
    )

    ncol = len(years)
    hover_cols = ["location_id", "location_state", "n", "mae", "mae_state"]

    for r, s in enumerate(stages, start=1):
        for c, y in enumerate(years, start=1):
            sub = df[
                (df["stage_id"] == s) & (df["year"] == y)
            ].dropna(subset=["latitude", "longitude", color_col, size_col])
            if sub.empty:
                continue

            max_size = max(float(sub[size_col].max()), 1e-9)
            sizeref = max_size / (12**2)
            sub_cd = sub.reindex(columns=hover_cols)

            show_scale = bool(r == 1 and c == ncol)
            fig.add_trace(
                go.Scattermap(
                    lat=sub["latitude"],
                    lon=sub["longitude"],
                    mode="markers",
                    marker=dict(
                        size=sub[size_col],
                        sizemode="area",
                        sizeref=sizeref,
                        sizemin=3,
                        color=sub[color_col],
                        cmin=cmin,
                        cmax=cmax,
                        colorscale="Turbo",
                        opacity=0.85,
                        showscale=show_scale,
                        colorbar=(
                            dict(title="MAE (days)", thickness=14, len=0.75, y=0.5)
                            if show_scale
                            else None
                        ),
                    ),
                    customdata=sub_cd.to_numpy(),
                    hovertemplate=(
                        "Location: %{customdata[0]}<br>"
                        "State: %{customdata[1]}<br>"
                        "n: %{customdata[2]}<br>"
                        "MAE: %{customdata[3]:.2f} days<br>"
                        "State mean MAE: %{customdata[4]:.2f} days<br>"
                    ),
                    showlegend=False,
                ),
                row=r,
                col=c,
            )

            idx = (r - 1) * ncol + c
            key = "map" if idx == 1 else f"map{idx}"
            fig.update_layout(
                **{key: dict(style=map_style, center=center, zoom=zoom)}
            )

    fig.update_layout(
        template="plotly_white",
        title=dict(text="MAE Map (location-level)", x=0.5, xanchor="center"),
        margin=dict(l=20, r=20, t=80, b=20),
        font=dict(family="Arial, sans-serif", size=16),
    )
    fig.for_each_annotation(lambda a: a.update(font=dict(size=14)))
    return fig


# ────────────────────────────────────────────────────────────
# Upload helpers
# ────────────────────────────────────────────────────────────
def _upload_html(s3: S3Manager, fig: go.Figure, name: str, prefix: str) -> str:
    """Write a Plotly figure as HTML and upload to S3."""
    key = f"{prefix}/{name}"
    html_str = fig.to_html(include_plotlyjs="cdn")
    s3.upload_file(html_str, key)
    logger.info("Uploaded %s → s3://%s/%s", name, s3.bucket_name, key)
    return key


def _upload_csv(s3: S3Manager, df: pd.DataFrame, name: str, prefix: str) -> str:
    key = f"{prefix}/{name}"
    s3.upload_file(df, key)
    logger.info("Uploaded %s → s3://%s/%s", name, s3.bucket_name, key)
    return key


# ────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────
def run(
    pred_key: str = DEFAULT_PRED_KEY,
    eval_key: str = DEFAULT_EVAL_KEY,
    output_prefix: str = OUTPUT_PREFIX,
) -> None:
    """Execute the full model-evaluation pipeline."""
    # ── Config & services ──────────────────────────────────
    config_path = Path(__file__).resolve().parents[1] / "config" / "weeviltrak_v2.3.yml"
    config = WeevillTrakConfig(config_path)
    bucket = config.config.get("data_source", {}).get("s3_bucket", "sps-ds-bucket")
    s3 = S3Manager(bucket)

    # ── Load data ──────────────────────────────────────────
    pred_df = load_predictions(s3, pred_key)
    evaluation_df = load_evaluation(s3, eval_key)

    # ── Metrics ────────────────────────────────────────────
    metrics_df = compute_metrics(evaluation_df)
    logger.info("Metrics summary:\n%s", metrics_df.to_string(index=False))

    # ── Build location-level aggregation for map ───────────
    plot_df = (
        evaluation_df.dropna(
            subset=["error_days", "latitude", "longitude", "location_state"]
        )
        .groupby(
            [
                "year", "stage_id", "location_id",
                "latitude", "longitude", "location_state", "location_name",
            ],
            as_index=False,
        )
        .agg(
            n=("error_days", "size"),
            rmse=("error_days", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            mae=("error_days", lambda x: float(np.mean(np.abs(x)))),
            bias=("error_days", "mean"),
            median_abs_error=("error_days", lambda x: float(np.median(np.abs(x)))),
            within_7d=("error_days", lambda x: float(np.mean(np.abs(x) <= 7))),
        )
    )
    plot_df["mae_state"] = plot_df.groupby(
        ["year", "stage_id", "location_state"]
    )["mae"].transform("mean")

    # ── Generate plots ─────────────────────────────────────
    fig_scatter = plot_predicted_vs_actual(evaluation_df, metrics_df)
    fig_mae_map = plot_mae_scattermap_grid(plot_df)

    # ── Upload to S3 ───────────────────────────────────────
    _upload_csv(s3, metrics_df, "metrics_summary.csv", output_prefix)
    _upload_html(s3, fig_scatter, "predicted_vs_actual_doy.html", output_prefix)
    _upload_html(s3, fig_mae_map, "mae_map.html", output_prefix)

    logger.info("All outputs uploaded to s3://%s/%s/", bucket, output_prefix)


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WeevilTrak model evaluation")
    p.add_argument("--pred-key", default=DEFAULT_PRED_KEY, help="S3 key for predictions CSV")
    p.add_argument("--eval-key", default=DEFAULT_EVAL_KEY, help="S3 key for evaluation CSV")
    p.add_argument("--output-prefix", default=OUTPUT_PREFIX, help="S3 prefix for outputs")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    args = _parse_args()
    run(
        pred_key=args.pred_key,
        eval_key=args.eval_key,
        output_prefix=args.output_prefix,
    )
