from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "outputs/cache/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.backtesting import BacktestingFramework
from app.settings import setup_logger

logger = setup_logger()


def build_stage1_event_windows(
    weevil_data: pd.DataFrame,
    years: list[int],
    days_before: int,
    days_after: int,
) -> pd.DataFrame:
    """Build one prediction-date row per Stage 1 event window."""
    events = weevil_data.copy()
    events["stage_date"] = pd.to_datetime(events["stage_date"]).dt.normalize()
    events["location_id"] = events["location_id"].astype(str)
    events["stage_id"] = pd.to_numeric(events["stage_id"], errors="coerce").astype(int)
    events = events[
        (events["stage_id"] == 1)
        & (events["year"].isin(years))
    ].copy()

    rows = []
    for row in events.itertuples(index=False):
        stage_date = pd.to_datetime(row.stage_date).normalize()
        for prediction_date in pd.date_range(
            stage_date - pd.Timedelta(days=days_before),
            stage_date + pd.Timedelta(days=days_after),
            freq="D",
        ):
            rows.append(
                {
                    "location_id": str(row.location_id),
                    "stage_id": 1,
                    "event_year": int(row.year),
                    "stage_date": stage_date,
                    "prediction_date": pd.to_datetime(prediction_date).normalize(),
                    "latitude": float(row.latitude),
                    "longitude": float(row.longitude),
                }
            )

    windows = pd.DataFrame(rows)
    if windows.empty:
        return windows

    windows = windows.drop_duplicates(
        subset=["location_id", "stage_id", "event_year", "stage_date", "prediction_date"],
        keep="first",
    ).copy()
    windows["days_from_event"] = (
        windows["prediction_date"] - windows["stage_date"]
    ).dt.days
    windows = windows.sort_values(
        ["event_year", "location_id", "stage_date", "prediction_date"]
    ).reset_index(drop=True)
    return windows


def _plot_predictions(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    work = df.copy()

    for _, group in work.groupby(["event_year", "location_id", "stage_date"], sort=False):
        ax.plot(
            group["days_from_event"],
            group["predicted_days"],
            color="#2f5bea",
            alpha=0.10,
            linewidth=0.8,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Issue 1 Validation: Stage 1 PredictedDays Around Actual Stage Date")
    ax.set_xlabel("Days From Actual Stage 1 Date")
    ax.set_ylabel("Predicted Days")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_summary(
    df: pd.DataFrame,
    output_path: Path,
    figure_path: Path,
    csv_path: Path,
    years: list[int],
    days_before: int,
    days_after: int,
    window_years: int | None,
) -> None:
    if df.empty:
        lines = [
            "# Issue 1 Signed Target Validation Summary",
            "",
            "未生成任何预测结果。",
        ]
    else:
        focus_days = {-1, 0, 1}
        min_per_event = (
            df.groupby(["event_year", "location_id", "stage_date"], as_index=False)["predicted_days"]
            .min()
            .rename(columns={"predicted_days": "min_predicted_days"})
        )
        focus_df = df[df["days_from_event"].isin(focus_days)].copy()
        focus_abs_error_median = pd.to_numeric(
            focus_df["abs_error_days"], errors="coerce"
        ).median()
        focus_termination_count = int(
            focus_df.groupby(["event_year", "location_id", "stage_date"])["predicted_days"]
            .apply(lambda s: (pd.to_numeric(s, errors="coerce") <= 0).any())
            .sum()
        ) if not focus_df.empty else 0
        lines = [
            "# Issue 1 Signed Target Validation Summary",
            "",
            "## 配置",
            "",
            "- 模型配置：`app/config/weeviltrak_v2.3.yml`",
            f"- 测试年份：`{', '.join(map(str, years))}`",
            f"- 训练窗口：`{window_years} 年滑动窗口`" if window_years is not None else "- 训练窗口：expanding window",
            "- 验证对象：仅历史 `location_id`，只看 `Stage 1`",
            f"- 时间窗：`[-{days_before}, +{days_after}]` 围绕真实 `stage_date`",
            "",
            "## 结果文件",
            "",
            f"- 图：`{figure_path}`",
            f"- 明细：`{csv_path}`",
            "",
            "## 摘要",
            "",
            f"- 预测行数：{len(df)}",
            f"- 唯一 location_id：{df['location_id'].nunique()}",
            f"- 唯一事件窗口：{df[['event_year', 'location_id', 'stage_date']].drop_duplicates().shape[0]}",
            f"- 全局最小 predicted_days：{df['predicted_days'].min():.3f}",
            f"- 全局最大 predicted_days：{df['predicted_days'].max():.3f}",
            f"- 各事件窗口最小 predicted_days 的中位数：{min_per_event['min_predicted_days'].median():.3f}",
            f"- 发生终止（窗口内任一天 predicted_days == 0）的事件窗口数：{int(df.groupby(['event_year', 'location_id', 'stage_date'])['predicted_days'].apply(lambda s: (pd.to_numeric(s, errors='coerce') <= 0).any()).sum())}",
            f"- 中位绝对日期误差（天）：{pd.to_numeric(df['abs_error_days'], errors='coerce').median():.3f}",
            f"- 核心点位 `days_from_event ∈ {{-1, 0, +1}}` 的中位绝对日期误差（天）：{focus_abs_error_median:.3f}",
            f"- 核心点位 `days_from_event ∈ {{-1, 0, +1}}` 内发生终止的事件窗口数：{focus_termination_count}",
            "",
            "## 业务目标",
            "",
            "- 目标1：是否终止。观察 `predicted_days` 是否在真实事件附近 7 天内收敛到 0。",
            "- 目标2：预测是否准确。观察整个窗口内 `predicted_stage_date` 与真实 `stage_date` 的误差。",
            "",
            "## 说明",
            "",
            "- 当前只记录现象，不给出通过/不通过结论。",
            "- 主图展示业务输出 `predicted_days`，横轴为 `days_from_event`。",
            "- 核心摘要额外强调 `days_from_event = -1, 0, +1`，用于观察最接近业务判断点的表现。",
        ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _predict_for_windows(
    framework: BacktestingFramework,
    model,
    event_windows: pd.DataFrame,
    processed_weather_unique: pd.DataFrame,
    model_train_cutoff_date: str,
) -> pd.DataFrame:
    prediction_dates = sorted(
        pd.to_datetime(event_windows["prediction_date"]).dt.date.unique()
    )
    parts = []

    for d in prediction_dates:
        date_mask = pd.to_datetime(event_windows["prediction_date"]).dt.date == d
        locations_for_date = sorted(
            event_windows.loc[date_mask, "location_id"].astype(str).unique().tolist()
        )
        if not locations_for_date:
            continue

        out_df = framework.predict_with_model_for_date(
            model=model,
            test_date=d,
            locations_for_date=locations_for_date,
            model_train_cutoff_date=model_train_cutoff_date,
            processed_weather_unique=processed_weather_unique,
        )
        if out_df is None or out_df.empty:
            continue

        key_cols = ["prediction_date", "location_id", "stage_id"]
        if out_df.duplicated(subset=key_cols, keep=False).any():
            out_df = framework.aggregate_predictions_unique(out_df)

        out_df = out_df.copy()
        out_df["prediction_date"] = pd.to_datetime(out_df["prediction_date"]).dt.normalize()
        out_df["location_id"] = out_df["location_id"].astype(str)
        parts.append(out_df)

    if not parts:
        return pd.DataFrame()

    preds = pd.concat(parts, ignore_index=True)
    detail = event_windows.merge(
        preds[
            [
                "location_id",
                "stage_id",
                "prediction_date",
                "raw_signed_days",
                "predicted_days",
                "predicted_stage_date",
            ]
        ],
        on=["location_id", "stage_id", "prediction_date"],
        how="left",
    )
    return detail.sort_values(
        ["event_year", "location_id", "stage_date", "prediction_date"]
    ).reset_index(drop=True)


def run_validation(
    config_path: str,
    years: list[int],
    days_before: int,
    days_after: int,
    output_dir: Path,
    window_years: int | None,
) -> pd.DataFrame:
    framework = BacktestingFramework(config_path=config_path, output_dir=str(output_dir))

    max_year = max(years)
    latest_date = f"{max_year}-12-31"
    weevil_data = framework.data_service.pull_weevil_data(today=latest_date)
    event_windows = build_stage1_event_windows(
        weevil_data=weevil_data,
        years=years,
        days_before=days_before,
        days_after=days_after,
    )
    if event_windows.empty:
        raise RuntimeError("No Stage 1 event windows found for the requested years.")

    test_dates = sorted(pd.to_datetime(event_windows["prediction_date"]).dt.date.unique())
    processed_weather_unique = framework.prepare_weather(weevil_data, test_dates)

    detail_parts = []
    for year in years:
        cutoff_date = pd.Timestamp(year=year, month=3, day=1)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")
        logger.info("Running Issue 1 validation for year=%s cutoff=%s", year, cutoff_str)

        model = framework.train_model_for_cutoff(
            cutoff_date=cutoff_date,
            processed_weather_unique=processed_weather_unique,
            training_window_years=window_years,
        )
        year_windows = event_windows[event_windows["event_year"] == year].copy()
        if year_windows.empty:
            continue

        detail_parts.append(
            _predict_for_windows(
                framework=framework,
                model=model,
                event_windows=year_windows,
                processed_weather_unique=processed_weather_unique,
                model_train_cutoff_date=cutoff_str,
            )
        )

    if not detail_parts:
        return pd.DataFrame()

    detail = pd.concat(detail_parts, ignore_index=True)
    detail["actual_stage_date"] = pd.to_datetime(detail["stage_date"]).dt.normalize()
    detail["predicted_stage_date"] = pd.to_datetime(detail["predicted_stage_date"], errors="coerce").dt.normalize()
    detail["error_days"] = (
        detail["predicted_stage_date"] - detail["actual_stage_date"]
    ).dt.days
    detail["abs_error_days"] = pd.to_numeric(detail["error_days"], errors="coerce").abs()
    detail["terminated"] = pd.to_numeric(detail["predicted_days"], errors="coerce") <= 0
    return detail


def _parse_years(value: str) -> list[int]:
    years = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not years:
        raise ValueError("At least one test year is required.")
    return sorted(set(years))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Issue 1 signed target validation.")
    parser.add_argument(
        "--config",
        default="app/config/weeviltrak_v2.3.yml",
        help="Config path.",
    )
    parser.add_argument(
        "--test-years",
        default="2023,2024,2025",
        help="Comma-separated test years, e.g. 2023,2024,2025.",
    )
    parser.add_argument(
        "--days-before",
        type=int,
        default=3,
        help="Days before actual Stage 1 date to include in the validation window.",
    )
    parser.add_argument(
        "--days-after",
        type=int,
        default=3,
        help="Days after actual Stage 1 date to include in the validation window.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/implemention_results",
        help="Directory for generated outputs.",
    )
    parser.add_argument(
        "--window-years",
        type=int,
        default=5,
        help="Rolling training window in years. Use 0 to disable and run expanding window.",
    )
    args = parser.parse_args()

    years = _parse_years(args.test_years)
    window_years = None if args.window_years == 0 else args.window_years
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_path = output_dir / "issue1_signed_target_validation.png"
    csv_path = output_dir / "issue1_signed_target_validation.csv"
    summary_path = output_dir / "issue1_signed_target_validation_summary.md"

    predictions = run_validation(
        config_path=args.config,
        years=years,
        days_before=args.days_before,
        days_after=args.days_after,
        output_dir=output_dir,
        window_years=window_years,
    )
    predictions.to_csv(csv_path, index=False)
    _plot_predictions(predictions, figure_path)
    _write_summary(
        predictions,
        summary_path,
        figure_path,
        csv_path,
        years=years,
        days_before=args.days_before,
        days_after=args.days_after,
        window_years=window_years,
    )
    logger.info("Saved Issue 1 validation outputs under %s", output_dir)


if __name__ == "__main__":
    main()
