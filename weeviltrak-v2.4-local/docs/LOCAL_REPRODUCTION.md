# Local reproduction guide

## Scope and safety boundary

The local entry point is `python -m app.local.cli`. It performs no network
requests and has no S3 upload or Redshift write path. All generated files go to
relative paths under `data/local/` or `outputs/`. The original `../weeviltrak/`
directory and its source exports are never modified.

The deterministic mock weather generator validates wiring only. It is not an
approximation of the production weather product and cannot support scientific
or performance claims.

## Environment installation

Use 64-bit CPython 3.11 and run commands from this project root.

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt
python -m unittest discover -s tests/unit -v
```

`pyproject.toml` points the private `griddedweather` dependency to the bundled
offline compatibility package. The copied `poetry.lock` came from the production
tree and is intentionally retained as audit evidence, but it is out of sync with
that path substitution. Regenerate and review it before using `poetry install`.
`requirements-local.txt` is the authoritative local installation surface here.

Copy `.env.example` to `.env` only when convenient. Local mode requires no
secret. Never add AWS/Redshift values to tracked files.

## Data placement

The supplied canonical label snapshot is read-only input:

```text
data_exports/v24_redshift_training_events_2026-08-21/
  redshift_raw_snapshot_through_2026-08-21.csv
  v24_canonical_training_event_labels_2023-08-21_to_2026-08-21.csv
  export_summary.csv
```

Real local research needs a daily engineered feature CSV or Parquet. Put it at
`data/local/engineered_daily_features.parquet` or pass `--features PATH`. It must
contain one row per date/location and these columns:

```text
date, location_id, place_id, latitude, longitude,
cumu_gdd_air, rolling_gdd_air, rolling_humidity_mean_pct,
cumu_precip_total_mm, doy, cumu_cdd_air, rolling_cdd_air
```

The calculations must remain causal, seasonal, use a March 1 pest-year start,
and use the configured 14-day rolling window. `stage_id` is expanded by the
local workflow using canonical IDs `[1, 4, 2, 3]`.

On an authorized AWS-connected host, export the weather cache without changing
it, transfer the export through the approved secure channel, verify its checksum,
then place it under `data/local/`. The private weather library is not mocked for
research data acquisition.

## Smoke-test data

```bash
python -m app.local.cli prepare-mock
```

This writes `data/local/mock_engineered_daily_features.csv` and a metadata
sidecar explicitly marking the file as synthetic and unsuitable for research.

## Training

```bash
python -m app.local.cli train \
  --config app/config/weeviltrak_v2.4_local.yml \
  --events data_exports/v24_redshift_training_events_2026-08-21/v24_canonical_training_event_labels_2023-08-21_to_2026-08-21.csv \
  --features data/local/mock_engineered_daily_features.csv \
  --artifacts outputs/local/artifacts
```

For real research, change only `--features` to the authorized export. Training
keeps the signed target, 10 post-event days, three-year window, coverage gate,
LightGBM parameters and chronological OOF Ridge termination training.

Artifacts are written beneath `outputs/local/artifacts/`: Base and termination
pickles, frozen config, label coverage and a manifest with SHA-256 hashes.

## Generate predictions

The feature file must contain the entire visible trajectory from March 1 through
the requested date because learned termination uses lagged history.

```bash
python -m app.local.cli predict \
  --date 2026-05-01 \
  --features data/local/mock_engineered_daily_features.csv \
  --artifacts outputs/local/artifacts \
  --output outputs/local/predictions_2026-05-01.csv
```

The output retains raw signed days, clipped business days, event date, canonical
business name and learned-termination audit columns.

## Walk-forward validation

```bash
python -m app.local.cli walk-forward \
  --config app/config/weeviltrak_v2.4_local.yml \
  --events data_exports/v24_redshift_training_events_2026-08-21/v24_canonical_training_event_labels_2023-08-21_to_2026-08-21.csv \
  --features data/local/mock_engineered_daily_features.csv \
  --output-dir outputs/backtesting/v2.4-local
```

For each validation year, the Base model is fit strictly on earlier events.
Termination is fit only from earlier walk-forward trajectories. Do not compare
mock metrics to production or report them as model quality.

## Tests and checks

```bash
python -m compileall -q app tests vendor/griddedweather
python -m unittest discover -s tests/unit -v
python -m app.local.cli --help
```

The original AWS commands remain reference-only. They need approved credentials,
the proprietary weather package, spatial coverage, historical weather cache and
approved model artifacts; see `docs/operations/v24_training_release_runbook.md`.
