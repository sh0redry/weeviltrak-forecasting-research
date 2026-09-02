# Local replication audit

## Project structure and entry points

The maintained code is under `app/`; notebooks and scripts are supporting
research evidence. The source documentation claims a `tests/` directory, but it
was absent in the copied source. This replica adds focused local tests.

| Directory | Audited contents |
| --- | --- |
| `app/config` | v2.1–v2.5 YAML policies and config loader; v2.4 is canonical |
| `app/models` | Random Forest and LightGBM managers plus factory/abstract interface |
| `app/pipeline` | train/predict, prediction-only, walk-forward, hindcast, evaluation and termination |
| `app/services` | Redshift access, weather acquisition/features, canonical/manual events and coverage gate |
| `scripts` | backtesting, hindcast, validation, reporting, weather maintenance and S3 release tooling |
| `data_exports` | frozen raw Redshift snapshot, canonical v2.4 labels and export summary |
| `manual` | client inference code, schemas and release-bundle instructions; actual pickles are absent |
| `notebooks` | production/experimental evidence plus large sensor investigations and saved outputs |
| `docs` | architecture, data risks, runbooks, handoff, model/version and archived plans |

| Surface | Entry point | Role |
| --- | --- | --- |
| Production train/predict | `app/pipeline/train_predict.py` | Redshift events → weather → features → Base model → OOF termination → S3 |
| Production prediction-only | `app/pipeline/predict_only.py` | Load approved S3 artifacts and predict |
| Walk-forward | `app/pipeline/backtesting.py` | Historical yearly/cutoff validation; S3 upload defaults exist |
| Hindcast | `app/pipeline/hindcast.py` | Historical what-if runs |
| Evaluation | `app/pipeline/model_eval.py` | Error metrics and plots, originally S3-oriented |
| Client inference | `manual/inference/weeviltrak_inference.py` | Local immutable pickle bundle |
| Local replica | `app/local/cli.py` | Credential-free local train/predict/walk-forward |

## v2.4 data and model flow

1. Read pest events and filter by cutoff.
2. Canonicalize Stage 1→1, Phase 1 midpoint→4, Stage 2→2 and Stage 3/Phase 2→3;
   exclude `12/31` placeholders and optionally add curated manual events.
3. Pull gridded weather, map points to place IDs, and calculate seasonal
   cumulative plus 14-day rolling GDD/CDD/humidity/precipitation features.
4. Join each daily feature row to its future event and train
   `signed_days_to_event`, including ten real post-event feature days.
5. Fit LightGBM with the configured decreasing monotonic constraint on
   cumulative GDD.
6. Generate chronological OOF Base trajectories and fit Ridge termination.
7. Emit raw signed days, non-negative business days, event dates and stable
   Stage/Phase business labels.

The local workflow preserves those definitions. Only its explicit `prepare-mock`
command synthesizes inputs, solely for infrastructure verification.

## Dependencies

Core local runtime: Python 3.11, pandas, NumPy, scikit-learn, SciPy, LightGBM,
PyYAML and PyArrow. The production project additionally declares boto3,
redshift-connector, geopandas/shapely, tables, pyreadr, notebooks and plotting
tools. `griddedweather==0.4.10` came from a private package source. The bundled
compatibility package supplies import-safe local object storage and deliberately
fails any attempted live weather query.

The copied `poetry.lock` records the production dependency resolution. It is not
claimed as a newly verified local lock because regenerating it can access public
package indexes. Use `requirements-local.txt` for this replica until a reviewed
Python 3.11 lock is generated.

## EC2/AWS coupling found

- `PipelineBase` eagerly initializes Redshift, S3 and the private weather client.
- Base and termination models are loaded/saved through S3 manager calls.
- prediction coverage is read from an S3 key and predictions are uploaded.
- backtesting defaults include S3 upload and optional local deletion.
- release creation stages and copies immutable objects inside S3.
- one prediction docstring used `/home/ec2-user/...`; the replica replaces the
  example with a relative path.
- a committed runtime log and PID file came from an EC2 execution; neither is
  source and both are ignored in the replica.
- no standalone EC2 provisioning script, systemd unit, Dockerfile or Terraform
  definition exists in the supplied directory.

## Environment variables

Production code reads `S3_BUCKET_NAME`, `REDSHIFT_HOST`, `REDSHIFT_PORT`,
`REDSHIFT_USER`, `REDSHIFT_PASSWORD`, `DATABASE_HOST`, `DATABASE_PORT` and
`DATABASE_NAME`; AWS SDK credential/profile/region variables may also affect
boto3. Documentation also names `REDSHIFT_DATABASE`, while code primarily reads
`DATABASE_NAME`, an inconsistency retained and documented in `.env.example`.

Local variables are non-secret path selectors: `WEEVILTRAK_MODE`,
`WEEVILTRAK_CONFIG`, `WEEVILTRAK_EVENTS`, `WEEVILTRAK_ENGINEERED_FEATURES` and
`WEEVILTRAK_LOCAL_STORE`.

## Missing external inputs and local alternatives

| Missing input | Production source | Local alternative |
| --- | --- | --- |
| Historical weather/cache | private package + Redshift/S3 | authorized engineered-feature CSV/Parquet; deterministic mock only for smoke tests |
| Spatial coverage | S3 key `weeviltrak_data/spatial_coverage.csv` | `place_id` supplied in engineered features |
| Approved Base/Ridge pickles | S3 model keys | train new local candidates into `outputs/`; never treat them as approved production artifacts |
| Private weather library | private Fury repository | bundled fail-closed compatibility API |
| Live pest events | Redshift table | supplied dated canonical export |

## Files that must not be published without review

Do not push these to a public GitHub repository:

- any `.env`, AWS config/credentials, SSH/TLS keys or credential-bearing shell history;
- `data_exports/`, especially the raw Redshift snapshot and canonical label
  export (names, precise locations, researcher/source metadata and proprietary data);
- `data/local/`, weather caches, spatial coverage and transferred source data;
- `outputs/`, logs, failure records, prediction files and evaluation details;
- `app/services/weather_updated_locations.log` and `.pid`;
- `model_testing_2025/`, `manual/artifacts/`, pickle/joblib files, manifests tied
  to controlled releases and any proprietary model artifact;
- notebooks with saved outputs, embedded source data, credentials, internal
  bucket paths or organization-specific analysis, until cleared and sanitized;
- `poetry.lock` until its private package-source reference is replaced by a
  reviewed local/public lock;
- `app/services/all_checks_weather_locations.csv` (precise coordinates);
- the private dependency URL/token or any downloaded proprietary package.

The repository-local `.gitignore` implements conservative defaults. Before a
public release, use `git status --ignored`, scan the full history for secrets,
strip notebook outputs, and obtain data/model licensing approval. `.gitignore`
does not remove secrets already committed to history.
