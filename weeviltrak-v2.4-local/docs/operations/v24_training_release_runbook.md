# v2.4 Training and Release Runbook

## Safety boundary

Training, weather pulls, Redshift reads, S3 publication, and artifact
replacement are different operations. Never run a command that can publish or
replace an artifact merely to inspect the system. Use a dated local output
directory and preserve the config, cutoff, logs, metrics, and manifest.

## Prerequisites

- Python 3.11 and Poetry: `poetry install`
- Required S3/Redshift credentials in environment variables; never commit them.
- A documented training cutoff and an approved data snapshot.
- Read [data contract and risks](../data/historical_data_contract_and_risks.md).

## Candidate training workflow

1. Confirm `app/config/weeviltrak_v2.4.yml` is the intended config.
2. Pull/canonicalize events and verify the recent three-year direct-label
   coverage gate for IDs `1, 4, 2, 3`.
3. If coverage blocks training, **do not weaken the policy ad hoc**. Continue
   using the last approved artifact through prediction-only flow and record the
   gap for review.
4. Train Base LightGBM on the approved training matrix.
5. Build chronological OOF Base trajectories and train learned Ridge
   termination only from those trajectories.
6. Validate feature schema, artifact loadability, trajectory/termination
   outputs, and versioned backtesting results before any publication.

The main orchestrator is `app/pipeline/train_predict.py`. It accesses external
systems and its module entry point uses a built-in prediction date, so inspect
the code/config and use a controlled environment before invoking it.

## Backtesting and evidence

- Use versioned outputs under `outputs/backtesting/v2.4/`.
- Distinguish trajectory-point metrics from termination-decision metrics; they
  answer different questions.
- Preserve the evaluation population, cutoff policy, weather snapshot, and
  comparison version with every claimed metric.
- Experimental notebooks are evidence, not a replacement for app behavior.

## Immutable 2027 release bundle

Create a controlled bundle with:

```bash
poetry run python scripts/release/create_v24_2027_release.py \
  --cutoff-date YYYY-MM-DD \
  --release-id v2.4-2027-YYYYMMDD \
  --train-and-publish
```

The release must contain Base and termination pickles, frozen config, and a
`manifest.json` with cutoff and hashes. Follow the client-facing
[`manual/`](../../manual/README.md) for local inference behavior.

## Post-release operation and rollback

- During season, daily weather/features update predictions; the released model
  does not retrain.
- Archive each input/output with release ID.
- First seven visible days publish Base output; termination may govern output
  from day eight onward.
- On a release defect, stop publication, retain the failed bundle and inputs,
  and roll back to the last approved immutable bundle. Do not overwrite a
  release in place.
