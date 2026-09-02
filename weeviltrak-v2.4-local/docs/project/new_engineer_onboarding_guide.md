# Welcome to WeevilTrak: A Guide for the Next Engineer

## What this project does

WeevilTrak forecasts the timing of Annual Bluegrass Weevil (ABW) lifecycle
events for a golf-course location.  It combines historical scouting events,
location coordinates, and accumulated weather conditions to answer a practical
question: **how many days remain until the next biologically meaningful ABW
event?**

The maintained production model is **v2.4**.  It is a Python machine-learning
pipeline, rather than a web application.  Its main deliverables are versioned
model artifacts and daily prediction outputs.  The repository also contains
notebooks used to validate model behavior and investigate the quality of the
historical data.

The four business outputs are:

| Canonical ID | Output shown to users | Meaning |
| ---: | --- | --- |
| 1 | Stage 1 | Peak activity of overwintered adults |
| 4 | Phase 1 | Midpoint between Stage 1 and Stage 2 |
| 2 | Stage 2 | Peak activity of early instar larvae inside the stem |
| 3 | Stage 3 / Phase 2 | Peak activity of later larvae; Phase 2 is the same business event |

The order of the IDs is deliberately not chronological: Phase 1 uses canonical
ID `4` for compatibility with the original Stage IDs.  Do not infer business
meaning from a raw ID.  Always use the canonical event mapping.

## The model in plain English

The v2.4 system has two layers.

1. The **Base Layer** is a LightGBM model.  For each location, day, and target
   event, it predicts `signed_days_to_event`: a positive value before the
   event, zero on the event date, and a negative value afterwards.  It sees
   causal weather-derived seasonal features, the target ID, and coordinates.
2. The **termination layer** is a Ridge model trained on chronological,
   out-of-fold Base trajectories.  It assesses the evolving Base prediction and
   determines when the event should be considered reached.  This is the final
   business-facing event date; it is not just a charting enhancement.

The first seven visible prediction days in a client release show the Base
output, because there is not yet enough trajectory history for the termination
features.  From day eight onward, the termination layer may govern the final
output.  Daily input data changes predictions, but it does **not** retrain the
released model during a season.

Phase 1 deserves special attention.  Pre-2026 historical Phase 1 labels are
constructed as the midpoint of paired observed Stage 1 and Stage 2 dates.  The
pipeline may also use eligible quality-controlled observed Phase 1 labels.  It
does not manufacture Stage 1 or Stage 2 observations from a Phase 1 record.

## Read these documents in this order

This order will give you the current operating model before you encounter
historical analysis or legacy configurations.

1. [`README.md`](../../README.md) for repository setup and common commands.
2. [`AGENTS.md`](../../AGENTS.md) for working conventions and the current model
   contract.  If you use an AI coding agent, it is its primary repository
   context.
3. [v2.4 architecture](../architecture/weeviltrak_model_process_overview.md)
   for the end-to-end data and model flow.
4. [Historical data contract and risks](../data/historical_data_contract_and_risks.md)
   before querying, interpreting, or changing any event data.
5. [v2.4 training and release runbook](../operations/v24_training_release_runbook.md)
   before training, backtesting, publishing, or replacing artifacts.
6. [Open risks and decisions](open_risks_and_decisions.md) for unresolved work
   that needs an owner or a decision from the data team/mentor.
7. [`manual/README.md`](../../manual/README.md) if you are working on the
   customer-facing immutable pickle bundle or local inference example.

The rest of `docs/`, `archive/`, and many experiment notebooks contain useful
history.  They are evidence, not automatically current policy.  In particular,
do not revive `v2.5` or older model behavior merely because a historical
notebook uses it.

## Read the code in this order

The following reading path mirrors the actual flow of a prediction.

1. [`app/config/weeviltrak_v2.4.yml`](../../app/config/weeviltrak_v2.4.yml):
   model policy, feature schema, canonical outputs, training window, and
   artifact/output locations.
2. [`app/services/canonical_events.py`](../../app/services/canonical_events.py):
   the authoritative translation from raw Stage/Phase records to the four
   model targets, including Phase 1 labels.
3. [`app/services/manual_events.py`](../../app/services/manual_events.py):
   the 11 curated, mentor-provided observations injected at runtime.  They are
   not a Redshift write-back mechanism.
4. [`app/services/data_preparation_service.py`](../../app/services/data_preparation_service.py):
   event pull, weather processing, causal feature engineering, and train/test
   matrix construction.
5. [`app/models/factory.py`](../../app/models/factory.py) and
   [`app/models/model_manager.py`](../../app/models/model_manager.py): Base
   model creation, fitting, prediction, and serialization.
6. [`app/pipeline/termination.py`](../../app/pipeline/termination.py):
   OOF termination training and final decision logic.
7. [`app/pipeline/train_predict.py`](../../app/pipeline/train_predict.py) and
   [`app/pipeline/predict_only.py`](../../app/pipeline/predict_only.py):
   training/release orchestration and approved-artifact inference.
8. [`app/pipeline/backtesting.py`](../../app/pipeline/backtesting.py):
   the historical evaluation path.

When modifying production behavior, follow the code path rather than copying a
notebook cell.  Notebooks are useful for investigation and visualization; the
app code and tests are the maintained implementation.

## Data sources and the most important safety rules

The operational pest-event table is
`europe_dna.europe_dna_sps_weeviltrak` in Redshift.  Weather features are built
from the gridded-weather cache in S3.  Historical Excel files are an important
reference, but they are not currently a clean one-to-one mirror of Redshift.

Known findings from the audit:

- `12/31` dates are placeholders, not event observations.
- Confirmed month/day-swapped dates are concentrated in specific 2020, 2023,
  and 2024 source batches.
- Some Redshift records have no corresponding Excel record and must not be
  declared wrong just because they are unmatched.
- The transition to Phase labels in 2026 means raw source Stage IDs no longer
  have a stable business interpretation.

Therefore:

- Canonicalize every event through `canonical_events.py` before model use.
- Do not bulk-correct dates based only on a notebook comparison.
- Do not turn missing weather features into zero without an approved policy.
- Treat any Redshift or S3 write as an explicit release/data-governance action,
  never as a convenient way to make an experiment pass.

## How to start working safely

Use Python 3.11 and Poetry:

```bash
poetry install
poetry env info
poetry run pytest tests/ -q
```

The test suite is the first safe confirmation that your local environment is
usable.  External data operations need valid S3 and Redshift credentials; do
not assume they are available in a test environment.

Before a production-facing code change, run the targeted tests closest to your
change and then the full suite.  The most valuable current checks are:

```bash
poetry run pytest tests/unit/services/test_canonical_events.py -q
poetry run pytest tests/unit/models/test_learned_termination.py -q
poetry run pytest tests/unit/services/test_post_event_zero_training.py -q
poetry run pytest tests/unit/pipeline/test_signed_target_outputs.py -q
poetry run pytest tests/ -q
```

Do not invoke the training module simply to inspect its command-line options:
the current entry point includes a built-in prediction-date workflow and can
attempt external data access.  Read the code and use the controlled release
workflow described in the runbook instead.

## Training, evaluation, and release principles

The intentional v2.4 training policy uses the most recent **three years** of
eligible data.  Earlier observations are excluded because their collection
process is less reliable.  A direct-label coverage gate checks all four
canonical targets before an artifact can be replaced.  A blocked retraining run
is a safe outcome: use the last approved artifact through the prediction-only
pipeline and escalate the missing coverage.

When evaluating a candidate, do not mix these metrics:

- **Trajectory-point metrics** measure daily remaining-days predictions.
- **Termination-decision metrics** measure the final operational date at which
  the system considers the event reached.

Both are useful, but they answer different questions.  Preserve the evaluation
population, data cutoff, weather snapshot, model version, and metric definition
with every result.  Generated files under `outputs/` are local evidence, not
source code or permanent source data.

The 2027 deployment bundle is an immutable directory containing the Base and
termination pickle artifacts, frozen configuration, and a manifest with hashes
and cutoff metadata.  Create and validate it with the release script only after
the candidate has passed the required checks.  The client-facing deployment
contract is documented in [`manual/`](../../manual/README.md).

## Useful notebooks, with the right expectations

| Notebook | What it is for |
| --- | --- |
| `notebooks/experiments/v24_whitepaper_figures.ipynb` | Reproducible v2.4 performance figures and tables. |
| `notebooks/experiments/stage_phase_2026_mapping_and_phase1_derivation.ipynb` | Evidence for the Stage/Phase mapping and Phase 1 strategies. |
| `notebooks/experiments/redshift_excel_2024_alignment_audit.ipynb` | Data-quality audit; do not treat it as an ingestion or repair script. |
| `notebooks/experiments/expanded_stages_pre2026_backtest.ipynb` | Historical Stage 1–7 research; not the production v2.4 target set. |
| `notebooks/experiments/v24_2027_release_deployment_smoke_test.ipynb` | Demonstrates immutable-bundle deployment behavior. |

Run notebooks with the Poetry Jupyter kernel.  Confirm their inputs and output
directories before execution; their saved outputs may be stale, local, or tied
to a historical snapshot.

## A practical first-week checklist

1. Set up the Poetry environment and pass the test suite.
2. Trace a small prediction from config through canonicalization, feature
   construction, Base inference, and termination.
3. Read the data audit before interpreting a surprising historical date.
4. Load, but do not replace, an approved release bundle and run the deployment
   smoke-test notebook.
5. Choose one small, well-tested issue.  Add or update the nearest unit test,
   run the full suite, and update the relevant current documentation.
6. Before proposing a release or data correction, review the runbook and open
   risks with the technical owner and data team.

## Final handoff principle

The safest way to extend WeevilTrak is to preserve its explicit boundaries:
canonical event semantics, causal features, chronological termination training,
the three-year coverage policy, immutable release artifacts, and a clear
separation between evidence notebooks and production code.  If a change crosses
one of those boundaries, make the decision visible in the config, tests,
documentation, and release record rather than hiding it in a notebook or a
local output file.
