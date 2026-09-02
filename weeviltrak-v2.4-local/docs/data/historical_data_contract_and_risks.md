# Historical Data Contract and Known Risks

## Scope

This document defines how engineering should interpret the primary data sources
for v2.4. It is not authorization to alter Redshift or historical files.

## Source roles

| Source | Role | Engineering rule |
| --- | --- | --- |
| `europe_dna.europe_dna_sps_weeviltrak` | Operational pest-event source | Pull read-only through `DataPreparationService`; canonicalize before modeling. |
| Historical WhatsApp Excel delivery | Data-quality reference and audit input | Preserve as received; use explicit `MM/DD/YYYY` parsing. |
| `s3://sps-ds-bucket/gridded-weather/cache/10by10/` | Weather cache | Needed to rebuild historical feature rows; cache completeness must be checked. |
| `s3://sps-ds-bucket/weeviltrak_data/raw_data/` | Raw incoming data landing area | Inspect provenance before any ingestion decision. |
| `app/services/manual_events.py` | Curated supplemental events | Runtime-only input; never write these rows back to Redshift. |

## Canonical event contract

Raw Redshift Stage/Phase labels are not stable across the 2026 transition.
`app/services/canonical_events.py` is mandatory for production training and
evaluation.

| Raw meaning | Canonical ID | Business output |
| --- | ---: | --- |
| Stage 1 | 1 | Stage 1 |
| Stage 2 | 2 | Stage 2 |
| Stage 3 or Phase 2 | 3 | Stage 3 / Phase 2 |
| Paired Stage 1/2 midpoint; eligible controlled Phase 1 | 4 | Phase 1 |

Never fabricate Stage 1 or Stage 2 labels from an observed Phase 1 date.
Placeholder dates, especially `12/31`, are not direct event labels.

## Excel–Redshift audit findings

The audit notebook is
[`notebooks/experiments/redshift_excel_2024_alignment_audit.ipynb`](../../notebooks/experiments/redshift_excel_2024_alignment_audit.ipynb).
Its full-history comparison is evidence, not a data-repair script.

Using the 2016–2026 Excel delivery and the frozen Redshift snapshot:

- 3,825 of 3,896 Excel unique events (98.2%) are exact Redshift matches or
  strict month/day-swap recoveries.
- 377 high-confidence date-swap events are concentrated in three source files:
  2020 (111), 2023 (147), and 2024 (119).
- Redshift contains substantially more unique events than the Excel delivery;
  it is not currently a one-to-one mirror.
- After separately excluding `12/31` placeholders and non-placeholder records
  after the Excel as-of cutoff, 1,971 Redshift review candidates remain.

### Interpretation rules

1. A confirmed month/day-swap group may be proposed for correction only with
   source-file provenance and strict same-location/Stage matching.
2. `Redshift unresolved` is a matching status, not proof of an invalid date.
3. Records absent from Excel, Stage-label mismatches, and multi-year coverage
   differences require data-owner review; do not bulk-shift their dates.
4. Preserve the audit snapshot and exported review CSVs with any claim about
   data quality, because Redshift is mutable.

## Weather and feature contract

v2.4 Base Layer input features are:

```text
stage_id
cumu_gdd_air
rolling_gdd_air
cumu_cdd_air
rolling_cdd_air
rolling_humidity_mean_pct
cumu_precip_total_mm
doy
latitude
longitude
```

Feature calculations are seasonal and causal. Client inference must supply
engineered seasonal history, not one row of raw weather measurements. See
[`manual/DATA_CONTRACT.md`](../../manual/DATA_CONTRACT.md).

## Data-quality gates before training or release

- Confirm the intended cutoff date and snapshot provenance.
- Run canonicalization and recent direct-label coverage checks.
- Validate coordinates, dates, Stage/Phase mapping, and weather availability.
- Do not silently replace missing engineered features with zero.
- Keep local audit outputs under `outputs/`; do not treat them as source data.
