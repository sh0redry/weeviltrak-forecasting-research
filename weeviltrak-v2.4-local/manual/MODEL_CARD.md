# Model Card: WeevilTrak v2.4

## Purpose

WeevilTrak estimates timing of Annual Bluegrass Weevil lifecycle events from
location, calendar, and weather-derived features.

## Business outputs

| Canonical ID | Business output |
| ---: | --- |
| 1 | Stage 1 |
| 4 | Phase 1 |
| 2 | Stage 2 |
| 3 | Stage 3 / Phase 2 |

Phase 1 is a direct model target. Historical Phase 1 labels are primarily the
midpoint of paired observed Stage 1 and Stage 2 dates; verified observed 2026
Phase 1 labels are also eligible. Phase 2 is the same business event as Stage
3 and maps to canonical ID 3.

## Architecture

1. A LightGBM Base Layer predicts signed days remaining to each event.
2. A Ridge termination layer uses the causal history of Base predictions to
   calibrate and lock the business-facing date.
3. The first seven visible daily predictions for each location/event retain
   the Base output. The termination layer may govern the published output from
   the eighth visible day onward.

## Scope and limitations

- The model is a forecast aid, not a replacement for scouting or agronomic
  judgement.
- It requires valid feature engineering and weather coverage for each site.
- The 2027 release is inference-only during the season: new daily weather
  updates change predictions but do not retrain the model.
- Consult the release `manifest.json` for the exact training cutoff, feature
  schema, artifact hashes, and validation record.
