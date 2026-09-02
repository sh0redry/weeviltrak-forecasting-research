# Release Notes

## v2.4 2027 deployment policy

- Four direct business targets: Stage 1, Phase 1, Stage 2, Stage 3 / Phase 2.
- LightGBM Base Layer plus learned Ridge termination layer.
- Recent three-year training-window policy with coverage gating for retraining.
- Verified observed 2026 Phase 1 labels are eligible direct Phase 1 training
  labels; they are never fabricated as Stage 1 or Stage 2 labels.
- Phase 2 is mapped to Stage 3 / Phase 2.
- New daily deployment warm-up: first seven visible days use Base output;
  learned termination can publish from day eight.

Use the `manifest.json` delivered with an artifact bundle as the authoritative
record of a specific release's hashes and training cutoff.
