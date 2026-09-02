# WeevilTrak v2.4 Client Delivery Manual

This directory is the client-facing entry point for the WeevilTrak v2.4
release. Start with [QUICKSTART.md](QUICKSTART.md), then use the notebook in
`notebooks/` with the local release bundle.

## Contents

| Path | Purpose |
| --- | --- |
| `INSTALL.md` | Python environment and dependency setup |
| `QUICKSTART.md` | Local pickle loading and daily prediction workflow |
| `MODEL_CARD.md` | Model purpose, targets, inputs, and limitations |
| `DATA_CONTRACT.md` | Required input and output fields |
| `DEPLOYMENT_GUIDE.md` | Daily operational process and termination policy |
| `artifacts/` | Place the immutable release bundle here |
| `inference/` | Local, pickle-based inference helper |
| `examples/` | Schema-valid example input and output contract |
| `notebooks/` | Customer-facing walkthrough |

## Delivery boundary

No credentials, production Redshift data, weather caches, or generated run
outputs belong in this directory. Model artifacts are binary release files and
must be delivered separately or placed in `artifacts/` before use.
