# WeevilTrak v2.4 — local research replica

This directory is a non-destructive local replica of the original `weeviltrak/`
tree. The source tree is unchanged. The local profile preserves v2.4 model and
research semantics while replacing EC2/S3 paths with relative filesystem paths.

The production pipeline still exists under `app/pipeline/` for reference. Its
live Redshift, S3 and private `griddedweather` operations are not invoked by the
local CLI. See [Local reproduction guide](docs/LOCAL_REPRODUCTION.md) for the
complete setup and command sequence, and [audit report](docs/LOCAL_AUDIT.md) for
architecture, external dependencies, risks and publication exclusions.

## Safe quick start

Use Python 3.11 from this directory:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt

# Synthetic data is only an infrastructure smoke test.
python -m app.local.cli prepare-mock
python -m app.local.cli train
python -m app.local.cli predict --date 2026-05-01
python -m app.local.cli walk-forward
python -m unittest discover -s tests/unit -v
```

For research results, replace the generated mock CSV with an authorized local
export of daily engineered features matching the schema documented in the
reproduction guide. Never draw model-performance conclusions from mock weather.
