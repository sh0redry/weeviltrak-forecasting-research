# Quick Start: Local Daily Inference

1. Put the release files in `manual/artifacts/`:

   - `base_model.pkl`
   - `termination_model.pkl`
   - `config.yml`
   - `manifest.json`

2. Prepare a CSV with one row per `prediction_date` × location × `stage_id`.
   The required columns are described in [DATA_CONTRACT.md](DATA_CONTRACT.md).

3. Run local inference:

```python
import pandas as pd
from manual.inference.weeviltrak_inference import LocalWeevilTrakRelease

release = LocalWeevilTrakRelease.load("manual/artifacts")
history = pd.read_csv("manual/examples/sample_daily_features.csv")
trajectory = release.predict_trajectory(history)
today = release.predict_today(history)
today.to_csv("predictions_today.csv", index=False)
```

`predict_trajectory()` returns the full visible seasonal trajectory. It is the
correct input to learned termination because its rolling features use only the
predictions available on or before each prediction date. `predict_today()`
returns the latest available day from that trajectory.

The example CSV demonstrates field names only. A real deployment must provide
daily weather-derived features for the full visible season, not a single day
of raw weather readings.
