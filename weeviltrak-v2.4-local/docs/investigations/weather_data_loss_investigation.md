# Weather Data Loss Investigation Report

**Date:** 2026-03-11  
**File Changed:** `app/services/data_preparation_service.py`  
**Status:** Fixed and verified

---

## Summary

`query_weather_data_for_weeviltrak_locs()` was returning only **1,302 rows** (covering ~3 location_ids) instead of the expected **~31,500 rows** (covering all ~70 unique grid points across 341 location/year requests). The root cause was a floating-point precision mismatch in the `griddedweather` library's `location_id` merge logic, which silently dropped `location_id` for 96% of weather rows. A downstream `_in_bounds` filter then discarded all rows without a valid `location_id`.

---

## Investigation Steps

### 1. Log File Analysis (`app_20260311_145748.log`)

From the log file of a previous run, we observed:

| Metric | Value |
|--------|-------|
| Weevil records pulled | 922 |
| Location/year requests built | 341 (70 unique grid points) |
| Cache index hits | 340 (but `cached_parts=0`) |
| Redshift parallel fetch | 340 gaps → 680 tasks → **31,620 rows** |
| Final weather records returned | **1,395** |

**Key finding:** 31,620 rows were fetched from Redshift, but only 1,395 survived to the final output — a **95.6% data loss**.

### 2. Diagnostic Instrumentation

Added temporary logging to `query_weather_data_for_weeviltrak_locs()` to trace data at each processing step:

```
[DIAG] After pull_weather_data: shape=(31434, 36)
[DIAG] location_id: 3 unique, 30132 NaN out of 31434 rows
[DIAG] location_id sample values: ['13', '27', '3']
[DIAG] Sample location_year_keys: ['nan__2015', '13__2015', '27__2015', ...]
[DIAG] Sample bounds_map keys: ['10__2015', '10__2016', '10__2017', ...]
[DIAG] Matched keys: 14 out of 25 in data, 339 in bounds_map
[DIAG] _in_bounds filter: 31434 -> 1302 rows (30132 dropped)
```

### 3. Root Cause Identified

The `griddedweather` library's `pull_weather_data()` method merges `location_id` back onto weather data using a **three-column join**:

```python
# griddedweather/client.py line ~237
weather = weather.merge(
    mapping,
    on=["place_id", "centroid_lat", "centroid_lon"],
    how="left",
)
```

The `mapping` DataFrame contains centroid coordinates from the initial `match_nearest_centroids()` lookup (which queries a centroids-only parquet/table for a reference date). The actual weather data rows contain coordinates from the weather archive table itself. These **floating-point lat/lon values do not match exactly** across the two sources due to precision differences, causing the merge to produce `NaN` for `location_id` on most rows.

Only 3 out of 70+ grid points happened to have exact float matches — those 3 location_ids were the only ones that survived.

The downstream `_in_bounds()` filter constructs a `location_year_key` like `"{location_id}__{year}"`. For rows with `location_id = NaN`, this becomes `"nan__2015"` etc., which never matches anything in `bounds_map`, so those rows are dropped.

**Data flow with the bug:**
```
Redshift → 31,620 rows
  ↓ griddedweather merge on (place_id, centroid_lat, centroid_lon)
  → 96% of rows get location_id = NaN (float mismatch)
  ↓ _in_bounds filter (needs valid location_year_key)
  → 30,132 rows dropped
  ↓ Final output
  → 1,395 rows (only 3 locations)
```

---

## Fix Applied

### New method: `_remap_location_id()`

Added a static method to `DataPreparationService` that bypasses the unreliable coordinate-based merge:

1. Extracts unique `(place_id, centroid_lat, centroid_lon)` from the returned weather data
2. Extracts unique `(location_id, lat, lon)` from the original `loc_requests`
3. For each location, finds the nearest grid centroid using Euclidean distance on degrees (sufficient at 10×10 km resolution)
4. Builds a clean `place_id → location_id` mapping
5. Drops the broken `location_id` column and merges the correct mapping on `place_id` alone

### Code location

```
app/services/data_preparation_service.py
├── query_weather_data_for_weeviltrak_locs()  # Added call to _remap_location_id after pull
└── _remap_location_id()                       # New method (lines ~275-325)
```

### Key change in `query_weather_data_for_weeviltrak_locs()`:

```python
df = self.weather_client.pull_weather_data(loc_requests)
df = self._normalize_weather_columns(df)

if df.empty:
    ...

# FIX: Rebuild location_id mapping using place_id only
if "place_id" in df.columns:
    df = self._remap_location_id(df, loc_requests)
```

---

## Verification

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Rows returned | 1,302 | **31,527** |
| Unique location_ids | 3 | **70** |
| Location/year combos matched | 14/339 | **339/339** |
| Data loss | 95.8% | **0%** |

The fix was verified by re-running the weather pull cell in `notebooks/test_train_predict.ipynb` (cell 12, Step 2).

---

## Why the Bug Was Silent

1. The `griddedweather` library uses a `left` merge — no error is raised when coordinates don't match; `location_id` simply becomes `NaN`.
2. The `_in_bounds` filter treats missing `location_year_key` as "out of bounds" (returns `False`), silently dropping rows rather than raising.
3. The method still returned a valid DataFrame (just much smaller), so no exception propagated upstream.
4. The 3 locations that did match happened to produce a plausible-looking (non-empty) result, masking the problem.

---

## Recommendations

1. **Upstream fix:** Consider filing an issue on the `griddedweather` library to merge on `place_id` alone (or use rounded/tolerance-based coordinate matching) instead of exact float comparison.
2. **Validation:** Add a post-fetch assertion in the pipeline that checks `weather_data` covers all expected `location_id` values from `weevil_data`, not just a subset.
3. **Cache re-read efficiency:** The S3 cache index is re-read on every single request (340 times in the log). Consider caching it in memory for the duration of a single `pull_weather_data` call.
