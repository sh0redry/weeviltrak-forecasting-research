# griddedweather S3 Cache Investigation

**Date:** 2026-03-18  
**Status:** Root cause identified and verified

---

## Problem

Weather data was being re-downloaded from Redshift on every notebook run, even when using identical inputs. Expected behavior: the `griddedweather` package should cache fetched data to S3 and serve subsequent requests from cache.

---

## Investigation

Created `scripts/diagnose_weather_cache.py` to test each layer of the caching system.

### Findings

| Component | Status | Details |
|-----------|--------|---------|
| WeatherConfig | OK | Bucket: `sps-ds-bucket`, prefix: `gridded-weather/cache/10by10/` |
| Cache index on S3 | OK | 419 entries, 134 unique place_ids, date range 2015–2026 |
| Cache parquet files | OK | 420 files on S3, 3/3 test reads successful |
| Cache read (hit) | OK | Index correctly identifies matching segments |
| Cache write-back | **Blocked** | `upload_file()` hangs during SSO token refresh on slow network |

### Root Cause

The `griddedweather` S3 manager calls `_refresh_sso_if_needed()` before every upload. This method spawns `subprocess.run(["aws", "sso", "login", ...])`, which **hangs indefinitely** when the network is slow or the SSO token needs interactive re-authentication.

Since the write-back never completes:
1. Newly fetched Redshift data is not persisted to S3 cache
2. Next run sees the same gaps in the cache index → re-fetches from Redshift
3. Cycle repeats

### Secondary Issue

Even when the cache index reports a hit (`Cache index hits: 1 segments`), the `_load_place_cache` method sometimes returns `cached_parts=0`. This causes unnecessary Redshift re-fetches even for cached data. The consolidation step then reads the same file successfully, so the data is not lost — but the initial cache read fails silently.

---

## Verification

After refreshing the SSO session (`aws sso login`) on a stable connection:

- Cache write-back completed successfully (parquet uploaded, index updated)
- Full diagnostic passed end-to-end

---

## Recommendations

1. **Before long runs:** refresh SSO with `aws sso login --profile DevAlphaAccess-266735803824`
2. **Verify session:** `aws sts get-caller-identity --profile DevAlphaAccess-266735803824`
3. **On EC2:** IAM roles avoid this issue entirely — no SSO refresh needed
4. **Upstream fix (griddedweather):** `_refresh_sso_if_needed()` should have a timeout or skip the refresh when the token is still valid, rather than spawning an interactive subprocess unconditionally
