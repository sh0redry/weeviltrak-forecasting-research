# Addressing Matt's Questions (Plain Language)

**How do we get the dates for stages 1, 2, and 3?**

- We use the same approach as the US protocol. For each site, we record the date when each main stage is first seen:
  - **Stage 1:** First date adults are active on the turf (vacuum or soap flush)
  - **Stage 2:** First date eggs or tiny larvae are found in stems
  - **Stage 3:** First date large larvae are found in the thatch/soil (and/or visible turf damage)
- Each week, field staff check and record which stage is most common. The first week a new stage is seen, that date is marked as the "stage date" for that stage.
- This is exactly how the US team did it, so our results will be directly comparable.
- No technical steps are needed from field staff—just fill out the weekly report as usual. Henry will handle all the data analysis and matching up with the model.

If you have questions or need help with stage identification, just ask in the WhatsApp group or contact Henry directly.

# WeevilTrak Canada Field Validation Protocol — 2026

**Version:** 1.2  
**Date:** March 24, 2026  
**Prepared by:** Henry Qu  
**Canadian Lead:** Matt Legg  
**US WeevilTrak Support:** Lisa Beirn (US protocol reference: I02-ABW-LAB-2025)  
**Project Sponsor:** Katie Dodson  

---

## 1. Objective

Evaluate the performance of the WeevilTrak v2.1 model (trained on US Northeast data, 2018–2025) for nowcasting Annual Bluegrass Weevil (ABW, *Listronotus maculicollis*) life-stage timing at Canadian sites. The 2026 field season will collect ground-truth observations to:

1. Measure nowcast accuracy (MAE, within-3-day and within-5-day rates) at Canadian locations
2. Identify systematic biases (e.g., latitude-driven prediction drift at higher latitudes)
3. Determine whether the model generalises acceptably or whether Canadian-specific training data is required for a future v2.2 retrain
4. Establish a repeatable scouting protocol for ongoing Canadian model calibration

---

## 2. Background & Known Limitations

| Factor | Detail |
|--------|--------|
| **Training data** | ~10 years of US-only observations (NE corridor), no Canadian sites in current training set |
| **Canadian data limiations** | Location ID '5' (Guelph, ON) is currently hardcoded out of training data |
| **Confidence at high latitudes** | Model uses latitude/longitude as features — predictions extrapolate beyond training range for ON/QC |
| **Geographic display** | Prediction maps currently show results extending to the Arctic — boundary restriction needed |
| **Weather data source** | 10×10 km gridded weather via S3 cache — coverage for ON/QC must be verified before season start |

---

## 3. Cooperator Sites

| Region | Cooperator | Affiliation | # Sites | Notes |
|--------|-----------|-------------|---------|-------|
| Toronto / GTA (ON) | Matt Legg | Syngenta SPS | 2 | Primary — vacuum sampling |
| London / SW Ontario | Bryce Allan | Syngenta SPS | 1 | Soap flush sampling |
| Blainville (QC) | Pat Moir | Syngenta SPS | 1 | Soap flush sampling |
| Ottawa (ON) | TBD | 3rd-party golf course | 1 | Soap flush sampling |
| Nova Scotia | TBD | 3rd-party agronomist | 1 | *Pending confirmation* |

**Access:** Restricted to invited cooperators only — no public sharing of prediction maps or data.

---

## 4. Scouting Period & Frequency

| Parameter | Value |
|-----------|-------|
| **Start date** | April 1, 2026 |
| **End date** | July 1, 2026 |
| **Frequency** | Weekly (every 7 days ± 1 day) |
| **Total weeks** | ~13 scouting visits per site |
| **Critical window** | May 1 – June 15 (peak ABW activity in ON/QC) |

> **Note:** Lisa suggested potential higher frequencies (e.g., 2× per week) may be needed during the critical window to capture rapid stage transitions. Discuss with Matt and cooperators to determine feasibility.

---

## 5. What the Model Produces (For Cooperator Reference)

The model is a **nowcasting** system: given the latest available weather data for a location, it estimates how many days remain until each ABW life stage occurs. It does not forecast into the future — it reflects the current state based on accumulated weather conditions (GDD, precipitation, humidity) through the most recent data. Cooperators should think of WeevilTrak as answering: *"Based on conditions so far this season, when should we expect each stage?"*

**Life stages tracked (aligned with US protocol treatment timings):**

| Stage ID | Stage Name | Field Indicator | US Treatment Timing |
|----------|-----------|-----------------|---------------------|
| 1 | Adults (overwintered) | Adult weevils active on turf surface, detectable by vacuum or soap flush | 'A' timing — foliar adulticide |
| 2 | Eggs / Early larvae (1st instar) | Eggs laid in grass stems; 1st instar larvae beginning to feed inside stems | 'B' timing — systemic (water in lightly) |
| 3 | Late larvae (3rd+ instars exiting stems) | L3–L5 in thatch/soil, feeding on crowns and roots; visible turf damage | 'C' timing — systemic (water in lightly) |

---

## 6. Data Collection Requirements

### 6.1 Site Registration (One-Time Setup — Before April 1)

Each cooperator must provide:

| Field | Format | Example | Purpose |
|-------|--------|---------|---------|
| **Site name** | Text | "Toronto Golf Club — East Fairway" | Human-readable label |
| **Latitude** | Decimal degrees (WGS84), 5+ decimals | 43.65107 | Model prediction grid lookup |
| **Longitude** | Decimal degrees (WGS84), 5+ decimals | -79.34731 | Model prediction grid lookup |
| **Province** | 2-letter code | ON, QC, NS | Regional analysis |
| **Turfgrass type** | Text | "Annual bluegrass / Poa annua fairway" | Context |
| **Management notes** | Text | "Irrigated, mowed 3×/week at 12mm" | Context for damage assessment |

**How to get coordinates:** Open Google Maps on your phone, long-press on the scout location, and copy the lat/lon that appears.

### 6.2 Weekly Scouting Report

Each weekly visit produces **one report per site** with the following data:

#### A. Mandatory Fields

| Field | Format | Notes |
|-------|--------|-------|
| **Date** | YYYY-MM-DD | Exact date of scouting visit |
| **Cooperator name** | Text | Who performed the scouting |
| **Site name** | Text | Must match registered site name |
| **Predominant life stage observed** | 1, 2, or 3 | Single dominant stage at that date |
| **Adult count** | Number per sq ft | Adults from soap flush or vacuum (see protocol below) |
| **Sampling method — adults** | "vacuum" or "soap flush" | Standardised |
| **Sampling area — adults** | sq ft | Area sampled (for per-sq-ft calculation) |
| **Larval count** | Number per sq ft | From salt-water extraction (see protocol below) |
| **Larval stages — separate** | Count per stage | Report larvae, pupae, and teneral adults **separately** |
| **Sampling area — larvae** | sq ft (cores × core area) | For per-sq-ft calculation |
| **Turf quality** | 1–9 scale | 1=dead, 6=minimum acceptable, 9=excellent. Every visit. |
| **ABW feeding damage** | % | Estimated % feeding damage. At minimum report at final larval assessment. |

#### B. Standardised Sampling Protocols

**Adult sampling — Soap flush (all cooperators except Matt):**

*Aligned with US protocol (Beirn I02-ABW-LAB-2025): weekly from early April until late-larval stage.*

1. Select a consistent sampling area (at minimum 1 sq ft per sample point, 3–5 sample points)
2. Mix approximately 30 mL of lemon-scented dish soap in 4 L of water
3. Pour the solution evenly over the sample area
4. Wait 5–10 minutes; count and collect all adults that surface
5. **Do not re-sample the same area** on subsequent visits — use a fresh area each week
6. **Report as number of adults per square foot** (total adults ÷ total area sampled)
7. Record the total area sampled (sq ft)

**Adult sampling — Vacuum (Matt Legg only):**
1. Use a modified leaf blower/vacuum on 3+ transects
2. Empty catch container after each transect and count adults
3. Record the total area sampled and report as adults per sq ft

**Larval sampling — Salt-water extraction (all cooperators):**

*Aligned with US protocol: weekly from Stage 1 (adult emergence) through emergence of 1st generation adults. Use consistent sampling area across all visits.*

1. Collect 3–4 soil/turf cores per sampling area (standard cup-cutter or 10 cm diameter × 5 cm depth)
2. Soak cores in saturated salt-water solution for 15–20 minutes
3. **Count and report each life stage separately:**
   - Larvae (specify instar if identifiable: L1, L2, L3, L4, L5)
   - Pupae
   - Teneral (newly emerged, soft-bodied) adults
4. **Report as number of each life stage per square foot** (count ÷ total core area in sq ft)
5. Record sampling method and total area sampled

> **Note:** The US protocol also collects final larval samples from treated plots to assess spray program success. For Canadian validation sites without treatment plots, collect final-assessment larvae from the standard monitoring area.

#### C. Turf Quality & Damage Assessments

*Aligned with US protocol reporting requirements.*

| Assessment | Scale | Frequency | Notes |
|-----------|-------|-----------|-------|
| **Turf quality** | 1–9 (1=dead, 6=min acceptable, 9=excellent) | Every visit (weekly) + at trial initiation | US protocol requires every 14 days; we collect weekly for finer resolution |
| **ABW feeding damage** | % estimated damage | At minimum at final larval assessment; ideally every visit | Visual estimate of percentage of turf area showing ABW feeding injury |

#### D. Phenological & Environmental Observations (Required)

*US protocol requires plant phenology and GDDs at time of each assessment. These observations are critical for comparing WeevilTrak nowcast timing against traditional phenological indicators.*

| Observation | Format | Purpose |
|-------------|--------|----------|
| **Cumulative GDD** | Number (base 50°F or 10°C) | Record local GDD from weather station or GDD tracker app. Key cross-reference for model validation. |
| **Forsythia bloom status** | Not yet / Blooming / Past peak | Traditional ABW adult emergence indicator |
| **Flowering dogwood status** | Not yet / Blooming / Past peak | Traditional ABW larval timing indicator |
| **Other plant phenology** | Free text | Any notable phenological events (lilac bloom, etc.) |
| **Soil temperature at 5 cm** | °C (if available) | Helps validate GDD alignment |
| **Local air temperature (high/low)** | °C | Cross-reference with gridded weather data |
| **Recent rainfall** | Light / Moderate / Heavy / None | Cross-reference with precipitation feature |

> **Note on GDD:** The model calculates GDD internally from gridded weather data (base 50°F, cumulative from March 1). Cooperator-reported GDD from local weather stations provides an independent ground-truth check on the gridded data.

---

## 7. Reporting System

### Primary: WhatsApp Group

- **Group name:** TBD (Henry to create private group)
- **Access:** Invited cooperators only
- Cooperators post weekly scouting reports directly to the group using the template below
- WhatsApp is preferred for ease of mobile reporting from the field

### Backup: Email

- If WhatsApp is inaccessible, send weekly report to [Henry.Qu@syngenta.com](mailto:Henry.Qu@syngenta.com) and [Matthew.Legg@syngenta.com](mailto:Matthew.Legg@syngenta.com)

### Reference: US Sampling Protocol (Beirn I02-ABW-LAB-2025)

The Canadian protocol is aligned with Lisa Beirn's US WeevilTrak 2.0 Model Validation protocol. Key alignment points:

| Aspect | US Protocol | Canadian Protocol |
|--------|------------|-------------------|
| Adult sampling | Soap flush, report per sq ft | Same (+ vacuum for Matt) |
| Larval sampling | Salt-water or heat extraction, 3–4 cores, report larvae/pupae/teneral adults separately per sq ft | Same (salt-water) |
| Turf quality | 1–9 scale, every 14 days | 1–9 scale, every 7 days (weekly visits) |
| Feeding damage | % at final larval assessment | Same |
| Phenology & GDD | Required at each assessment | Required |
| Experimental design | RCB, 5 reps, 5'×5' plots, separate adult & larval monitoring plots | Simplified: single monitoring area per site (no treatment comparison) |
| Insecticide treatments | 3 timings (A/B/C) aligned to Stages 1/2/3 | No treatments — validation only |

**Key difference:** The US protocol is a full insecticide trial with treated and untreated plots. The Canadian protocol is model validation only — cooperators scout a single representative area per site without insecticide treatment comparisons.

### Weekly Report Template (Copy-Paste for Each Visit)

```
📋 Weekly ABW Scouting Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Date: [YYYY-MM-DD]
Cooperator: [Your Name]
Site: [Site Name]

🔬 Life Stage Assessment
- Predominant stage: [1 / 2 / 3]
- Adults: [count] per sq ft via [vacuum / soap flush]
  Area sampled: [X sq ft]
- Larvae per sq ft: [count]
  - Larvae (instar): [count, e.g. L1=2, L3=5]
  - Pupae: [count]
  - Teneral adults: [count]
  Area sampled: [X sq ft from Y cores]

🌿 Turf Condition
- Turf quality: [1-9]
- ABW feeding damage: [X%]
- Notes: [any relevant observations]

🌸 Phenology & GDD
- Cumulative GDD (base 50°F): [number or N/A]
- Forsythia: [Not yet / Blooming / Past peak]
- Dogwood: [Not yet / Blooming / Past peak]
- Other phenology: [notes]
- Soil temp (5cm): [°C or N/A]
- Air temp (high/low): [°C]
```

---

## 8. How Collected Data Will Be Used

### 8.1 Data Pipeline Integration

Cooperator observations will be entered into the WeevilTrak Redshift database with the following mapping:

| Field Report | → | Database Column | Type |
|-------------|---|-----------------|------|
| Date | → | `stage_date` | date |
| Site lat/lon | → | `latitude`, `longitude` | float |
| Province | → | `location_state` | string ("ON", "QC", "NS") |
| Predominant stage | → | `stage_id` | integer (1–3) |
| Site name | → | `location_name` | string |

Each site will receive a unique `location_id` in the system. Henry will handle data entry — cooperators just submit reports.

### 8.2 Evaluation Against Model Predictions

For each cooperator observation, we will:

1. **Run the model** for that site's coordinates using the latest available weather data up to and including `stage_date` (nowcast)
2. **Compare** nowcast stage date vs. actual observed `stage_date`
3. **Calculate error** = `nowcast_stage_date − actual_stage_date` (in days)

### 8.3 Success Metrics

| Metric | Target (US Baseline) | Canadian Threshold | Interpretation |
|--------|--------------------|--------------------|----------------|
| **MAE** (days) | ~5–7 days | ≤ 10 days | Average prediction offset |
| **Within 3 days** | ~40–50% | ≥ 25% | Tight accuracy |
| **Within 5 days** | ~55–65% | ≥ 35% | Operational accuracy |
| **Within 7 days** | ~70–80% | ≥ 50% | Useful guidance |
| **Mean signed error** | ~0 | Any direction | Systematic early/late bias |
| **P90 absolute error** | ~12 days | ≤ 18 days | Worst-case bound |

**Decision framework after 2026 season:**

| Outcome | Action |
|---------|--------|
| Canadian MAE ≤ 7 days, within-5d ≥ 50% | Model generalises well — enable Canadian predictions with current model |
| Canadian MAE 7–12 days | Model shows promise — retrain v2.2 with Canadian data appended |
| Canadian MAE > 12 days or strong bias | Model does not generalise — needs dedicated Canadian training data and/or additional features (e.g., photoperiod) |

---

## 9. Geographic Boundary Restrictions

### Action Items (Henry — Pre-Season)

1. **Restrict prediction display**: Cap northern prediction boundary for Ontario at approximately **46.5°N** (roughly Sudbury line) and for Quebec at approximately **47.5°N** (roughly Saguenay line)
   - Matt to provide specific geographic boundaries based on known ABW distribution
2. **Consider hiding Canadian view entirely** until validation data supports prediction quality — no need to expose prediction maps to Canadian customers before we have a proper ABW strategy
3. **Weather data verification**: Confirm that gridded weather cache at `s3://sps-ds-bucket/gridded-weather/cache/10by10/` has coverage for all cooperator site coordinates before April 1

---

## 10. Timeline & Milestones

| Date | Milestone | Owner |
|------|-----------|-------|
| **March 28** | Call to discuss model UI + cooperator setup | Henry + Matt (+ Katie optional) |
| **March 31** | Matt finalises cooperator list and confirms sites | Matt |
| **March 31** | Henry verifies weather data coverage for all Canadian sites | Henry |
| **April 1** | Scouting begins — cooperators submit site registration | All cooperators |
| **April 1** | WhatsApp reporting group created and cooperators invited | Henry |
| **April 30** | First month check-in — review data quality and completeness | Henry + Matt |
| **May 15** | Mid-season interim analysis (early predictions vs. observed) | Henry |
| **July 1** | Scouting concludes | All cooperators |
| **July 15** | Final data compilation and quality check | Henry |
| **August 1** | Evaluation report: Canadian accuracy metrics vs. US baseline | Henry |
| **August 15** | Go/no-go decision on Canadian model deployment | Henry + Katie + Matt |

---

## 11. Open Questions

1. ~~**US sampling protocol alignment**~~ — **RESOLVED.** Lisa Beirn's protocol (I02-ABW-LAB-2025) received. Canadian protocol now aligned on sampling units (per sq ft), separate life-stage reporting, turf quality scale (1–9), feeding damage %, and phenology/GDD requirements.
2. **ABW insecticide sensitivity** — US protocol asks cooperators to report any known ABW insecticide sensitivity at trial sites. Canadian cooperators should also note if there is known insecticide resistance at their site.
3. **Nova Scotia cooperator** — Is the 3rd-party agronomist confirmed? If so, does ABW pressure exist that far east?
4. **University of Guelph data** — Can any historical data from Guelph be obtained to supplement the 2026 field season?
5. **Vacuum vs. soap flush comparability** — Matt uses vacuum, others use soap flush. Both report per sq ft, but detection efficiency may differ. Do we need a sampling-method correction factor, or do we treat all counts as relative indices only?
6. **Additional sites in subsequent years** — If 2026 results are promising, plan to expand to ~10 sites for 2027?
7. **Heat extraction alternative** — US protocol allows heat extraction as an alternative to salt-water for larvae. Do any Canadian cooperators prefer this method?

---

## Appendix A: Equipment Checklist Per Cooperator

- [ ] Bucket (4 L) + lemon-scented dish soap (for soap flush)
- [ ] Salt (NaCl, table salt or pool salt) + container (for larval salt-water extraction)
- [ ] Standard cup-cutter or turf plug cutter (~10 cm / 4" diameter) — know your core area in sq ft
- [ ] White tray or pan (for sorting specimens)
- [ ] Forceps / tweezers
- [ ] Hand lens or magnifying glass (10×–20× for larval instar ID and separating pupae/teneral adults)
- [ ] Smartphone (for WhatsApp reporting + GDD tracker app)
- [ ] Soil thermometer (for 5 cm soil temperature readings)
- [ ] Measuring tape or marked area for consistent sq ft sampling
- [ ] Printed/saved copy of this protocol or the weekly report template

## Appendix B: ABW Life Stage Quick Reference

| Stage | Description | Size | Key Features | US Treatment | When (Typical NE US) |
|-------|-------------|------|--------------|--------------|---------------------|
| **1 — Adults** | Overwintered adults emerging from leaf litter | 3–5 mm | Dark snout beetle, walks on turf surface | 'A' — foliar adulticide | April – early May |
| **2 — Eggs/1st Instar** | Eggs in stems; 1st instar larvae beginning to feed inside stems | < 2 mm | Tiny, legless, creamy white, inside stem | 'B' — systemic, water in | Mid-May |
| **3 — Late Larvae (3rd+ exiting stems)** | L3–L5 exiting stems into thatch/soil, feeding on crowns and roots | 3–8 mm | Larger, C-shaped, creamy with brown head | 'C' — systemic, water in | Late May – June |
| **Pupae** | Pupation in soil | 3–5 mm | Immobile, light brown, in soil chamber | — | June |
| **Teneral adults** | Newly emerged soft-bodied adults | 3–5 mm | Lighter colour than mature adults, soft exoskeleton | — | June – July |

**Note:** Canadian timing may run 1–3 weeks later than NE US depending on spring progression — this is exactly what the validation aims to quantify.

## Appendix C: US Protocol Reference (Beirn I02-ABW-LAB-2025)

Lisa Beirn's full US protocol ("WeevilTrak 2.0 Model Validation") is an RCB insecticide trial design with 5 replications and 5'×5' plots. It includes treated plots with three application timings (A/B/C corresponding to WeevilTrak Stages 1/2/3) using Scimitar GC, Acelepryn, and Ference. The Canadian protocol adopts the US monitoring methodology (sampling units, life-stage reporting, phenology requirements) but omits the insecticide treatment component since the Canadian effort is model validation only.

Key US protocol requirements carried forward to Canada:
- Adults reported **per square foot** via soap flush
- Larvae, pupae, and teneral adults counted and reported **separately per square foot**
- Turf quality on **1–9 scale**
- **ABW feeding damage %** at final assessment
- **Plant phenology and GDD** reported at each assessment
- **Sampling methods and area** documented
