# Data

All files in `data/processed_v2/` are written by `src/make_dataset_v2.py` from the raw EnerNOC and BDG2 data (see the main README for sources, licences and download steps). The raw data is not included.

## How the dataset is built

1. **Interval data to hourly kW.** EnerNOC 5-minute kWh readings are converted from UTC to site local time, multiplied by 12 to give average kW, cleaned (zero readings treated as dropouts, gaps up to 1 h interpolated) and resampled to 15-minute and hourly means. BDG2 hourly kWh is already average kW in local time; gaps up to 3 h are interpolated. Building-years with 10% or more missing readings are dropped.
2. **Synthetic monthly bills.** For each calendar month: energy (kWh, sum) and peak demand. The billed peak is the highest 15-minute average kW for EnerNOC and the highest hourly kW for BDG2 (no 15-minute data). Months with more than 20% missing hours are dropped; only building-years with 12 complete bills enter `dataset.csv`.
3. **Shape descriptors and labels.** Normalized weekday and weekend 24-hour profiles (divided by the weekday mean) give:
   - `day_night_ratio` = weekday mean of hours 9 to 16 / weekday mean of hours 0 to 5
   - `weekend_weekday_ratio` = mean weekend profile / mean weekday profile

   The archetype is assigned by fixed rules (thresholds in `numbers.json`):

   | rule (applied in order) | archetype |
   |---|---|
   | `day_night_ratio` < 1.3 | `flat_7day` |
   | `weekend_weekday_ratio` >= 0.85 | `daytime_7day` |
   | `day_night_ratio` >= 1.8 | `daytime_weekday` |
   | otherwise | `weekday_moderate` |

   Meters dominated by spikes (annual load factor 0.05 or less) are dropped before labelling. A k-means clustering (K = 4) of the profiles is run only as a check against the rules (adjusted Rand index in `numbers.json`).
4. **Duplicates.** Within each source, building-years whose 48-value profiles correlate above 0.9995 with a different building are treated as duplicate meters and dropped (30 rows). EnerNOC duplicate groups with conflicting industry labels get `building_type = unknown`.
5. **Augmentation.** Bills are recomputed with billing-cycle start offsets of 0, 3, 6, ..., 27 days (10 copies per building-year) for `dataset_aug.csv`. `offset_days = 0` rows are the calendar-month bills in `dataset.csv`.

Rows from one building must stay in one split: group train, validation and test splits by `building_key` (BDG2 buildings appear for both 2016 and 2017, and each building-year has 10 augmented copies).

## Files in this repository

| file | size | rows | one row is |
|---|---|---|---|
| `dataset.csv` | 1.2 MB | 2,191 | one building-year with 12 calendar-month bills |
| `dataset_aug.csv` | 12.5 MB | 21,896 | one building-year at one billing-cycle offset (training file used by the notebook) |
| `sites.csv` | 0.9 MB | 2,387 | one building-year (after duplicate removal), with shape descriptors and labels |
| `numbers.json` | 1 KB | | summary counts, label thresholds, k-means check |

`make_dataset_v2.py` also writes `bills.csv` (one row per monthly bill), `profiles_weekday_hourly.csv` and `profiles_weekend_hourly.csv` (normalized 24-hour profiles per building-year, columns `uid`, `h00` to `h23`) and `sites_all_with_duplicates.csv` (audit trail of duplicate removal). These are not included in the repository and are rebuilt by the script.

## dataset.csv and dataset_aug.csv columns

Columns, in file order (`offset_days` is in `dataset_aug.csv` only):

`uid, offset_days, kwh_m01 ... kwh_m12, peak_kw_billed_m01 ... peak_kw_billed_m12, source, building_key, building_type, sub_type, sq_ft, lat, lng, region, resolution_min, year, archetype, cluster, label_margin, hours_above_80`

| column | role | description |
|---|---|---|
| `uid` | id | building-year id, `enernoc_<site_id>` or `bdg2_<building_id>_<year>` |
| `offset_days` | id | billing-cycle start offset in days, 0 = calendar months |
| `kwh_m01` to `kwh_m12` | input | monthly energy, kWh |
| `peak_kw_billed_m01` to `peak_kw_billed_m12` | input | monthly billed peak demand, kW (15-minute for EnerNOC, hourly for BDG2) |
| `building_type` | input | harmonized type: `education`, `assembly_public`, `office_commercial`, `lodging`, `other`, `industrial`, `retail_food`, `healthcare`, `unknown` |
| `sq_ft` | input | floor area, square feet (anonymized for EnerNOC) |
| `lat` | input | latitude (randomized slightly for EnerNOC) |
| `region` | input | `US_East`, `US_Central`, `US_Mountain`, `US_West`, `UK`, `Ireland` (from the time zone) |
| `resolution_min` | input | 15 for EnerNOC, 60 for BDG2; used as the `hourly_source` flag |
| `source` | metadata | `enernoc` or `bdg2` |
| `building_key` | group key | one id per physical building; use it for grouped splits |
| `sub_type` | metadata | source sub-category (not used as an input) |
| `lng` | metadata | longitude (not used as an input) |
| `year` | metadata | data year: 2012 (EnerNOC), 2016 or 2017 (BDG2) |
| `archetype` | label | target class (rules above) |
| `cluster` | label | integer form of `archetype`: 0 `flat_7day`, 1 `daytime_7day`, 2 `daytime_weekday`, 3 `weekday_moderate` |
| `label_margin` | label-side | distance of the building's descriptors to the nearest rule threshold; below 0.1 = borderline label |
| `hours_above_80` | label-side | hours in the year above 80% of the annual hourly peak |

`label_margin`, `hours_above_80`, `cluster` and `archetype` are label-side columns computed from the interval data. They are never model inputs; the notebook asserts that none of them reaches the feature table. The notebook uses `label_margin` only to report scores on confident test rows.

The model inputs are not the raw columns but 55 features built from them by `make_features` in the notebook (also in `src/predict.py`): 12 monthly kWh shares, 12 monthly peaks relative to the annual maximum, 12 monthly load factors, log annual kWh, log floor area, latitude, an hourly-source flag, and one-hot building type and region.

## sites.csv columns

`uid, source, source_id, building_type, sub_type, sq_ft, lat, lng, tz, region, resolution_min, pct_missing, year, annual_kwh, peak_kw_hourly, load_factor, hours_above_70, hours_above_80, hours_above_90, nb_window_coincidence, day_night_ratio, evening_day_ratio, weekend_weekday_ratio, peak_hour_wd, peak_kw_15min, interval_uplift, building_key, dup_of, keep, archetype, label_margin, cluster`

| column | description |
|---|---|
| `uid`, `source`, `building_key`, `building_type`, `sub_type`, `sq_ft`, `lat`, `lng`, `region`, `resolution_min`, `year`, `archetype`, `cluster`, `label_margin` | as above |
| `source_id` | site or building id in the original dataset |
| `tz` | IANA time zone |
| `pct_missing` | percent of readings missing before interpolation |
| `annual_kwh` | annual energy, kWh |
| `peak_kw_hourly` | highest hourly average kW in the year |
| `load_factor` | mean hourly kW / peak hourly kW |
| `hours_above_70`, `hours_above_80`, `hours_above_90` | hours above 70, 80, 90% of the annual hourly peak |
| `nb_window_coincidence` | highest kW on winter weekdays (December to March) from 16:00 to 20:00, divided by the annual peak |
| `day_night_ratio` | weekday hours 9 to 16 / hours 0 to 5 (label descriptor) |
| `evening_day_ratio` | weekday hours 17 to 21 / hours 9 to 16 |
| `weekend_weekday_ratio` | mean weekend profile / mean weekday profile (label descriptor) |
| `peak_hour_wd` | hour of the weekday profile maximum |
| `peak_kw_15min` | highest 15-minute kW (EnerNOC only; empty for BDG2) |
| `interval_uplift` | `peak_kw_15min / peak_kw_hourly` (EnerNOC only) |
| `dup_of` | empty for kept rows (duplicates are only in `sites_all_with_duplicates.csv`) |
| `keep` | `True` for every row in this file |

## numbers.json

Summary of the build: rows kept, unique buildings, rows by source, duplicates dropped, ML row counts (`ml_rows`, `ml_rows_aug`), the label thresholds (`DN_FLAT`, `DN_DAY`, `WE_7DAY`), number of borderline rows (`label_margin` < 0.1), the k-means check (silhouette and adjusted Rand index against the rules), and class, building type and region counts. The class, type and region counts in `numbers.json` are over `sites.csv` (2,387 rows), not over `dataset.csv`.

Class counts in `dataset.csv` (2,191 rows): `flat_7day` 812, `daytime_weekday` 584, `daytime_7day` 429, `weekday_moderate` 366.

## Known limitations

- Labels come from fixed thresholds on two descriptors, so buildings near a threshold have soft labels (`label_margin`).
- BDG2 billed peaks are hourly, not 15-minute, so they are lower than a utility's 15-minute billed demand would be; `resolution_min` marks the difference.
- Geography is the United States, the United Kingdom and Ireland; data years are 2012, 2016 and 2017.
- Building types are unevenly represented: `education` is the largest type (939 of 2,387 building-years in `sites.csv`) and EnerNOC contributes only 72 building-years.
