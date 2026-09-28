# Phase 6 — Operational data acquisition and the first real dataset

**Status: one authentic dataset has been acquired, validated, checksummed and
provenance-recorded. Real-data training has NOT run, and the report below says
exactly why.**

Phase 5 established that IMDAA, MERA and MOSDAC all require a registered
account. Phase 6 searched for an alternative and found one: a **keyless public
OGC service that IMD's own website calls**. That yielded a genuine,
provenance-tracked dataset of 154 real station observations over Uttarakhand.

Evidence and limitations are recorded here rather than asserted in code
comments alone.

---

## 1. The source that actually works

| Property | Value |
|---|---|
| Endpoint | `https://reactjs.imd.gov.in/geoserver/imd/ows` |
| Protocol | OGC **WFS 1.1.0** `GetCapabilities` / `GetFeature`, GeoJSON output |
| Authentication | **None.** No key, token, account or registration |
| EPSG | 4326 — no reprojection needed |
| Provenance of the endpoint | referenced directly by `https://mausam.imd.gov.in/`, which calls it via `fetch()` to populate its own dashboards |
| Layers discovered | 30 WFS layers |
| Observation layers used | `imd:synop_data_layer`, `imd:aws_data_layer` |

AOI used: `77.5, 29.0, 80.5, 31.5` (the project's canonical Uttarakhand box).

### Layers that carry observations

| Layer | AOI hits | Contents |
|---|---|---|
| `imd:aws_data_layer` | 147 | AWS stations: temperature, RH, pressure, wind, rainfall |
| `imd:synop_data_layer` | 7 | SYNOP: surface state + 3/6/12/24 h rainfall accumulations |
| `imd:metar_data_layer` | 2 | METAR (includes VIDN Dehradun) |
| `imd:radar_station_status` | 3 | radar availability (Surkandaji) |
| `imd:subdiv_rainfall_now` | 4 | subdivision recent daily rainfall |

---

## 2. The dataset that was acquired

Command (run 2026-09-26):

```
python -m app.ingestion.realtime.acquire --bbox 77.5,29.0,80.5,31.5
```

| Field | Value |
|---|---|
| Source | IMD GeoServer OGC |
| Product | `imd:synop_data_layer` + `imd:aws_data_layer` |
| Data class | `observation` (genuine, not synthetic) |
| Records | **154** (7 SYNOP + 147 AWS) |
| Distinct stations | 154 |
| Valid times | 11 distinct (2026-03-13 … 2026-09-26) |
| Native resolution | **point observations** (no gridding) |
| Variables | air/dewpoint temperature, RH, MSLP, wind speed/direction, cloud oktas, rainfall |
| Raw artefact | `data/realtime/raw/*.geojson` (immutable, exact validated bytes) |
| Processed artefact | `data/realtime/processed/imd_ogc_stations_*.json` |
| Raw SHA-256 (SYNOP) | `59a6caa015394fab…` |
| Raw SHA-256 (AWS) | `b9479e208bd73fe7…` |
| Provenance rows | `dataset_provenance`, `ingest_key` unique-constrained |

Raw and processed files are stored in **separate directories**, so any processed
value can be traced back to the exact bytes it came from.

### Two physical-validity bugs the data itself exposed

These are why the values were inspected rather than trusted, and both are now
pinned by tests:

1. **`rain_sel` is hours, not minutes.** The AWS layer publishes an accumulation
   window per station. Read as minutes, a 26 mm / 10-unit record produced
   **156 mm/h**, and the worst case in the AOI was **10 800 mm/h** — physically
   impossible. Read as *hours* the same records give 2.6 mm/h and at most
   180 mm/h, which is credible for Uttarakhand.
2. **A zero window divides by zero.** `rain_sel = 0.0` occurs on 16 records. The
   window is treated as unknown and the rate is left missing; the accumulation
   itself is preserved.

Guards now in place: `MAX_PLAUSIBLE_RAIN_RATE_MMH = 300` (the world 1-hour
record is ~350 mm/h) withholds impossible rates rather than publishing them, and
`_bounded()` rejects out-of-range sentinels (temperature, pressure, humidity,
cloud, wind direction) as missing. `NULL`, `-999` and `-9999` are never read as
zero.

### Known data-quality limitations (stated, not worked around)

* The AWS `time` field arrives as `1970-01-01T06:15:00Z` — an epoch artefact. It
  is **ignored**; AWS records are marked `time_precision="day"`.
* `rain_sel` is `NULL` on 108 of 147 AWS rows, so most AWS rainfall rates are
  legitimately missing.
* This is a **snapshot**, not a time series: one latest observation per station.

---

## 3. Why real-data training did not run

`assess_readiness()` compares the acquired data against the model's real
contract. Verdict: **`partial` — `can_train: false`**.

| Requirement | Status |
|---|---|
| Input channels (12) | **0 of 12 present** |
| Contiguous 30-min frames (6) | 11 distinct times, but per-station, not a series |
| Targets (thunderstorm, rain_class, cloudburst, flood) | not available |

The 12 required channels are satellite radiances (TIR1, TIR2, WV, VIS, SWIR,
MIR, CTT, cooling rate, WV anomaly), upper-air diagnostics (IWV, CAPE) and DEM
elevation. **Surface station observations cannot supply any of them.** Feeding
this dataset into the network would mean writing zeros or synthetic values into
real-data channels, which is exactly what the specification forbids.

No checkpoint was retrained, no synthetic checkpoint was replaced, and no test
metrics were produced — because there is nothing legitimate to score.

---

## 4. Still blocked, and why

| Source | Blocker | What unblocks it |
|---|---|---|
| IMDAA reanalysis | `POST /download/request` needs a session; `/auth/me` returns 401 | Register at rds.ncmrwf.gov.in, then request 2008–2020 pressure-level subsets |
| MERA rainfall | same account gate | Same account; covers 2020–2025 (rainfall only) |
| MOSDAC INSAT-3D/3DR | Downloads require an **approved** account; anonymous is metadata-only | Register + approval, then use the official `mdapi.py` client |
| INSAT L1B → BT conversion | Needs published per-channel calibration coefficients | Obtain the INSAT-3D radiometric calibration document |
| IMD gridded rainfall | No documented keyless endpoint | IMDAA gridded rainfall, or an IMD data request |

**The IMDAA-only split (train 2008–2014, val 2015–2017, test 2018–2020) remains
the correct target** and is unaffected by anything acquired here; the OGC
snapshot cannot contribute to it.

---

## 5. What was implemented

* `app/ingestion/realtime/imd_ogc.py` — WFS client with strict response
  validation. GeoServer answers errors with **HTTP 200** (an HTML page or an
  `ows:ExceptionReport`), so structure is checked, not just status.
* `app/ingestion/realtime/acquire.py` — the acquisition entry point
  (`python -m app.ingestion.realtime.acquire`).
* `app/ingestion/realtime/readiness.py` — the training-readiness gate.
* `sources.py` now reports the OGC source as `not_probed` offline and
  `available` when probed live, so `/health` never claims a connection it has
  not just verified.
* Tests: `tests/test_imd_ogc.py` (38) using recorded fixtures; the live test is
  marked `live_source` and skipped unless `SIHPS_LIVE_SOURCE_TESTS=1`.

---

## 6. Next milestone

1. Register for **NCMRWF RDS** and download IMDAA pressure-level subsets for
   2008–2020. This is the single highest-value action: it unblocks the upper-air
   channels (IWV, CAPE) *and* the whole required split.
2. In parallel, request **MERA** for 2020–2025 so the required 2024–2025 test
   window has a rainfall reference.
3. Once pressure levels are on disk, implement `IMDAAConnector.fetch()`:
   vertical interpolation (log-pressure) onto the 2 km grid and a documented
   CAPE calculation, recording both as `ResamplingRecord` / derived-variable
   entries.
4. Only then re-run the readiness gate; it should move from `partial` toward
   `ready`, and real-data training becomes legitimate.

