# Phase 5 — Real-world data sources: access findings and current limits

> **Superseded in part by Phase 6.** This document records the Phase 5 access
> survey. Phase 6 subsequently found a *keyless* source that this survey had
> missed — IMD's public GeoServer OGC service — and used it to acquire a real
> station-observation dataset. See
> [`phase6_real_data_acquisition.md`](phase6_real_data_acquisition.md) for the
> acquired dataset, its checksums, and the current blockers. The findings below
> about MOSDAC/IMDAA/MERA account gating still stand.

Verified on **2026-09-25** by querying the live services from the development
host. Every statement below is a recorded observation, not an assumption.
This document is the evidence base for the adapter design in
`backend/app/ingestion/realtime/`.

> **Nothing in this repository has ingested a real observation or reanalysis
> field.** The adapters exist, they validate and checksum real files, and they
> refuse to fabricate data — but no real dataset has been downloaded, so no
> independent meteorological validation has been performed. See
> [Validation status](#6-validation-status).

---

## 1. Source access summary

| Source | Access mechanism | Credentials required | Verified result |
|---|---|---|---|
| **MOSDAC INSAT-3D/3DR** | Official client `mdapi.py` + `config.json` from [`/software/mdapi.zip`](https://www.mosdac.gov.in/software/mdapi.zip) | **Yes — approved account** | Manual states: *"To download any data using this application, you must first authenticate yourself using valid credentials"* |
| **IMDAA reanalysis** | NCMRWF RDS portal, public catalog API | Catalog: no. Bulk download: **yes** | `GET /api/datasets/catalog` returned 9 datasets, unauthenticated, HTTP 200 |
| **MERA rainfall** | Same portal, slug `mera` | As above | Published: hourly, 4 km, **2020–2025** |
| **IMD stations/gridded** | — | — | `mausam.imd.gov.in` reachable (HTTP 200); `eclims.imd.gov.in` and `rts.imd.gov.in` **did not resolve** (DNS failure); no documented credential-free machine-readable endpoint found |
| **SRTM / CartoDEM** | `rasterio` GeoTIFF | No (public domain / open) | Already supported by `app.ingestion.terrain`; no new dependency added |

### MOSDAC access tiers (published Data Access Policy)

| Profile | Access |
|---|---|
| Anonymous | Metadata, imagery and open data, NRT |
| Registered general user | 3-day latency |
| Registered privileged user | All data, including NRT |

Documented search parameters: `datasetId` (required, e.g. `3SIMG_L1B_STD`),
`startTime`/`endTime` (`YYYY-MM-DD`), `boundingBox`
(`minLon,minLat,maxLon,maxLat`), `gId`, and `count` (**maximum 100** per request).

### IMDAA / MERA published coverage

| Slug | Title | Period | Cadence | Resolution |
|---|---|---|---|---|
| `hourly-pressure` | IMDAA 3-hourly pressure levels | 1979–2020 | 3-hourly | ~12 km |
| `imdaa-daily` | IMDAA daily single level | 1979–2020 | daily | ~12 km |
| `mera` | MERA hourly rainfall analysis | 2020–2025 | hourly | 4 km |

---

## 2. The required evaluation split cannot be built

The requirements specify **train 2015–2022, validation 2023, test 2024–2025**.
Against the coverage above:

* IMDAA ends in **2020**, so the training window is not fully covered.
* MERA starts in **2020** and holds **only rainfall** — no upper-air state — so
  the 2023 validation and 2024–2025 test windows are covered by a rainfall-only
  product, insufficient for a multi-channel nowcasting model.

`assess_split_feasibility()` computes this from the published ranges and reports
it rather than degrading the split silently:

```
feasible: False
unsatisfiable: ['train']
proposed:  {'train': (2008, 2014), 'val': (2015, 2017), 'test': (2018, 2020)}
notes:
  - Proposed alternative (upper-air, IMDAA only): all years inside 1979-2020.
  - The required 2024-2025 test window is covered ONLY by MERA ... which
    contains no upper-air state.
  - Splits covered by a rainfall-only product (no upper-air state): val, test.
  - Unsatisfiable as specified: train 2015-2022. No missing observation has
    been synthesised to fill these years.
```

The **documented alternative** is train 2008–2014 / val 2015–2017 / test
2018–2020, entirely inside the published IMDAA record. This is a proposal, not a
substitute: the original split remains the target, and closing the gap requires
the sources listed under [Next steps](#next-steps).

---

## 3. Resolution: the model grid is not native to any real source

The canonical grid is **2 km / 30 min** (`SIHPS_GRID_RES_KM=2.0`,
`FRAME_MINUTES=30`). No accessible source matches that natively:

| Channel | Native source resolution | Target | Operation |
|---|---|---|---|
| INSAT TIR/VIS | 4 km, 15 min | 2 km, 30 min | spatial upsample + temporal composite |
| INSAT WV/SWIR/MIR | 8 km, 15 min | 2 km, 30 min | spatial upsample + temporal composite |
| IMDAA | ~12 km, 3-hourly | 2 km, 30 min | spatial upsample + **6× temporal repeat** |
| MERA | 4 km, hourly | 2 km, 30 min | spatial upsample + **2× temporal aggregation** |

Every one of these steps is recorded as a `ResamplingRecord` on the provenance
record (`stage`, `method`, `source_resolution`, `target_resolution`), so a
30-minute frame derived from 3-hourly reanalysis is always identifiable as an
estimate rather than an observation. `composite_temporal()` returns a `None`
record when the source cadence already matches, so the pipeline never claims a
resampling it did not perform.

**Downsampling note.** Aligning coarse reanalysis onto a finer 2 km grid does not
create information. The effective spatial skill is bounded by the coarsest
native input; the pipeline documents this rather than implying 2 km physical
resolution.

---

## 4. Missing-data policy

* A masked cell is `NaN`, **never** `0.0`. A zero rainfall reading and a missing
  reading are different facts and are stored differently
  (`IMDObservation.is_missing`).
* A temporal bucket with no observations stays `NaN`; it is not filled with a
  spatial mean or a zero frame.
* Channels absent from a source stay `NaN` and are listed in `channels_absent`
  on the cube metadata, with a quality warning.
* A cube whose missing fraction exceeds 50% is **rejected** — `build_cube`
  returns `ok=False` and no `ObservationCube`.

---

## 5. Provenance and de-duplication

Every ingested dataset produces a `DatasetProvenance` persisted to the
`dataset_provenance` table (migration `0002_dataset_provenance`, additive only).

* `ingest_key = sha256(source | product | acquired_at | checksum)` is
  **uniquely constrained**, so re-running a batch is a no-op rather than a
  duplicate row. `record_dataset_provenance` returns `(row, created)` and also
  handles a concurrent-insert race via `IntegrityError`.
* File identity is a streaming **SHA-256** of the file bytes.
* Credentials are never stored: `DatasetProvenance.to_dict()` runs a recursive
  redaction pass, and `scrub_detail()` masks credential-bearing keys while
  remaining valid JSON.

---

## 6. Validation status

`sihps-evaluate` emits a `validation_capability` block, and
`GET /api/v1/health` exposes `data_sources`. With no real data ingested:

```
status: no_real_data_accessible
validation_performed: false
NOT EVALUATED: CSI / POD / FAR against observed rain events
NOT EVALUATED: Brier Skill Score against climatology
NOT EVALUATED: Reliability / calibration curves from real forecasts
NOT EVALUATED: CRPS of a calibrated ensemble
NOT EVALUATED: Skill by storm event across the 2024-2025 test window
NOT EVALUATED: Lead-time skill decay in the Indian monsoon regime
```

Metrics computed on the synthetic demo remain **agreement with a generator**,
not forecasting skill. The existing invariant
`CheckpointInfoResponse.observational_validation` stays hard-coded `false` and
is enforced by tests.

---

## Next steps

1. **Register a MOSDAC account** and obtain approval, then download INSAT-3D
   L1B/L2 granules for the AOI with `mdapi.py` into `SIHPS_MOSDAC_DIR`.
2. **Register for NCMRWF RDS** and request IMDAA pressure-level subsets for
   2008–2020 into `SIHPS_IMDAA_DIR`; this unlocks the documented alternative
   split.
3. **Request MERA rainfall** for 2020–2025 to cover the required test window —
   accepting that it validates rainfall targets only, not upper-air skill.
4. Implement `MOSDACConnector.fetch()` and `IMDAAConnector.fetch()`: L1B
   geolocation to a lat/lon grid, radiance → brightness temperature using the
   published calibration coefficients, and IMDAA vertical interpolation with CAPE
   derivation. Both currently raise `NotImplementedError` rather than returning
   a partially-populated cube.
5. Investigate a legitimate IMD station-data channel (IMDAA gridded rainfall or
   an IMD data request) to build an observational target for the risk heads.
6. Only after 1–3: retrain and re-evaluate on real data, and require an explicit
   configuration change to promote a new checkpoint.

