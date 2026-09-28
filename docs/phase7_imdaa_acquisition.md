# Phase 7 - IMDAA Pressure-Level Data and the Authenticated Acquisition Path

**Status: credential-ready implementation complete. No IMDAA data was acquired.**

This phase implements the authenticated NCMRWF RDS workflow end to end and
verifies the real, scientific interpolation and diagnostic code against
controlled profiles. It stops short of a download because no RDS account is
available, and says so precisely rather than substituting anything.

---

## 1. What the RDS portal actually is (verified 2026-09-26)

Probed live against `https://rds.ncmrwf.gov.in/api` using its published
`/api/openapi.json` (FastAPI, OpenAPI 3.1). Results:

| Endpoint | Auth | Observed response without credentials |
|---|---|---|
| `GET /` | no | `200 {"message":"Backend IMDAA is running ?"}` |
| `GET /datasets/catalog` | no | `200`, 9 datasets |
| `GET /datasets/{slug}` | no | `200`, full metadata |
| `POST /auth/login` | n/a | `422` on a syntactically invalid email (real validator) |
| `GET /auth/me` | **yes** | `401 {"detail":"Not authenticated"}` |
| `GET /jobs/my` | **yes** | `401` |
| `GET /jobs/id/{uuid}` | **yes** | `401` |
| `POST /jobs` | **yes** | `401` |
| `POST /jobs` + bogus bearer | **yes** | `401` |

**Conclusion: the metadata is open; every endpoint that moves data is gated.**

### 1.1 Exact authentication workflow

From the portal's own `components/schemas`:

1. **Register** - `POST /auth/register` with `RegisterRequest`:
   `first_name`, `last_name`, `institution`, `country`, `email`, `password`
   (all required). Registration is a manual, external action on the website.
2. **Login** - `POST /auth/login` with `LoginRequest` `{email, password}`
   returns `TokenResponse {access_token: str, token_type: "bearer"}`.
3. **Authorise** - send `Authorization: Bearer <access_token>` on:
   - `POST /jobs` with `JobCreateRequest {request_payload: JobPayloadSchema}`
   - `GET /jobs/id/{job_id}` -> `JobStatusResponse
     {id, status, last_error, created_at, completed_at}`
   - `GET /jobs/my`
4. **Completion** - the portal also exposes a `JobWebhookPayload` with
   `external_job_id`, `status` in `{COMPLETED, FAILED}`, `file_url`, `error`.
   This client polls rather than depending on webhook delivery.

### 1.2 `JobPayloadSchema` (the documented request fields)

`dataset_type` is the **only** required field. The rest are optional:
`year`, `month[]`, `day[]`, `time[]`, `frequency`, `variables[]`,
`pressure_level[]`, `data_format`, `download_format`, `soil`, `wind`, `cloud`,
`others`, `rain_and_snow`, `radiation_and_heat`, `temperature_and_pressure`,
`daily_sum`, `daily_mean`, `daily_maximum`, `daily_minimum`,
`initialization_day`, `forecast_day`, `instantaneous`, `accumulated`,
`averaged`, `area`.

The geographic box key is **`area`**, typed by `GeoArea` as
`{type: str, north?: str, south?: str, east?: str, west?: str}` - note the
values are **strings**, which `build_sample_request` reproduces exactly.

### 1.3 Products in the catalog

The catalog returns 9 datasets, including:

- `hourly-pressure` - "IMDAA 3-hourly data on pressure levels from 1979 to 2020"
- `imdaa-daily` - "IMDAA daily data on single level from 1979 to 2020"

Published IMDAA record: **1979-2020, ~12 km, 3-hourly on pressure levels**.

The standard pressure-level axis is documented in
`rds_client.IMDAA_STANDARD_LEVELS_HPA`
(1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100
hPa). This is used **only for documentation**; the levels actually present are
always read from the downloaded file and never assumed.

---

## 2. What was implemented

### 2.1 `app/ingestion/realtime/rds_client.py`

- `RDSCredentials` - reads `SIHPS_RDS_EMAIL` / `SIHPS_RDS_PASSWORD`.
  `__repr__` and `public_status()` redact both fields, so a credential cannot
  leak through a log line, an error message, or a provenance record.
- `RDSClient` - `submit_job` -> `poll_job` -> `wait_for_job` -> `download`.
- The bearer token is held **in memory only**, never persisted, and only its
  length is ever logged or reported.
- `wait_for_job` **raises** when a job does not reach a terminal state, so
  "still running" can never be mistaken for "data acquired".
- `download` rejects an HTML body served with HTTP 200 - the same failure mode
  seen on the IMD OGC service, where a login redirect masquerades as data.
- No credential is hardcoded and no endpoint is accessed by bypassing auth.

### 2.2 `app/ingestion/realtime/imdaa_netcdf.py`

**`validate_imdaa_file`** checks a real file and reports exactly what is in it:
recognised variables and their units, coordinate ranges, CRS, the time axis,
the pressure axis, and physical-range violations. Out-of-range values are
**counted and warned, never silently clipped**. A file that is not recognisable
IMDAA is rejected outright rather than partially accepted.

Missing-value handling honours the project contract: `NaN`, `-999`, `-9999`
and the NetCDF fill sentinels are **missing, never zero**.

**`interpolate_to_levels`** - linear in `ln(p)`, the standard treatment for
meteorological pressure-level data, with an added **gap rule**: a target level
is filled only when its bracketing valid levels are adjacent
(`max_gap_levels=1`, configurable). This was a deliberate correction made during
testing: the first version happily interpolated across a missing level, which
would have manufactured a value two levels from any observation.

**`derive_cape`** - delegates to the project's existing
`app.physics.parcel_ascent` (Bolton 1980 thermodynamics). CAPE is derived
**only** when temperature *and* specific humidity are both present on at least
three levels. Without humidity the result is `NaN`; it is never estimated from
temperature or relative humidity alone, because that is not the same quantity.

**`read_pressure_fields` / `derive_iwv` / `derive_channel_fields` / `build_observation_cube`**
- close the loop from a downloaded file to a model-ready cube. `read_pressure_fields`
  pulls the real ``(time, level, lat, lon)`` temperature and humidity, normalising
  degC/g-per-kg if an operator-supplied file uses them (recorded, never silent),
  flipping a descending latitude axis, and building a `GridSpec` for the file's
  own extent.
- `derive_iwv` integrates ``(1/g)·∫q dp`` by the trapezoidal rule. A column with
  any missing humidity returns `NaN` **in full**, so a truncated column can never
  report a small-but-plausible total.
- `derive_channel_fields` derives CAPE and IWV on the source grid, then warps them
  onto the model grid with `GridSpec.resample_bilinear`. Target cells outside the
  file's own lat/lon extent are `NaN`, **not** clamped to the edge value.
- `build_observation_cube` returns a genuine `ObservationCube` with all 12
  channels. Only `cape` and `iwv` are filled. The other ten are `NaN`, and
  `metadata["channels_available"]` / `["channels_missing"]` / `["trainable"]`
  record the split so no consumer can mistake it for a complete input tensor.
  It raises rather than returning an all-NaN cube.

`IMDAAConnector.fetch()` is now implemented on the same path: it walks the
operator's local directory, skips (and records) unusable files, and concatenates
the real cubes. An empty directory raises `FileNotFoundError` naming the download
step; it does not generate data to avoid raising.

### 2.3 `app/ingestion/realtime/imdaa_acquire.py`

- `acquire_sample` - the full path: request -> job -> poll -> download ->
  validate -> derive -> align -> provenance.
- `assess_split_coverage` - computes, **from files actually on disk**, whether
  the target split can be filled. The time axis is authoritative; the filename
  is only a fallback, so a mislabelled file is visible rather than silently
  placed in the wrong split.
- CLI: `--check` (access status), `--split-coverage` (year coverage), and a
  default mode that acquires one explicitly configured sample.

---

## 3. What was verified, and what was not

**Verified live:** the access boundary in section 1. The metadata endpoints
respond; the data endpoints return 401.

**Verified offline (47 tests in `tests/test_imdaa_auth.py`):** credential
redaction, request-schema fidelity, job-state classification, refusal to
report an unfinished job, NetCDF validation, fill-value handling, log-p
interpolation and its gap rule, CAPE derivation and its NaN-on-missing-humidity
guarantee, and split-coverage honesty.

**Not verified, because it did not happen:** any IMDAA download, any real
NetCDF, any real CAPE value, any resampling to the canonical grid, and any
training. The NetCDF files in the test suite are **synthetic fixtures built by
the test itself**, labelled as such in their attributes. They are not IMDAA
data.

### 3.1 A caveat found in existing code

The project's existing `app.physics.parcel_ascent` returns non-zero CAPE even
for an absolutely stable profile (a warm column with a lapse rate well under
the dry adiabat). Its own docstring describes it as suitable for
"nowcasting-scale instability screening" and notes that no entrainment or ice
phase is modelled. The CAPE tests therefore assert **ordering and finiteness**
rather than absolute meteorological thresholds. This is a pre-existing
characteristic of the physics module, out of scope for Phase 7, and is flagged
here rather than papered over by weakening an assertion.


---

## 4. Readiness: what would still block training

Even with a full IMDAA record, the model would **not** be trainable, because:

| Requirement | Status |
|---|---|
| IMDAA pressure-level inputs (T, q, u, v, z) | blocked on credentials |
| 6 contiguous 30-minute frames per window | **not satisfied** - IMDAA is 3-hourly on pressure levels |
| 12 model channels (incl. satellite radiances) | **missing** - no INSAT L1B |
| Model targets / labels | **missing** |
| DEM / terrain features | **missing** |

`assess_split_coverage` currently returns `no_data`, and the split verdict
lists 7/3/3 missing years for train/val/test respectively.

---

## 5. Exact external action required

1. **Register** at <https://rds.ncmrwf.gov.in/> (email + password).
2. **Set** `SIHPS_RDS_EMAIL` and `SIHPS_RDS_PASSWORD` in the environment.
   Never commit real values; `.env.example` documents the variables.
3. **Verify** the credentials:
   ```
   python -m app.ingestion.realtime.imdaa_acquire --check
   ```
   Expect `availability: available`.
4. **Acquire a small sample** (not bulk):
   ```
   python -m app.ingestion.realtime.imdaa_acquire \
       --dataset-type hourly-pressure --frequency 3H \
       --year 2019 --month 07 --day 01 \
       --pressure-levels 1000,925,850,700,500,300 \
       --variables t,q,u,v,z
   ```
5. **Check split coverage** with `--split-coverage`.

Only after steps 1-4 succeed should the sample be validated, resampled to the
canonical grid with a recorded `ResamplingRecord`, and the remaining channel
gaps (satellite radiances, labels, DEM) addressed before any training is
attempted.

**No existing checkpoint was modified or replaced by this phase.**

---

## 6. Follow-ups carried forward

- Quarantine the pre-rainfall-fix IMD OGC artifacts and their provenance rows
  (Phase 6 hygiene), keeping one canonical post-fix acquisition.
- Correct `valid_from` / `valid_to` to use parsed observation times rather than
  retrieval times.
- Replace the distinct-date "contiguous frame" heuristic in `readiness.py` with
  a real per-station temporal-contiguity check (the OGC snapshot yields zero).
- Wire OGC acquisition into the Celery/API job lifecycle
  (queued/downloading/processing/completed/failed).
- Add mocked-transport tests for retry, HTTP 5xx/4xx, malformed content type,
  duplicate acquisition and partial downloads.
- Benchmark `app.physics.parcel_ascent` against a trusted sounding before any
  CAPE value is used for training.

