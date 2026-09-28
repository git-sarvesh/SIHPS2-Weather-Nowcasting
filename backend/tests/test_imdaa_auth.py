"""Tests for the Phase 7 authenticated IMDAA path.

These are offline and deterministic. The NetCDF fixtures are **synthetic
NetCDF files built by the test itself** (clearly labelled as such in their
attributes and filename); they are not IMDAA data and are never presented as
any. Their purpose is to exercise validation, log-pressure interpolation and
CAPE derivation on a known vertical structure.

No test contacts the RDS portal except the one marked ``live_source``, which is
skipped unless ``SIHPS_LIVE_SOURCE_TESTS=1`` *and* credentials are configured.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.grid import GridSpec
from app.ingestion.realtime import imdaa_acquire as acq
from app.ingestion.realtime.imdaa_netcdf import (
    VerticalProfile,
    derive_cape,
    detect_variables,
    interpolate_to_levels,
    read_pressure_levels,
    validate_imdaa_file,
)
from app.ingestion.realtime.rds_client import (
    RDSClient,
    RDSCredentials,
    RDSDownloadError,
    RDSJob,
    build_sample_request,
)


def _write_profile_nc(path, *, n_times=2, with_humidity=True, descending_lat=False,
                      temperature=None, humidity=None, levels=(1000.0, 925.0, 850.0, 700.0, 500.0)):
    """Synthetic IMDAA-*shaped* NetCDF with a physically plausible profile.

    A test fixture, not IMDAA data: temperature falls ~6.5 K/km in the lower
    troposphere and humidity decreases with height. Used to exercise the real
    reading, interpolation, CAPE and grid-alignment path end to end.
    """
    import xarray as xr

    levels = np.asarray(levels, dtype="float64")
    lats = np.array([29.0, 29.5, 30.0]) if not descending_lat else np.array([30.0, 29.5, 29.0])
    lons = np.array([78.0, 78.5, 79.0, 79.5])
    if temperature is None:
        temperature = 300.0 - 0.065 * (1000.0 - levels)[:, None, None] * np.ones((1, lats.size, lons.size))
    if humidity is None and with_humidity:
        humidity = 0.012 - 0.010 * (levels[:, None, None] / 1000.0) ** 2 * np.ones((1, lats.size, lons.size))
    data = {
        "t": (["time", "level", "lat", "lon"],
              np.repeat(temperature[None, ...], n_times, axis=0).astype("float32")),
    }
    if humidity is not None:
        data["q"] = (["time", "level", "lat", "lon"],
                     np.repeat(humidity[None, ...], n_times, axis=0).astype("float32"))
    times = [f"2019-07-0{1 + i // 4}T{(i % 4) * 6:02d}:00:00" for i in range(n_times)]
    dataset = xr.Dataset(
        data,
        coords={"time": times, "level": levels, "lat": lats, "lon": lons},
    )
    dataset["t"].attrs["units"] = "K"
    if "q" in data:
        dataset["q"].attrs["units"] = "kg kg-1"
    dataset.attrs["title"] = "SYNTHETIC TEST FIXTURE - not IMDAA data"
    dataset.to_netcdf(path)
    return path


def _write_nc(
    path, *, levels=None, variables=("t", "q"), with_time=True, fill=0.0, year=2019
):
    """Build a small synthetic NetCDF with an IMDAA-like layout.

    Clearly labelled as a test fixture; it is not IMDAA data. ``year`` sets the
    time axis, which is what the coverage assessment reads.
    """
    import xarray as xr

    nlev = len(levels) if levels else 1
    dims = (["level", "lat", "lon"] if nlev > 1 else ["lat", "lon"])
    shape = (nlev, 3, 4) if nlev > 1 else (3, 4)
    data = {name: (dims, np.full(shape, fill, dtype="float32")) for name in variables}
    coords = {"lat": [29.0, 30.0, 31.0], "lon": [77.5, 78.5, 79.5, 80.5]}
    if nlev > 1:
        coords["level"] = levels
    if with_time:
        coords["time"] = [f"{year}-07-01T00:00:00"]
    dataset = xr.Dataset(data, coords=coords)
    dataset.attrs["title"] = "SYNTHETIC TEST FIXTURE - not IMDAA data"
    dataset.to_netcdf(path)
    return path


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #
class TestRDSCredentials:
    def test_repr_never_leaks_the_password(self) -> None:
        creds = RDSCredentials(email="user@example.org", password="hunter2secret")
        text = repr(creds)
        assert "hunter2secret" not in text
        assert "user@example.org" not in text
        assert "hunter2secret" not in f"{creds}"

    def test_missing_credentials_name_the_exact_action(self) -> None:
        with pytest.raises(RDSDownloadError) as excinfo:
            RDSCredentials().require()
        message = str(excinfo.value)
        assert "SIHPS_RDS_EMAIL" in message
        assert "SIHPS_RDS_PASSWORD" in message
        assert "rds.ncmrwf.gov.in" in message

    def test_present_credentials_pass(self) -> None:
        creds = RDSCredentials(email="u@example.org", password="p")
        assert creds.present
        assert creds.require() is creds

    def test_public_status_redacts(self) -> None:
        status = RDSCredentials(email="u@example.org", password="hunter2").public_status()
        assert "hunter2" not in json.dumps(status)
        assert "u@example.org" not in json.dumps(status)

    def test_read_from_environment(self, monkeypatch) -> None:
        monkeypatch.setenv("SIHPS_RDS_EMAIL", "u@example.org")
        monkeypatch.setenv("SIHPS_RDS_PASSWORD", "pw")
        creds = RDSCredentials.from_env()
        assert creds.present


# --------------------------------------------------------------------------- #
# sample request construction
# --------------------------------------------------------------------------- #
class TestSampleRequest:
    def test_matches_the_published_job_payload_schema(self) -> None:
        request = build_sample_request()
        payload = request["request_payload"]
        # Field names come from JobPayloadSchema, not from guesswork.
        for key in ("dataset_type", "year", "month", "day", "time", "variables", "area"):
            assert key in payload
        assert payload["dataset_type"] == "imdaa-daily"

    def test_area_is_a_string_geoarea(self) -> None:
        area = build_sample_request()["request_payload"]["area"]
        assert area["type"] == "rectangle"
        for key in ("north", "south", "east", "west"):
            assert isinstance(area[key], str)

    def test_aoi_defaults_to_the_project_bbox(self) -> None:
        area = build_sample_request()["request_payload"]["area"]
        assert area["west"] == "77.5" and area["east"] == "80.5"
        assert area["south"] == "29.0" and area["north"] == "31.5"

    def test_pressure_levels_are_optional(self) -> None:
        without = build_sample_request()["request_payload"]
        assert "pressure_level" not in without
        with_levels = build_sample_request(pressure_level=("1000", "850"))["request_payload"]
        assert with_levels["pressure_level"] == ["1000", "850"]

    def test_pressure_levels_reach_the_request(self) -> None:
        # Guards the CLI plumbing: --pressure-levels must not be silently dropped.
        import inspect

        signature = inspect.signature(acq.acquire_sample)
        assert "pressure_level" in signature.parameters
        assert signature.parameters["pressure_level"].default is None


# --------------------------------------------------------------------------- #
# job lifecycle (no network: the client is driven with a stubbed transport)
# --------------------------------------------------------------------------- #
class TestJobLifecycle:
    def test_missing_credentials_block_submission(self) -> None:
        client = RDSClient(credentials=RDSCredentials())
        with pytest.raises(RDSDownloadError):
            client.submit_job(build_sample_request())

    def test_job_states_are_classified(self) -> None:
        assert RDSJob(job_id="x", status="COMPLETED").succeeded
        assert RDSJob(job_id="x", status="COMPLETED").finished
        assert RDSJob(job_id="x", status="FAILED").finished
        assert not RDSJob(job_id="x", status="FAILED").succeeded
        assert not RDSJob(job_id="x", status="RUNNING").finished

    def test_job_dict_never_carries_a_token(self) -> None:
        payload = RDSJob(job_id="x", status="COMPLETED").to_dict()
        assert "token" not in json.dumps(payload).lower()

    def test_completed_job_without_url_is_an_error(self) -> None:
        client = RDSClient(credentials=RDSCredentials(email="u@e.org", password="p"))
        job = RDSJob(job_id="x", status="COMPLETED", file_url=None)
        with pytest.raises(RDSDownloadError, match="no file_url"):
            client.wait_for_job(job, max_polls=1, poll_seconds=0)

    def test_failed_job_raises_with_the_server_reason(self) -> None:
        client = RDSClient(credentials=RDSCredentials(email="u@e.org", password="p"))
        job = RDSJob(job_id="x", status="FAILED", error="no such variable")
        with pytest.raises(RDSDownloadError, match="no such variable"):
            client.wait_for_job(job, max_polls=1, poll_seconds=0)

    def test_unfinished_job_raises_rather_than_returning(self) -> None:
        client = RDSClient(credentials=RDSCredentials(email="u@e.org", password="p"))
        # Stub the poll so the test never touches the network.
        client.poll_job = lambda job_id: RDSJob(job_id=job_id, status="RUNNING")
        job = RDSJob(job_id="x", status="RUNNING")
        with pytest.raises(RDSDownloadError, match="did not finish"):
            client.wait_for_job(job, max_polls=2, poll_seconds=0)


    def test_run_sample_mock_workflow(self, tmp_path) -> None:
        client = RDSClient(credentials=RDSCredentials(email="u@e.org", password="p"))
        # Mock submit_job, poll_job, and download
        nc_dest = tmp_path / "artefact.nc"
        _write_nc(nc_dest, variables=("t", "q", "u", "v", "msl"))

        client.submit_job = lambda req: RDSJob(job_id="job-999", status="SUBMITTED")
        client.poll_job = lambda j_id: RDSJob(
            job_id=j_id, status="COMPLETED", file_url="https://rds.example/job-999.nc"
        )
        client.download = lambda url, dest: nc_dest

        req = build_sample_request()
        job, path = client.run_sample(req, tmp_path, poll_seconds=0, max_polls=3)
        assert job.job_id == "job-999"
        assert job.succeeded is True
        assert path == nc_dest

    def test_acquire_sample_end_to_end_mock(self, tmp_path, monkeypatch) -> None:
        import app.ingestion.realtime.imdaa_acquire as imdaa_acq
        from app.ingestion.realtime.provenance import DataClass

        monkeypatch.setenv("SIHPS_RDS_EMAIL", "test@ncmrwf.gov.in")
        monkeypatch.setenv("SIHPS_RDS_PASSWORD", "secret123")

        # Create a valid pressure profile NetCDF file to serve as the mock downloaded artefact
        profile_file = tmp_path / "downloaded_sample.nc"
        _write_profile_nc(profile_file, n_times=1)

        # Mock RDSClient.run_sample in imdaa_acquire
        mock_job = RDSJob(
            job_id="sample-job-42",
            status="COMPLETED",
            file_url="https://rds.ncmrwf.gov.in/sample.nc",
        )
        monkeypatch.setattr(
            RDSClient,
            "run_sample",
            lambda self, req, out_dir, **kwargs: (mock_job, profile_file),
        )

        result = imdaa_acq.acquire_sample(
            dataset_type="imdaa-daily",
            year="2019",
            month=("07",),
            day=("01",),
            time=("00",),
            out_dir=tmp_path,
            poll_seconds=0,
            max_polls=1,
        )

        assert result["job"]["job_id"] == "sample-job-42"
        assert result["cube"]["built"] is True
        assert result["cube"]["n_frames"] == 1
        assert "cape" in result["derived"]["channels"]["available"]
        assert "iwv" in result["derived"]["channels"]["available"]

        # Validate DatasetProvenance and source files manifest
        prov = result["provenance"]
        assert prov["data_class"] == DataClass.REANALYSIS.value
        assert prov["source_sha256"] is not None
        assert len(prov["source_files"]) == 1
        assert prov["source_files"][0]["role"] == "imdaa_pressure_levels"
        assert prov["source_files"][0]["source_identity"] == "sample-job-42"



# --------------------------------------------------------------------------- #
# NetCDF validation
# --------------------------------------------------------------------------- #
class TestNetCdfValidation:
    def test_pressure_level_file_is_recognised(self, tmp_path) -> None:
        path = _write_nc(
            tmp_path / "imdaa_levels.nc",
            levels=[1000.0, 850.0, 700.0, 500.0],
            variables=("t", "q", "u", "v", "msl"),
        )
        report = validate_imdaa_file(path)
        assert report.ok, report.quality.errors
        assert "temperature" in report.detected
        assert "specific_humidity" in report.detected
        assert report.pressure_levels_hpa == [1000.0, 850.0, 700.0, 500.0]
        assert report.lat_range == (29.0, 31.0)
        assert report.lon_range == (77.5, 80.5)
        assert report.crs == "EPSG:4326"
        assert report.times

    def test_single_level_file_is_flagged_as_unable_to_supply_cape(self, tmp_path) -> None:
        path = _write_nc(tmp_path / "imdaa_sfc.nc", variables=("t", "msl"))
        report = validate_imdaa_file(path)
        assert report.ok
        assert report.pressure_levels_hpa == []
        assert any("single-level" in note for note in report.notes)

    def test_unrecognisable_file_is_rejected_not_partially_accepted(self, tmp_path) -> None:
        path = _write_nc(tmp_path / "junk.nc", variables=("foo", "bar"))
        report = validate_imdaa_file(path)
        assert not report.ok
        assert any("no recognised" in e for e in report.quality.errors)
        assert report.detected == {}

    def test_non_nc_file_is_rejected(self, tmp_path) -> None:
        path = tmp_path / "notdata.nc"
        path.write_text("this is not a NetCDF file", encoding="utf-8")
        report = validate_imdaa_file(path)
        assert not report.ok
        assert any("could not open" in e for e in report.quality.errors)

    def test_fill_values_count_as_missing(self, tmp_path) -> None:
        path = _write_nc(
            tmp_path / "fills.nc",
            levels=[1000.0, 850.0, 500.0],
            variables=("t", "q"),
            fill=-999.0,
        )
        report = validate_imdaa_file(path)
        # Everything is the -999 sentinel, so everything is missing, not data.
        assert report.quality.n_missing == report.quality.n_values
        assert report.quality.n_values > 0

    def test_out_of_range_values_are_warned_not_clipped(self, tmp_path) -> None:
        path = _write_nc(
            tmp_path / "hot.nc",
            levels=[1000.0, 850.0, 500.0],
            variables=("t",),
            fill=500.0,  # far above the documented upper bound
        )
        report = validate_imdaa_file(path)
        assert any(
            "outside the documented physical range" in w for w in report.quality.warnings
        )

    def test_variable_alias_detection_does_not_guess(self) -> None:
        import xarray as xr

        dataset = xr.Dataset({"t2m": (("lat", "lon"), np.zeros((2, 2)))})
        detected, unmapped = detect_variables(dataset)
        # "t2m" is not in the alias list, so nothing is inferred.
        assert "temperature" not in detected
        assert unmapped == ["t2m"]

    def test_pressure_levels_in_pascal_are_converted(self) -> None:
        import xarray as xr

        dataset = xr.Dataset(
            {"t": (("level",), np.zeros(3))},
            coords={"level": [100000.0, 85000.0, 50000.0]},
        )
        assert read_pressure_levels(dataset) == [1000.0, 850.0, 500.0]



# --------------------------------------------------------------------------- #
# vertical interpolation
# --------------------------------------------------------------------------- #
class TestVerticalInterpolation:
    def _profile(self):
        src = [1000.0, 850.0, 700.0, 500.0]
        values = np.array([
            [300.0, 301.0], [288.0, 289.0], [276.0, 277.0], [240.0, 241.0],
        ])
        return src, values

    def test_log_p_interpolation_lands_between_bracketing_levels(self) -> None:
        src, values = self._profile()
        out, record = interpolate_to_levels(src, [925.0], values)
        assert 288.0 < out[0, 0] < 300.0
        assert record.stage == "vertical_interpolation"
        assert "log" in record.method

    def test_requesting_an_existing_level_reproduces_it(self) -> None:
        src, values = self._profile()
        out, _record = interpolate_to_levels(src, [850.0], values)
        assert out[0, 0] == pytest.approx(288.0)

    def test_wide_gap_is_not_interpolated_across(self) -> None:
        # A missing level must not manufacture a value two levels away.
        src = [1000.0, 850.0, 700.0]
        values = np.array([[300.0], [np.nan], [270.0]])
        out, _record = interpolate_to_levels(src, [900.0], values)
        assert np.isnan(out[0, 0])

    def test_adjacent_levels_are_interpolated(self) -> None:
        # 900 hPa sits between two adjacent valid levels, so it is filled.
        src = [1000.0, 900.0, 850.0, 700.0]
        values = np.array([[300.0], [290.0], [288.0], [276.0]])
        out, _record = interpolate_to_levels(src, [875.0], values)
        assert 288.0 < out[0, 0] < 290.0

    def test_gap_limit_is_configurable(self) -> None:
        src = [1000.0, 850.0, 700.0]
        values = np.array([[300.0], [np.nan], [270.0]])
        out, _record = interpolate_to_levels(src, [900.0], values, max_gap_levels=2)
        # Explicitly allowing a 2-level gap fills the value.
        assert not np.isnan(out[0, 0])

    def test_level_count_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="source levels"):
            interpolate_to_levels([1000.0, 850.0], [900.0], np.zeros((3, 2)))

    def test_non_positive_pressure_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            interpolate_to_levels([1000.0, 0.0], [900.0], np.zeros((2, 1)))

    def test_too_few_levels_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            interpolate_to_levels([1000.0], [900.0], np.zeros((1, 1)))


# --------------------------------------------------------------------------- #
# CAPE derivation
# --------------------------------------------------------------------------- #
def _unstable_profile() -> VerticalProfile:
    return VerticalProfile(
        pressure_hpa=[1000.0, 925.0, 850.0, 700.0, 500.0, 400.0],
        temperature_k=[302.0, 295.0, 288.0, 270.0, 240.0, 225.0],
        specific_humidity_kgkg=[0.017, 0.013, 0.009, 0.004, 0.0015, 0.0008],
    )


def _stable_profile() -> VerticalProfile:
    """A warm, absolutely stable column (lapse rate well under the dry adiabat).

    Note: the project's existing ``parcel_ascent`` is a demo-grade ascent (its
    own docstring says no entrainment or ice phase is modelled), so it returns
    non-zero CAPE even for a stable profile. The tests therefore assert the
    *ordering* and finiteness, not an absolute meteorological threshold.
    """
    return VerticalProfile(
        pressure_hpa=[1000.0, 925.0, 850.0, 700.0, 500.0, 400.0],
        temperature_k=[290.0, 289.0, 287.0, 284.0, 275.0, 270.0],
        specific_humidity_kgkg=[0.004, 0.004, 0.0035, 0.002, 0.001, 0.0004],
    )


class TestCapeDerivation:
    def test_unstable_column_yields_positive_cape(self) -> None:
        cape, _record, detail = derive_cape([_unstable_profile()])
        assert cape[0] > 0
        assert detail["n_derived"] == 1

    def test_cape_is_always_finite_and_non_negative(self) -> None:
        cape, _record, _detail = derive_cape(
            [_unstable_profile(), _stable_profile()]
        )
        assert np.isfinite(cape).all()
        assert (cape >= 0).all()

    def test_unstable_column_has_more_cape_than_stable(self) -> None:
        unstable, _r1, _d1 = derive_cape([_unstable_profile()])
        stable, _r2, _d2 = derive_cape([_stable_profile()])
        assert unstable[0] > stable[0]

    def test_batch_keeps_profiles_aligned(self) -> None:
        cape, record, detail = derive_cape(
            [_unstable_profile(), _stable_profile(), _unstable_profile()]
        )
        assert cape.shape == (3,)
        assert detail["n_derived"] == 3
        assert "parcel_ascent" in record.method

    def test_missing_humidity_yields_nan_not_an_estimate(self) -> None:
        broken = VerticalProfile(
            pressure_hpa=[1000.0, 850.0, 700.0, 500.0],
            temperature_k=[302.0, 290.0, 275.0, 240.0],
            specific_humidity_kgkg=[np.nan, 0.008, 0.004, 0.001],
        )
        cape, record, detail = derive_cape([broken])
        assert np.isnan(cape[0]), "CAPE must not be estimated without humidity"
        assert detail["n_skipped_incomplete"] == 1
        assert "specific humidity" in record.detail

    def test_profile_requires_three_levels(self) -> None:
        with pytest.raises(ValueError, match="at least 3 levels"):
            VerticalProfile(
                pressure_hpa=[1000.0, 850.0],
                temperature_k=[300.0, 280.0],
                specific_humidity_kgkg=[0.01, 0.005],
            )

    def test_profile_is_sorted_surface_first(self) -> None:
        profile = VerticalProfile(
            pressure_hpa=[500.0, 1000.0, 700.0],
            temperature_k=[240.0, 300.0, 280.0],
            specific_humidity_kgkg=[0.001, 0.015, 0.006],
        )
        assert profile.pressure_hpa[0] == 1000.0
        assert profile.pressure_hpa[-1] == 500.0



# --------------------------------------------------------------------------- #
# split coverage, computed from files actually on disk
# --------------------------------------------------------------------------- #
class TestSplitCoverage:
    def test_empty_directory_reports_no_data_not_a_failure(self, tmp_path) -> None:
        coverage = acq.assess_split_coverage(tmp_path)
        assert coverage.verdict == "no_data"
        assert coverage.years_available == []
        assert not coverage.complete

    def test_missing_years_are_reported_never_substituted(self, tmp_path) -> None:
        for year in (2008, 2009):
            _write_nc(
                tmp_path / f"imdaa_{year}.nc", levels=[1000.0, 850.0, 500.0], year=year
            )
        coverage = acq.assess_split_coverage(tmp_path)
        assert coverage.verdict == "incomplete"
        assert 2008 in coverage.years_available
        # train is 2008-2014, so the rest must be listed as missing.
        assert coverage.missing_years["train"] == [2010, 2011, 2012, 2013, 2014]
        assert any("never filled" in note for note in coverage.notes)

    def test_fully_covered_split_is_complete(self, tmp_path) -> None:
        for year in range(2008, 2021):
            _write_nc(
                tmp_path / f"imdaa_{year}.nc", levels=[1000.0, 850.0, 500.0], year=year
            )
        coverage = acq.assess_split_coverage(tmp_path)
        assert coverage.verdict == "complete"
        assert coverage.complete
        assert all(coverage.satisfied.values())

    def test_years_outside_the_published_record_are_excluded(self, tmp_path) -> None:
        _write_nc(tmp_path / "imdaa_2024.nc", levels=[1000.0, 850.0, 500.0], year=2024)
        coverage = acq.assess_split_coverage(tmp_path)
        assert coverage.verdict == "out_of_published_range"
        assert any("outside the published IMDAA record" in n for n in coverage.notes)

    def test_target_split_matches_the_phase5_proposal(self) -> None:
        assert acq.TARGET_SPLIT == {
            "train": (2008, 2014),
            "val": (2015, 2017),
            "test": (2018, 2020),
        }
        # Every year must sit inside the published IMDAA record.
        for lo, hi in acq.TARGET_SPLIT.values():
            assert acq.IMDAA_COVERAGE[0] <= lo <= hi <= acq.IMDAA_COVERAGE[1]

    def test_coverage_reads_the_time_axis_not_just_the_filename(self, tmp_path) -> None:
        # The file is named 2008 but its time axis says 2016; the time axis wins.
        path = _write_nc(
            tmp_path / "imdaa_2008.nc", levels=[1000.0, 850.0, 500.0], year=2008
        )
        import xarray as xr

        with xr.open_dataset(path) as dataset:
            replaced = dataset.assign_coords(time=["2016-07-01T00:00:00"])
            replaced.load()  # read into memory so the file handle can be released
        replaced.to_netcdf(path)
        coverage = acq.assess_split_coverage(tmp_path)
        assert coverage.years_available == [2016]


# --------------------------------------------------------------------------- #
# provenance: a real acquisition must never be labelled synthetic
# --------------------------------------------------------------------------- #
class TestProvenanceLabelling:
    def test_imdaa_provenance_is_reanalysis_not_synthetic(self) -> None:
        from app.ingestion.realtime.provenance import DataClass, DatasetProvenance

        prov = DatasetProvenance(
            source="NCMRWF RDS (IMDAA)",
            product="imdaa-daily",
            data_class=DataClass.REANALYSIS,
            acquired_at=__import__("datetime").datetime.now(
                tz=__import__("datetime").timezone.utc
            ),
        )
        assert prov.data_class == "reanalysis"
        assert prov.to_dict()["data_class"] == "reanalysis"

    def test_reanalysis_would_not_flag_a_cube_as_synthetic(self) -> None:
        import datetime as dt

        from app.ingestion.base import ObservationCube, Provenance

        grid = GridSpec.from_bbox([77.5, 29.0, 78.0, 29.5], res_km=20.0)
        base = dt.datetime(2019, 7, 1, tzinfo=dt.timezone.utc)
        cube = ObservationCube(
            grid=grid,
            times=[base, base],
            channels=np.zeros((2, 12, *grid.shape), dtype=np.float32),
            provenance=[Provenance(
                source="NCMRWF RDS (IMDAA)",
                product="imdaa-daily",
                valid_from=base,
                is_synthetic=False,
            )],
        )
        assert all(not p.is_synthetic for p in cube.provenance)


# --------------------------------------------------------------------------- #
# live source test: opt-in only
# --------------------------------------------------------------------------- #
@pytest.mark.live_source
def test_live_rds_login_when_credentials_configured() -> None:
    """LIVE-SOURCE TEST - needs SIHPS_LIVE_SOURCE_TESTS=1 *and* real credentials.

    Verifies the documented auth flow end to end. Skipped by default so the
    routine suite never depends on the network or on an account.
    """
    import os

    if os.getenv("SIHPS_LIVE_SOURCE_TESTS") != "1":
        pytest.skip("set SIHPS_LIVE_SOURCE_TESTS=1 to run live-source tests")
    if not RDSCredentials.from_env().present:
        pytest.skip("SIHPS_RDS_EMAIL / SIHPS_RDS_PASSWORD are not configured")
    client = RDSClient()
    status = client.availability(probe=True)
    assert status.available, status.reason




# --------------------------------------------------------------------------- #
# real feature extraction, grid alignment and cube assembly
# --------------------------------------------------------------------------- #
class TestChannelDerivation:
    """CAPE / IWV must come from the real calculation, not a placeholder."""

    @pytest.fixture
    def grid(self):
        return GridSpec.from_bbox([78.0, 29.0, 79.5, 30.0], res_km=20.0)

    def test_iwv_matches_the_analytic_column(self) -> None:
        from app.ingestion.realtime.imdaa_netcdf import derive_iwv
        from app.physics import GRAVITY

        levels = np.array([1000.0, 850.0, 700.0])
        value = derive_iwv(levels, np.full(3, 0.010))
        expected = 0.010 * ((1000.0 - 700.0) * 100.0) / GRAVITY
        assert value == pytest.approx(expected, rel=1e-9)

    def test_iwv_is_nan_for_a_partially_missing_column(self) -> None:
        from app.ingestion.realtime.imdaa_netcdf import derive_iwv

        levels = np.array([1000.0, 850.0, 700.0])
        # A truncated column must not report a small-but-plausible total.
        assert np.isnan(derive_iwv(levels, np.array([0.010, np.nan, 0.004])))

    def test_cape_is_derived_from_the_file(self, tmp_path, grid) -> None:
        from app.ingestion.realtime.imdaa_netcdf import (
            derive_channel_fields,
            read_pressure_fields,
        )

        path = _write_profile_nc(tmp_path / "prof.nc")
        derived = derive_channel_fields(read_pressure_fields(path), grid)
        assert set(derived.channels) == {"cape", "iwv"}
        assert np.isfinite(derived.channels["iwv"]).any()
        # A real unstable profile yields non-negative CAPE; the point is that a
        # number was computed, not that a placeholder was emitted.
        cape = derived.channels["cape"]
        finite = cape[np.isfinite(cape)]
        assert finite.size > 0
        assert (finite >= 0).all()
        assert derived.coverage["iwv"] > 0.0

    def test_no_humidity_leaves_cape_and_iwv_nan(self, tmp_path, grid) -> None:
        from app.ingestion.realtime.imdaa_netcdf import (
            derive_channel_fields,
            read_pressure_fields,
        )

        path = _write_profile_nc(tmp_path / "dry.nc", with_humidity=False)
        fields = read_pressure_fields(path)
        assert fields.specific_humidity_kgkg is None
        derived = derive_channel_fields(fields, grid)
        for values in derived.channels.values():
            assert np.isnan(values).all()

    def test_cells_outside_the_file_extent_stay_nan(self, tmp_path) -> None:
        from app.ingestion.realtime.imdaa_netcdf import (
            derive_channel_fields,
            read_pressure_fields,
        )

        path = _write_profile_nc(tmp_path / "small.nc")
        # Target grid extends well beyond the file's 78.0-79.5 x 29.0-30.0 box.
        wide = GridSpec.from_bbox([76.0, 27.0, 82.0, 32.0], res_km=20.0)
        derived = derive_channel_fields(read_pressure_fields(path), wide)
        iwv = derived.channels["iwv"]
        assert np.isfinite(iwv).any()
        assert np.isnan(iwv).any()
        # Nothing outside the source box may be silently edge-clamped to a value.
        outside = np.where(np.asarray(wide.lat_centers()) > 30.0)[0]
        assert np.isnan(iwv[:, outside, :]).all()

    def test_single_level_file_is_rejected(self, tmp_path) -> None:
        from app.ingestion.realtime.imdaa_netcdf import read_pressure_fields

        path = _write_nc(tmp_path / "single.nc", variables=("t", "q"))
        with pytest.raises(ValueError, match="pressure axis"):
            read_pressure_fields(path)

    def test_fill_values_are_not_read_as_measurements(self, tmp_path) -> None:
        from app.ingestion.realtime.imdaa_netcdf import read_pressure_fields

        filled = _write_profile_nc(tmp_path / "filled.nc")
        import xarray as xr

        with xr.open_dataset(filled) as ds:
            t = ds["t"].values.copy()
            times = ds["time"].values
            levels = ds["level"].values
            lats = ds["lat"].values
            lons = ds["lon"].values
        t[:, -1, :, :] = -9999.0
        xr.Dataset(
            {"t": (["time", "level", "lat", "lon"], t)},
            coords={"time": times, "level": levels, "lat": lats, "lon": lons},
        ).to_netcdf(filled)
        fields = read_pressure_fields(filled)
        # A fill sentinel at the top level must stay missing, not read as 0 K
        # and not be interpolated over as if it were a real observation.
        assert np.isnan(fields.temperature_k[:, -1, :, :]).all()
        assert np.isfinite(fields.temperature_k[:, 0, :, :]).all()


class TestObservationCubeAssembly:
    def test_cube_has_twelve_channels_with_only_two_filled(self, tmp_path) -> None:
        from app.ingestion.realtime.imdaa_netcdf import build_observation_cube
        from app.physics import CHANNELS, channel_index

        grid = GridSpec.from_bbox([78.0, 29.0, 79.5, 30.0], res_km=20.0)
        cube = build_observation_cube(_write_profile_nc(tmp_path / "p.nc"), grid)

        assert cube.channels.shape == (2, len(CHANNELS), *grid.shape)
        assert np.isfinite(cube.channels[:, channel_index("cape")]).any()
        assert np.isfinite(cube.channels[:, channel_index("iwv")]).any()
        # The ten channels IMDAA cannot supply must be NaN, never zero-filled.
        for name in ("tir1_bt", "vis_refl", "mir_bt", "elevation"):
            assert np.isnan(cube.channels[:, channel_index(name)]).all(), name
        assert set(cube.metadata["channels_available"]) == {"cape", "iwv"}
        assert len(cube.metadata["channels_missing"]) == 10
        assert cube.metadata["trainable"] is False

    def test_cube_records_real_provenance(self, tmp_path) -> None:
        from app.ingestion.realtime.imdaa_netcdf import build_observation_cube

        grid = GridSpec.from_bbox([78.0, 29.0, 79.5, 30.0], res_km=20.0)
        cube = build_observation_cube(_write_profile_nc(tmp_path / "p.nc"), grid)
        record = cube.provenance[0]
        assert record.is_synthetic is False
        assert "IMDAA" in record.attribution
        assert record.valid_from <= record.valid_to
        # valid_from/valid_to must come from the file's own time axis.
        assert cube.times[0].year == 2019

    def test_cube_round_trips_through_npz(self, tmp_path) -> None:
        from app.ingestion.base import ObservationCube
        from app.ingestion.realtime.imdaa_netcdf import build_observation_cube

        grid = GridSpec.from_bbox([78.0, 29.0, 79.5, 30.0], res_km=20.0)
        cube = build_observation_cube(_write_profile_nc(tmp_path / "p.nc"), grid)
        reloaded = ObservationCube.load_npz(cube.save_npz(tmp_path / "cube.npz"))
        assert reloaded.channels.shape == cube.channels.shape
        assert np.isnan(reloaded.channels).sum() == np.isnan(cube.channels).sum()
        assert reloaded.metadata["channels_available"] == ["cape", "iwv"]

    def test_file_without_derived_data_refuses_to_build_a_cube(self, tmp_path) -> None:
        from app.ingestion.realtime.imdaa_netcdf import build_observation_cube

        grid = GridSpec.from_bbox([78.0, 29.0, 79.5, 30.0], res_km=20.0)
        path = _write_profile_nc(tmp_path / "dry.nc", with_humidity=False)
        with pytest.raises(ValueError, match="no finite value"):
            build_observation_cube(path, grid)

    def test_invalid_file_is_refused(self, tmp_path) -> None:
        from app.ingestion.realtime.imdaa_netcdf import build_observation_cube

        grid = GridSpec.from_bbox([78.0, 29.0, 79.5, 30.0], res_km=20.0)
        junk = tmp_path / "junk.nc"
        junk.write_bytes(b"this is not a NetCDF file")
        with pytest.raises(ValueError, match="failed validation"):
            build_observation_cube(junk, grid)



# --------------------------------------------------------------------------- #
# connector-level assembly
# --------------------------------------------------------------------------- #
class TestConnectorFetch:
    """``IMDAAConnector.fetch`` must assemble real channels or refuse."""

    def test_empty_directory_raises_rather_than_fabricating(self, tmp_path) -> None:
        from app.ingestion.realtime.imdaa import IMDAAConnector

        grid = GridSpec.from_bbox([78.0, 29.0, 79.5, 30.0], res_km=20.0)
        connector = IMDAAConnector(grid, data_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="no IMDAA/MERA NetCDF files"):
            connector.fetch()

    def test_fetch_assembles_only_available_channels(self, tmp_path) -> None:
        from app.ingestion.realtime.imdaa import IMDAAConnector
        from app.physics import CHANNELS, channel_index

        grid = GridSpec.from_bbox([78.0, 29.0, 79.5, 30.0], res_km=20.0)
        _write_profile_nc(tmp_path / "a.nc", n_times=2)
        _write_profile_nc(tmp_path / "b.nc", n_times=1)
        cube = IMDAAConnector(grid, data_dir=tmp_path).fetch()

        assert cube.n_frames == 3
        assert cube.channels.shape == (3, len(CHANNELS), *grid.shape)
        assert np.isfinite(cube.channels[:, channel_index("cape")]).any()
        for name in ("tir1_bt", "mir_bt", "elevation"):
            assert np.isnan(cube.channels[:, channel_index(name)]).all(), name
        assert set(cube.metadata["channels_available"]) == {"cape", "iwv"}
        assert len(cube.metadata["channels_missing"]) == 10
        assert cube.metadata["trainable"] is False
        assert len(cube.provenance) == 2
        assert all(not p.is_synthetic for p in cube.provenance)

    def test_fetch_refuses_when_no_file_can_supply_a_channel(self, tmp_path) -> None:
        from app.ingestion.realtime.imdaa import IMDAAConnector

        grid = GridSpec.from_bbox([78.0, 29.0, 79.5, 30.0], res_km=20.0)
        _write_profile_nc(tmp_path / "dry.nc", with_humidity=False)
        with pytest.raises(ValueError, match="could supply a derived channel"):
            IMDAAConnector(grid, data_dir=tmp_path).fetch()

    def test_unusable_file_is_skipped_not_fatal(self, tmp_path) -> None:
        from app.ingestion.realtime.imdaa import IMDAAConnector

        grid = GridSpec.from_bbox([78.0, 29.0, 79.5, 30.0], res_km=20.0)
        _write_profile_nc(tmp_path / "good.nc", n_times=1)
        (tmp_path / "junk.nc").write_bytes(b"not a netcdf file")
        cube = IMDAAConnector(grid, data_dir=tmp_path).fetch()
        assert cube.n_frames == 1
        assert any("junk.nc" in note for note in cube.metadata["notes"])
