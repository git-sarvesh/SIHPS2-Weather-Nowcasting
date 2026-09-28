"""Tests for the Phase 5 real-data ingestion layer.

Constraints honoured here:

* **No downloads.** Every fixture is a few kilobytes written to ``tmp_path``.
* **No synthetic data passed off as real.** Numeric fixtures are explicitly
  labelled with a ``data_class`` and tests assert that labelling survives the
  pipeline, so a real-data path can never inherit a synthetic flag.
* **No credentials in output.** Tests assert secrets are *redacted*, not merely
  absent.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from app.db.migrations import applied_revisions, init_db
from app.db.repository import list_dataset_provenance, record_dataset_provenance, scrub_detail
from app.db.session import SessionFactory
from app.grid import GridSpec
from app.ingestion.base import ObservationCube, WindowSpec
from app.ingestion.realtime import (
    AlignedField,
    Availability,
    CanonicalPipeline,
    Credentials,
    DataClass,
    DatasetProvenance,
    IMDStationConnector,
    MOSDACConnector,
    access,
    assess_split_feasibility,
    build_windows,
    composite_temporal,
    decode_brightness_temperature,
    file_sha256,
    ingest_key,
    load_station_csv,
    normalise_units,
    validate_dataset,
)
from app.ingestion.realtime.imdaa import IMDAAConnector, PRODUCTS
from app.ingestion.realtime.mosdac import MOSDAC_PASSWORD_ENV, MOSDAC_USER_ENV
from app.physics import CHANNELS
from app.training.validation_report import (
    EVALUATIONS_REQUIRING_REAL_DATA,
    build_capability_report,
)


@pytest.fixture
def grid() -> GridSpec:
    """Small canonical grid, same 30-minute cadence contract as production."""
    return GridSpec.from_bbox([78.0, 30.0, 79.0, 31.0], res_km=12.0)


@pytest.fixture
def source_grid() -> GridSpec:
    """Coarser 'source' grid standing in for a ~12 km reanalysis grid."""
    return GridSpec.from_bbox([78.0, 30.0, 79.0, 31.0], res_km=30.0)


@pytest.fixture
def small_cube(grid: GridSpec) -> ObservationCube:
    """A tiny fully-populated cube, used only for window-geometry tests."""
    times = [
        dt.datetime(2020, 7, 1, tzinfo=dt.timezone.utc) + dt.timedelta(minutes=30 * i)
        for i in range(20)
    ]
    shape = (len(times), 12, *grid.shape)
    return ObservationCube(
        grid=grid,
        times=times,
        channels=np.ones(shape, dtype=np.float32),
        quality=np.ones(shape, dtype=np.float32),
    )


def aligned(grid: GridSpec, channel: str, unit: str, data_class: str) -> AlignedField:
    """A fully-observed constant field, for shape/label assertions only."""
    return AlignedField(
        data=np.full(grid.shape, 250.0, dtype=np.float32),
        valid=np.ones(grid.shape, dtype=bool),
        channel=channel,
        unit=unit,
        data_class=data_class,
    )


def half_hourly(n: int = 4) -> list[dt.datetime]:
    base = dt.datetime(2020, 7, 1, tzinfo=dt.timezone.utc)
    return [base + dt.timedelta(minutes=30 * i) for i in range(n)]



class TestCredentials:
    def test_repr_never_leaks_the_secret(self) -> None:
        creds = Credentials(username="alice", password="s3cr3t-token-value")
        text = repr(creds)
        assert "s3cr3t-token-value" not in text
        assert "<set:" in text
        assert "s3cr3t-token-value" not in f"{creds}"

    def test_missing_credentials_raise_with_actionable_steps(self) -> None:
        with pytest.raises(access.CredentialMissing) as excinfo:
            Credentials().require(
                source="MOSDAC INSAT-3D", env_hint=(MOSDAC_USER_ENV, MOSDAC_PASSWORD_ENV)
            )
        message = str(excinfo.value)
        assert MOSDAC_USER_ENV in message
        assert MOSDAC_PASSWORD_ENV in message
        assert "approved" in message.lower()
        assert "password=" not in message

    def test_credentials_read_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MOSDAC_USER_ENV, "alice")
        monkeypatch.setenv(MOSDAC_PASSWORD_ENV, "hunter2")
        creds = access.credential_status(MOSDAC_USER_ENV, MOSDAC_PASSWORD_ENV)
        assert creds.present
        assert creds.public_status()["username"] != "alice"


class TestProvenance:
    def test_checksum_is_stable_and_content_addressed(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.bin", tmp_path / "b.bin"
        a.write_bytes(b"payload")
        b.write_bytes(b"payload")
        assert file_sha256(a) == file_sha256(b)
        b.write_bytes(b"different")
        assert file_sha256(a) != file_sha256(b)

    def test_ingest_key_is_deterministic(self) -> None:
        stamp = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
        assert ingest_key("IMDAA", "p", stamp, "abc") == ingest_key("IMDAA", "p", stamp, "abc")
        assert ingest_key("IMDAA", "p", stamp, "abc") != ingest_key("IMDAA", "p", stamp, "def")

    def test_secret_looking_fields_are_scrubbed(self) -> None:
        prov = DatasetProvenance(
            source="MOSDAC",
            product="3SIMG_L1B_STD",
            data_class=DataClass.OBSERVATION,
            acquired_at=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
            checksum="deadbeef",
            processing_config={"password": "hunter2", "datasetId": "3SIMG_L1B_STD"},
        )
        payload = json.dumps(prov.to_dict())
        assert "hunter2" not in payload
        assert "3SIMG_L1B_STD" in payload

    def test_data_class_is_validated(self) -> None:
        with pytest.raises(ValueError):
            DatasetProvenance(
                source="x",
                product="y",
                data_class="definitely-real",
                acquired_at=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
            )

    def test_overlaps_grid(self, grid: GridSpec) -> None:
        prov = DatasetProvenance(
            source="s",
            product="p",
            data_class=DataClass.REANALYSIS,
            acquired_at=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
            min_lon=77.0, min_lat=29.0, max_lon=82.0, max_lat=32.0,
        )
        assert prov.overlaps_grid(grid)
        prov.max_lon = 77.1
        assert not prov.overlaps_grid(grid)


class TestUnitConversion:
    def test_knots_to_metres_per_second(self, grid: GridSpec) -> None:
        field_ = aligned(grid, "tir1_bt", "knot", DataClass.OBSERVATION)
        field_.data = np.full(grid.shape, 10.0, dtype=np.float32)
        normalise_units(field_, "knot", "m s-1", "wind_speed")
        assert field_.unit == "m s-1"
        assert np.allclose(field_.data, 10.0 * 0.514444, atol=1e-3)

    def test_celsius_to_kelvin_is_affine(self, grid: GridSpec) -> None:
        field_ = aligned(grid, "tir1_bt", "C", DataClass.OBSERVATION)
        field_.data = np.full(grid.shape, 20.0, dtype=np.float32)
        normalise_units(field_, "C", "K", "temperature")
        assert np.allclose(field_.data, 293.15, atol=1e-3)

    def test_unknown_conversion_raises_rather_than_passing_through(self, grid: GridSpec) -> None:
        field_ = aligned(grid, "cape", "furlongs", DataClass.OBSERVATION)
        with pytest.raises(ValueError, match="no conversion"):
            normalise_units(field_, "furlongs", "J kg-1", "cape")

    def test_brightness_temperature_inverts_the_planck_form(self) -> None:
        # T = c2 / ln(1 + c1/L)  <=>  L = c1 / (exp(c2/T) - 1)
        c1, c2, target = 1.0, 1000.0, 250.0
        radiance = c1 / (np.exp(c2 / target) - 1.0)
        recovered = float(np.atleast_1d(decode_brightness_temperature(radiance, c1, c2))[0])
        assert abs(recovered - target) < 0.5


class TestMissingData:
    def test_masked_cells_become_nan_not_zero(self, grid: GridSpec) -> None:
        mask = np.ones(grid.shape, dtype=bool)
        mask[0, 0] = False
        field_ = AlignedField(
            data=np.zeros(grid.shape, dtype=np.float32),  # ambiguous if kept
            valid=mask,
            channel="tir1_bt",
            unit="K",
            data_class=DataClass.OBSERVATION,
        )
        assert np.isnan(field_.data[0, 0])
        assert field_.missing_fraction == pytest.approx(1.0 / grid.size, rel=1e-3)

    def test_validation_measures_missing_fraction(self, grid: GridSpec) -> None:
        frames = np.ones((3, *grid.shape), dtype=np.float32)
        mask = np.zeros((3, *grid.shape), dtype=bool)
        mask[0] = True
        report = validate_dataset(frames, grid=grid, valid_mask=mask)
        assert report.n_values == frames.size
        assert 0.6 < report.missing_fraction < 0.7

    def test_validation_fails_when_mostly_missing(self, grid: GridSpec) -> None:
        frames = np.ones((3, *grid.shape), dtype=np.float32)
        mask = np.zeros((3, *grid.shape), dtype=bool)
        mask[0, 0, 0] = True
        report = validate_dataset(frames, grid=grid, valid_mask=mask)
        assert not report.passed
        assert any("missing fraction" in e for e in report.errors)

    def test_validation_rejects_wrong_shape(self, grid: GridSpec) -> None:
        report = validate_dataset(np.ones((3, 2, 2), dtype=np.float32), grid=grid)
        assert not report.passed

    def test_out_of_range_values_are_warned_not_clipped(self, grid: GridSpec) -> None:
        frames = np.full((1, *grid.shape), 400.0, dtype=np.float32)
        report = validate_dataset(
            frames,
            grid=grid,
            expected_channels=["tir1_bt"],
            channel_bounds={"tir1_bt": (180.0, 320.0)},
        )
        assert report.passed


class TestTemporalCompositing:
    def test_native_cadence_is_passed_through_without_a_record(self, grid: GridSpec) -> None:
        frames = [np.full(grid.shape, float(i), dtype=np.float32) for i in range(4)]
        out, out_times, record = composite_temporal(frames, half_hourly(4))
        assert record is None, "no resampling happened, so none must be claimed"
        assert len(out) == 4
        assert out_times == half_hourly(4)

    def test_two_observations_in_one_bucket_are_averaged(self, grid: GridSpec) -> None:
        base = dt.datetime(2020, 7, 1, tzinfo=dt.timezone.utc)
        frames = [
            np.full(grid.shape, 0.0, dtype=np.float32),
            np.full(grid.shape, 4.0, dtype=np.float32),
        ]
        times = [base + dt.timedelta(minutes=10), base + dt.timedelta(minutes=20)]
        out, _times, record = composite_temporal(frames, times)
        assert record is not None
        assert "nanmean" in record.method
        assert len(out) == 1
        assert np.allclose(out[0], 2.0)

    def test_fully_unobserved_bucket_stays_nan(self, grid: GridSpec) -> None:
        base = dt.datetime(2020, 7, 1, tzinfo=dt.timezone.utc)
        frame = np.full(grid.shape, np.nan, dtype=np.float32)
        out, _times, _record = composite_temporal(
            [frame, frame],
            [base + dt.timedelta(hours=3), base + dt.timedelta(minutes=230)],
        )
        assert np.isnan(out).all()

    def test_mismatched_frame_and_time_counts_raise(self, grid: GridSpec) -> None:
        base = dt.datetime(2020, 7, 1, tzinfo=dt.timezone.utc)
        with pytest.raises(ValueError, match="must correspond"):
            composite_temporal(
                [np.zeros(grid.shape, np.float32)],
                [base, base + dt.timedelta(minutes=30)],
            )


class TestSpatialAlignment:
    def test_aligning_records_the_resampling_step(
        self, grid: GridSpec, source_grid: GridSpec
    ) -> None:
        pipe = CanonicalPipeline(
            grid, source="test", product="fixture", data_class=DataClass.REANALYSIS
        )
        result = pipe.align_field(
            np.full(source_grid.shape, 280.0, dtype=np.float32),
            np.ones(source_grid.shape, dtype=bool),
            source_grid=source_grid,
            channel="tir1_bt",
            unit="K",
        )
        assert result.data.shape == grid.shape
        assert len(result.resampling) == 1
        assert result.resampling[0].stage == "align_spatial"
        assert "12 km" in result.resampling[0].target_resolution

    def test_nearest_method_is_recorded_as_nearest(
        self, grid: GridSpec, source_grid: GridSpec
    ) -> None:
        pipe = CanonicalPipeline(grid, source="t", product="f")
        result = pipe.align_field(
            np.full(source_grid.shape, 280.0, dtype=np.float32),
            np.ones(source_grid.shape, dtype=bool),
            source_grid=source_grid,
            channel="tir1_bt",
            unit="K",
            method="nearest",
        )
        assert result.resampling[0].method == "nearest"

    def test_misaligned_input_is_rejected(
        self, grid: GridSpec, source_grid: GridSpec
    ) -> None:
        pipe = CanonicalPipeline(grid, source="t", product="f")
        with pytest.raises(ValueError, match="source grid"):
            pipe.align_field(
                np.ones((3, 3), dtype=np.float32),
                np.ones((3, 3), dtype=bool),
                source_grid=source_grid,
                channel="tir1_bt",
                unit="K",
            )


class TestWindows:
    def test_window_geometry_matches_the_training_contract(
        self, small_cube: ObservationCube
    ) -> None:
        spec = WindowSpec(seq_len=6, horizon=12, frame_minutes=30, stride=1)
        windows = build_windows(small_cube, spec)
        assert windows, "a 20-frame cube supports 6+12=18 frames per window"
        index, history, future = windows[0]
        assert index == 0
        assert history == slice(0, 6)
        assert future == slice(6, 18)

    def test_too_short_a_cube_yields_no_windows(self, grid: GridSpec) -> None:
        tiny = ObservationCube(
            grid=grid,
            times=half_hourly(5),
            channels=np.ones((5, 12, *grid.shape), dtype=np.float32),
            quality=np.ones((5, 12, *grid.shape), dtype=np.float32),
        )
        spec = WindowSpec(seq_len=6, horizon=12, frame_minutes=30)
        assert build_windows(tiny, spec) == []


class TestStationCsv:
    def test_short_header_parses(self, tmp_path: Path) -> None:
        path = tmp_path / "rain.csv"
        path.write_text(
            "station_id,lat,lon,time,rain_mm\n"
            "STN001,30.25,78.75,2020-07-01T00:30:00Z,12.5\n",
            encoding="utf-8",
        )
        rows = load_station_csv(path)
        assert len(rows) == 1
        assert rows[0].station_id == "STN001"
        assert rows[0].rain_mm == pytest.approx(12.5)
        assert rows[0].time.tzinfo is not None

    def test_verbose_header_and_quality_flag(self, tmp_path: Path) -> None:
        path = tmp_path / "rain.csv"
        path.write_text(
            "Station ID,Latitude,Longitude,Date,Rainfall,Quality Flag\n"
            "STN002,30.5,79.0,01-07-2020 06:00,3.2,good\n",
            encoding="utf-8",
        )
        rows = load_station_csv(path)
        assert rows[0].quality_flag == "good"
        assert rows[0].rain_mm == pytest.approx(3.2)
        # A naive timestamp must be interpreted as UTC, not left naive.
        assert rows[0].time.utcoffset() is not None

    def test_missing_rainfall_stays_missing_not_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "rain.csv"
        path.write_text(
            "station_id,lat,lon,time,rain_mm\n"
            "STN003,30.5,79.0,2020-07-01T00:00:00Z,\n"
            "STN004,30.6,79.1,2020-07-01T00:00:00Z,0.0\n",
            encoding="utf-8",
        )
        rows = load_station_csv(path)
        assert rows[0].is_missing
        assert np.isnan(rows[0].rain_mm)
        # A genuine zero must stay distinguishable from a missing value.
        assert not rows[1].is_missing
        assert rows[1].to_dict()["rain_mm"] == 0.0
        assert rows[0].to_dict()["rain_mm"] is None

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("station_id,lat\nSTN005,30.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="required column"):
            load_station_csv(path)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            load_station_csv(path)



class TestConnectorAvailability:
    def test_mosdac_without_credentials_reports_the_blocker(
        self, grid: GridSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(MOSDAC_USER_ENV, raising=False)
        monkeypatch.delenv(MOSDAC_PASSWORD_ENV, raising=False)
        status = MOSDACConnector(grid, data_dir=tmp_path).availability()
        assert not status.available
        assert status.availability is Availability.NEEDS_CREDENTIALS
        assert MOSDAC_USER_ENV in status.manual_instructions
        assert "mosdac.gov.in" in status.manual_instructions.lower()
        assert "password=" not in status.manual_instructions.lower()

    def test_mosdac_with_staged_files_is_available(
        self, grid: GridSpec, tmp_path: Path
    ) -> None:
        (tmp_path / "3SIMG_L1B_STD.nc").write_bytes(b"fake granule")
        status = MOSDACConnector(grid, data_dir=tmp_path).availability()
        assert status.available
        assert status.availability is Availability.AVAILABLE
        assert status.details["n_local_files"] == 1

    def test_mosdac_fetch_refuses_rather_than_fabricating(
        self, grid: GridSpec, tmp_path: Path
    ) -> None:
        """Phase 8.5 implemented ``fetch``; with nothing staged it still refuses.

        The intent of this test is unchanged: no granule means no radiance, no
        brightness temperature and no cube - never a filled substitute.
        """
        with pytest.raises(FileNotFoundError, match="no INSAT L1B/L2 granule"):
            MOSDACConnector(grid).fetch()
        with pytest.raises(FileNotFoundError, match="no INSAT L1B/L2 granule"):
            MOSDACConnector(grid, data_dir=tmp_path).fetch()

    def test_mosdac_fetch_refuses_an_unreadable_granule(
        self, grid: GridSpec, tmp_path: Path
    ) -> None:
        """A staged file that is not a granule yields an error, not a cube."""
        (tmp_path / "junk.nc").write_bytes(b"this is not a NetCDF granule")
        with pytest.raises(ValueError, match="could supply an INSAT channel"):
            MOSDACConnector(grid, data_dir=tmp_path).fetch()

    def test_mosdac_credential_status_never_exposes_values(
        self, grid: GridSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(MOSDAC_USER_ENV, "alice")
        monkeypatch.setenv(MOSDAC_PASSWORD_ENV, "hunter2")
        status = MOSDACConnector(grid, data_dir=tmp_path).availability()
        rendered = json.dumps(status.to_dict())
        assert "hunter2" not in rendered
        assert "alice" not in rendered
        assert status.details["credentials"]["present"] is True

    def test_imdaa_without_files_reports_manual_download(
        self, grid: GridSpec, tmp_path: Path
    ) -> None:
        status = IMDAAConnector(grid, data_dir=tmp_path).availability(timeout=0.0)
        assert not status.available
        assert status.availability is Availability.NEEDS_MANUAL_DOWNLOAD
        assert "rds.ncmrwf.gov.in" in status.manual_instructions

    def test_imd_station_connector_reports_no_automatic_endpoint(self, grid: GridSpec) -> None:
        status = IMDStationConnector(grid).availability()
        assert not status.available
        assert status.details["legacy_portals"]

    def test_connectors_declare_their_data_class(self, grid: GridSpec) -> None:
        # A reanalysis connector must not present itself as an observation.
        assert IMDAAConnector(grid).data_class == "reanalysis"
        assert MOSDACConnector(grid).data_class == "observation"
        assert IMDStationConnector(grid).data_class == "observation"

    def test_unknown_product_is_rejected(self, grid: GridSpec) -> None:
        with pytest.raises(ValueError, match="unknown IMDAA product"):
            IMDAAConnector(grid, product="not-a-product")


class TestSplitFeasibility:
    def test_required_split_is_not_feasible(self) -> None:
        report = assess_split_feasibility()
        assert report.feasible is False
        assert "train" in report.unsatisfiable

    def test_proposed_alternative_stays_inside_real_coverage(self) -> None:
        report = assess_split_feasibility()
        # Every proposed year must be covered by a published product.
        for name, (lo, hi) in report.proposed.items():
            assert any(p.covers(lo) and p.covers(hi) for p in PRODUCTS.values()), name

    def test_rainfall_only_coverage_is_called_out(self) -> None:
        joined = " ".join(assess_split_feasibility().notes)
        assert "rainfall-only" in joined
        assert "2024-2025" in joined

    def test_imdaa_coverage_ends_in_2020(self) -> None:
        assert PRODUCTS["hourly-pressure"].end_year == 2020
        assert not PRODUCTS["hourly-pressure"].covers(2024)

    def test_mera_covers_the_test_window_as_rainfall_only(self) -> None:
        mera = PRODUCTS["mera"]
        assert mera.covers(2024) and mera.covers(2025)
        assert mera.product_type == "Rainfall"

    def test_a_fully_covered_split_is_reported_feasible(self) -> None:
        report = assess_split_feasibility(
            {"train": (2010, 2012), "val": (2013, 2014), "test": (2015, 2016)}
        )
        assert report.feasible
        assert report.unsatisfiable == []



@pytest.fixture
def session_factory(tmp_path: Path):
    """An isolated SQLite database, migrated to the latest revision."""
    factory = SessionFactory(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    init_db(factory.engine)
    yield factory
    factory.dispose()


def provenance(**overrides) -> DatasetProvenance:
    payload = {
        "source": "IMDAA",
        "product": "hourly-pressure",
        "data_class": DataClass.REANALYSIS,
        "acquired_at": dt.datetime(2019, 7, 1, tzinfo=dt.timezone.utc),
        "checksum": "abc123",
        "attribution": "NCMRWF",
    }
    payload.update(overrides)
    return DatasetProvenance(**payload)


class TestProvenancePersistence:
    def test_migration_adds_the_provenance_table(self, session_factory) -> None:
        revisions = applied_revisions(session_factory.engine)
        assert "0001_initial_schema" in revisions
        assert "0002_dataset_provenance" in revisions

    def test_migration_is_idempotent(self, session_factory) -> None:
        assert init_db(session_factory.engine)["applied_now"] == []

    def test_migration_adds_no_columns_to_existing_tables(self, session_factory) -> None:
        from sqlalchemy import inspect

        tables = set(inspect(session_factory.engine).get_table_names())
        # Backward compatibility: the three original tables are untouched.
        assert {"forecast_runs", "risk_cells", "audit_records"} <= tables
        assert "dataset_provenance" in tables

    def test_record_and_read_back(self, session_factory) -> None:
        with session_factory.scope() as session:
            row, created = record_dataset_provenance(session, provenance())
            assert created
            assert row.data_class == DataClass.REANALYSIS
            rows = list_dataset_provenance(session)
            assert len(rows) == 1
            assert rows[0].ingest_key == row.ingest_key

    def test_duplicate_ingestion_is_prevented(self, session_factory) -> None:
        with session_factory.scope() as session:
            _row, first = record_dataset_provenance(session, provenance())
            _row, second = record_dataset_provenance(session, provenance())
            assert first is True
            assert second is False, "the same product must not be ingested twice"
            assert len(list_dataset_provenance(session)) == 1

    def test_a_different_checksum_is_a_different_dataset(self, session_factory) -> None:
        with session_factory.scope() as session:
            record_dataset_provenance(session, provenance(checksum="aaa"))
            _row, created = record_dataset_provenance(session, provenance(checksum="bbb"))
            assert created
            assert len(list_dataset_provenance(session)) == 2

    def test_filtering_by_data_class(self, session_factory) -> None:
        with session_factory.scope() as session:
            record_dataset_provenance(session, provenance(product="p1"))
            record_dataset_provenance(
                session, provenance(product="p2", data_class=DataClass.SYNTHETIC, checksum="z")
            )
            real = list_dataset_provenance(session, data_class=DataClass.REANALYSIS)
            assert len(real) == 1
            assert real[0].product == "p1"

    def test_details_are_scrubbed_before_storage(self, session_factory) -> None:
        prov = provenance(processing_config={"password": "hunter2", "ok": "value"})
        with session_factory.scope() as session:
            row, _ = record_dataset_provenance(session, prov)
            rendered = json.dumps(row.detail)
            assert "hunter2" not in rendered
            assert "value" in rendered

    def test_scrub_detail_removes_credential_patterns(self) -> None:
        cleaned = scrub_detail({"note": "password=hunter2", "n": 1})
        assert "hunter2" not in json.dumps(cleaned)
        assert cleaned["n"] == 1



class TestValidationCapabilityReport:
    """The report must never overstate what has been validated."""

    def test_no_ingested_real_data_means_no_validation(self) -> None:
        report = build_capability_report(real_datasets_ingested=0)
        assert report.validation_performed is False
        assert report.status == "no_real_data_accessible"
        payload = report.to_dict()
        assert payload["real_datasets_ingested"] == 0
        assert "No independently validated skill" in payload["disclaimer"]

    def test_ingested_data_alone_does_not_imply_validation(self) -> None:
        # Data present but the required split is impossible: still not validated.
        report = build_capability_report(real_datasets_ingested=5)
        assert report.validation_performed is False

    def test_blocked_evaluations_are_specific(self) -> None:
        report = build_capability_report()
        payload = report.to_dict()
        assert payload["blocked_evaluations"]
        for entry in payload["blocked_evaluations"]:
            assert entry["evaluation"]
            assert len(entry["blocked_because"]) > 20, "each blocker must be explained"

    def test_report_names_the_coverage_shortfall(self) -> None:
        payload = build_capability_report().to_dict()
        joined = " ".join(payload["notes"])
        assert "2020" in joined
        assert "synthesised" in joined.lower() or "synthesized" in joined.lower()

    def test_split_infeasibility_is_reported(self) -> None:
        payload = build_capability_report().to_dict()
        assert payload["split_feasible"] is False
        assert "train" in payload["unsatisfiable_splits"]

    def test_every_blocked_entry_has_a_reason(self) -> None:
        for name, reason in EVALUATIONS_REQUIRING_REAL_DATA:
            assert name and reason



# --------------------------------------------------------------------------- #
# INSAT reader, NetCDF validation and channel derivations
# --------------------------------------------------------------------------- #
class TestINSATNetCDFReader:
    """Validate genuine INSAT-3D/3DR NetCDF granules and derived channels."""

    def _write_insat_nc(
        self,
        path: Path,
        *,
        channels: dict[str, tuple[str, float]] | None = None,
        units: dict[str, str] | None = None,
        curvilinear: bool = False,
        no_time: bool = False,
        no_coords: bool = False,
    ) -> Path:
        import xarray as xr

        if channels is None:
            channels = {
                "tir1_bt": ("TIR1_BT", 240.0),
                "tir2_bt": ("TIR2_BT", 235.0),
                "wv_bt": ("WV_BT", 230.0),
                "vis_refl": ("VIS_REFL", 0.5),
            }
        units = units or {}

        if no_coords:
            coords = {}
            dims = ("y", "x")
            shape = (4, 4)
        elif curvilinear:
            lats_2d = np.tile(np.linspace(29.0, 31.0, 4)[:, None], (1, 4))
            lons_2d = np.tile(np.linspace(77.5, 80.5, 4)[None, :], (4, 1))
            coords = {"latitude": (("y", "x"), lats_2d), "longitude": (("y", "x"), lons_2d)}
            dims = ("y", "x")
            shape = (4, 4)
        else:
            coords = {
                "lat": np.linspace(29.0, 31.0, 4),
                "lon": np.linspace(77.5, 80.5, 4),
            }
            if not no_time:
                coords["time"] = ["2019-07-01T00:00:00"]
                dims = ("time", "lat", "lon")
                shape = (1, 4, 4)
            else:
                dims = ("lat", "lon")
                shape = (4, 4)

        data_vars = {}
        for channel, (var_name, val) in channels.items():
            arr = np.full(shape, val, dtype=np.float32)
            attrs = {}
            if channel in units:
                attrs["units"] = units[channel]
            elif channel.endswith("_bt"):
                attrs["units"] = "K"
            elif channel.endswith("_refl"):
                attrs["units"] = "-"
            data_vars[var_name] = (dims, arr, attrs)

        ds = xr.Dataset(data_vars, coords=coords)
        ds.attrs["title"] = "TEST INSAT GRANULE"
        ds.to_netcdf(path)
        return path

    def test_valid_insat_file_passes(self, tmp_path: Path) -> None:
        from app.ingestion.realtime.insat_netcdf import validate_insat_file

        nc_path = self._write_insat_nc(tmp_path / "valid.nc")
        report = validate_insat_file(nc_path, product_id="3D_IMG_L1B")
        assert report.ok is True
        assert "tir1_bt" in report.detected
        assert "wv_bt" in report.detected
        assert report.lat_range == (29.0, 31.0)
        assert report.lon_range == (77.5, 80.5)
        assert len(report.times) == 1

    def test_missing_file_fails_validation(self, tmp_path: Path) -> None:
        from app.ingestion.realtime.insat_netcdf import validate_insat_file

        missing = tmp_path / "absent.nc"
        report = validate_insat_file(missing)
        assert report.ok is False
        assert any("does not exist" in e for e in report.quality.errors)

    def test_unrecognised_variables_fail_validation(self, tmp_path: Path) -> None:
        import xarray as xr
        from app.ingestion.realtime.insat_netcdf import validate_insat_file

        ds = xr.Dataset(
            {"unrelated_var": (("lat", "lon"), np.ones((2, 2)))},
            coords={"lat": [29.0, 30.0], "lon": [78.0, 79.0], "time": ["2019-07-01T00:00:00"]},
        )
        nc_path = tmp_path / "unrecognised.nc"
        ds.to_netcdf(nc_path)

        report = validate_insat_file(nc_path)
        assert report.ok is False
        assert any("no recognised INSAT channel variable" in e for e in report.quality.errors)


    def test_radiance_encoded_unit_fails_validation(self, tmp_path: Path) -> None:
        from app.ingestion.realtime.insat_netcdf import validate_insat_file

        nc_path = self._write_insat_nc(
            tmp_path / "radiance.nc",
            units={"tir1_bt": "mW m-2 sr-1 (cm-1)-1"},
        )
        report = validate_insat_file(nc_path)
        assert report.ok is False
        assert any("spectral radiance" in e for e in report.quality.errors)

    def test_curvilinear_swath_grid_fails_validation(self, tmp_path: Path) -> None:
        from app.ingestion.realtime.insat_netcdf import validate_insat_file

        nc_path = self._write_insat_nc(tmp_path / "swath.nc", curvilinear=True)
        report = validate_insat_file(nc_path)
        assert report.ok is False
        assert any("curvilinear (2-D) latitude/longitude found" in e for e in report.quality.errors)

    def test_missing_geolocation_coordinates_fails_validation(self, tmp_path: Path) -> None:
        from app.ingestion.realtime.insat_netcdf import validate_insat_file

        nc_path = self._write_insat_nc(tmp_path / "no_coords.nc", no_coords=True)
        report = validate_insat_file(nc_path)
        assert report.ok is False
        assert any("no latitude/longitude coordinates" in e for e in report.quality.errors)

    def test_missing_time_fails_validation(self, tmp_path: Path) -> None:
        from app.ingestion.realtime.insat_netcdf import validate_insat_file

        nc_path = self._write_insat_nc(tmp_path / "no_time.nc", no_time=True)
        report = validate_insat_file(nc_path)
        assert report.ok is False
        assert any("no valid time" in e for e in report.quality.errors)

    def test_channel_derivations_and_math(self) -> None:
        import datetime as dt
        from app.ingestion.realtime.insat_netcdf import (
            add_derived_channels,
            derive_ctt,
            derive_ctt_cooling_rate,
            derive_wv_bt_anomaly,
        )

        # 1. derive_ctt
        tir1 = np.array([[[240.0, 150.0], [350.0, np.nan]]])  # 150 < 180, 350 > 340 (out of bounds)
        ctt = derive_ctt(tir1)
        assert ctt[0, 0, 0] == pytest.approx(240.0)
        assert np.isnan(ctt[0, 0, 1])  # out of physical range (180..340 K)
        assert np.isnan(ctt[0, 1, 0])  # out of physical range
        assert np.isnan(ctt[0, 1, 1])  # originally nan

        # 2. derive_ctt_cooling_rate
        # 3 frames, 30 min apart
        times = [
            dt.datetime(2019, 7, 1, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2019, 7, 1, 0, 30, tzinfo=dt.timezone.utc),
            dt.datetime(2019, 7, 1, 1, 0, tzinfo=dt.timezone.utc),
        ]
        ctt_series = np.array([
            [[250.0, 250.0]],  # t0
            [[248.0, 240.0]],  # t1: -2 K in 0.5 h -> -4 K/h (valid); -10 K in 0.5 h -> -20 K/h (out of range [-10, 10])
            [[251.0, 250.0]],  # t2: +3 K in 0.5 h -> +6 K/h (valid)
        ])
        rates, notes = derive_ctt_cooling_rate(ctt_series, times)
        assert np.isnan(rates[0, 0, 0])  # first frame is NaN
        assert rates[1, 0, 0] == pytest.approx(-4.0)
        assert np.isnan(rates[1, 0, 1])  # out of physical range [-10, 10] masked to NaN
        assert rates[2, 0, 0] == pytest.approx(6.0)

        # 3. derive_wv_bt_anomaly
        wv = np.array([
            [[230.0, 240.0], [250.0, np.nan]]  # median of [230, 240, 250] is 240.0
        ])
        anomaly = derive_wv_bt_anomaly(wv)
        assert anomaly[0, 0, 0] == pytest.approx(-10.0)
        assert anomaly[0, 0, 1] == pytest.approx(0.0)
        assert anomaly[0, 1, 0] == pytest.approx(10.0)
        assert np.isnan(anomaly[0, 1, 1])

        # 4. add_derived_channels in place
        channel_dict = {"tir1_bt": tir1, "wv_bt": wv}
        notes = add_derived_channels(channel_dict, times[:1])
        assert "ctt" in channel_dict
        assert "ctt_cooling_rate" in channel_dict
        assert "wv_bt_anomaly" in channel_dict
        assert len(notes) > 0

    def test_build_insat_cube_full_pipeline(self, tmp_path: Path, grid: GridSpec) -> None:
        from app.ingestion.realtime.insat_netcdf import build_insat_cube
        from app.ingestion.realtime.provenance import file_sha256

        nc_path = self._write_insat_nc(tmp_path / "insat_cube_source.nc")
        cube = build_insat_cube(nc_path, grid, product_id="3D_IMG_L1B")

        assert cube.grid == grid
        assert cube.n_frames == 1
        assert cube.channels.shape == (1, 12, *grid.shape)
        # IMDAA and DEM channels (iwv, cape, elevation) are NaN
        assert np.all(np.isnan(cube.channel("iwv")))
        assert np.all(np.isnan(cube.channel("cape")))
        assert np.all(np.isnan(cube.channel("elevation")))

        # Check provenance and manifest
        assert len(cube.provenance) == 1
        prov = cube.provenance[0]
        assert prov.is_synthetic is False
        assert prov.source_sha256 == file_sha256(nc_path)
        assert len(prov.source_files) == 1
        assert prov.source_files[0].sha256 == file_sha256(nc_path)
        assert prov.source_files[0].role == "insat_l1b"

