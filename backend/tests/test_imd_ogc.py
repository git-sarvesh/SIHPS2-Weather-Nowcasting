"""Tests for the Phase 6 live IMD GeoServer OGC acquisition path.

These are **offline and deterministic**: they use trimmed excerpts of real IMD
responses recorded in ``fixtures/imd_ogc_sample.json``. No network call is made
except in the explicitly marked ``@pytest.mark.live_source`` test, which is
skipped unless ``SIHPS_LIVE_SOURCE_TESTS=1``.

The behaviours pinned here are the ones that would otherwise produce a
plausible-looking but wrong dataset: unit errors, sentinel values, HTML error
pages returned with HTTP 200, and division by a zero accumulation window.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from app.ingestion.realtime.imd_ogc import (
    IMD_OGC_BASE_URL,
    MAX_PLAUSIBLE_RAIN_RATE_MMH,
    OGCFeatureSet,
    OGCResponseError,
    _coordinates,
    _parse_day,
    _parse_aws_features,
    _parse_synop_features,
    _parse_feature_collection,
    _to_float,
    fetch_layer,
    list_layers,
    parse_features,
)
from app.ingestion.realtime.imd_ogc import LayerSpec
from app.ingestion.realtime.readiness import REQUIRED_CHANNELS, assess_readiness

FIXTURE = Path(__file__).parent / "fixtures" / "imd_ogc_sample.json"


def _load() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _feature_set(layer: str, payload: dict) -> OGCFeatureSet:
    raw = json.dumps(payload).encode("utf-8")
    return OGCFeatureSet(
        layer=layer,
        features=payload["features"],
        raw=raw,
        url="https://example.invalid/ows",
        retrieved_at=datetime(2026, 9, 26, tzinfo=timezone.utc),
        content_type="application/json",
    )


class TestMissingAndSentinelValues:
    def test_none_and_null_become_nan_not_zero(self) -> None:
        for value in (None, "NULL", "null", "", "NA", "NaN", "-"):
            assert np.isnan(_to_float(value)), value

    def test_real_zero_survives(self) -> None:
        assert _to_float(0) == 0.0
        assert _to_float("0.0") == 0.0

    def test_weather_sentinels_become_nan(self) -> None:
        # -999 / -9999 are WMO "no data" markers.
        assert np.isnan(_to_float(-999))
        assert np.isnan(_to_float("-9999"))

    def test_non_numeric_becomes_nan(self) -> None:
        assert np.isnan(_to_float("light rain"))


class TestSynopParsing:
    def _records(self):
        return _parse_synop_features(_feature_set("imd:synop_data_layer", _load()["synop"]))

    def test_parses_both_stations(self) -> None:
        assert len(self._records()) == 2

    def test_timestamp_uses_the_real_utc_hour(self) -> None:
        record = self._records()[0]
        assert record.valid_time == datetime(2026, 9, 26, 12, tzinfo=timezone.utc)
        assert record.time_precision == "hour"

    def test_physical_units_are_preserved(self) -> None:
        record = self._records()[0]
        assert record.temperature_c == pytest.approx(21.4)
        assert record.relative_humidity_pct == pytest.approx(96.0)
        assert record.pressure_hpa == pytest.approx(1003.0)

    def test_rainfall_uses_the_shortest_published_window(self) -> None:
        record = self._records()[0]
        # 8 mm over 3 h = 2.667 mm/h, not over the 24 h window.
        assert record.rain_window_h == pytest.approx(3.0)
        assert record.rain_source_field == "3hrlyrain"
        assert record.rain_mm_per_h == pytest.approx(8.0 / 3.0)

    def test_null_pressure_stays_missing_not_zero(self) -> None:
        record = [r for r in self._records() if r.station_id == "42147"][0]
        assert np.isnan(record.pressure_hpa)
        # Its 3/6/12 h accumulations are all null, so the 24 h window is used
        # rather than a shorter one being invented.
        assert record.rain_window_h == pytest.approx(24.0)
        assert record.rain_source_field == "24hrlyrain"
        assert record.rain_mm_per_h == pytest.approx(71.0 / 24.0)

    def test_variable_summary_marks_observed_fields(self) -> None:
        summary = {e["variable"]: e for e in self._records()[0].variable_summary()}
        assert summary["air_temperature"]["observed"] is True
        assert summary["rainfall_rate"]["unit"] == "mm h-1"


class TestAwsParsing:
    """The AWS layer carried two real unit/sentinel traps."""

    def _records(self):
        return _parse_aws_features(_feature_set("imd:aws_data_layer", _load()["aws"]))

    def test_rain_sel_is_hours_not_minutes(self) -> None:
        # 26 mm over a 10 h accumulation = 2.6 mm/h. Reading `rain_sel` as
        # minutes produced 156 mm/h, which is the bug these tests pin.
        record = [r for r in self._records() if r.station_id == "A0B21F2A"][0]
        assert record.rain_window_h == pytest.approx(10.0)
        assert record.rain_mm_per_h == pytest.approx(2.6)

    def test_zero_window_is_rejected_not_divided_by(self) -> None:
        record = [r for r in self._records() if r.station_id == "ZERO_WINDOW"][0]
        assert np.isnan(record.rain_window_h)
        assert np.isnan(record.rain_mm_per_h)
        # The accumulation itself is still preserved.
        assert record.rain_amount_mm == pytest.approx(33.5)

    def test_null_window_leaves_the_rate_missing(self) -> None:
        record = [r for r in self._records() if r.station_id == "NULL_WINDOW"][0]
        assert np.isnan(record.rain_window_h)
        assert np.isnan(record.rain_mm_per_h)

    def test_implausible_rate_is_withheld(self) -> None:
        # 113 mm over 0.5 h = 226 mm/h is physically absurd; it is withheld.
        record = [r for r in self._records() if r.station_id == "IMPOSSIBLE"][0]
        assert np.isnan(record.rain_mm_per_h)
        assert record.rain_amount_mm == pytest.approx(113.0)

    def test_epoch_time_field_is_ignored(self) -> None:
        record = [r for r in self._records() if r.station_id == "A0B21F2A"][0]
        # The fixture's `time` is 1970-01-01; the day field must be used.
        assert record.valid_time == datetime(2026, 8, 21, tzinfo=timezone.utc)
        assert record.time_precision == "day"
        assert record.valid_time.year != 1970

    def test_sentinel_measurements_become_missing(self) -> None:
        record = [r for r in self._records() if r.station_id == "SENTINEL"][0]
        assert np.isnan(record.temperature_c)
        assert np.isnan(record.pressure_hpa)

    def test_plausibility_bound_is_documented(self) -> None:
        assert 0 < MAX_PLAUSIBLE_RAIN_RATE_MMH <= 400



class TestResponseValidation:
    """GeoServer returns errors as HTTP 200, so structure must be checked."""

    def test_html_error_page_is_rejected(self) -> None:
        with pytest.raises(OGCResponseError, match="HTML page"):
            _parse_feature_collection("<!DOCTYPE html><html>oops</html>", "imd:x")

    def test_ogc_exception_report_is_rejected(self) -> None:
        body = '<?xml version="1.0"?><ows:ExceptionReport><ows:Exception/></ows:ExceptionReport>'
        with pytest.raises(OGCResponseError, match="ExceptionReport"):
            _parse_feature_collection(body, "imd:x")

    def test_non_json_is_rejected(self) -> None:
        with pytest.raises(OGCResponseError, match="not valid JSON"):
            _parse_feature_collection("not json at all", "imd:x")

    def test_wrong_top_level_type_is_rejected(self) -> None:
        with pytest.raises(OGCResponseError, match="FeatureCollection"):
            _parse_feature_collection('{"type": "Feature"}', "imd:x")

    def test_features_not_a_list_is_rejected(self) -> None:
        with pytest.raises(OGCResponseError, match="not a list"):
            _parse_feature_collection('{"type":"FeatureCollection","features":{}}', "imd:x")

    def test_malformed_feature_is_rejected(self) -> None:
        body = '{"type":"FeatureCollection","features":[{"nope":1}]}'
        with pytest.raises(OGCResponseError, match="not a valid GeoJSON feature"):
            _parse_feature_collection(body, "imd:x")

    def test_empty_collection_is_valid_not_an_error(self) -> None:
        # An empty AOI is a real answer, not a failure.
        assert _parse_feature_collection('{"type":"FeatureCollection","features":[]}', "imd:x") == []


class TestGeometryAndDates:
    def test_coordinates_are_returned_as_lon_lat(self) -> None:
        feature = {"geometry": {"type": "Point", "coordinates": [77.63, 29.02]}}
        assert _coordinates(feature) == (77.63, 29.02)

    def test_non_point_geometry_is_rejected(self) -> None:
        with pytest.raises(OGCResponseError, match="Point"):
            _coordinates({"geometry": {"type": "LineString", "coordinates": []}})

    def test_day_field_parsing(self) -> None:
        assert _parse_day("2026-09-26Z") == datetime(2026, 9, 26, tzinfo=timezone.utc)
        assert _parse_day(None) is None
        assert _parse_day("nonsense") is None


class TestLayerSpec:
    def test_request_is_wfs_1_1_geojson(self) -> None:
        spec = LayerSpec("imd:synop_data_layer", "d", (77.5, 29.0, 80.5, 31.5))
        params = spec.request_params()
        assert params["service"] == "WFS"
        assert params["version"] == "1.1.0"
        assert params["outputFormat"] == "application/json"
        assert params["srsName"] == "EPSG:4326"
        assert params["bbox"] == "77.5,29.0,80.5,31.5,EPSG:4326"

    def test_endpoint_is_the_keyless_imd_geoserver(self) -> None:
        assert IMD_OGC_BASE_URL == "https://reactjs.imd.gov.in/geoserver/imd/ows"


class TestParserDispatch:
    def test_unknown_layer_raises_rather_than_guessing(self) -> None:
        fetched = _feature_set("imd:unknown_layer", {"type": "FeatureCollection", "features": []})
        with pytest.raises(OGCResponseError, match="no verified parser"):
            parse_features(fetched)

    def test_known_layers_dispatch(self) -> None:
        data = _load()
        assert parse_features(_feature_set("imd:synop_data_layer", data["synop"]))
        assert parse_features(_feature_set("imd:aws_data_layer", data["aws"]))



class TestDatasetReadiness:
    def test_station_data_cannot_train_the_model(self) -> None:
        report = assess_readiness(
            dataset_name="IMD station snapshot",
            is_synthetic=False,
            n_records=154,
            n_stations=154,
            available_channels=[],
            distinct_times=11,
        )
        assert report.can_train is False
        assert report.verdict == "partial"
        assert len(report.channels_missing) == len(REQUIRED_CHANNELS)

    def test_every_missing_channel_has_a_stated_reason(self) -> None:
        report = assess_readiness(
            dataset_name="x", is_synthetic=False, n_records=1, n_stations=1,
            available_channels=[], distinct_times=1,
        )
        for name in report.channels_missing:
            assert name in report.blockers
            assert report.blockers[name]

    def test_temporal_shortfall_is_explained(self) -> None:
        report = assess_readiness(
            dataset_name="x", is_synthetic=False, n_records=10, n_stations=10,
            available_channels=[], distinct_times=2, required_frames=6,
        )
        assert "__temporal__" in report.blockers
        assert "not a time series" in report.blockers["__temporal__"]

    def test_full_contract_would_be_ready(self) -> None:
        report = assess_readiness(
            dataset_name="full", is_synthetic=False, n_records=100, n_stations=5,
            available_channels=list(REQUIRED_CHANNELS), distinct_times=50,
        )
        assert report.verdict == "ready"
        assert report.can_train is True
        assert report.channels_missing == []

    def test_empty_dataset_is_blocked_not_partial(self) -> None:
        report = assess_readiness(
            dataset_name="empty", is_synthetic=False, n_records=0, n_stations=0,
            available_channels=[], distinct_times=0,
        )
        assert report.verdict == "blocked"
        assert report.can_train is False

    def test_synthetic_data_is_never_called_ready_for_skill(self) -> None:
        report = assess_readiness(
            dataset_name="synthetic", is_synthetic=True, n_records=100, n_stations=5,
            available_channels=list(REQUIRED_CHANNELS), distinct_times=50,
        )
        assert any("not forecasting skill" in n for n in report.notes)


@pytest.mark.live_source
def test_live_imd_geoserver_is_reachable() -> None:
    """LIVE-SOURCE TEST - opt in with SIHPS_LIVE_SOURCE_TESTS=1.

    Verifies the keyless endpoint still answers and that the observation layers
    this adapter depends on are still published. Skipped by default so the
    routine suite never depends on the network.
    """
    import os

    if os.getenv("SIHPS_LIVE_SOURCE_TESTS") != "1":
        pytest.skip("set SIHPS_LIVE_SOURCE_TESTS=1 to run live-source tests")
    layers = list_layers(timeout=60.0)
    assert layers, "IMD OGC GetCapabilities returned no layers"
    assert "imd:synop_data_layer" in layers
    assert "imd:aws_data_layer" in layers

