"""Common spatio-temporal grid definitions.

The whole system (ingestion, model tensors, risk cells, GeoJSON export) shares a
single :class:`GridSpec`: a regular lat/lon raster anchored on the AOI bounding
box with a metric resolution of 1-3 km and a 30 minute temporal cadence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sequence

#: Mean earth radius in km (spherical approximation is adequate at 1-3 km).
EARTH_RADIUS_KM = 6371.0088

#: Temporal cadence of the aligned cube (minutes).
FRAME_MINUTES = 30


def km_per_deg_lat() -> float:
    """Length of one degree of latitude in km."""
    return math.pi * EARTH_RADIUS_KM / 180.0


def km_per_deg_lon(lat_deg: float) -> float:
    """Length of one degree of longitude at ``lat_deg`` in km."""
    return math.pi * EARTH_RADIUS_KM * math.cos(math.radians(lat_deg)) / 180.0


@dataclass(frozen=True, slots=True)
class GridSpec:
    """Regular lat/lon raster covering ``[min_lon, max_lon) x [min_lat, max_lat)``."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    res_km: float
    nx: int
    ny: int

    # ------------------------------------------------------------------ ctors
    @classmethod
    def from_bbox(
        cls,
        bbox: Sequence[float],
        res_km: float = 2.0,
        *,
        shape: tuple[int, int] | None = None,
    ) -> "GridSpec":
        """Build a grid from ``(min_lon, min_lat, max_lon, max_lat)``."""
        min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)
        if not (min_lon < max_lon and min_lat < max_lat):
            raise ValueError(f"invalid bbox {bbox!r}")
        lat_mid = 0.5 * (min_lat + max_lat)
        deg_per_km_lon = 1.0 / km_per_deg_lon(lat_mid)
        deg_per_km_lat = 1.0 / km_per_deg_lat()
        if shape is not None:
            ny, nx = int(shape[0]), int(shape[1])
        else:
            nx = max(8, int(round((max_lon - min_lon) / (res_km * deg_per_km_lon))))
            ny = max(8, int(round((max_lat - min_lat) / (res_km * deg_per_km_lat))))
        return cls(
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
            res_km=res_km,
            nx=nx,
            ny=ny,
        )

    @classmethod
    def from_dict(cls, payload: dict) -> "GridSpec":
        """Rebuild a grid from :meth:`to_dict` output."""
        return cls(
            min_lon=payload["min_lon"],
            min_lat=payload["min_lat"],
            max_lon=payload["max_lon"],
            max_lat=payload["max_lat"],
            res_km=payload["res_km"],
            nx=payload["nx"],
            ny=payload["ny"],
        )

    # ------------------------------------------------------------- geometry
    @property
    def shape(self) -> tuple[int, int]:
        """``(ny, nx)`` raster shape."""
        return self.ny, self.nx

    @property
    def size(self) -> int:
        """Total number of cells."""
        return self.nx * self.ny

    @property
    def d_lon(self) -> float:
        return (self.max_lon - self.min_lon) / self.nx

    @property
    def d_lat(self) -> float:
        return (self.max_lat - self.min_lat) / self.ny

    @property
    def effective_res_km(self) -> tuple[float, float]:
        """Actual ``(x_km, y_km)`` cell size after snapping to whole cells."""
        lat_mid = 0.5 * (self.min_lat + self.max_lat)
        return self.d_lon * km_per_deg_lon(lat_mid), self.d_lat * km_per_deg_lat()

    def lon_centers(self) -> list[float]:
        return [self.min_lon + (i + 0.5) * self.d_lon for i in range(self.nx)]

    def lat_centers(self) -> list[float]:
        """Latitude centres, south -> north (row 0 is the southern edge)."""
        return [self.min_lat + (j + 0.5) * self.d_lat for j in range(self.ny)]

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        """Return ``(lat, lon)`` of a cell centre."""
        return self.min_lat + (row + 0.5) * self.d_lat, self.min_lon + (col + 0.5) * self.d_lon

    def index_of(self, lat: float, lon: float) -> tuple[int, int] | None:
        """Map ``(lat, lon)`` -> ``(row, col)``; ``None`` when outside the AOI."""
        if not (self.min_lon <= lon < self.max_lon and self.min_lat <= lat < self.max_lat):
            return None
        col = min(self.nx - 1, max(0, int((lon - self.min_lon) / self.d_lon)))
        row = min(self.ny - 1, max(0, int((lat - self.min_lat) / self.d_lat)))
        return row, col

    def cell_polygon(self, row: int, col: int) -> list[list[float]]:
        """GeoJSON ring (closed) ``[[lon, lat], ...]`` for a raster cell."""
        x0 = self.min_lon + col * self.d_lon
        y0 = self.min_lat + row * self.d_lat
        x1, y1 = x0 + self.d_lon, y0 + self.d_lat
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]

    def iter_cells(self) -> Iterator[tuple[int, int]]:
        """Iterate ``(row, col)`` in row-major order."""
        for row in range(self.ny):
            for col in range(self.nx):
                yield row, col

    def contains(self, lat: float, lon: float) -> bool:
        return self.min_lon <= lon <= self.max_lon and self.min_lat <= lat <= self.max_lat

    def resample_nearest(self, src: "GridSpec", src_field):
        """Nearest-neighbour resampling of a 2-D array from ``src`` onto this grid."""
        import numpy as np

        arr = np.asarray(src_field)
        if arr.shape != src.shape:
            raise ValueError(f"src_field shape {arr.shape} != src shape {src.shape}")
        rows = np.clip(((np.asarray(self.lat_centers()) - src.min_lat) / src.d_lat).astype(int), 0, src.ny - 1)
        cols = np.clip(((np.asarray(self.lon_centers()) - src.min_lon) / src.d_lon).astype(int), 0, src.nx - 1)
        return arr[np.ix_(rows, cols)]

    def resample_bilinear(self, src: "GridSpec", src_field):
        """Bilinear resampling of a 2-D array from ``src`` onto this grid.

        This is the operator used to align 0.12 deg IMDAA reanalysis fields onto
        the 1-3 km satellite grid (Part 1.2 of the specification). Pure NumPy so
        the alignment path has no SciPy/GDAL hard requirement.
        """
        import numpy as np

        arr = np.asarray(src_field, dtype=np.float64)
        if arr.shape != src.shape:
            raise ValueError(f"src_field shape {arr.shape} != src shape {src.shape}")
        x = np.clip((np.asarray(self.lon_centers()) - src.min_lon) / src.d_lon - 0.5, 0, src.nx - 1 - 1e-9)
        y = np.clip((np.asarray(self.lat_centers()) - src.min_lat) / src.d_lat - 0.5, 0, src.ny - 1 - 1e-9)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, src.nx - 1), np.minimum(y0 + 1, src.ny - 1)
        wx, wy = x - x0, y - y0
        out = np.zeros(self.shape, dtype=np.float64)
        for r in range(self.ny):
            top = arr[y0[r], x0] * (1.0 - wx) + arr[y0[r], x1] * wx
            bot = arr[y1[r], x0] * (1.0 - wx) + arr[y1[r], x1] * wx
            out[r, :] = top * (1.0 - wy[r]) + bot * wy[r]
        return out

    def to_dict(self) -> dict:
        """Serialisable description (used by ``/health`` and XAI metadata)."""
        x_km, y_km = self.effective_res_km
        return {
            "min_lon": self.min_lon,
            "min_lat": self.min_lat,
            "max_lon": self.max_lon,
            "max_lat": self.max_lat,
            "nx": self.nx,
            "ny": self.ny,
            "res_km": self.res_km,
            "effective_res_km": {"x": round(x_km, 4), "y": round(y_km, 4)},
            "crs": "EPSG:4326",
            "frame_minutes": FRAME_MINUTES,
        }


def utc_floor_to_cadence(dt, minutes: int = FRAME_MINUTES):
    """Floor a datetime to the nearest ``minutes`` boundary."""
    dt = dt.replace(second=0, microsecond=0)
    return dt.replace(minute=(dt.minute // minutes) * minutes)
