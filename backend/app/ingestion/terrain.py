"""Terrain processing: DEM loading, D8 hydrology, TWI and exposure (Part 1.3).

Supports SRTM (30 m) / CartoDEM GeoTIFFs through ``rasterio`` when available and
falling back to the synthetic DEM generator (``app.ingestion.synthetic``) in demo
mode. Heavy hydrological derivatives are computed with pure NumPy so the risk
engine works in any environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.grid import GridSpec, km_per_deg_lat, km_per_deg_lon
from app.ingestion.base import (
    DataConnector,
    Provenance,
    SourceFile,
    TerrainStack,
    source_file_from_disk,
    utcnow,
)
from app.logging_conf import get_logger
from app.nputils import gaussian_smooth

logger = get_logger("ingestion.terrain")

#: ESRI D8 flow-direction codes -> (row_delta, col_delta).
#: Rows increase northwards (row 0 is the southern domain edge), so northward
#: neighbours have ``+1`` row offsets.
D8_OFFSETS: dict[int, tuple[int, int]] = {
    1: (0, 1),     # E
    2: (-1, 1),    # SE
    4: (-1, 0),    # S
    8: (-1, -1),   # SW
    16: (0, -1),   # W
    32: (1, -1),   # NW
    64: (1, 0),    # N
    128: (1, 1),   # NE
}
#: D8 code -> distance weight in cell units.
D8_DISTANCE: dict[int, float] = {
    1: 1.0, 2: np.sqrt(2.0), 4: 1.0, 8: np.sqrt(2.0),
    16: 1.0, 32: np.sqrt(2.0), 64: 1.0, 128: np.sqrt(2.0),
}
#: D8 code -> 8-connected (dy, dx) index shifts for vectorised neighbour gathering.
D8_SHIFTS: dict[int, tuple[int, int]] = D8_OFFSETS

#: Runoff coefficient proxy by land-use class (higher => faster runoff => more flood risk).
LAND_USE_RUNOFF: dict[int, float] = {0: 0.30, 1: 0.20, 2: 0.45, 3: 0.70, 4: 0.85, 5: 0.95}

LAND_USE_LABELS: dict[int, str] = {
    0: "barren_rock", 1: "forest", 2: "grassland", 3: "cropland", 4: "built_up", 5: "water_body",
}


@dataclass(slots=True)
class DEMTile:
    """A DEM raster plus its georeferencing metadata."""

    elevation_m: np.ndarray
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    source: str = "SRTM"
    path: str | None = None


def slope_aspect(elevation_m: np.ndarray, dx_m: float, dy_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope [deg] and aspect [deg from north, clockwise] from a DEM.

    Uses Horn's 3x3 kernel (the same estimator GDAL uses for ``gdaldem slope``).
    """
    arr = np.asarray(elevation_m, dtype=np.float64)
    padded = np.pad(arr, 1, mode="edge")
    # Horn's kernel
    dzdx = (
        (padded[:-2, 2:] + 2.0 * padded[1:-1, 2:] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2.0 * padded[1:-1, :-2] + padded[2:, :-2])
    ) / (8.0 * max(dx_m, 1e-6))
    dzdy = (
        (padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2.0 * padded[:-2, 1:-1] + padded[:-2, 2:])
    ) / (8.0 * max(dy_m, 1e-6))
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = (np.degrees(np.arctan2(dzdy, -dzdx)) + 360.0) % 360.0
    return slope, aspect


def _shift(arr: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Return ``out`` with ``out[j, i] = arr[j + dr, i + dc]`` (NaN off-grid).

    ``dr``/``dc`` are row/column offsets in the D8 sense (row increases north).
    """
    ny, nx = arr.shape
    out = np.full((ny, nx), np.nan, dtype=np.float64)
    src_y = slice(max(0, dr), ny - max(0, -dr))
    dst_y = slice(max(0, -dr), ny - max(0, dr))
    src_x = slice(max(0, dc), nx - max(0, -dc))
    dst_x = slice(max(0, -dc), nx - max(0, dc))
    out[dst_y, dst_x] = arr[src_y, src_x]
    return out


def d8_flow_direction(elevation_m: np.ndarray, dx_m: float, dy_m: float) -> np.ndarray:
    """Steepest-descent D8 flow direction per cell (ESRI codes, 0 = pit/sink).

    Parameters
    ----------
    elevation_m:
        DEM elevation [m] on the model grid.
    dx_m, dy_m:
        Cell size [m] in the x (longitude) and y (latitude) directions.
    """
    arr = np.asarray(elevation_m, dtype=np.float64)
    ny, nx = arr.shape
    best_slope = np.full((ny, nx), -np.inf, dtype=np.float64)
    direction = np.zeros((ny, nx), dtype=np.int16)
    for code, (dr, dc) in D8_SHIFTS.items():
        dist = float(np.hypot(dc * dx_m, dr * dy_m)) or 1.0
        neighbour = _shift(arr, dr, dc)
        slope = (arr - neighbour) / dist
        slope = np.where(np.isfinite(slope), slope, -np.inf)
        # Strict descent: the priority-flood filled DEM guarantees every cell has
        # at least one strictly lower neighbour (or is a terminal boundary sink),
        # which keeps the D8 network acyclic. Ties are removed beforehand by the
        # deterministic micro-gradient applied to the hydrology DEM.
        mask = (slope > best_slope) & (slope > 0)
        if mask.any():
            best_slope[mask] = slope[mask]
            direction[mask] = code
    return direction


def _flat_neighbour_index(direction: np.ndarray, dr: int, dc: int) -> tuple[np.ndarray, np.ndarray]:
    """Flattened downstream index (clipped) plus an in-bounds mask."""
    ny, nx = direction.shape
    rows, cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    tgt_r = rows + dr
    tgt_c = cols + dc
    in_bounds = (tgt_r >= 0) & (tgt_r < ny) & (tgt_c >= 0) & (tgt_c < nx)
    idx = np.clip(tgt_r, 0, ny - 1) * nx + np.clip(tgt_c, 0, nx - 1)
    return idx, in_bounds


def _downstream_indices(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flattened ``(downstream_index, valid_mask)`` for every cell.

    Cells whose D8 path leaves the domain (or that sit in a pit, ``dir == 0``)
    are marked invalid so they terminate the flow network.
    """
    flat = np.zeros(direction.size, dtype=np.int64)
    valid = direction.ravel() > 0
    for code, (dr, dc) in D8_SHIFTS.items():
        idx, in_bounds = _flat_neighbour_index(direction, dr, dc)
        sel = (direction.ravel() == code) & in_bounds.ravel()
        flat[sel] = idx.ravel()[sel]
        valid &= ~((direction.ravel() == code) & ~in_bounds.ravel())
    return flat, valid & (direction.ravel() > 0)


def flow_accumulation_d8(elevation_m: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Upstream contributing cell count via D8 (each cell contributes 1).

    Uses Kahn's topological ordering of the flow graph, seeded in descending
    elevation order for determinism. Cells caught in residual flat-area loops
    (possible on pure flats) are drained once after the main pass, and their
    count is reported by the caller through the returned array only - i.e. the
    algorithm always terminates with a finite accumulation.
    """
    from collections import deque

    elev = np.asarray(elevation_m, dtype=np.float64).ravel()
    downstream, valid = _downstream_indices(direction)
    n = direction.size
    in_degree = np.zeros(n, dtype=np.int64)
    np.add.at(in_degree, downstream[valid], 1)

    acc = np.ones(n, dtype=np.float64)
    processed = np.zeros(n, dtype=bool)
    seeds = np.flatnonzero(in_degree == 0)
    queue: deque[int] = deque(int(c) for c in seeds[np.argsort(-elev[seeds])])
    while queue:
        cell = queue.popleft()
        processed[cell] = True
        if not valid[cell]:
            continue
        target = int(downstream[cell])
        acc[target] += acc[cell]
        in_degree[target] -= 1
        if in_degree[target] == 0 and not processed[target]:
            queue.append(target)

    leftover = np.flatnonzero(~processed)
    for cell in leftover:
        if valid[cell]:
            acc[int(downstream[cell])] += acc[cell]
    return acc.reshape(direction.shape)


def drainage_basins(direction: np.ndarray) -> np.ndarray:
    """Drainage basin id per cell from the D8 network (union-find on flow paths)."""
    n = direction.size
    parent = np.arange(n, dtype=np.int64)

    def find(idx: int) -> int:
        root = idx
        while parent[root] != root:
            root = parent[root]
        while parent[idx] != root:  # path compression
            parent[idx], idx = root, parent[idx]
        return root

    downstream, valid = _downstream_indices(direction)
    for cell in np.flatnonzero(valid):
        ra, rb = find(int(cell)), find(int(downstream[cell]))
        if ra != rb:
            parent[rb] = ra

    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    _, basin = np.unique(roots, return_inverse=True)
    return basin.reshape(direction.shape)


def topographic_wetness_index(flow_accumulation: np.ndarray, slope_deg: np.ndarray) -> np.ndarray:
    """Topographic Wetness Index ``ln(a / tan(beta))`` with a 1-cell minimum slope."""
    sca = np.maximum(np.asarray(flow_accumulation, dtype=np.float64), 1.0)
    beta = np.radians(np.clip(np.asarray(slope_deg, dtype=np.float64), 0.5, 89.9))
    return np.log(sca / np.tan(beta))


def land_use_proxy(elevation_m: np.ndarray, slope_deg: np.ndarray, flow_accumulation: np.ndarray) -> np.ndarray:
    """Land-cover proxy class (0-5) when no official LULC raster is supplied.

    Classes: 0 barren/rock, 1 forest, 2 grassland, 3 cropland, 4 built-up,
    5 water body. The proxy follows elevation/slope/valley position, which is the
    dominant control on Himalayan land cover, and is explicitly flagged as a
    proxy in the API payloads so it is never mistaken for measured LULC.
    """
    elev = np.asarray(elevation_m, dtype=np.float64)
    slope = np.asarray(slope_deg, dtype=np.float64)
    acc = np.asarray(flow_accumulation, dtype=np.float64)
    high_water = acc >= np.percentile(acc, 99.0)
    classes = np.full(elev.shape, 2, dtype=np.int64)                    # grassland
    classes[(elev > 3000) | (slope > 35)] = 0                           # barren / rock
    classes[(elev > 1800) & (elev <= 3000) & (slope <= 35)] = 1          # forest
    classes[(elev > 400) & (elev <= 1800) & (slope <= 20)] = 3           # cropland
    classes[(elev <= 400) & (slope <= 8) & (acc >= np.percentile(acc, 80))] = 4  # built-up
    classes[high_water] = 5                                             # water body
    return classes


def fill_depressions(elevation_m: np.ndarray, *, epsilon: float = 1e-3, max_iterations: int = 4) -> np.ndarray:
    """Fill closed depressions with the priority-flood algorithm (Barnes et al. 2014).

    A DEM free of pits guarantees that every cell has a downslope path to the
    domain edge, which is a precondition for a meaningful D8 network and flow
    accumulation. Pits are raised to their lowest spill elevation (+epsilon).

    Parameters
    ----------
    elevation_m:
        Input DEM [m].
    epsilon:
        Small increment applied while flooding to guarantee drainage.
    max_iterations:
        Safety cap on heap iterations per cell (guards against pathological input).
    """
    import heapq

    elev = np.asarray(elevation_m, dtype=np.float64)
    ny, nx = elev.shape
    filled = np.full((ny, nx), np.inf, dtype=np.float64)
    heap: list[tuple[float, int, int]] = []
    for col in range(nx):
        heapq.heappush(heap, (float(elev[0, col]), 0, col))
        heapq.heappush(heap, (float(elev[ny - 1, col]), ny - 1, col))
    for row in range(ny):
        heapq.heappush(heap, (float(elev[row, 0]), row, 0))
        heapq.heappush(heap, (float(elev[row, nx - 1]), row, nx - 1))

    iterations = 0
    limit = ny * nx * max_iterations
    while heap and iterations < limit:
        level, row, col = heapq.heappop(heap)
        iterations += 1
        if level >= filled[row, col]:
            continue
        filled[row, col] = level
        for dr, dc in D8_OFFSETS.values():
            r2, c2 = row + dr, col + dc
            if 0 <= r2 < ny and 0 <= c2 < nx and not np.isfinite(filled[r2, c2]):
                heapq.heappush(heap, (max(level + epsilon, float(elev[r2, c2])), r2, c2))
    return np.where(np.isfinite(filled), filled, elev)


def river_network_mask(flow_accumulation: np.ndarray, percentile: float = 98.0) -> np.ndarray:
    """Boolean mask of the channel network (highest flow-accumulation cells)."""
    acc = np.asarray(flow_accumulation, dtype=np.float64)
    return acc >= np.percentile(acc, percentile)


def load_dem_geotiff(path: str | Path, bbox: tuple[float, float, float, float] | None = None) -> DEMTile:
    """Read a GeoTIFF DEM with ``rasterio`` (SRTM / CartoDEM).

    Parameters
    ----------
    path:
        GeoTIFF path.
    bbox:
        Optional ``(min_lon, min_lat, max_lon, max_lat)`` window to read.

    Raises
    ------
    RuntimeError
        When ``rasterio`` is unavailable - install ``backend/requirements-torch.txt``
        or run with ``SIHPS_DEMO_MODE=true`` to use the synthetic DEM.
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "rasterio is required to read GeoTIFF DEMs; install backend/requirements-torch.txt "
            "or run with SIHPS_DEMO_MODE=true"
        ) from exc

    with rasterio.open(path) as dataset:
        if bbox is not None:
            window = from_bounds(*bbox, transform=dataset.transform)
            elevation = dataset.read(1, window=window, boundless=True, fill_value=np.nan).astype(np.float64)
            transform = dataset.window_transform(window)
        else:
            elevation = dataset.read(1).astype(np.float64)
            transform = dataset.transform
        if dataset.nodata is not None:
            elevation = np.where(elevation == dataset.nodata, np.nan, elevation)
        left, top = transform * (0, 0)
        right, bottom = transform * (elevation.shape[1], elevation.shape[0])
        return DEMTile(
            elevation_m=elevation,
            min_lon=float(min(left, right)),
            min_lat=float(min(top, bottom)),
            max_lon=float(max(left, right)),
            max_lat=float(max(top, bottom)),
            source=Path(path).stem,
            path=str(path),
        )


#: Provider recorded on every operator-supplied DEM.
DEM_PROVIDER = "NASA/USGS (SRTM) / ISRO NRSC (CartoDEM)"

#: Attribution text reproduced on every DEM-derived artefact.
DEM_ATTRIBUTION = (
    "SRTM (NASA/USGS, 30 m) / CartoDEM (ISRO NRSC) elevation; "
    "derivatives (slope, D8 flow accumulation, TWI, basins) computed by SIHPS."
)


def dem_coverage_string(tile: DEMTile) -> str:
    """``"minLon-maxLonE minLat-maxLatN"`` from the tile's own georeferencing.

    Read from the raster transform, not from the model grid, so it states what
    the DEM actually covers rather than where it was placed.
    """
    return f"{tile.min_lon}-{tile.max_lon}E {tile.min_lat}-{tile.max_lat}N"


def dem_source_file(path: str | Path) -> SourceFile:
    """Phase 8.4 manifest entry for a DEM raster, hashed from its own bytes.

    The digest comes from the file, so a DEM that cannot be re-read records no
    hash at all - the entry stays *unknown*, which every observational check
    treats as a failure rather than a pass.
    """
    source = Path(path)
    return source_file_from_disk(
        source, provider=DEM_PROVIDER, role="dem", source_identity=source.stem
    )


def dem_provenance(tile: DEMTile, *, acquired_at: datetime | None = None) -> Provenance:
    """Provenance for a DEM read from a real raster.

    Records the SHA-256 of the raster's **actual bytes**, the file that digest
    came from, the provider, the tile's own spatial coverage and the physical
    unit of the one model channel a DEM supplies (``elevation``, metres). With no
    readable file the digest is ``None`` and the record states *unknown*: it never
    claims verification it cannot support.
    """
    path = Path(tile.path) if tile.path else None
    entry = dem_source_file(path) if path is not None else None
    if acquired_at is not None:
        stamp = acquired_at
    elif path is not None and path.is_file():
        stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    else:
        stamp = utcnow()
    return Provenance(
        source=f"{DEM_PROVIDER} DEM raster",
        product=f"elevation:{tile.source}",
        valid_from=stamp,
        fetched_at=stamp,
        path=str(path) if path else None,
        is_synthetic=False,
        attribution=DEM_ATTRIBUTION,
        source_sha256=entry.sha256 if entry else None,
        source_path=str(path) if path else None,
        source_bytes_available=bool(entry and entry.bytes_available),
        provider=DEM_PROVIDER,
        acquired_at=stamp,
        channel_units={"elevation": "m"},
        coverage=dem_coverage_string(tile),
        source_files=[entry] if entry else [],
    )



class TerrainProcessor(DataConnector):
    """Builds the :class:`TerrainStack` used by the flood head and risk engine."""

    source_name = "SRTM/CartoDEM"
    is_synthetic = True

    #: Attribution shown in UI/API payloads.
    ATTRIBUTION = (
        "SRTM (NASA/USGS, 30 m) / CartoDEM (ISRO NRSC) elevation; "
        "derivatives (slope, D8 flow accumulation, TWI, basins) computed by SIHPS."
    )

    def dem_from_geotiff(self, path: str | Path) -> DEMTile:
        """Load and regrid a real DEM GeoTIFF onto the model grid."""
        tile = load_dem_geotiff(
            path, bbox=(self.grid.min_lon, self.grid.min_lat, self.grid.max_lon, self.grid.max_lat)
        )
        self.is_synthetic = False
        return self._regrid_dem(tile)

    def _regrid_dem(self, tile: DEMTile) -> DEMTile:
        """Nearest-neighbour regrid of a DEM tile onto the model grid (NumPy)."""
        src = GridSpec.from_bbox(
            (tile.min_lon, tile.min_lat, tile.max_lon, tile.max_lat),
            1.0,
            shape=tile.elevation_m.shape,
        )
        resampled = self.grid.resample_nearest(src, np.nan_to_num(tile.elevation_m, nan=0.0))
        return DEMTile(
            elevation_m=resampled,
            min_lon=self.grid.min_lon,
            min_lat=self.grid.min_lat,
            max_lon=self.grid.max_lon,
            max_lat=self.grid.max_lat,
            source=tile.source,
            path=tile.path,
        )

    def build(self, dem: DEMTile, *, provenance: Provenance | None = None) -> TerrainStack:
        """Compute the full terrain stack (slope, aspect, D8 network, TWI, land use)."""
        elevation = np.asarray(dem.elevation_m, dtype=np.float64)
        if elevation.shape != self.grid.shape:
            raise ValueError(f"DEM shape {elevation.shape} != grid {self.grid.shape}")
        lat_mid = 0.5 * (self.grid.min_lat + self.grid.max_lat)
        dx_m = self.grid.d_lon * km_per_deg_lon(lat_mid) * 1000.0
        dy_m = self.grid.d_lat * km_per_deg_lat() * 1000.0
        smooth = gaussian_smooth(elevation, sigma=0.8)
        slope_deg, aspect_deg = slope_aspect(smooth, dx_m, dy_m)
        hydrology_dem = fill_depressions(smooth)
        # Deterministic micro-gradient (sub-millimetre) that removes exact ties on
        # filled flats, guaranteeing an acyclic D8 network.
        rows = np.arange(hydrology_dem.shape[0], dtype=np.float64)[:, None]
        cols = np.arange(hydrology_dem.shape[1], dtype=np.float64)[None, :]
        hydrology_dem = hydrology_dem + 1e-4 * (rows + 1.37 * cols)
        direction = d8_flow_direction(hydrology_dem, dx_m, dy_m)
        accumulation = flow_accumulation_d8(hydrology_dem, direction)
        basins = drainage_basins(direction)
        twi = topographic_wetness_index(accumulation, slope_deg)
        land_use = land_use_proxy(elevation, slope_deg, accumulation)
        logger.info(
            "terrain stack built",
            extra={
                "shape": str(elevation.shape),
                "n_basins": int(basins.max()) + 1,
                "max_flow_accumulation": float(accumulation.max()),
            },
        )
        prov = provenance
        if prov is None and dem.path:
            prov = dem_provenance(dem)
        return TerrainStack(
            grid=self.grid,
            elevation_m=elevation,
            slope_deg=slope_deg,
            aspect_deg=aspect_deg,
            flow_accumulation=accumulation,
            drainage_basin=basins,
            twi=twi,
            land_use=land_use,
            provenance=prov
            or Provenance(
                source=self.source_name,
                product="terrain_stack",
                valid_from=utcnow(),
                is_synthetic=self.is_synthetic,
                attribution=self.ATTRIBUTION,
            ),
        )

    def exposure_score(
        self,
        terrain: TerrainStack,
        *,
        weights: tuple[float, float, float, float] = (0.35, 0.25, 0.30, 0.10),
        smooth_sigma: float = 1.0,
    ) -> np.ndarray:
        """Terrain exposure in ``[0, 1]`` used by the flood risk engine (Part 3.1).

        Weight order: low elevation, steep *upstream* slope, high flow
        accumulation, land-use runoff coefficient. High values mark catchments
        where intense rainfall converts to flash flooding fastest.
        """
        w_elev, w_slope, w_flow, w_land = weights
        elev = np.asarray(terrain.elevation_m, dtype=np.float64)
        lo, hi = np.percentile(elev, [5, 95])
        elev_term = 1.0 - np.clip((elev - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

        slope = np.asarray(terrain.slope_deg, dtype=np.float64)
        padded = np.pad(slope, 2, mode="edge")
        upstream = np.maximum.reduce(
            [padded[dy : dy + slope.shape[0], dx : dx + slope.shape[1]] for dy in range(5) for dx in range(5)]
        )
        slope_term = np.clip(upstream / 45.0, 0.0, 1.0)

        flow_term = terrain.normalised_flow
        land_term = np.clip(
            np.vectorize(lambda value: LAND_USE_RUNOFF.get(int(value), 0.3))(terrain.land_use), 0.0, 1.0
        )

        exposure = w_elev * elev_term + w_slope * slope_term + w_flow * flow_term + w_land * land_term
        return gaussian_smooth(np.clip(exposure, 0.0, 1.0), sigma=smooth_sigma)

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["attribution"] = self.ATTRIBUTION
        return payload

