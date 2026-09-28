"""Risk engine: terrain-aware flood risk, compound fusion and categorisation (Part 3).

The engine turns raw model probabilities into actionable, terrain-aware risk:

* ``terrain_aware_flood_risk``: ``risk = flood_prob x exposure(elevation, slope,
  flow accumulation, land use)`` - a differentiable product, so gradients can flow
  from risk back to features (unified differentiable pipeline).
* ``compound_risk``: joint probability of the thunderstorm-and-cloudburst cascade,
  either under an independence assumption or with a Gaussian copula
  (SciPy-free implementation based on the bivariate normal CDF).
* ``overall_risk``: configurable weighted fusion, default ``(0.3, 0.4, 0.3)``.
* ``categorise``: LOW / MODERATE / HIGH / EXTREME thresholds.
* ``to_geojson``: risk cells / vectorised polygons for the dashboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from app.grid import GridSpec
from app.ingestion.base import TerrainStack
from app.logging_conf import get_logger
from app.nputils import gaussian_smooth

logger = get_logger("services.risk_engine")

#: Risk categories in ascending severity.
RISK_CATEGORIES: tuple[str, ...] = ("LOW", "MODERATE", "HIGH", "EXTREME")

__all__ = [
    "RISK_CATEGORIES",
    "RiskWeights",
    "RiskResult",
    "RiskEngine",
    "categorise",
    "compound_probability",
    "bivariate_normal_cdf",
]


def normal_cdf(z, *, vectorize: bool = True):
    """Standard normal CDF (NumPy ``erf`` based, SciPy-free)."""
    arr = np.asarray(z, dtype=np.float64)
    out = 0.5 * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0))) if vectorize else 0.5 * (
        1.0 + math.erf(float(arr) / math.sqrt(2.0))
    )
    return out


def normal_ppf(p) -> np.ndarray:
    """Inverse standard normal CDF via Acklam's rational approximation.

    Accurate to about 1e-9 over the whole open interval (0, 1), which is more than
    enough to build the copula marginals.
    """
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    plow, phigh = 0.02425, 1.0 - 0.02425
    out = np.zeros_like(p)
    lower = p < plow
    upper = p > phigh
    middle = ~(lower | upper)
    q = np.sqrt(-2.0 * np.log(p[lower]))
    out[lower] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )
    q = np.sqrt(-2.0 * np.log(1.0 - p[upper]))
    out[upper] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )
    q = p[middle] - 0.5
    r = q * q
    out[middle] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )
    return out


def bivariate_normal_cdf(h, k, rho):
    """Vectorised standard bivariate normal CDF ``Phi(h, k; rho)`` (SciPy-free).

    Uses the identity ``Phi2(h,k;rho) = Phi(h)Phi(k) + int_0^rho phi2(h,k;t) dt``
    with 24-point Gauss-Legendre quadrature, which is smooth in ``t`` on (-1, 1).
    """
    h = np.asarray(h, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    rho = np.clip(np.asarray(rho, dtype=np.float64), -0.999, 0.999)
    if rho.ndim > 1:
        rho_flat = rho
    else:
        rho_flat = rho
    nodes, weights = np.polynomial.legendre.leggauss(24)
    t = 0.5 * (nodes[None, ...] + 1.0) * rho_flat[..., None]  # (..., n)
    w = 0.5 * rho_flat[..., None] * weights[None, ...]
    one_minus = np.clip(1.0 - t**2, 1e-9, None)
    exponent = -(h[..., None] ** 2 - 2.0 * t * h[..., None] * k[..., None] + k[..., None] ** 2) / (
        2.0 * one_minus
    )
    density = np.exp(exponent) / (2.0 * math.pi * np.sqrt(one_minus))
    integral = np.sum(w * density, axis=-1)
    return np.clip(normal_cdf(h) * normal_cdf(k) + integral, 0.0, 1.0)


def compound_probability(p_a, p_b, *, rho: float = 0.6, use_copula: bool = True) -> np.ndarray:
    """Joint probability ``P(A and B)`` for two hazard probability fields.

    With ``use_copula=True`` a Gaussian copula with correlation ``rho`` models the
    physical dependence between hazards (thunderstorms and cloudbursts co-occur);
    otherwise the hazards are treated as independent.
    """
    a = np.clip(np.asarray(p_a, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    b = np.clip(np.asarray(p_b, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if not use_copula or abs(rho) < 1e-6:
        return a * b
    joint = bivariate_normal_cdf(normal_ppf(a), normal_ppf(b), np.full_like(a, float(rho)))
    # Copula output already includes the marginals; clip for numerical safety.
    return np.clip(joint, 0.0, np.minimum(a, b) * 1.0)


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class RiskWeights:
    """Configurable weights of the compound risk fusion."""

    thunderstorm: float = 0.3
    cloudburst: float = 0.4
    flood: float = 0.3

    def normalised(self) -> tuple[float, float, float]:
        total = self.thunderstorm + self.cloudburst + self.flood
        if total <= 0:
            raise ValueError("risk weights must be positive")
        return self.thunderstorm / total, self.cloudburst / total, self.flood / total


def categorise(risk: np.ndarray, thresholds: Sequence[float] = (0.3, 0.6, 0.85)) -> np.ndarray:
    """Map continuous risk to category codes ``0..3`` (LOW..EXTREME)."""
    low, moderate, high = (float(t) for t in thresholds)
    if not (0.0 < low < moderate < high < 1.0):
        raise ValueError(f"thresholds must be ascending in (0, 1), got {thresholds}")
    arr = np.asarray(risk, dtype=np.float64)
    codes = np.zeros(arr.shape, dtype=np.int8)
    codes[arr >= low] = 1
    codes[arr >= moderate] = 2
    codes[arr >= high] = 3
    return codes


@dataclass(slots=True)
class RiskResult:
    """Terrain-aware, compound risk fields for one lead time (or one run)."""

    grid: GridSpec
    init_time: Any
    lead_hours: list[float]
    thunderstorm: np.ndarray
    cloudburst: np.ndarray
    flood_probability: np.ndarray
    flood_risk: np.ndarray
    exposure: np.ndarray
    compound_storm_cloudburst: np.ndarray
    overall: np.ndarray
    category: np.ndarray
    thresholds: tuple[float, float, float] = (0.3, 0.6, 0.85)
    weights: RiskWeights = field(default_factory=RiskWeights)
    uncertainty: dict[str, np.ndarray] = field(default_factory=dict)
    confidence_intervals: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    model_version: str = "unknown"
    disclaimer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def category_names(self) -> np.ndarray:
        """Category names as an object array of strings."""
        names = np.asarray(RISK_CATEGORIES, dtype=object)
        return names[self.category]

    def area_fraction(self) -> dict[str, float]:
        """Fraction of the AOI in each risk category."""
        counts = np.bincount(self.category.ravel(), minlength=len(RISK_CATEGORIES))
        return {
            name: float(counts[index]) / float(self.category.size)
            for index, name in enumerate(RISK_CATEGORIES)
        }

    def hotspots(self, *, top_k: int = 5, min_risk: float = 0.6) -> list[dict[str, Any]]:
        """Highest-risk cells with their lat/lon and per-hazard breakdown."""
        mask = self.overall >= min_risk
        if not mask.any():
            mask = self.overall >= float(np.percentile(self.overall, 99.0))
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            return []
        scores = self.overall[rows, cols]
        order = np.argsort(-scores)[: max(1, int(top_k))]
        hotspots: list[dict[str, Any]] = []
        for index in order:
            row, col = int(rows[index]), int(cols[index])
            lat, lon = self.grid.cell_center(row, col)
            hotspots.append(
                {
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "row": row,
                    "col": col,
                    "overall_risk": round(float(self.overall[row, col]), 4),
                    "category": RISK_CATEGORIES[int(self.category[row, col])],
                    "thunderstorm": round(float(self.thunderstorm[row, col]), 4),
                    "cloudburst": round(float(self.cloudburst[row, col]), 4),
                    "flood": round(float(self.flood_risk[row, col]), 4),
                    "exposure": round(float(self.exposure[row, col]), 4),
                }
            )
        return hotspots

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable summary for the API/dashboard."""
        return {
            "init_time": self.init_time.isoformat() if hasattr(self.init_time, "isoformat") else str(self.init_time),
            "lead_hours": list(self.lead_hours),
            "model_version": self.model_version,
            "disclaimer": self.disclaimer,
            "weights": {
                "thunderstorm": self.weights.thunderstorm,
                "cloudburst": self.weights.cloudburst,
                "flood": self.weights.flood,
            },
            "thresholds": list(self.thresholds),
            "max_overall_risk": float(self.overall.max()),
            "mean_overall_risk": float(self.overall.mean()),
            "area_fraction": self.area_fraction(),
            "max_by_hazard": {
                "thunderstorm": float(self.thunderstorm.max()),
                "cloudburst": float(self.cloudburst.max()),
                "flood": float(self.flood_risk.max()),
                "flood_probability": float(self.flood_probability.max()),
                "compound": float(self.compound_storm_cloudburst.max()),
            },
            "hotspots": self.hotspots(top_k=5),
            "uncertainty": {
                key: {"mean_std": float(np.mean(value)), "max_std": float(np.max(value))}
                for key, value in self.uncertainty.items()
            },
            "metadata": self.metadata,
        }


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class RiskEngine:
    """Turns model probabilities into terrain-aware, compound, categorised risk."""

    def __init__(
        self,
        terrain: TerrainStack,
        *,
        exposure: np.ndarray | None = None,
        weights: RiskWeights | tuple[float, float, float] = RiskWeights(),
        thresholds: tuple[float, float, float] = (0.3, 0.6, 0.85),
        exposure_weights: tuple[float, float, float, float] = (0.35, 0.25, 0.30, 0.10),
        copula_rho: float = 0.6,
        use_copula: bool = True,
        disclaimer: str = "",
    ) -> None:
        self.terrain = terrain
        self.weights = weights if isinstance(weights, RiskWeights) else RiskWeights(*weights)
        self.thresholds = thresholds
        self.copula_rho = float(copula_rho)
        self.use_copula = bool(use_copula)
        self.disclaimer = disclaimer
        if exposure is None:
            from app.ingestion.terrain import TerrainProcessor

            processor = TerrainProcessor(terrain.grid, demo_mode=True)
            exposure = processor.exposure_score(terrain, weights=exposure_weights)
        self.exposure = np.clip(np.asarray(exposure, dtype=np.float64), 0.0, 1.0)
        if self.exposure.shape != terrain.grid.shape:
            raise ValueError(
                f"exposure shape {self.exposure.shape} != grid {terrain.grid.shape}"
            )

    # ------------------------------------------------------------------ core
    def compute(
        self,
        prediction: dict[str, np.ndarray],
        *,
        init_time,
        lead_hours: Sequence[float],
        step: int | None = None,
        model_version: str = "unknown",
        uncertainty: dict[str, np.ndarray] | None = None,
        confidence_intervals: dict[str, dict[str, np.ndarray]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RiskResult:
        """Fuse hazards into a :class:`RiskResult` for one lead time.

        Parameters
        ----------
        prediction:
            Arrays shaped ``(T_out, H, W)`` (or ``(S, T_out, H, W)`` for samples)
            keyed by ``thunderstorm``, ``cloudburst``, ``flood``.
        step:
            Forecast step index (defaults to the last step of the arrays).
        uncertainty, confidence_intervals:
            Optional MC statistics aligned with the selected step.
        """
        thunderstorm_stack = np.asarray(prediction["thunderstorm"], dtype=np.float64)
        cloudburst_stack = np.asarray(prediction["cloudburst"], dtype=np.float64)
        flood_stack = np.asarray(prediction["flood"], dtype=np.float64)
        index = -1 if step is None else int(step)
        thunderstorm = np.clip(thunderstorm_stack[index], 0.0, 1.0)
        cloudburst = np.clip(cloudburst_stack[index], 0.0, 1.0)
        flood_probability = np.clip(flood_stack[index], 0.0, 1.0)

        flood_risk = flood_probability * self.exposure
        compound = compound_probability(
            thunderstorm, cloudburst, rho=self.copula_rho, use_copula=self.use_copula
        )
        w_ts, w_cb, w_fl = self.weights.normalised()
        overall = np.clip(w_ts * thunderstorm + w_cb * cloudburst + w_fl * flood_risk, 0.0, 1.0)
        categories = categorise(overall, self.thresholds)

        selected_lead = list(lead_hours)
        if 0 <= (step if step is not None else len(selected_lead) - 1) < len(selected_lead):
            selected_lead = [selected_lead[step if step is not None else len(selected_lead) - 1]]

        return RiskResult(
            grid=self.terrain.grid,
            init_time=init_time,
            lead_hours=selected_lead,
            thunderstorm=thunderstorm,
            cloudburst=cloudburst,
            flood_probability=flood_probability,
            flood_risk=flood_risk,
            exposure=self.exposure,
            compound_storm_cloudburst=compound,
            overall=overall,
            category=categories,
            thresholds=self.thresholds,
            weights=self.weights,
            uncertainty={key: np.asarray(value) for key, value in (uncertainty or {}).items()},
            confidence_intervals=confidence_intervals or {},
            model_version=model_version,
            disclaimer=self.disclaimer,
            metadata=metadata or {},
        )

    def compute_all_leads(
        self,
        prediction: dict[str, np.ndarray],
        *,
        init_time,
        lead_hours: Sequence[float],
        model_version: str = "unknown",
    ) -> list[RiskResult]:
        """Compute a :class:`RiskResult` for every lead time."""
        thunderstorm = np.asarray(prediction["thunderstorm"])
        steps = thunderstorm.shape[0]
        results: list[RiskResult] = []
        for step in range(steps):
            lead = [lead_hours[step]] if step < len(lead_hours) else [float(step)]
            results.append(
                self.compute(
                    prediction,
                    init_time=init_time,
                    lead_hours=lead,
                    step=step,
                    model_version=model_version,
                )
            )
        return results

    # ------------------------------------------------------------- GeoJSON
    def vectorise(
        self,
        result: RiskResult,
        *,
        min_category: int = 1,
        risk_field: str = "overall",
        max_features: int = 6000,
    ) -> list[dict[str, Any]]:
        """Merge contiguous same-category cells into rectangle polygons.

        Raster-to-vector conversion keeps the payload small enough for MapLibre:
        only cells at or above ``min_category`` are emitted, and identical row runs
        in consecutive rows are merged into single rectangles.
        """
        category = np.asarray(result.category)
        risk = np.asarray(getattr(result, risk_field), dtype=np.float64)
        ny, nx = category.shape
        open_boxes: dict[tuple[int, int, int], dict[str, Any]] = {}
        boxes: list[dict[str, Any]] = []
        for row in range(ny):
            runs: list[tuple[int, int, int]] = []
            col = 0
            while col < nx:
                if category[row, col] < min_category:
                    col += 1
                    continue
                cat = int(category[row, col])
                start = col
                while col + 1 < nx and category[row, col + 1] == cat:
                    col += 1
                runs.append((start, col, cat))
                col += 1
            current: set[tuple[int, int, int]] = set()
            for col_start, col_end, cat in runs:
                key = (col_start, col_end, cat)
                current.add(key)
                row_max = float(risk[row, col_start : col_end + 1].max())
                if key in open_boxes:
                    box = open_boxes[key]
                    box["row1"] = row
                    box["risk"] = max(box["risk"], row_max)
                else:
                    open_boxes[key] = {
                        "row0": row,
                        "row1": row,
                        "col0": col_start,
                        "col1": col_end,
                        "category": cat,
                        "risk": row_max,
                    }
            for key in [k for k in open_boxes if k not in current]:
                boxes.append(open_boxes.pop(key))
        boxes.extend(open_boxes.values())
        boxes.sort(key=lambda box: -box["risk"])
        return boxes[: int(max_features)]

    def to_geojson(
        self,
        result: RiskResult,
        *,
        min_category: int = 1,
        risk_field: str = "overall",
        max_features: int = 6000,
        event_type: str = "compound",
    ) -> dict[str, Any]:
        """GeoJSON ``FeatureCollection`` of risk polygons (dashboard + alerting)."""
        grid = result.grid
        boxes = self.vectorise(
            result, min_category=min_category, risk_field=risk_field, max_features=max_features
        )
        risk_array = np.asarray(getattr(result, risk_field), dtype=np.float64)
        cell_area_km2 = grid.effective_res_km[0] * grid.effective_res_km[1]
        hazard = {
            "flood_risk": "flood",
            "thunderstorm": "thunderstorm",
            "cloudburst": "cloudburst",
            "compound_storm_cloudburst": "compound",
        }.get(risk_field, "compound")
        valid_time = (
            result.init_time.isoformat() if hasattr(result.init_time, "isoformat") else str(result.init_time)
        )
        features: list[dict[str, Any]] = []
        for box in boxes:
            row0, row1, col0, col1 = box["row0"], box["row1"], box["col0"], box["col1"]
            x0 = grid.min_lon + col0 * grid.d_lon
            y0 = grid.min_lat + row0 * grid.d_lat
            x1 = grid.min_lon + (col1 + 1) * grid.d_lon
            y1 = grid.min_lat + (row1 + 1) * grid.d_lat
            risk_slice = risk_array[row0 : row1 + 1, col0 : col1 + 1]
            features.append(
                {
                    "type": "Feature",
                    "id": f"{event_type}-{row0}-{col0}-{col1}-{row1}",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
                    },
                    "properties": {
                        "event_type": event_type,
                        "hazard": hazard,
                        "risk_category": RISK_CATEGORIES[int(box["category"])],
                        "risk_category_code": int(box["category"]),
                        "risk_max": round(float(box["risk"]), 4),
                        "risk_mean": round(float(risk_slice.mean()), 4),
                        "area_km2": round(
                            float((row1 - row0 + 1) * (col1 - col0 + 1) * cell_area_km2), 3
                        ),
                        "n_cells": int((row1 - row0 + 1) * (col1 - col0 + 1)),
                        "lead_hours": result.lead_hours,
                        "valid_time": valid_time,
                        "model_version": result.model_version,
                        "disclaimer": result.disclaimer,
                    },
                }
            )
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "grid": grid.to_dict(),
                "n_features": len(features),
                "min_category": RISK_CATEGORIES[int(min_category)],
                "risk_field": risk_field,
                "model_version": result.model_version,
                "disclaimer": result.disclaimer,
                "summary": result.summary(),
            },
        }

    # ------------------------------------------------------------ point api
    def point_risk(self, result: RiskResult, lat: float, lon: float) -> dict[str, Any]:
        """Risk breakdown for a single point (``GET /api/v1/risk/point``)."""
        index = result.grid.index_of(lat, lon)
        if index is None:
            raise ValueError(f"({lat}, {lon}) lies outside the model AOI")
        row, col = index
        payload: dict[str, Any] = {
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "row": row,
            "col": col,
            "lead_hours": result.lead_hours,
            "init_time": result.init_time.isoformat()
            if hasattr(result.init_time, "isoformat")
            else str(result.init_time),
            "hazards": {
                "thunderstorm": round(float(result.thunderstorm[row, col]), 4),
                "cloudburst": round(float(result.cloudburst[row, col]), 4),
                "flood_probability": round(float(result.flood_probability[row, col]), 4),
            },
            "terrain_exposure": round(float(result.exposure[row, col]), 4),
            "flood_risk": round(float(result.flood_risk[row, col]), 4),
            "compound_storm_cloudburst": round(float(result.compound_storm_cloudburst[row, col]), 4),
            "overall_risk": round(float(result.overall[row, col]), 4),
            "risk_category": RISK_CATEGORIES[int(result.category[row, col])],
            "model_version": result.model_version,
            "disclaimer": result.disclaimer,
        }
        if result.uncertainty:
            payload["uncertainty"] = {
                key: round(float(value[row, col]), 4) for key, value in result.uncertainty.items()
            }
        if result.confidence_intervals:
            payload["confidence_intervals"] = {
                key: {
                    "lower": round(float(value["lower"][row, col]), 4),
                    "upper": round(float(value["upper"][row, col]), 4),
                }
                for key, value in result.confidence_intervals.items()
            }
        return payload
