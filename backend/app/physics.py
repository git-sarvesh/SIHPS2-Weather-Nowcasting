"""Physical constants, thermodynamic formulas and channel definitions.

Everything physical lives here so that the ingestion connectors, the model
normalisation layer, the risk engine and the XAI consistency checker all agree
on units, bounds and formulas. Formulas are the standard meteorological forms
(see e.g. Bolton 1980, Stull 2011) implemented in NumPy with no SciPy dependency.

References
----------
* Bolton, D. (1980), Mon. Wea. Rev. 108 - saturation vapour pressure, LCL.
* Stull, R. (2011), JAMC 50 - wet-bulb / equivalent potential temperature.
* Doswell & Rasmussen (1994) - CAPE/CIN from parcel ascent.
* Arkin & Meisner (1987) - GOES precipitation index style rain-rate proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Constants (SI unless suffixed)
# --------------------------------------------------------------------------- #
T0_KELVIN = 273.15
P0_HPA = 1000.0
R_D = 287.05           # specific gas constant, dry air [J kg-1 K-1]
R_V = 461.5            # specific gas constant, water vapour [J kg-1 K-1]
CP_D = 1005.0          # specific heat, dry air [J kg-1 K-1]
CP_V = 1870.0          # specific heat, water vapour [J kg-1 K-1]
GRAVITY = 9.80665      # [m s-2]
L_V = 2.501e6          # latent heat of vaporisation [J kg-1]
EPSILON = R_D / R_V    # ~0.622
KAPPA = R_D / CP_D     # ~0.286

#: Physical scale heights / bounds used for normalisation and validation.
TIR_BT_BOUNDS = (180.0, 320.0)     # K, INSAT TIR1/TIR2 window
WV_BT_BOUNDS = (190.0, 280.0)      # K, 6.8 um water-vapour channel
CTT_BOUNDS = (180.0, 300.0)        # K, cloud-top temperature
COOLING_RATE_BOUNDS = (-10.0, 10.0)  # K h-1
WV_ANOMALY_BOUNDS = (-25.0, 25.0)  # K, deviation from background
IWV_BOUNDS = (0.0, 70.0)           # mm, integrated water vapour
CAPE_BOUNDS = (0.0, 5000.0)        # J kg-1
ELEVATION_BOUNDS = (0.0, 8000.0)   # m


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """Description of one model input channel."""

    name: str
    unit: str
    vmin: float
    vmax: float
    description: str
    derived: bool = False

    def normalise(self, values) -> np.ndarray:
        """Scale physical values to ``[0, 1]`` using physically meaningful bounds."""
        arr = np.asarray(values, dtype=np.float32)
        return np.clip((arr - self.vmin) / (self.vmax - self.vmin), 0.0, 1.0)

    def denormalise(self, values) -> np.ndarray:
        """Inverse of :meth:`normalise`."""
        arr = np.asarray(values, dtype=np.float32)
        return arr * (self.vmax - self.vmin) + self.vmin


#: Channel order of the input tensor ``(B, T, C, H, W)`` with ``C == 12``.
CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("tir1_bt", "K", *TIR_BT_BOUNDS, "INSAT-3D TIR1 10.8 um brightness temperature"),
    ChannelSpec("tir2_bt", "K", *TIR_BT_BOUNDS, "INSAT-3D TIR2 12.0 um brightness temperature"),
    ChannelSpec("wv_bt", "K", *WV_BT_BOUNDS, "INSAT-3D water-vapour 6.8 um brightness temperature"),
    ChannelSpec("vis_refl", "-", 0.0, 1.0, "Visible 0.65 um reflectance"),
    ChannelSpec("swir_refl", "-", 0.0, 1.0, "SWIR 1.6 um reflectance (cloud phase)"),
    ChannelSpec("mir_bt", "K", 250.0, 350.0, "Middle-IR 3.9 um brightness temperature"),
    ChannelSpec("ctt", "K", *CTT_BOUNDS, "Cloud-top temperature (derived from TIR1)", derived=True),
    ChannelSpec("ctt_cooling_rate", "K h-1", *COOLING_RATE_BOUNDS, "dCTT/dt - convective growth rate", derived=True),
    ChannelSpec("wv_bt_anomaly", "K", *WV_ANOMALY_BOUNDS, "Water-vapour BT anomaly vs. background field", derived=True),
    ChannelSpec("iwv", "mm", *IWV_BOUNDS, "Integrated water vapour (IMDAA column)", derived=True),
    ChannelSpec("cape", "J kg-1", *CAPE_BOUNDS, "Convective available potential energy (IMDAA)", derived=True),
    ChannelSpec("elevation", "m", *ELEVATION_BOUNDS, "DEM elevation (SRTM/CartoDEM)", derived=True),
)

CHANNEL_INDEX: dict[str, int] = {spec.name: i for i, spec in enumerate(CHANNELS)}
N_CHANNELS: int = len(CHANNELS)

CHANNEL_INDEX: dict[str, int] = {spec.name: i for i, spec in enumerate(CHANNELS)}
N_CHANNELS: int = len(CHANNELS)


# --------------------------------------------------------------------------- #
# Channel provenance and semantics (Phase 8.4)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ChannelMeta:
    """Provenance and data semantics for one model channel.

    Phase 8.4 requires every channel to state where it comes from, how it is
    derived, what range is physical, and what a missing value means. Recording
    this explicitly is what stops a station field's unit being attached to an
    unrelated satellite channel.

    Attributes
    ----------
    name, unit:
        Identity and physical unit, mirroring :class:`ChannelSpec`.
    source:
        The data source that must provide this channel. A channel whose source is
        not acquired has no value, and stays missing - it is never approximated.
    derivation:
        ``observed`` for a direct measurement, ``derived`` when computed from
        other channels or columns, or a named transform.
    vmin, vmax:
        Physical range used for validation and normalisation, from
        :class:`ChannelSpec`. Not a clamp applied to data.
    missing:
        What an absent value means for this channel, and the rule the ingestion
        layer must follow. Absent is always NaN, never zero.
    requires_acquisition:
        True when the channel cannot be produced without acquiring its source.
    """

    name: str
    unit: str
    source: str
    derivation: str
    vmin: float
    vmax: float
    missing: str
    requires_acquisition: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "source": self.source,
            "derivation": self.derivation,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "missing": self.missing,
            "requires_acquisition": self.requires_acquisition,
        }


#: Unit every absent channel carries: no measurement, not a zero.
MISSING_SENTINEL_UNIT = "NaN"

_INSAT = "INSAT-3D/3DR (MOSDAC) L1B/L2 imagery"
_IMDAA = "IMDAA reanalysis (NCMRWF RDS)"
_DEM = "SRTM/CartoDEM elevation"

#: The 12 model channels, with provenance and missing-data semantics.
CHANNEL_META: tuple[ChannelMeta, ...] = (
    ChannelMeta("tir1_bt", "K", _INSAT, "observed", *TIR_BT_BOUNDS,
                "no INSAT L1B/L2 granule covers this cell; NaN, never 0 K"),
    ChannelMeta("tir2_bt", "K", _INSAT, "observed", *TIR_BT_BOUNDS,
                "no INSAT L1B/L2 granule covers this cell; NaN, never 0 K"),
    ChannelMeta("wv_bt", "K", _INSAT, "observed", *WV_BT_BOUNDS,
                "no INSAT water-vapour granule covers this cell; NaN"),
    ChannelMeta("vis_refl", "-", _INSAT, "observed", 0.0, 1.0,
                "no INSAT visible granule (daylight only); NaN, never 0.0"),
    ChannelMeta("swir_refl", "-", _INSAT, "observed", 0.0, 1.0,
                "no INSAT SWIR granule covers this cell; NaN, never 0.0"),
    ChannelMeta("mir_bt", "K", _INSAT, "observed", 250.0, 350.0,
                "no INSAT middle-IR granule covers this cell; NaN"),
    ChannelMeta("ctt", "K", f"derived from {_INSAT} tir1_bt", "derived", *CTT_BOUNDS,
                "parent tir1_bt is missing, so CTT is missing; NaN, never 0 K"),
    ChannelMeta("ctt_cooling_rate", "K h-1",
                "finite difference of ctt over 30-minute frames", "derived",
                *COOLING_RATE_BOUNDS,
                "needs at least two consecutive valid CTT frames; NaN when fewer"),
    ChannelMeta("wv_bt_anomaly", "K",
                "wv_bt minus the field median background", "derived",
                *WV_ANOMALY_BOUNDS,
                "parent wv_bt is missing; NaN, never 0 K"),
    ChannelMeta("iwv", "mm", f"{_IMDAA} pressure-level specific humidity",
                "derived", *IWV_BOUNDS,
                "no valid humidity column (levels or values missing); NaN, "
                "never 0 mm. Partial columns are NaN, not a truncated total."),
    ChannelMeta("cape", "J kg-1",
                f"{_IMDAA} pressure-level temperature and specific humidity",
                "derived", *CAPE_BOUNDS,
                "no valid T+q column; NaN, never 0 J kg-1. Not estimated from "
                "temperature or relative humidity alone."),
    ChannelMeta("elevation", "m", _DEM, "observed", *ELEVATION_BOUNDS,
                "no DEM tile covers this cell; NaN, never 0 m (0 m is a real "
                "elevation and must not stand in for missing)"),
)

CHANNEL_META_INDEX: dict[str, ChannelMeta] = {m.name: m for m in CHANNEL_META}


def channel_metadata_payload() -> dict[str, dict[str, object]]:
    """The full 12-channel metadata registry, for provenance records and API."""
    return {name: meta.to_dict() for name, meta in CHANNEL_META_INDEX.items()}


def channel_units() -> dict[str, str]:
    """``channel -> unit`` for every model channel, from :data:`CHANNEL_META`."""
    return {name: meta.unit for name, meta in CHANNEL_META_INDEX.items()}

#: Channel groups used by the physical-consistency checker (Part 4.3).
CONVECTIVE_CHANNELS = ("ctt", "ctt_cooling_rate", "wv_bt_anomaly", "mir_bt", "tir1_bt", "cape")
MOISTURE_CHANNELS = ("iwv", "wv_bt", "wv_bt_anomaly")
TERRAIN_CHANNELS = ("elevation",)

#: Rain-level class labels for the cloudburst / extreme-rainfall head.
RAIN_CLASSES = ("no_rain", "light", "heavy", "extreme")
RAIN_CLASS_THRESHOLDS_MMH = (0.1, 7.5, 35.0)  # light <7.5, heavy <35, extreme >=35 mm/h


def channel_index(name: str) -> int:
    """Index of a channel inside the input tensor."""
    try:
        return CHANNEL_INDEX[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown channel {name!r}; known: {sorted(CHANNEL_INDEX)}") from exc


def normalise_channels(cube: np.ndarray) -> np.ndarray:
    """Normalise a ``(..., C, H, W)`` cube channel-wise to ``[0, 1]``."""
    arr = np.asarray(cube, dtype=np.float32)
    if arr.shape[-3] != N_CHANNELS:
        raise ValueError(f"expected {N_CHANNELS} channels, got {arr.shape[-3]}")
    out = np.empty_like(arr)
    for i, spec in enumerate(CHANNELS):
        out[..., i, :, :] = spec.normalise(arr[..., i, :, :])
    return out


def denormalise_channels(cube: np.ndarray) -> np.ndarray:
    """Inverse of :func:`normalise_channels`."""
    arr = np.asarray(cube, dtype=np.float32)
    out = np.empty_like(arr)
    for i, spec in enumerate(CHANNELS):
        out[..., i, :, :] = spec.denormalise(arr[..., i, :, :])
    return out


# --------------------------------------------------------------------------- #
# Thermodynamics
# --------------------------------------------------------------------------- #
def saturation_vapour_pressure_hpa(t_kelvin) -> np.ndarray:
    """Saturation vapour pressure over water (Bolton 1980), hPa."""
    t_c = np.asarray(t_kelvin, dtype=np.float64) - T0_KELVIN
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))


def vapour_pressure_hpa(p_hpa, mixing_ratio_kgkg) -> np.ndarray:
    """Vapour pressure from pressure and water-vapour mixing ratio."""
    w = np.asarray(mixing_ratio_kgkg, dtype=np.float64)
    p = np.asarray(p_hpa, dtype=np.float64)
    return p * w / (EPSILON + w)


def mixing_ratio_kgkg(p_hpa, t_kelvin, rh_percent) -> np.ndarray:
    """Water-vapour mixing ratio [kg/kg] from pressure, temperature and RH [%]."""
    es = saturation_vapour_pressure_hpa(t_kelvin)
    e = np.clip(np.asarray(rh_percent, dtype=np.float64), 0.0, 100.0) / 100.0 * es
    e = np.minimum(e, 0.99 * np.asarray(p_hpa, dtype=np.float64))
    return EPSILON * e / (np.asarray(p_hpa, dtype=np.float64) - e)


def relative_humidity_percent(p_hpa, t_kelvin, mixing_ratio) -> np.ndarray:
    """Relative humidity [%] from pressure, temperature and mixing ratio."""
    e = vapour_pressure_hpa(p_hpa, mixing_ratio)
    es = saturation_vapour_pressure_hpa(t_kelvin)
    return np.clip(100.0 * e / es, 0.0, 100.0)


def dewpoint_kelvin(p_hpa, mixing_ratio) -> np.ndarray:
    """Dew-point temperature [K] from pressure and mixing ratio (inverse Bolton)."""
    e = np.clip(vapour_pressure_hpa(p_hpa, mixing_ratio), 1e-3, None)
    ln = np.log(e / 6.112)
    return 243.5 * ln / (17.67 - ln) + T0_KELVIN


def potential_temperature(t_kelvin, p_hpa) -> np.ndarray:
    """Dry potential temperature [K]."""
    return np.asarray(t_kelvin, dtype=np.float64) * (P0_HPA / np.asarray(p_hpa, dtype=np.float64)) ** KAPPA


def lcl_temperature_kelvin(t_kelvin, td_kelvin) -> np.ndarray:
    """Lifting condensation level temperature [K] (Bolton 1980, eq. 21)."""
    t_c = np.asarray(t_kelvin, dtype=np.float64) - T0_KELVIN
    td_c = np.asarray(td_kelvin, dtype=np.float64) - T0_KELVIN
    td_c = np.minimum(td_c, t_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / (td_c - 56.0) + np.log(np.clip(t_c, 1e-3, None) / np.clip(td_c, 1e-3, None)) / 800.0
        out = 1.0 / np.where(np.abs(inv) < 1e-9, np.nan, inv) + 56.0
    return np.where(np.isfinite(out), out, td_c) + T0_KELVIN


def parcel_ascent(
    p_levels_hpa,
    t_env_k,
    q_env_kgkg,
    *,
    n_substeps: int = 12,
) -> dict[str, float | np.ndarray]:
    """Pseudo-adiabatic parcel ascent used to derive CAPE / CIN / LI.

    The parcel is launched from the lowest (highest pressure) level with the
    environmental temperature and moisture, lifted dry-adiabatically until
    saturation and pseudo-adiabatically above the LCL. No entrainment or ice
    phase is modelled - adequate for nowcasting-scale instability screening.

    Parameters
    ----------
    p_levels_hpa:
        1-D pressure levels [hPa], ordered surface first (descending pressure).
    t_env_k, q_env_kgkg:
        Environmental temperature [K] and specific humidity [kg/kg] on those levels.

    Returns
    -------
    dict with ``cape``, ``cin``, ``lcl_pressure_hpa``, ``lfc_pressure_hpa``,
    ``lifted_index``, ``parcel_temperature_k`` and ``buoyancy`` (profile array).
    """
    p = np.asarray(p_levels_hpa, dtype=np.float64)
    t_env = np.asarray(t_env_k, dtype=np.float64)
    q_env = np.asarray(q_env_kgkg, dtype=np.float64)
    if not (p.ndim == t_env.ndim == q_env.ndim == 1 and p.size == t_env.size == q_env.size):
        raise ValueError("pressure/temperature/humidity must be 1-D arrays of equal length")
    if p.size < 3:
        raise ValueError("need at least 3 levels for a parcel ascent")
    order = np.argsort(-p)
    p, t_env, q_env = p[order], t_env[order], q_env[order]

    t_parcel = float(t_env[0])
    w_parcel = float(q_env[0])  # total water conserved until condensation
    parcel_t = np.empty_like(p)
    parcel_t[0] = t_parcel
    lcl_p = float(p[0])
    saturated = False

    for i in range(1, p.size):
        p_lo, p_hi = p[i - 1], p[i]
        dp = (p_hi - p_lo) / n_substeps
        for step in range(1, n_substeps + 1):
            p_prev = p_lo + (step - 1) * dp
            p_step = p_lo + step * dp
            rs = float(mixing_ratio_kgkg(p_step, t_parcel, 100.0))
            if not saturated and w_parcel >= rs:
                saturated = True
                lcl_p = p_step
            if saturated:
                denom = CP_D + (L_V**2) * rs * EPSILON / (R_D * t_parcel**2)
                dtdp = (R_D * t_parcel + L_V * rs) / (p_step * denom)
                # `dp` is NEGATIVE for a rising parcel (pressure decreases), and
                # dT/dp is POSITIVE, so the parcel must COOL. Accumulating with
                # `-=` instead of `+=` warms a rising parcel, which is physically
                # impossible and produced large spurious CAPE in stable profiles.
                t_parcel += dtdp * dp
            else:
                t_parcel *= (p_step / max(p_prev, 1e-6)) ** KAPPA
        parcel_t[i] = t_parcel

    rs_env = np.array([float(mixing_ratio_kgkg(pi, ti, 100.0)) for pi, ti in zip(p, t_env)])
    tv_env = t_env * (1.0 + 0.608 * np.clip(q_env, 0.0, None))
    tv_parcel = parcel_t * (1.0 + 0.608 * np.minimum(w_parcel, rs_env))
    buoyancy = GRAVITY * (tv_parcel - tv_env) / tv_env

    dz = np.zeros_like(p)
    dz[1:] = (R_D * 0.5 * (tv_env[1:] + tv_env[:-1]) / GRAVITY) * np.log(p[:-1] / p[1:])
    dz = np.clip(dz, 0.0, None)

    positive = buoyancy > 0
    lfc_idx = int(np.argmax(positive)) if positive.any() else -1
    cape = float(np.sum(buoyancy[1:][positive[1:]] * dz[1:][positive[1:]])) if lfc_idx > 0 else 0.0
    cin_mask = (~positive) & (np.arange(p.size) <= max(lfc_idx, 0))
    cin = float(-np.sum(buoyancy[1:][cin_mask[1:]] * dz[1:][cin_mask[1:]])) if cin_mask.any() else 0.0

    t500_env = float(np.interp(500.0, p[::-1], t_env[::-1]))
    t500_parcel = float(np.interp(500.0, p[::-1], parcel_t[::-1]))
    return {
        "cape": max(cape, 0.0),
        "cin": max(cin, 0.0),
        "lcl_pressure_hpa": lcl_p,
        "lfc_pressure_hpa": float(p[lfc_idx]) if lfc_idx > 0 else float("nan"),
        "lifted_index": t500_env - t500_parcel,
        "parcel_temperature_k": parcel_t,
        "buoyancy": buoyancy,
    }


def k_index(t850_k, td850_k, t700_k, td700_k, t500_k) -> np.ndarray:
    """K-Index [K] - mid-level thunderstorm potential (>=30 suggests convection)."""
    t8 = np.asarray(t850_k, dtype=np.float64) - T0_KELVIN
    td8 = np.asarray(td850_k, dtype=np.float64) - T0_KELVIN
    t7 = np.asarray(t700_k, dtype=np.float64) - T0_KELVIN
    td7 = np.asarray(td700_k, dtype=np.float64) - T0_KELVIN
    t5 = np.asarray(t500_k, dtype=np.float64) - T0_KELVIN
    return (t8 - t5) + td8 - (t7 - td7)


def total_totals_index(t850_k, td850_k, t500_k) -> np.ndarray:
    """Total Totals Index [K] (>=50 indicates severe convection potential)."""
    return (
        np.asarray(t850_k, dtype=np.float64)
        + np.asarray(td850_k, dtype=np.float64)
        - 2.0 * np.asarray(t500_k, dtype=np.float64)
    )


def showalter_stability_index() -> None:  # pragma: no cover - documented placeholder
    """Showalter index requires 850/500 hPa levels; use ``parcel_ascent`` (LI)."""
    raise NotImplementedError("use parcel_ascent()['lifted_index'] instead")


# --------------------------------------------------------------------------- #
# Derived satellite / moisture features
# --------------------------------------------------------------------------- #
def cloud_top_temperature(tir1_bt_k) -> np.ndarray:
    """Cloud-top temperature [K] proxy from the 10.8 um window channel.

    For optically thick convective tops the 10.8 um brightness temperature is a
    good CTT estimate (semi-transparent cirrus requires a split-window or CO2
    slicing correction - out of scope for this prototype and flagged in the XAI
    consistency check).
    """
    return np.asarray(tir1_bt_k, dtype=np.float64)


def cooling_rate(ctt_sequence_k, dt_hours: float = 0.5) -> np.ndarray:
    """Temporal derivative dCTT/dt [K h-1] over a ``(T, H, W)`` CTT sequence.

    Uses second-order central differences in time with replicated edges; a
    strongly negative rate marks rapid cloud-top growth / overshooting tops.
    """
    ctt = np.asarray(ctt_sequence_k, dtype=np.float64)
    if ctt.ndim != 3:
        raise ValueError("ctt_sequence_k must be (T, H, W)")
    if ctt.shape[0] < 2:
        return np.zeros((1,) + ctt.shape[1:], dtype=np.float64)
    grad = np.gradient(ctt, dt_hours, axis=0)
    return grad


def water_vapour_anomaly(wv_bt_k, background_k=None) -> np.ndarray:
    """Water-vapour brightness-temperature anomaly [K].

    When ``background_k`` (a smooth climatological / time-mean field) is not
    supplied the spatial median of the frame is used as the background, which
    highlights localised mid-level moistening (negative anomaly = moist).
    """
    wv = np.asarray(wv_bt_k, dtype=np.float64)
    bg = np.median(wv) if background_k is None else np.asarray(background_k, dtype=np.float64)
    return wv - bg


def split_window_difference(tir1_bt_k, tir2_bt_k) -> np.ndarray:
    """Brightness-temperature difference TIR1-TIR2 [K]; positive => thin cirrus."""
    return np.asarray(tir1_bt_k, dtype=np.float64) - np.asarray(tir2_bt_k, dtype=np.float64)


def rain_rate_from_ctt(ctt_k, iwv_mm=None) -> np.ndarray:
    """Cloudburst-relevant rainfall-rate proxy [mm h-1] from cloud-top temperature.

    Simple Arkin/Meisner-style linear mapping between cold cloud tops and rain
    rate, modulated by column moisture. Used to *label* synthetic training data
    and as a fallback when no gauge/radar product is available; it is not a
    quantitative precipitation estimate.
    """
    ctt = np.asarray(ctt_k, dtype=np.float64)
    warm, cold = 280.0, 200.0  # no rain above `warm`, max scaling at `cold`
    frac = np.clip((warm - ctt) / (warm - cold), 0.0, 1.0)
    rate = 36.0 * frac**2.0
    if iwv_mm is not None:
        iwv = np.asarray(iwv_mm, dtype=np.float64)
        rate = rate * np.clip(0.55 + 0.65 * iwv / 50.0, 0.55, 1.45)
    return rate


def rain_class_from_rate(rate_mmh) -> np.ndarray:
    """Map a rainfall-rate field [mm h-1] to the 4 rainfall classes (0..3)."""
    from app.physics import RAIN_CLASS_THRESHOLDS_MMH

    rate = np.asarray(rate_mmh, dtype=np.float64)
    cls = np.zeros(rate.shape, dtype=np.int64)
    cls[rate >= RAIN_CLASS_THRESHOLDS_MMH[0]] = 1
    cls[rate >= RAIN_CLASS_THRESHOLDS_MMH[1]] = 2
    cls[rate >= RAIN_CLASS_THRESHOLDS_MMH[2]] = 3
    return cls


def moisture_flux_convergence(u_ms, v_ms, q_kgkg, dx_km, dy_km, *, scale: float = 1e7) -> np.ndarray:
    """Low-level moisture flux convergence ``-div(qV)`` in ``1e-7 kg kg-1 s-1``.

    Central differences on the model grid; the leading edge of a convergence
    maximum is the classic nowcasting precursor for cloudburst initiation.
    """
    u = np.asarray(u_ms, dtype=np.float64)
    v = np.asarray(v_ms, dtype=np.float64)
    q = np.asarray(q_kgkg, dtype=np.float64)
    if not (u.shape == v.shape == q.shape):
        raise ValueError("u, v and q must share the same shape")
    dx = max(float(dx_km), 1e-6) * 1000.0
    dy = max(float(dy_km), 1e-6) * 1000.0
    dqu_dx = np.gradient(q * u, axis=1) / dx
    dqv_dy = np.gradient(q * v, axis=0) / dy
    return -(dqu_dx + dqv_dy) * scale


def channel_importance_template() -> dict[str, float]:
    """Uniform prior over the 12 input channels.

    Used as the fallback attribution when the model backend cannot provide
    gradient based attributions (e.g. NumPy reference backend or first boot).
    """
    weight = 1.0 / N_CHANNELS
    return {spec.name: weight for spec in CHANNELS}


def summarise_profile(values: Iterable[float], labels: Sequence[str]) -> dict[str, float]:
    """Zip a 1-D profile with labels into a mapping (helper for API payloads)."""
    return {label: float(value) for label, value in zip(labels, values)}
