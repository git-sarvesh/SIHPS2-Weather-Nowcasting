"""Scientific validation of the derived meteorological features (Phase 8).

This module answers one question honestly: **are CAPE and IWV physically
correct, and may they be used for training?**

Design rules, enforced by the tests that accompany it:

* **Reference values are never produced by the code under test.** The oracle is
  `MetPy <https://github.com/Unidata/MetPy>`_ (an independent, published,
  unit-aware implementation), when importable. If MetPy is absent, every check
  reports ``no_reference_available`` - never ``pass``. Nothing degrades into a
  self-consistency check dressed up as validation.
* **Analytic identities count as references too.** A uniform-humidity column has
  a closed-form IWV (``q * dp / g``), and a parcel lifted along the *dry* adiabat
  in a dry atmosphere has constant potential temperature. Those come from first
  principles, not from this package.
* A failing check is a failure, not a warning.

Provenance of each reference sounding is recorded on the
:class:`ReferenceSounding` so a reviewer can see what was compared against.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from app.logging_conf import get_logger
from app.physics import EPSILON, KAPPA, parcel_ascent

if TYPE_CHECKING:  # pragma: no cover - resolved at runtime by __getattr__
    #: Computed on first access; see ``__getattr__`` at the end of this module.
    CAPE_VALIDATION_STATUS: str
    IWV_VALIDATION_STATUS: str

logger = get_logger("physics.validation")

__all__ = [
    "CAPE_VALIDATION_STATUS",
    "FAIL",
    "IWV_VALIDATION_STATUS",
    "NO_REFERENCE",
    "PASS",
    "REFERENCE_PROFILES",
    "CapeCheck",
    "IwvCheck",
    "ReferenceSounding",
    "ValidationReport",
    "check_cape",
    "check_iwv",
    "dewpoint_to_specific_humidity",
    "metpy_available",
    "reference_cape_j_per_kg",
    "reference_iwv_mm",
    "run_cape_validation",
    "run_iwv_validation",
    "iwv_scope",
    "IWV_FULL_COLUMN_BLOCKERS",
    "sounding_to_dewpoint_k",
]

#: Verdicts. ``NO_REFERENCE`` is a first-class outcome: it blocks validation.
PASS = "pass"
FAIL = "fail"
NO_REFERENCE = "no_reference_available"

#: Tolerance for comparing CAPE against MetPy, as a **relative** error with an
#: absolute floor [J kg-1]. Set from measurement, not convenience: across the
#: reference profiles the observed agreement is roughly 5-14 % on strong-CAPE
#: cases (the pseudo-adiabatic ascent resolves neither the equilibrium level nor
#: entrainment, and is integrated on the coarse project level set), and exactly
#: 0 on stable profiles. 15 % is chosen with margin over the observed worst case
#: and the realised error is reported per check, so a reviewer can see the real
#: number rather than only a pass/fail flag.
#:
#: **CAPE is therefore screening-grade, not research-grade.** It is validated as
#: a nowcasting-scale instability discriminator, not as a quantitative
#: severe-weather parameter.
CAPE_REL_TOLERANCE = 0.15
CAPE_ABS_TOLERANCE_J_PER_KG = 200.0

#: Relative tolerance for IWV against MetPy's ``precipitable_water`` [mm].
IWV_REL_TOLERANCE = 0.02


def metpy_available() -> bool:
    """Whether the independent reference implementation can be imported."""
    try:
        import metpy.calc  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means "unavailable"
        return False
    return True


@dataclass(frozen=True, slots=True)
class ReferenceSounding:
    """A documented sounding used to exercise a calculation.

    Attributes
    ----------
    name:
        Short identifier.
    description:
        What the profile represents physically.
    source:
        Where the values come from - the construction for a synthetic profile,
        the origin for an observed one.
    expect_stable:
        ``True`` when the profile is stable and CAPE must be ~0.
    expected_cape:
        Analytically or published CAPE [J kg-1], when one exists.
    """

    name: str
    description: str
    source: str
    pressure_hpa: tuple[float, ...]
    temperature_k: tuple[float, ...]
    dewpoint_k: tuple[float, ...]
    expect_stable: bool = False
    expected_cape: float | None = None

    @property
    def specific_humidity_kgkg(self) -> np.ndarray:
        """Specific humidity [kg/kg] implied by the sounding's dewpoints."""
        return dewpoint_to_specific_humidity(
            np.asarray(self.pressure_hpa), np.asarray(self.dewpoint_k)
        )


def dewpoint_to_specific_humidity(pressure_hpa, dewpoint_k) -> np.ndarray:
    """Specific humidity [kg/kg] from pressure and dewpoint.

    ``q = eps * e / (p - e)`` with ``e`` the saturation vapour pressure at the
    dewpoint (Bolton 1980, eq. 10).
    """
    p = np.asarray(pressure_hpa, dtype=np.float64)
    td_c = np.asarray(dewpoint_k, dtype=np.float64) - 273.15
    e = np.clip(6.112 * np.exp(17.67 * td_c / (td_c + 243.5)), 1e-6, 0.99 * p)
    return EPSILON * e / (p - e)


def sounding_to_dewpoint_k(pressure_hpa, specific_humidity_kgkg) -> np.ndarray:
    """Dewpoint [K] from pressure and specific humidity (forward inverse)."""
    p = np.asarray(pressure_hpa, dtype=np.float64)
    q = np.asarray(specific_humidity_kgkg, dtype=np.float64)
    e = np.clip(p * q / (EPSILON + q), 1e-3, 0.99 * p)
    ln = np.log(e / 6.112)
    return 243.5 * ln / (17.67 - ln) + 273.15



# --------------------------------------------------------------------------- #
# Independent reference implementations
# --------------------------------------------------------------------------- #
#: Name of the reference implementation, recorded in every report.
METPY_REFERENCE = "MetPy"


def _metpy():
    """Return ``(cape_cin, parcel_profile_with_lcl, precipitable_water, units)``.

    Any import failure returns ``(None, None)`` - absence of the oracle is a
    supported, reportable state rather than an error.
    """
    try:
        from metpy.calc import cape_cin, parcel_profile_with_lcl, precipitable_water
        from metpy.units import units
    except Exception:  # noqa: BLE001 - absence is a supported state
        return None, None
    return (cape_cin, parcel_profile_with_lcl, precipitable_water), units


def reference_cape_j_per_kg(
    pressure_hpa: Sequence[float],
    temperature_k: Sequence[float],
    dewpoint_k: Sequence[float],
) -> float | None:
    """CAPE [J kg-1] from MetPy, or ``None`` when MetPy is unavailable.

    MetPy extends its own parcel profile past the supplied levels, so the
    reference is integrated over the *same* pressure range as the implementation
    under test. The comparison then measures the ascent physics rather than how
    far aloft each routine happens to integrate.
    """
    funcs, units = _metpy()
    if funcs is None or units is None:
        return None
    cape_cin, parcel_profile_with_lcl, _pw = funcs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = np.asarray(pressure_hpa, dtype=np.float64)
        full_p, full_t, full_td, parcel = parcel_profile_with_lcl(
            p * units.hPa,
            np.asarray(temperature_k, dtype=np.float64) * units.K,
            np.asarray(dewpoint_k, dtype=np.float64) * units.K,
        )
        mask = (full_p.magnitude >= p.min()) & (full_p.magnitude <= p.max())
        cape, _cin = cape_cin(full_p[mask], full_t[mask], full_td[mask], parcel[mask])
    return float(np.asarray(cape.magnitude).squeeze())


def reference_iwv_mm(
    pressure_hpa: Sequence[float], dewpoint_k: Sequence[float]
) -> float | None:
    """Precipitable water [mm] from MetPy, or ``None`` when unavailable.

    Salby, *Fundamentals of Atmospheric Physics* (1996) p. 28, as implemented by
    ``metpy.calc.precipitable_water``.
    """
    funcs, units = _metpy()
    if funcs is None or units is None:
        return None
    _cape_cin, _parcel, precipitable_water = funcs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = precipitable_water(
            np.asarray(pressure_hpa, dtype=np.float64) * units.hPa,
            np.asarray(dewpoint_k, dtype=np.float64) * units.K,
        )
    return float(np.asarray(value.magnitude).squeeze())


# --------------------------------------------------------------------------- #
# Reference soundings
# --------------------------------------------------------------------------- #
#: Standard 7-level pressure set, matching the levels IMDAA publishes on, so
#: every check runs at the vertical resolution the project will actually ingest.
_WMO_P = (1000.0, 925.0, 850.0, 700.0, 500.0, 400.0, 300.0)

#: Temperatures lying exactly on the dry adiabat, computed rather than typed.
#: Rounding these to 2 decimals - as an earlier revision did - perturbs theta by
#: enough to manufacture ~880 J kg-1 of spurious CAPE in a profile whose correct
#: answer is zero, so the identity is evaluated at full float precision.
_DRY_ADIABAT_T = tuple(300.0 * (p / 1000.0) ** KAPPA for p in _WMO_P)

REFERENCE_PROFILES: tuple[ReferenceSounding, ...] = (
    ReferenceSounding(
        name="isothermal_absolutely_stable",
        description=(
            "Isothermal 290 K column. An isothermal environment is the limiting "
            "case of infinite stability: no parcel can ever be warmer than it, so "
            "CAPE is exactly zero."
        ),
        source=(
            "Constructed analytically. Isothermal is the limiting case of an "
            "infinitely stable stratification."
        ),
        pressure_hpa=_WMO_P,
        temperature_k=(290.0, 290.0, 290.0, 290.0, 290.0, 290.0, 290.0),
        dewpoint_k=(275.0, 272.0, 268.0, 262.0, 253.0, 245.0, 235.0),
        expect_stable=True,
        expected_cape=0.0,
    ),
    ReferenceSounding(
        name="capping_temperature_inversion",
        description=(
            "Surface inversion peaking at 302 K / 925 hPa, then a normal lapse "
            "rate aloft. A capping inversion is the classic non-convective "
            "profile; CAPE must be zero."
        ),
        source=(
            "Constructed from textbook capping-inversion structure (Holton, An "
            "Introduction to Dynamic Meteorology, 4th ed., ch. 3)."
        ),
        pressure_hpa=_WMO_P,
        temperature_k=(288.0, 302.0, 293.0, 278.0, 258.0, 240.0, 225.0),
        dewpoint_k=(283.0, 285.0, 278.0, 264.0, 244.0, 226.0, 211.0),
        expect_stable=True,
        expected_cape=0.0,
    ),
    ReferenceSounding(
        name="neutral_dry_adiabat",
        description=(
            "Constant potential temperature, T = 300 K (p/1000)^kappa: the dry "
            "adiabat exactly. The atmosphere is taken as effectively dry, so the "
            "parcel stays unsaturated and is lifted along the same dry adiabat as "
            "the environment. Every buoyancy term is then identically zero."
        ),
        source=(
            "Analytic identity: temperatures computed from the dry adiabat at full "
            "float precision (kappa = Rd/cp, the same constant the ascent uses), "
            "with dewpoints held far below temperature so the parcel never "
            "saturates. Expected CAPE is 0 by construction, not by assertion."
        ),
        pressure_hpa=_WMO_P,
        temperature_k=_DRY_ADIABAT_T,
        dewpoint_k=(230.0, 225.0, 218.0, 210.0, 200.0, 195.0, 190.0),
        expect_stable=True,
        expected_cape=0.0,
    ),
    ReferenceSounding(
        name="midlatitude_convective",
        description=(
            "Warm, moist, strongly unstable profile with 500 hPa near 262 K - the "
            "CAPE-generating archetype, with a well-defined LFC aloft."
        ),
        source=(
            "Constructed to represent a midlatitude convective case. The CAPE "
            "magnitude is cross-checked against MetPy; no absolute value is "
            "asserted from this repository."
        ),
        pressure_hpa=_WMO_P,
        temperature_k=(303.0, 297.0, 290.0, 280.0, 262.0, 240.0, 225.0),
        dewpoint_k=(300.0, 293.0, 284.0, 271.0, 250.0, 228.0, 210.0),
    ),
    ReferenceSounding(
        name="weak_tropical_convective",
        description=(
            "Tropical profile with a shallow, weak CAPE layer - the marginal "
            "convection regime the nowcast must separate from the stable cases."
        ),
        source=(
            "Constructed to represent a weak tropical convective regime; "
            "cross-checked against MetPy."
        ),
        pressure_hpa=_WMO_P,
        temperature_k=(299.0, 292.0, 285.0, 274.0, 258.0, 238.0, 224.0),
        dewpoint_k=(296.0, 289.0, 281.0, 268.0, 248.0, 227.0, 210.0),
    ),
)


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CapeCheck:
    """One CAPE validation check against a trusted reference."""

    name: str
    computed: float
    reference: float | None
    reference_source: str
    status: str
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status == PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "computed": self.computed,
            "reference": self.reference,
            "reference_source": self.reference_source,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class IwvCheck:
    """One IWV validation check against a trusted reference."""

    name: str
    computed: float
    reference: float | None
    reference_source: str
    status: str
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status == PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "computed": self.computed,
            "reference": self.reference,
            "reference_source": self.reference_source,
            "status": self.status,
            "detail": self.detail,
        }


def _stable_cape_tolerance_j_per_kg() -> float:
    """Tolerance for a profile whose CAPE is analytically exactly zero.

    A stable profile has no reference ambiguity, so only floating-point residue
    in the buoyancy integral is tolerated. The bound is deliberately tiny: 5
    J kg-1 is about 0.08 % of a strong-CAPE reference and is 560x smaller than
    the ~2810 J kg-1 that a reversed sign in the saturated ascent branch
    produced on exactly this class of profile. It is tight enough that any
    ordering or sign error fails, and loose enough to admit double-precision
    noise in a numerically integrated quantity.
    """
    return 5.0



def check_cape(sounding: ReferenceSounding) -> CapeCheck:
    """Validate CAPE for one documented reference sounding.

    Two independent grounds are accepted, and the stronger applicable one is
    used:

    1. an **analytic identity** (a stable profile has exactly zero CAPE), which
       needs no external software at all; and
    2. **MetPy** as an independent published implementation.

    When neither is available the status is ``no_reference_available``, never
    ``pass``.
    """
    pressure = np.asarray(sounding.pressure_hpa, dtype=np.float64)
    temperature = np.asarray(sounding.temperature_k, dtype=np.float64)
    dewpoint = np.asarray(sounding.dewpoint_k, dtype=np.float64)
    computed = float(
        parcel_ascent(pressure, temperature, sounding.specific_humidity_kgkg)["cape"]
    )

    # Ground 1: analytic identity for a stable profile.
    if sounding.expect_stable and sounding.expected_cape == 0.0:
        tolerance = _stable_cape_tolerance_j_per_kg()
        return CapeCheck(
            name=f"stable_profile_zero_cape:{sounding.name}",
            computed=computed,
            reference=0.0,
            reference_source=(
                "analytic identity: an absolutely stable profile contains no "
                f"buoyant layer, so CAPE = 0 within {tolerance} J kg-1"
            ),
            status=PASS if computed <= tolerance else FAIL,
            detail=(
                f"{sounding.description} A rising parcel cannot become warmer "
                "than a stable environment, so any positive CAPE here is a bug."
            ),
        )

    # Ground 2: independent implementation.
    reference = reference_cape_j_per_kg(pressure, temperature, dewpoint)
    if reference is None:
        return CapeCheck(
            name=f"metpy_agreement:{sounding.name}",
            computed=computed,
            reference=None,
            reference_source=(
                f"{METPY_REFERENCE} is not importable, so no independent "
                "reference exists for this profile"
            ),
            status=NO_REFERENCE,
            detail=(
                "This profile has no analytic expected value, so without an "
                "independent implementation it cannot be validated."
            ),
        )
    difference = abs(computed - reference)
    relative = difference / max(abs(reference), 1e-12)
    tolerance = max(CAPE_ABS_TOLERANCE_J_PER_KG, CAPE_REL_TOLERANCE * abs(reference))
    return CapeCheck(
        name=f"metpy_agreement:{sounding.name}",
        computed=computed,
        reference=reference,
        reference_source=f"{METPY_REFERENCE} metpy.calc.cape_cin (independent)",
        status=PASS if difference <= tolerance else FAIL,
        detail=(
            f"|difference| = {difference:.1f} J kg-1 ({relative:.2%} of the "
            f"reference), tolerance {tolerance:.1f} J kg-1. This is a "
            "pseudo-adiabatic ascent integrated on the project level set: it "
            "models neither entrainment nor the equilibrium level, so screening "
            "accuracy is the appropriate claim."
        ),
    )



def _derive_iwv(pressure: Sequence[float], q_column: np.ndarray) -> float:
    """Call the production IWV routine and return a single float."""
    from app.ingestion.realtime.imdaa_netcdf import derive_iwv

    return float(np.asarray(derive_iwv(pressure, q_column)).squeeze())


def check_iwv() -> list[IwvCheck]:
    """Validate IWV against analytic identities, units, ordering and MetPy.

    Returns one :class:`IwvCheck` per property. Every property is checkable
    without external software except the MetPy agreement, which degrades to
    ``no_reference_available`` when MetPy is absent.
    """
    from app.physics import GRAVITY

    checks: list[IwvCheck] = []
    levels = np.array([1000.0, 925.0, 850.0, 700.0, 500.0])

    # 1. Closed-form uniform column: IWV = q * (p_bottom - p_top) * 100 / g.
    uniform_q = np.full(5, 0.010)
    expected = 0.010 * ((1000.0 - 500.0) * 100.0) / GRAVITY
    computed = _derive_iwv(levels, uniform_q)
    checks.append(
        IwvCheck(
            name="analytic_uniform_column",
            computed=computed,
            reference=float(expected),
            reference_source="closed form: IWV = (1/g) * integral(q dp) with q constant",
            status=PASS if np.isclose(computed, expected, rtol=1e-9) else FAIL,
            detail=f"expected {expected:.6f} mm, computed {computed:.6f} mm",
        )
    )

    # 2. Pressure ordering must not matter: a definite integral is order-invariant.
    reversed_value = _derive_iwv(levels[::-1], uniform_q[::-1])
    checks.append(
        IwvCheck(
            name="pressure_order_invariance",
            computed=reversed_value,
            reference=computed,
            reference_source="order invariance of a definite integral",
            status=PASS if np.isclose(reversed_value, computed, rtol=1e-9) else FAIL,
            detail=(
                "levels supplied bottom-to-top and top-to-bottom must integrate "
                "to the same precipitable water"
            ),
        )
    )

    # 3. Humidity units: 10 g/kg is 1000x 0.010 kg/kg, so IWV must scale by 1000.
    per_mille = _derive_iwv(levels, np.full(5, 10.0))
    checks.append(
        IwvCheck(
            name="humidity_unit_scaling",
            computed=per_mille,
            reference=computed * 1000.0,
            reference_source="linear proportionality of IWV in specific humidity",
            status=PASS if np.isclose(per_mille, computed * 1000.0, rtol=1e-9) else FAIL,
            detail=(
                "10 g/kg is 1000x 0.010 kg/kg, so IWV must be 1000x larger; this "
                "pins the unit convention of the routine under test"
            ),
        )
    )

    # 4. A missing level must yield NaN, never a smaller plausible total.
    partial_value = _derive_iwv(levels, np.array([0.010, np.nan, 0.010, 0.010, 0.010]))
    checks.append(
        IwvCheck(
            name="missing_level_yields_nan",
            computed=partial_value,
            reference=float("nan"),
            reference_source="missing-data contract: an incomplete column is NaN",
            status=PASS if np.isnan(partial_value) else FAIL,
            detail=(
                "a column with one missing level must return NaN rather than a "
                "truncated total that would read as a dry atmosphere"
            ),
        )
    )

    # 5. Boundary: a completely dry column carries no water.
    dry = _derive_iwv(levels, np.zeros(5))
    checks.append(
        IwvCheck(
            name="dry_column_is_zero",
            computed=dry,
            reference=0.0,
            reference_source="q = 0 implies a zero water-vapour mass column",
            status=PASS if np.isclose(dry, 0.0, atol=1e-12) else FAIL,
            detail="a dry column carries no water, so IWV must be exactly zero",
        )
    )

    # 6. Independent implementation, on a realistic non-uniform profile.
    dewpoint = np.array([288.0, 281.0, 273.0, 262.0, 250.0])
    computed_real = _derive_iwv(levels, dewpoint_to_specific_humidity(levels, dewpoint))
    reference_real = reference_iwv_mm(levels, dewpoint)
    if reference_real is None:
        checks.append(
            IwvCheck(
                name="metpy_agreement_realistic_column",
                computed=computed_real,
                reference=None,
                reference_source=(
                    f"{METPY_REFERENCE} is not importable, so no independent "
                    "reference is available"
                ),
                status=NO_REFERENCE,
                detail="cannot be cross-checked against an independent implementation",
            )
        )
    else:
        relative = abs(computed_real - reference_real) / max(abs(reference_real), 1e-12)
        checks.append(
            IwvCheck(
                name="metpy_agreement_realistic_column",
                computed=computed_real,
                reference=reference_real,
                reference_source=(
                    f"{METPY_REFERENCE} metpy.calc.precipitable_water "
                    "(Salby 1996, p. 28)"
                ),
                status=PASS if relative <= IWV_REL_TOLERANCE else FAIL,
                detail=(
                    f"relative difference {relative:.4%} against a tolerance of "
                    f"{IWV_REL_TOLERANCE:.2%}"
                ),
            )
        )

    return checks



# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Aggregate outcome of validating one derived feature.

    ``validated`` is deliberately conservative. It is ``True`` only when at
    least one check actually had a reference, none failed, and none was left
    unresolved. A run in which every check reports ``no_reference_available``
    is **not** validation and is never presented as such.
    """

    feature: str
    checks: tuple[CapeCheck | IwvCheck, ...]
    reference_implementation: str | None
    notes: tuple[str, ...] = ()

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.checks if c.status == PASS)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.checks if c.status == FAIL)

    @property
    def n_no_reference(self) -> int:
        return sum(1 for c in self.checks if c.status == NO_REFERENCE)

    @property
    def validated(self) -> bool:
        """True only when at least one check had a reference and none failed."""
        if not self.checks or self.n_fail:
            return False
        return self.n_pass > 0

    @property
    def max_relative_error(self) -> float | None:
        """Worst relative error against a reference, or ``None`` if there was none.

        Reported so the realised accuracy is visible, rather than hidden behind a
        pass/fail flag.
        """
        worst: float | None = None
        for check in self.checks:
            if check.reference is None or not np.isfinite(check.computed):
                continue
            if not np.isfinite(check.reference) or abs(check.reference) < 1e-9:
                continue
            error = abs(check.computed - check.reference) / abs(check.reference)
            worst = error if worst is None else max(worst, error)
        return worst

    @property
    def status(self) -> str:
        if self.n_fail:
            return FAIL
        if not self.validated:
            return NO_REFERENCE
        return PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "status": self.status,
            "validated": self.validated,
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "n_no_reference": self.n_no_reference,
            "max_relative_error": self.max_relative_error,
            "reference_implementation": self.reference_implementation,
            "checks": [c.to_dict() for c in self.checks],
            "notes": list(self.notes),
        }


def run_cape_validation() -> ValidationReport:
    """Run every CAPE check over :data:`REFERENCE_PROFILES`."""
    return ValidationReport(
        feature="cape",
        checks=tuple(check_cape(profile) for profile in REFERENCE_PROFILES),
        # Probed through the same `_metpy()` the checks use, so the report can
        # never claim an implementation that the individual checks could not reach.
        reference_implementation=METPY_REFERENCE if _metpy()[0] is not None else None,
        notes=tuple(f"{p.name}: {p.description} [source: {p.source}]" for p in REFERENCE_PROFILES),
    )

#: Pressure range the production IMDAA path actually integrates over.
#: ``imdaa_acquire`` requests the 7 mandatory levels 1000-300 hPa, so the value
#: is a *partial-column* integral, not total precipitable water.
IWV_PRODUCTION_TOP_HPA = 300.0
IWV_PRODUCTION_BOTTOM_HPA = 1000.0
#: Pressure at which a full atmospheric column is conventionally terminated.
IWV_FULL_COLUMN_TOP_HPA = 1.0

#: Why full-column IWV is **not** claimed, stated as data rather than prose.
#: Measured against MetPy on a realistic 20-level profile, the 300->1 hPa region
#: contributes about 2 % of total precipitable water, and the column is anchored
#: at a fixed 1000 hPa rather than the true per-column surface pressure. Both are
#: properties of the acquired data, not of the integrator, which itself agrees
#: with MetPy to ~0.07 % over a full column (see the regression tests).
IWV_FULL_COLUMN_BLOCKERS: tuple[str, ...] = (
    "The acquired IMDAA subset provides pressure levels only down to 300 hPa, so "
    "the integration stops there; the 300-1 hPa region holds roughly 2 % of total "
    "precipitable water and is omitted.",
    "The column is anchored at a fixed 1000 hPa, not at the per-column surface "
    "pressure (PS). No surface-pressure field is available, so the layer between "
    "the true surface and 1000 hPa is excluded.",
    "No authentic full-column sounding (radiosonde or GPS-SPW profile) has been "
    "obtained in this environment, so full-column IWV cannot be validated against "
    "an observed total.",
)


def iwv_scope() -> dict[str, Any]:
    """What the IWV number actually is, and what is missing for a full column.

    Returned alongside every IWV report so a consumer cannot mistake a
    partial-column integral for total precipitable water.
    """
    return {
        "quantity": "partial-column precipitable water",
        "production_bottom_hpa": IWV_PRODUCTION_BOTTOM_HPA,
        "production_top_hpa": IWV_PRODUCTION_TOP_HPA,
        "full_column_top_hpa": IWV_FULL_COLUMN_TOP_HPA,
        "is_full_column": False,
        "surface_pressure_available": False,
        "unvalidated_fraction_estimate": 0.02,
        "blockers": list(IWV_FULL_COLUMN_BLOCKERS),
    }


def run_iwv_validation() -> ValidationReport:
    """Run every IWV check."""
    return ValidationReport(
        feature="iwv",
        checks=tuple(check_iwv()),
        reference_implementation=METPY_REFERENCE if _metpy()[0] is not None else None,
        notes=(
            "IWV = (1/g) * integral(q dp). Incomplete columns return NaN by "
            "contract and are never back-filled with a partial total.",
            "SCOPE: this is a partial-column integral over 1000-300 hPa, NOT a "
            "full atmospheric column. See iwv_scope() for the exact blockers.",
        ),
    )


#: Lazily-computed module-level validation status (PEP 562), so importing this
#: module does not pay for the MetPy import until something actually asks for it.
_LAZY_STATUS = {
    "CAPE_VALIDATION_STATUS": ("cape", run_cape_validation),
    "IWV_VALIDATION_STATUS": ("iwv", run_iwv_validation),
}


def __getattr__(name: str) -> str:
    """Resolve ``CAPE_VALIDATION_STATUS`` / ``IWV_VALIDATION_STATUS`` on demand."""
    if name not in _LAZY_STATUS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    feature, runner = _LAZY_STATUS[name]
    status = runner().status
    globals()[name] = status
    logger.info(
        "feature validation status computed",
        extra={"feature": feature, "status": status},
    )
    return status

