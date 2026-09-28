"""Numerical helper routines shared across the pipeline (NumPy only).

Deliberately avoids SciPy so the core nowcasting path installs on any Python
3.10+ interpreter with NumPy. Only *reading* satellite tiles (HDF5/NetCDF) and
GeoTIFF DEMs requires optional packages (h5py/xarray/rasterio).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "gaussian_kernel",
    "gaussian_smooth",
    "box_smooth",
    "local_extreme",
    "bilinear_resize",
    "norm_robust",
    "finite_differences",
    "sigmoid",
    "softmax",
    "logit",
    "to_uint8_rgb",
    "patch_grid",
    "rbf_interpolate",
]


def gaussian_kernel(sigma: float = 1.0, radius: int | None = None) -> np.ndarray:
    """Return a normalised 1-D Gaussian kernel."""
    sigma = max(float(sigma), 1e-3)
    radius = int(radius if radius is not None else max(1, round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    return k / k.sum()


def gaussian_smooth(field: np.ndarray, sigma: float = 1.0, *, mode: str = "reflect") -> np.ndarray:
    """Separable Gaussian smoothing of a 2-D field (edges handled by padding)."""
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("gaussian_smooth expects a 2-D array")
    if sigma <= 0:
        return arr.copy()
    k = gaussian_kernel(sigma)
    r = (k.size - 1) // 2
    padded = np.pad(arr, ((r, r), (r, r)), mode=mode)
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, padded)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, tmp)
    return out


def box_smooth(field: np.ndarray, size: int = 3) -> np.ndarray:
    """Uniform (box) mean filter; ``size`` is forced odd."""
    arr = np.asarray(field, dtype=np.float64)
    size = max(1, int(size) | 1)
    r = size // 2
    padded = np.pad(arr, ((r, r), (r, r)), mode="reflect")
    csum = np.pad(padded.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode="constant")
    ny, nx = arr.shape
    return (
        csum[size : size + ny, size : size + nx]
        - csum[0:ny, size : size + nx]
        - csum[size : size + ny, 0:nx]
        + csum[0:ny, 0:nx]
    ) / float(size * size)


def local_extreme(field: np.ndarray, size: int = 5, *, mode: str = "max") -> np.ndarray:
    """Sliding ``max``/``min`` over a ``size x size`` window."""
    arr = np.asarray(field, dtype=np.float64)
    size = max(1, int(size) | 1)
    r = size // 2
    padded = np.pad(arr, ((r, r), (r, r)), mode="reflect")
    ny, nx = arr.shape
    out = np.full_like(arr, -np.inf if mode == "max" else np.inf)
    reduce_fn = np.maximum if mode == "max" else np.minimum
    for dy in range(size):
        for dx in range(size):
            out = reduce_fn(out, padded[dy : dy + ny, dx : dx + nx])
    return out


def bilinear_resize(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D field to ``shape`` with bilinear interpolation."""
    arr = np.asarray(field, dtype=np.float64)
    ny, nx = int(shape[0]), int(shape[1])
    if arr.shape == (ny, nx):
        return arr.copy()
    y = np.linspace(0, arr.shape[0] - 1, ny)
    x = np.linspace(0, arr.shape[1] - 1, nx)
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1, x1 = np.minimum(y0 + 1, arr.shape[0] - 1), np.minimum(x0 + 1, arr.shape[1] - 1)
    wy, wx = (y - y0)[:, None], (x - x0)[None, :]
    top = arr[np.ix_(y0, x0)] * (1 - wx) + arr[np.ix_(y0, x1)] * wx
    bot = arr[np.ix_(y1, x0)] * (1 - wx) + arr[np.ix_(y1, x1)] * wx
    return top * (1 - wy) + bot * wy


def norm_robust(field: np.ndarray, *, lower_q: float = 2.0, upper_q: float = 98.0) -> np.ndarray:
    """Robust percentile normalisation to ``[0, 1]``."""
    arr = np.asarray(field, dtype=np.float64)
    lo, hi = np.nanpercentile(arr, [lower_q, upper_q])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def finite_differences(field: np.ndarray, spacing: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(d/dy, d/dx)`` central differences of a 2-D field."""
    arr = np.asarray(field, dtype=np.float64)
    return np.gradient(arr, spacing, axis=0), np.gradient(arr, spacing, axis=1)


def sigmoid(x) -> np.ndarray:
    """Numerically stable logistic function."""
    arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(arr)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    exp_x = np.exp(arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def logit(p, eps: float = 1e-6) -> np.ndarray:
    """Inverse sigmoid with clipping (used by temperature scaling)."""
    arr = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(arr / (1.0 - arr))


def softmax(x, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    arr = np.asarray(x, dtype=np.float64)
    shifted = arr - np.max(arr, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def to_uint8_rgb(field: np.ndarray, *, lower_q: float = 1.0, upper_q: float = 99.0) -> np.ndarray:
    """Map a scalar field to an RGB uint8 image (colour-blind friendly ramp)."""
    norm = norm_robust(field, lower_q=lower_q, upper_q=upper_q)
    stops = np.array(
        [
            [0.19, 0.07, 0.23],
            [0.13, 0.36, 0.65],
            [0.10, 0.70, 0.63],
            [0.75, 0.85, 0.25],
            [0.98, 0.55, 0.10],
            [0.78, 0.12, 0.10],
        ]
    )
    idx = norm * (len(stops) - 1)
    lo = np.clip(np.floor(idx).astype(int), 0, len(stops) - 1)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    w = (idx - lo)[..., None]
    rgb = stops[lo] * (1 - w) + stops[hi] * w
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def patch_grid(ny: int, nx: int, patch: int, stride: int) -> list[tuple[int, int, int, int]]:
    """Enumerate patch bounds ``(y0, y1, x0, x1)`` tiling a raster."""
    bounds: list[tuple[int, int, int, int]] = []
    for y in range(0, max(1, ny - patch + 1), max(1, stride)):
        for x in range(0, max(1, nx - patch + 1), max(1, stride)):
            bounds.append((y, min(y + patch, ny), x, min(x + patch, nx)))
    return bounds


def rbf_interpolate(points: np.ndarray, values: np.ndarray, grid_y, grid_x, *, sigma: float = 25.0) -> np.ndarray:
    """Gaussian RBF (Nadaraya-Watson) interpolation of scattered observations.

    Used to spread IMD surface station reports onto the model grid before the
    values are used for validation or as a model input channel.
    """
    pts = np.asarray(points, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be (N, 2) as (y, x) grid coordinates")
    gy = np.asarray(grid_y, dtype=np.float64)[:, None]
    gx = np.asarray(grid_x, dtype=np.float64)[None, :]
    dy = gy - pts[:, 0][None, None, :]
    dx = gx - pts[:, 1][None, None, :]
    w = np.exp(-(dy**2 + dx**2) / (2.0 * max(sigma, 1e-6) ** 2))
    return np.sum(w * vals[None, None, :], axis=-1) / (np.sum(w, axis=-1) + 1e-9)
