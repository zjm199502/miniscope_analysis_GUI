"""Estimate the typical cell diameter from (max - mean) projection.

Bright blobs are found with a scale-normalised Laplacian of Gaussian; each
blob's best scale sigma gives its diameter (about 2*sqrt(2)*sigma for a
Gaussian-shaped cell). The median over the clearest blobs is returned.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def estimate_cell_diameter(img: np.ndarray, mask: np.ndarray | None = None,
                           d_min: float = 3.0, d_max: float = 30.0, n_blobs: int = 40) -> float | None:
    f = np.asarray(img, np.float32)
    if mask is not None:
        f = np.where(mask, f, np.median(f[mask]) if mask.any() else 0)
    # remove large structures first (glow, vessels wider than any cell)
    big = int(round(d_max))
    f = cv2.morphologyEx(f, cv2.MORPH_TOPHAT,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * big + 1, 2 * big + 1)))
    f = f / (f.std() or 1.0)
    sigmas = np.arange(d_min, d_max + 0.01, 0.5) / (2 * np.sqrt(2))
    stack = np.stack([-s * s * ndimage.gaussian_laplace(f, s) for s in sigmas])
    best = stack.max(0)
    arg = stack.argmax(0)
    peak = (best == ndimage.maximum_filter(best, size=int(2 * d_min) | 1)) & (best > 0)
    if mask is not None:
        peak &= ndimage.binary_erosion(mask, iterations=int(d_max))
    ys, xs = np.nonzero(peak)
    if len(ys) < 5:
        return None
    order = np.argsort(-best[ys, xs])[:n_blobs]
    s = sigmas[arg[ys[order], xs[order]]]
    inner = (s > sigmas[0]) & (s < sigmas[-1])       # drop blobs at the edge of the range
    if inner.sum() < 5:
        return None
    # x1.15: calibrated on synthetic data, where a slightly generous diameter
    # gives the best extraction recall
    return float(1.15 * np.median(2 * np.sqrt(2) * s[inner]))
