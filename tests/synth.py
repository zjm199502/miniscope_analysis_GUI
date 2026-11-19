"""Synthetic 1-photon movies with known ground truth for algorithm tests."""

import numpy as np
import cv2

from miniscope_gui.core.preprocess import MOVIE_FILE, TIMES_FILE
from miniscope_gui.core.workspace import Workspace


def make_movie(T=600, H=96, W=96, n_cells=12, fps=10.0, shifts=None, seed=1,
               cell_radius=3.0, noise=2.0, tau_s=0.8, fixed_centres=None, amplitude=40.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W]
    # cells away from the border
    centres = [] if fixed_centres is None else list(np.asarray(fixed_centres, float))
    if fixed_centres is not None and len(centres) != n_cells:
        raise ValueError("fixed_centres must contain n_cells positions")
    while len(centres) < n_cells:
        c = rng.uniform(12, [H - 12, W - 12])
        if all(np.hypot(*(c - d)) > 4 * cell_radius for d in centres):
            centres.append(c)
    centres = np.asarray(centres).reshape(-1, 2)
    A = np.stack([np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * cell_radius ** 2))
                  for cy, cx in centres]) if n_cells else np.empty((0, H, W))
    # spikes -> AR(1) calcium
    g = np.exp(-1 / (tau_s * fps))
    S = (rng.random((n_cells, T)) < 0.02) * rng.uniform(0.5, 1.5, (n_cells, T))
    C = np.zeros_like(S)
    for t in range(T):
        C[:, t] = (g * C[:, t - 1] if t else 0) + S[:, t]
    # structured background: big blob + vessels texture, fluctuating slowly
    bg = 60 + 50 * np.exp(-((yy - H / 2) ** 2 + (xx - W / 2) ** 2) / (2 * (0.45 * H) ** 2))
    vessels = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 2.5)
    vessels /= vessels.std()
    bg = bg - 25 * (vessels > 1.0)
    drift = 1 + 0.05 * np.sin(np.linspace(0, 6, T))
    movie = np.empty((T, H, W), np.float32)
    for t in range(T):
        f = bg * drift[t] + amplitude * np.tensordot(C[:, t], A, axes=1)
        if shifts is not None:
            M = np.float32([[1, 0, shifts[t, 0]], [0, 1, shifts[t, 1]]])
            f = cv2.warpAffine(f.astype(np.float32), M, (W, H), borderMode=cv2.BORDER_REFLECT)
        movie[t] = f + rng.normal(0, noise, (H, W))
    movie8 = np.clip(np.rint(movie), 0, 255).astype(np.uint8)
    return movie8, dict(A=A, C=C, S=S, centres=centres, g=g)


def workspace_from_movie(root, movie8, fps=10.0):
    ws = Workspace.create_or_open(root)
    np.save(ws.path(MOVIE_FILE), movie8)
    np.save(ws.path(TIMES_FILE), np.arange(movie8.shape[0]) / fps)
    ws.record_step("preprocess", {"fps_out": fps, "spatial_bin": 1}, {"shape": list(movie8.shape)})
    return ws
