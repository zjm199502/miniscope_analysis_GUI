"""Step 6: dF/F and spike inference.

dF/F
    F0 is a running low percentile of each cell's local fluorescence (the
    motion-corrected movie averaged over the cell's footprint, i.e. what
    the cell sits in). dF is the extracted trace (raw = denoised + residual),
    already expressed in the same units. dF/F = dF / F0.

Spikes
    OASIS (Friedrich, Zhou & Paninski 2017) for an AR(1) calcium model
    c[t] = g c[t-1] + s[t], with a minimal spike size s_min = k * noise.
    g comes from the indicator decay time: g = exp(-1 / (tau * fps)).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from scipy import ndimage

from .extraction import load_cells
from .workspace import Workspace

SIGNALS_FILE = "signals.npz"

INDICATORS = {      # approximate decay time constants (s) for 1-photon data
    "GCaMP6f": 0.7,
    "GCaMP6m": 1.2,
    "GCaMP6s": 1.8,
    "GCaMP7f": 0.6,
    "GCaMP8f": 0.3,
    "GCaMP8m": 0.5,
    "jGCaMP8s": 0.9,
}

try:  # optional speed-up
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        def deco(f):
            return f
        return deco(a[0]) if a and callable(a[0]) else deco


@dataclass
class SignalParams:
    indicator: str = "GCaMP6f"
    tau_s: float | None = None        # overrides the indicator value
    baseline_window_s: float = 60.0
    baseline_percentile: float = 10.0
    spike_threshold: float = 2.5      # s_min in units of noise std


@njit(cache=True)
def _oasis_ar1(y, g, smin):
    T = y.shape[0]
    v = np.empty(T); w = np.empty(T); t0 = np.empty(T, np.int64); ln = np.empty(T, np.int64)
    n = 0
    for k in range(T):
        v[n] = y[k]; w[n] = 1.0; t0[n] = k; ln[n] = 1
        n += 1
        while n > 1 and v[n - 1] < (g ** ln[n - 2]) * v[n - 2] + smin:
            gl = g ** ln[n - 2]
            v[n - 2] = (w[n - 2] * v[n - 2] + gl * w[n - 1] * v[n - 1]) / (w[n - 2] + gl * gl * w[n - 1])
            w[n - 2] = w[n - 2] + gl * gl * w[n - 1]
            ln[n - 2] = ln[n - 2] + ln[n - 1]
            n -= 1
    c = np.empty(T)
    for i in range(n):
        vi = v[i] if v[i] > 0 else 0.0
        for j in range(ln[i]):
            c[t0[i] + j] = vi * g ** j
    return c


def oasis_ar1(y: np.ndarray, g: float, smin: float = 0.0):
    """Return (denoised calcium c, spikes s)."""
    y = np.ascontiguousarray(y, dtype=np.float64)
    c = _oasis_ar1(y, float(g), float(smin))
    s = np.empty_like(c)
    s[0] = c[0]
    s[1:] = c[1:] - g * c[:-1]
    s[s < 1e-9] = 0.0
    return c, s


def noise_std(x: np.ndarray) -> float:
    d = np.diff(x)
    if len(d) == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def running_percentile(x: np.ndarray, window: int, q: float) -> np.ndarray:
    """Running percentile via a decimated grid (fast for long traces)."""
    T = len(x)
    window = max(3, min(int(window), T))
    step = max(1, window // 10)
    centres = np.arange(0, T, step)
    vals = np.array([np.percentile(x[max(0, c - window // 2):c + window // 2 + 1], q) for c in centres])
    return np.interp(np.arange(T), centres, vals)


def compute_signals(raw, F_local, fps, p: SignalParams):
    tau = p.tau_s if p.tau_s else INDICATORS.get(p.indicator, 0.7)
    g = float(np.exp(-1.0 / (tau * fps)))
    win = int(round(p.baseline_window_s * fps))
    K, T = raw.shape
    dff = np.zeros((K, T), np.float32)
    denoised = np.zeros((K, T), np.float32)
    spikes = np.zeros((K, T), np.float32)
    noise = np.zeros(K, np.float32)
    for k in range(K):
        F0 = running_percentile(F_local[k], win, p.baseline_percentile)
        F0 = np.maximum(F0, 1.0)
        x = raw[k] / F0
        # remove any slow residual baseline of the trace itself
        x = x - running_percentile(x, win, 20.0 if p.baseline_percentile < 20 else p.baseline_percentile)
        dff[k] = x
        sn = noise_std(x)
        noise[k] = sn
        c, s = oasis_ar1(x, g, p.spike_threshold * sn)
        denoised[k] = c
        spikes[k] = s
    return dict(dff=dff, denoised=denoised, spikes=spikes, noise=noise, g=g, tau=tau)


def run_signals(session, ws: Workspace, p: SignalParams,
                progress: Callable[[float, str], None] | None = None,
                should_stop: Callable[[], bool] | None = None) -> dict:
    progress = progress or (lambda f, m: None)
    cells = load_cells(ws)
    fps = frame_rate(ws)
    progress(0.1, "计算 ΔF/F 和 spike…")
    res = compute_signals(cells["raw"], cells["F_local"], fps, p)
    times = load_times(ws, cells["raw"].shape[1])
    np.savez_compressed(ws.path(SIGNALS_FILE), times=times, fps=fps,
                        **{k: v for k, v in res.items()})
    params = asdict(p)
    params.update(tau_used=res["tau"], g=res["g"])
    ws.record_step("signals", params, {"signals": SIGNALS_FILE})
    progress(1.0, "信号计算完成")
    return {"n_cells": int(res["dff"].shape[0])}


def frame_rate(ws: Workspace) -> float:
    st = ws.step("preprocess") or {}
    return float(st.get("params", {}).get("fps_out", 30.0))


def load_times(ws: Workspace, n: int) -> np.ndarray:
    from .preprocess import TIMES_FILE
    if ws.path(TIMES_FILE).exists():
        t = np.load(ws.path(TIMES_FILE))
        if len(t) == n:
            return t - t[0]
    return np.arange(n) / frame_rate(ws)


def load_signals(ws: Workspace) -> dict:
    z = np.load(ws.path(SIGNALS_FILE))
    return {k: z[k] for k in z.files}
