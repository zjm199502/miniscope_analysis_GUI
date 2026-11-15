"""Synthetic workspace with two kinetic cell types for the event-clustering tests."""

import numpy as np
import scipy.sparse as sp

from miniscope_gui.core.extraction import save_cells
from miniscope_gui.core.signals import SIGNALS_FILE
from miniscope_gui.core.workspace import Workspace

FPS = 20.0


def kernel(rise, decay, fps=FPS, length_s=12.0):
    t = np.arange(int(length_s * fps)) / fps
    k = np.exp(-t / decay) - np.exp(-t / rise)
    return k / k.max()


def make_kinetic_workspace(root, n_fast=30, n_slow=30, T_s=600.0, seed=0, noise=0.04):
    """Cells 0..n_fast-1: fast events (rise 0.05 s, decay 0.35 s, regular firing);
    the rest: slow events (rise 0.25 s, decay 1.6 s, bursty)."""
    rng = np.random.default_rng(seed)
    T = int(T_s * FPS)
    N = n_fast + n_slow
    t = np.arange(T) / FPS
    dff = np.zeros((N, T))
    truth = np.zeros(N, int)
    kf, ks = kernel(0.05, 0.35), kernel(0.25, 1.6)
    for i in range(N):
        slow = i >= n_fast
        truth[i] = int(slow)
        s = np.zeros(T)
        if slow:                                 # bursts of 2-3 events
            starts = np.flatnonzero(rng.random(T) < 0.012 / FPS * 1.0)
            for a in starts:
                for j in range(rng.integers(1, 3)):
                    b = a + int(j * rng.uniform(4, 8) * FPS)
                    if b < T:
                        s[b] = rng.lognormal(np.log(0.6), 0.3)
            s[rng.random(T) < 0.04 / FPS] = rng.lognormal(np.log(0.6), 0.3)
        else:
            s[rng.random(T) < 0.08 / FPS] = rng.lognormal(np.log(0.6), 0.3)
        gain = rng.uniform(0.7, 1.4)
        dff[i] = gain * np.convolve(s, ks if slow else kf)[:T] + rng.normal(0, noise, T)
    ws = Workspace.create_or_open(root)
    H = W = 64
    centres = rng.uniform(8, 56, (N, 2)).astype(np.float32)
    A = sp.lil_matrix((H * W, N))
    for k, (cy, cx) in enumerate(centres):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                A[(int(cy) + dy) * W + int(cx) + dx, k] = 1.0
    C = dff.astype(np.float32)
    save_cells(ws, A, C, C, np.ones_like(C), centres, np.ones(N), np.ones(N), np.ones(N, bool), (H, W))
    ws.record_step("extraction", {}, {})
    np.save(ws.path("bg_max.npy"), rng.random((H, W)).astype(np.float32))
    np.savez_compressed(ws.path(SIGNALS_FILE), times=t, fps=FPS, dff=C, denoised=C, spikes=np.zeros_like(C),
                        noise=np.full(N, noise), taus=np.ones(N))
    ws.record_step("signals", {}, {})
    return ws, truth
