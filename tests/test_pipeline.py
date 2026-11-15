import numpy as np
from scipy.optimize import linear_sum_assignment

from miniscope_gui.core.background import BackgroundParams, run_background
from miniscope_gui.core.export import export_results
from miniscope_gui.core.extraction import ExtractionParams, load_cells, run_extraction
from miniscope_gui.core.motion import MotionParams, run_motion_correction
from miniscope_gui.core.signals import SignalParams, load_signals, oasis_ar1, run_signals

from synth import make_movie, workspace_from_movie


def test_oasis_recovers_spikes():
    rng = np.random.default_rng(0)
    T, g = 2000, 0.9
    s = (rng.random(T) < 0.02) * 1.0
    c = np.zeros(T)
    for t in range(T):
        c[t] = (g * c[t - 1] if t else 0) + s[t]
    y = c + rng.normal(0, 0.1, T)
    chat, shat = oasis_ar1(y, g, smin=0.3)
    assert np.corrcoef(chat, c)[0, 1] > 0.95
    # detected events land on (or next to) true spikes
    det = np.where(shat > 0)[0]
    true = np.where(s > 0)[0]
    near = np.array([np.min(np.abs(true - d)) <= 1 for d in det])
    assert near.mean() > 0.9
    assert abs(len(det) - len(true)) < 0.2 * len(true)


def test_full_calcium_pipeline(tmp_path):
    movie, gt = make_movie(T=800, H=96, W=96, n_cells=10, fps=10.0, seed=4)
    ws = workspace_from_movie(tmp_path / "ws", movie, fps=10.0)
    run_motion_correction(None, ws, MotionParams(template_frames=100))
    run_background(None, ws, BackgroundParams(cell_diameter=8))
    out = run_extraction(None, ws, ExtractionParams(cell_diameter=7))
    cells = load_cells(ws)
    D = np.linalg.norm(gt["centres"][:, None] - cells["centres"][None], axis=2)
    r, c = linear_sum_assignment(D)
    hit = D[r, c] < 3
    assert hit.sum() >= 9, (out, hit.sum())
    assert out["n_cells"] <= 14
    corr = [np.corrcoef(gt["C"][i], cells["C"][j])[0, 1] for i, j in zip(r[hit], c[hit])]
    assert np.mean(corr) > 0.9
    run_signals(None, ws, SignalParams(tau_s=0.8))
    sig = load_signals(ws)
    assert sig["dff"].shape == cells["C"].shape
    assert np.isfinite(sig["dff"]).all()
    dcorr = [np.corrcoef(gt["C"][i], sig["denoised"][j])[0, 1] for i, j in zip(r[hit], c[hit])]
    assert np.mean(dcorr) > 0.85
    exp = export_results(ws)
    for f in ("dff.csv", "spikes.csv", "cells.csv", "results.mat"):
        assert (exp / f).exists()
    # cleanup removes the big intermediate movies; re-running extraction then explains why it can't
    from miniscope_gui.core.cleanup import cleanup_intermediate, intermediate_size
    import pytest
    assert intermediate_size(ws) > 0
    assert cleanup_intermediate(ws) > 0
    assert intermediate_size(ws) == 0 and ws.path("movie_mc.npy").exists()
    with pytest.raises(FileNotFoundError, match="去除背景"):
        run_extraction(None, ws, ExtractionParams(cell_diameter=7))


def test_review_toggle_and_merge(tmp_path):
    from miniscope_gui.core.review import merge_cells, set_accepted
    movie, gt = make_movie(T=400, H=80, W=80, n_cells=6, fps=10.0, seed=2)
    ws = workspace_from_movie(tmp_path / "ws", movie, fps=10.0)
    run_background(None, ws, BackgroundParams(cell_diameter=8, source="preprocess"))
    run_extraction(None, ws, ExtractionParams(cell_diameter=7))
    run_signals(None, ws, SignalParams())
    n = len(load_cells(ws)["C"])
    acc = set_accepted(ws, [0])
    assert not acc[0] and acc[1:].all()
    set_accepted(ws, [0])
    new = merge_cells(ws, [1, 2])
    cells = load_cells(ws)
    assert len(cells["C"]) == n - 1 and new == 1
    assert cells["A"].shape[1] == n - 1
    assert load_signals(ws)["dff"].shape[0] == n - 1


def test_signals_slow_cell_and_flat_baseline():
    from miniscope_gui.core.signals import SignalParams, compute_signals
    rng = np.random.default_rng(3)
    fps, T = 30.0, 9000
    def calcium(tau, rate):
        g = np.exp(-1 / (tau * fps))
        s = (rng.random(T) < rate / fps) * 1.0
        c = np.zeros(T)
        for t in range(T):
            c[t] = (g * c[t - 1] if t else 0) + s[t]
        return c, s
    c_slow, s_slow = calcium(3.0, 0.05)       # much slower than GCaMP6f
    flat = np.zeros(T)
    F = np.full((2, T), 100.0)
    raw = np.stack([10 * c_slow, flat]) + rng.normal(0, 0.5, (2, T)) \
        + 0.3 * np.sin(np.arange(T) / 200.0)   # slow wobble on both
    res = compute_signals(raw.astype(np.float32), F, fps, SignalParams(indicator="GCaMP6f"))
    n_true = int(s_slow.sum())
    n_slow = int((res["spikes"][0] > 0).sum())
    assert n_slow <= 2 * n_true + 2, (n_slow, n_true)
    assert int((res["spikes"][1] > 0).sum()) <= 3      # flat trace: (almost) no events
    assert 2.0 < res["taus"][0] < 4.0, res["taus"]
    # a normal GCaMP6f cell on the same slow wobble keeps a GCaMP6f-like decay
    c_fast, _ = calcium(0.7, 0.1)
    raw2 = (10 * c_fast + rng.normal(0, 0.5, T) + 0.3 * np.sin(np.arange(T) / 200.0))[None]
    res2 = compute_signals(raw2.astype(np.float32), F[:1], fps, SignalParams(indicator="GCaMP6f"))
    assert 0.45 < res2["taus"][0] < 1.1, res2["taus"]


def test_suspect_cells():
    import scipy.sparse as sp
    from miniscope_gui.core.extraction import suspect_cells
    H = W = 60
    yy, xx = np.mgrid[:H, :W]
    round_ = ((yy - 20) ** 2 + (xx - 20) ** 2 <= 16).astype(float)        # normal cell, d ~ 8
    long_ = ((abs(yy - 45) <= 1) & (abs(xx - 30) <= 15)).astype(float)    # vessel-like
    big = ((yy - 30) ** 2 + (xx - 45) ** 2 <= 144).astype(float)          # far too large
    piece = ((yy - 30) ** 2 + (xx - 30) ** 2 <= 9).astype(float)          # touches the big one
    A = sp.csc_matrix(np.stack([round_.ravel(), long_.ravel(), big.ravel(), piece.ravel()], 1))
    assert suspect_cells(A, (H, W), 8).tolist() == [False, True, True, True]


def test_estimate_cell_diameter():
    from miniscope_gui.core.cellsize import estimate_cell_diameter
    rng = np.random.default_rng(0)
    for r in (2.0, 3.5):
        movie, gt = make_movie(T=600, H=160, W=160, n_cells=30, cell_radius=r, seed=5, noise=4)
        img = movie.max(0).astype(np.float32) - movie.mean(0)
        d = estimate_cell_diameter(img)
        assert d is not None and abs(d - 2 * np.sqrt(2) * r) < 0.2 * 2 * np.sqrt(2) * r, (r, d)
