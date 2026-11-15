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
