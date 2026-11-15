import numpy as np

from miniscope_gui.core.motion import MotionParams, load_mc_movie, run_motion_correction

from synth import make_movie, workspace_from_movie


def _shifts(T, seed=3, amp=4.0):
    rng = np.random.default_rng(seed)
    s = np.cumsum(rng.normal(0, 0.6, (T, 2)), axis=0)
    s -= s.mean(axis=0)
    return np.clip(s, -amp, amp).astype(np.float32)


def test_rigid_recovers_shifts(tmp_path):
    T = 300
    true = _shifts(T)
    movie, _ = make_movie(T=T, shifts=true)
    ws = workspace_from_movie(tmp_path / "ws", movie)
    summ = run_motion_correction(None, ws, MotionParams(max_shift=10, template_frames=100))
    est = np.load(ws.path("mc_shifts.npy"))
    # estimated shift undoes the true one (up to a constant offset of the template)
    err = (est + true) - (est + true).mean(axis=0)
    assert np.abs(err).mean() < 0.3, np.abs(err).mean()
    assert summ["corr_after"] > summ["corr_before"]
    mc = load_mc_movie(ws)
    assert mc.shape == movie.shape and mc.dtype == np.uint8


def test_piecewise_runs(tmp_path):
    T = 60
    movie, _ = make_movie(T=T, shifts=_shifts(T))
    ws = workspace_from_movie(tmp_path / "ws", movie)
    summ = run_motion_correction(None, ws, MotionParams(piecewise=True, patch_size=48,
                                                       patch_overlap=16, template_frames=30))
    assert summ["corr_after"] > summ["corr_before"]
