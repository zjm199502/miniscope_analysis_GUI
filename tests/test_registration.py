"""Cross-session registration on synthetic sessions with known identities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import cv2
import numpy as np
import pytest
import scipy.sparse as sp

from miniscope_gui.core import registration as R
from miniscope_gui.core.extraction import save_cells
from miniscope_gui.core.signals import SIGNALS_FILE
from miniscope_gui.core.workspace import Workspace

SENSOR = 160          # binned sensor size
DIAM = 8


def _world(seed=0, n=110):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:                       # cells at least 0.9 diameter apart
        p = rng.uniform(10, SENSOR - 10, 2)
        if all(np.hypot(*(p - q)) > 0.9 * DIAM for q in pts):
            pts.append(p)
    return np.array(pts), rng.uniform(0.7, 1.3, n), rng.uniform(0.8, 1.25, (n, 2))


def make_session(root, cells, keep, angle=0.0, shift=(0.0, 0.0), crop=(0, 0, SENSOR, SENSOR),
                 warp=None, seed=0, T=0, activity=None):
    """Workspace whose accepted cells are cells[keep] seen through a rotation + shift.

    crop is (x, y, w, h) in binned sensor pixels. Returns the true world index of each cell.
    """
    rng = np.random.default_rng(seed)
    pts, amp, ax = cells
    x0, y0, w, h = crop
    c = np.array([SENSOR / 2, SENSOR / 2])
    a = np.radians(angle)
    Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    idx = np.array(keep)
    P = (pts[idx] - c) @ Rm.T + c + np.array(shift) + rng.normal(0, 0.4, (len(idx), 2))
    if warp is not None:
        P = P + warp(P)
    P = P - [y0, x0]
    ok = (P[:, 0] > 4) & (P[:, 0] < h - 4) & (P[:, 1] > 4) & (P[:, 1] < w - 4)
    idx, P = idx[ok], P[ok]
    yy, xx = np.mgrid[:h, :w]
    cols = []
    for k, (py, px) in zip(idx, P):
        sy, sx = ax[k] * DIAM / 4
        img = amp[k] * np.exp(-((yy - py) ** 2 / (2 * sy ** 2) + (xx - px) ** 2 / (2 * sx ** 2)))
        img[img < 0.1 * img.max()] = 0
        cols.append(img.ravel())
    # a few rejected cells must be ignored
    n_rej = 3
    for _ in range(n_rej):
        cols.append(np.roll(cols[0], 7))
    A = sp.csc_matrix(np.array(cols).T)
    N = A.shape[1]
    acc = np.r_[np.ones(len(idx), bool), np.zeros(n_rej, bool)]
    perm = rng.permutation(N)                 # cell numbers differ between days
    A, acc = A[:, perm], acc[perm]
    true = np.r_[idx, -np.ones(n_rej, int)][perm]
    ws = Workspace.create_or_open(root)
    ws.record_step("preprocess", {"spatial_bin": 2, "crop": [2 * x0, 2 * y0, 2 * w, 2 * h], "fps_out": 10.0}, {})
    Tn = max(T, 10)
    C = np.zeros((N, Tn), np.float32)
    if activity is not None:
        C[acc] = activity(true[acc])
    save_cells(ws, A, C, C, np.ones_like(C), np.zeros((N, 2), np.float32), np.ones(N), np.ones(N), acc, (h, w))
    ws.record_step("extraction", {"cell_diameter": DIAM}, {})
    # mean projection: a "blood vessel" texture fixed to the tissue, seen through the same transform
    tex = np.random.default_rng(99).normal(size=(SENSOR, SENSOR)).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (0, 0), 3) * 50 + 100
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    qy, qx = yy + y0 - shift[0] - c[0], xx + x0 - shift[1] - c[1]
    wy = np.cos(a) * qy + np.sin(a) * qx + c[0]          # inverse rotation back to tissue coordinates
    wx = -np.sin(a) * qy + np.cos(a) * qx + c[1]
    mean = cv2.remap(tex, wx.astype(np.float32), wy.astype(np.float32), cv2.INTER_LINEAR)
    np.save(ws.path("mean_projection.npy"), mean.astype(np.float32))
    if T:
        t = np.arange(T) / 10.0
        spikes = (C > 0.5).astype(np.float32)
        np.savez_compressed(ws.path(SIGNALS_FILE), times=t, fps=10.0, dff=C, denoised=C, spikes=spikes,
                            noise=np.ones(N), taus=np.ones(N))
        ws.record_step("signals", {}, {})
    return ws, true


def _accuracy(res, trues):
    """Fraction of truly shared cells grouped correctly, and number of wrong links."""
    S = len(trues)
    ids = res["cell_ids"]
    good = bad = total = 0
    for a in range(S):
        for b in range(a + 1, S):
            ta = {int(c): int(t) for c, t in enumerate(trues[a]) if t >= 0}
            tb = {int(c): int(t) for c, t in enumerate(trues[b]) if t >= 0}
            shared = set(ta.values()) & set(tb.values())
            total += len(shared)
            for row in ids:
                if row[a] >= 0 and row[b] >= 0:
                    if ta[int(row[a])] == tb[int(row[b])]:
                        good += 1
                    else:
                        bad += 1
    return good / total, bad


def test_three_days(tmp_path):
    cells = _world()
    n = len(cells[0])
    rng = np.random.default_rng(1)
    k1 = np.arange(n)
    k2 = np.sort(rng.choice(n, int(0.8 * n), replace=False))
    k3 = np.sort(rng.choice(n, int(0.75 * n), replace=False))
    ws1, t1 = make_session(tmp_path / "d1", cells, k1, crop=(10, 10, 140, 140), seed=1)
    ws2, t2 = make_session(tmp_path / "d2", cells, k2, angle=4.0, shift=(3.0, -5.0),
                           crop=(14, 6, 136, 140), seed=2)
    warp = lambda P: np.column_stack([1.5 * np.sin(P[:, 1] / 50), 1.5 * np.cos(P[:, 0] / 60)])
    ws3, t3 = make_session(tmp_path / "d3", cells, k3, angle=-3.0, shift=(-4.0, 2.0),
                           crop=(8, 12, 140, 136), warp=warp, seed=3)
    for transform in ("rigid", "nonrigid"):
        p = R.RegParams(sessions=[str(ws1.root), str(ws2.root), str(ws3.root)],
                        names=["Day1", "Day2", "Day3"], transform=transform)
        out = tmp_path / f"reg_{transform}"
        s = R.run_registration(p, out)
        res = R.load_registration(out)
        acc, bad = _accuracy(res, [t1, t2, t3])
        assert acc > 0.9 and bad <= 4, (transform, acc, bad)
        assert abs(s["alignment"][1]["rotation_deg"] - 4.0) < 1.0
        assert s["alignment"][1]["ncc_after"] > s["alignment"][1]["ncc_before"]
        # no group holds two cells of one session, and every accepted cell appears once
        for k, tr in enumerate((t1, t2, t3)):
            col = res["cell_ids"][:, k]
            col = col[col >= 0]
            assert len(col) == len(set(col)) == (tr >= 0).sum()
        assert (out / R.CSV_FILE).exists() and "Day1" in R.describe(s)
    # aligning on the mean projection (blood vessels) gives the same rotation
    p = R.RegParams(sessions=[str(ws1.root), str(ws2.root)], image="mean")
    s = R.run_registration(p, tmp_path / "reg_mean")
    assert abs(s["alignment"][1]["rotation_deg"] - 4.0) < 1.0, s["alignment"]
    acc, bad = _accuracy(R.load_registration(tmp_path / "reg_mean"), [t1, t2])
    assert acc > 0.9 and bad <= 3


def test_model_and_cluster():
    rng = np.random.default_rng(0)
    d1 = rng.rayleigh(1.0, 300); r1 = rng.beta(8, 2, 300)
    d0 = 12 * np.sqrt(rng.random(900)); r0 = rng.beta(1, 6, 900)
    m = R.PairModel(dmax=12, sigma=2).fit(np.r_[d1, d0], np.r_[r1, r0])
    assert abs(m.sigma - 1.0) < 0.3 and 0.15 < m.pi < 0.35
    p = m.p_same(np.r_[d1, d0], np.r_[r1, r0])
    assert (p[:300] > 0.5).mean() > 0.95 and (p[300:] > 0.5).mean() < 0.03
    # a cell missing on day 2 is still linked between day 1 and day 3;
    # conflicting links never put two cells of one session together
    edges = [(0.9, 0, 0, 2, 0), (0.95, 0, 1, 1, 0), (0.9, 1, 0, 2, 1), (0.8, 0, 1, 2, 1),
             (0.6, 0, 0, 1, 0)]
    assign, score = R.cluster([2, 1, 2], edges)
    rows = {tuple(r) for r in assign}
    assert (1, 0, 1) in rows and (0, -1, 0) in rows and len(assign) == 2


def test_bad_inputs(tmp_path):
    with pytest.raises(ValueError):
        R.run_registration(R.RegParams(sessions=[str(tmp_path)]), tmp_path / "o")
    (tmp_path / "x").mkdir(); (tmp_path / "x" / "f.txt").write_text("1")
    with pytest.raises(ValueError):
        R.check_output_folder(tmp_path / "x")


def _activity(seed, assembly=True, T=3000):
    def f(world):
        rng = np.random.default_rng(seed)
        n = len(world)
        spikes = rng.random((n, T)) < 0.01
        if assembly:
            common = rng.random(T) < 0.02
            members = np.isin(world, np.arange(15))
            spikes[members] |= common & (rng.random((members.sum(), T)) < 0.8)
        g = np.exp(-1 / 8)
        C = np.zeros((n, T), np.float32)
        for i in range(1, T):
            C[:, i] = g * C[:, i - 1] + spikes[:, i]
        return C + rng.normal(0, 0.05, C.shape).astype(np.float32)
    return f


def test_crossday(tmp_path):
    from miniscope_gui.core import crossday as X
    cells = _world(seed=5, n=60)
    n = len(cells[0])
    rng = np.random.default_rng(2)
    keep2 = np.sort(np.r_[np.arange(15), rng.choice(np.arange(15, n), 35, replace=False)])
    ws1, _ = make_session(tmp_path / "d1", cells, np.arange(n), seed=1, T=3000, activity=_activity(1))
    ws2, _ = make_session(tmp_path / "d2", cells, keep2, angle=2, shift=(2, 3), seed=2, T=3000,
                          activity=_activity(2))
    ws3, _ = make_session(tmp_path / "d3", cells, np.arange(n), angle=-2, seed=3, T=3000,
                          activity=_activity(3, assembly=False))
    out = tmp_path / "reg"
    R.run_registration(R.RegParams(sessions=[str(w.root) for w in (ws1, ws2, ws3)],
                                   names=["训练", "测试", "对照"]), out)
    s = X.run_reactivation(out, X.ReactivationParams(template=0, target=1, n_shuffle=200))
    assert s["n_assemblies"] >= 1 and 1 in s["reactivated"]
    r = X.load_crossday(out, "crossday_reactivation")
    top = r["members"][0]
    assert top.sum() >= 10 and (r["ids_template"][top] >= 0).all()
    s3 = X.run_reactivation(out, X.ReactivationParams(template=0, target=2, n_shuffle=200))
    assert 1 not in s3["reactivated"]
    st = X.run_stability(out, X.StabilityParams(n_shuffle=50))
    assert st["n_tracked_all"] >= 40
    r = X.load_crossday(out, "crossday_stability")
    assert r["pv"].shape == (3, 3) and np.isfinite(r["pv"][0, 1])
    assert "训练" in X.describe_reactivation(s) and X.describe_stability(st)
    assert len(X.export_crossday(out)) == 3


def test_stats_helpers():
    from scipy import stats
    from miniscope_gui.core import crossday as X
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 5, 40).astype(float), rng.normal(size=40)
    assert abs(X.spearman(a, b) - stats.spearmanr(a, b)[0]) < 1e-9
    for k, n, na, nb in ((3, 50, 10, 12), (0, 20, 5, 5), (7, 30, 8, 9)):
        assert abs(X.hypergeom_sf(k, n, na, nb) - stats.hypergeom.sf(k - 1, n, na, nb)) < 1e-9
