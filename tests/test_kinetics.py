"""Calcium-event kinetics and clustering on synthetic cells with known kinetics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pytest

from kinetics_synth import FPS, kernel, make_kinetic_workspace
from miniscope_gui.core.modeling import kinetics as K
from miniscope_gui.core.modeling.common import load_result
from miniscope_gui.core.modeling.report import describe, export_all


def test_measure_single_event():
    rng = np.random.default_rng(0)
    T = 400
    s = np.zeros(T); s[100] = 1.0; s[300] = 0.8
    y = np.convolve(s, kernel(0.03, 0.8))[:T] + rng.normal(0, 0.01, T)
    p = K.KineticsParams(smooth_s=0)
    F, pk, iso, wav = K.measure_cell(y, 0.01, FPS, p, 20, 120)
    assert len(pk) == 2 and abs(pk[0] - 101) <= 2 and iso.all()
    tau = F[:, K.EV_INDEX["decay_tau_s"]]
    assert np.all(np.abs(tau - 0.8) < 0.15), tau
    assert np.all(F[:, K.EV_INDEX["rise_s"]] < 0.15)
    assert abs(F[0, K.EV_INDEX["amp_dff"]] - 1.0) < 0.1
    assert abs(F[1, K.EV_INDEX["iei_s"]] - 10.0) < 0.2
    assert abs(np.nanmax(wav[0]) - 1.0) < 0.05


def test_fit_decay_and_helpers():
    t = np.arange(60) / FPS
    tau, r2 = K.fit_decay(2.0 * np.exp(-t / 0.5), FPS)
    assert abs(tau - 0.5) < 0.01 and r2 > 0.99
    assert np.isnan(K.fit_decay(np.ones(2), FPS)[0])
    assert K.ari([0, 0, 1, 1], [1, 1, 0, 0]) == 1.0
    assert abs(K.ari([0, 1, 0, 1, 0, 1], [0, 0, 0, 1, 1, 1])) < 0.3
    assert np.isnan(K.spearman(np.ones(10), np.arange(10)))
    assert K.burstiness(np.ones(20)) == -1.0
    g = np.repeat(np.arange(10), 20)
    v = np.random.default_rng(1).normal(size=200) + 3 * g
    assert K.icc1(v, g) > 0.9 and K.icc1(np.random.default_rng(2).normal(size=200), g) < 0.2


@pytest.fixture(scope="module")
def two_types(tmp_path_factory):
    ws, truth = make_kinetic_workspace(tmp_path_factory.mktemp("kin") / "ws", n_fast=25, n_slow=25, T_s=400)
    s = K.run_kinetics(None, ws, {"params": K.KineticsParams(n_boot=15, n_null=5)})
    return ws, truth, s


def test_two_cell_types_are_found(two_types):
    ws, truth, s = two_types
    r = load_result(ws, "kinetics")
    F, good = r["ev_feat"], r["ev_good"]
    tau = F[:, K.EV_INDEX["decay_tau_s"]]
    fast = good & (truth[r["ev_cell"]] == 0)
    slow = good & (truth[r["ev_cell"]] == 1)
    assert np.median(tau[slow]) > 2.5 * np.median(tau[fast])
    c = s["cell"]
    assert c["K"] == 2 and c["ari"] > 0.8
    lab = r["cell_label"]
    assert K.ari(truth[lab >= 0], lab[lab >= 0]) > 0.9
    assert lab[truth == 0].max() == 0 and (lab[truth == 1] == 1).all()   # cluster 1 = faster decay
    ev = s["event"]
    assert ev["K"] >= 2 and ev["purity_p"] < 0.05 and ev["icc"]["decay_tau_s"] > 0.6
    text = describe("kinetics", s)
    assert "可能的细胞类型" in text and "【细胞层面】" in text
    assert r["ev_wave_mean"].shape[0] == ev["K"] and r["cell_wave_mean"].shape[0] == 2


def test_export(two_types, tmp_path):
    ws, _, _ = two_types
    assert "kinetics" in export_all(ws, tmp_path)
    for f in ("kinetics_events.csv", "kinetics_cells.csv", "kinetics_clusters.csv", "kinetics_cluster_waveforms.csv"):
        assert (tmp_path / f).exists()
    head = (tmp_path / "kinetics_cells.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    assert "cell_cluster" in head and "decay_tau_s" in head


def test_one_type_is_not_split_into_stable_cell_types(tmp_path):
    ws, _ = make_kinetic_workspace(tmp_path / "ws", n_fast=30, n_slow=0, T_s=300, seed=1, noise=0.02)
    s = K.run_kinetics(None, ws, {"params": K.KineticsParams(n_boot=15, n_null=5)})
    c = s["cell"]
    assert c["K"] == 1 or c["ari"] < 0.8 or c["silhouette"] < 0.25


def test_too_few_events(tmp_path):
    ws, _ = make_kinetic_workspace(tmp_path / "ws", n_fast=2, n_slow=0, T_s=10)
    with pytest.raises(ValueError):
        K.run_kinetics(None, ws, {"params": K.KineticsParams(threshold_sd=50)})
