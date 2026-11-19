"""Modelling analyses on synthetic data with known structure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pytest
import scipy.sparse as sp

from miniscope_gui.core.behavior import BehaviorConfig, save_track
from miniscope_gui.core.extraction import save_cells
from miniscope_gui.core.modeling import decoder, embedding, ensembles, glm, hmm, info, network
from miniscope_gui.core.modeling.common import (block_folds, fdr_bh, load_dataset, load_result,
                                                 parse_times, zscore)
from miniscope_gui.core.modeling.report import ANALYSES, VIEWS, describe, export_all
from miniscope_gui.core.signals import SIGNALS_FILE
from miniscope_gui.core.sync import save_sync
from miniscope_gui.core.workspace import Workspace

FPS = 10.0
SHOCKS = (60.0, 150.0, 240.0)


def make_workspace(root, T=3600, N=40, seed=0, behavior=True):
    """A finished workspace (cells, signals, behaviour, sync) without running the pipeline.

    cells 0-5 are more active during freezing, 6-9 fire with speed, 10-15 are place
    cells, 16-19 respond to the 'shocks', 20-27 form an assembly, the rest is noise.
    """
    rng = np.random.default_rng(seed)
    ws = Workspace.create_or_open(root)
    t = np.arange(T) / FPS
    fz = np.zeros(T, bool)
    a = 200
    while a < T - 250:                       # irregular bouts, like real freezing
        n = int(rng.integers(80, 220))
        fz[a:a + n] = True
        a += n + int(rng.integers(150, 600))
    x = np.zeros(T); y = np.zeros(T)
    for i in range(1, T):
        step = 0.0 if fz[i] else 1.0
        x[i] = np.clip(x[i - 1] + step * rng.normal(0, 1.2), 0, 30)
        y[i] = np.clip(y[i - 1] + step * rng.normal(0, 1.2), 0, 30)
    speed = np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])) * FPS
    rate = np.full((N, T), 0.02)
    rate[0:6] += 0.25 * fz
    rate[6:10] += 0.02 * np.convolve(speed, np.ones(5) / 5, "same")
    for i in range(10, min(16, N)):
        cx, cy = rng.uniform(5, 25, 2)
        rate[i] += 0.5 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 30)
    for s in SHOCKS:
        rate[16:20, int(s * FPS):int(s * FPS) + 20] += 0.6
    common = rng.random(T) < 0.03
    rate[20:28] += 0.6 * np.convolve(common, np.ones(3), "same")
    spikes = (rng.random((N, T)) < np.clip(rate, 0, 1)).astype(np.float32)
    g = np.exp(-1 / (0.8 * FPS))
    C = np.zeros((N, T), np.float32)
    for i in range(1, T):
        C[:, i] = g * C[:, i - 1] + spikes[:, i]
    dff = (0.3 * C + rng.normal(0, 0.05, (N, T))).astype(np.float32)
    H = W = 64
    centres = rng.uniform(8, 56, (N, 2)).astype(np.float32)
    A = sp.lil_matrix((H * W, N))
    for k, (cy, cx) in enumerate(centres):
        A[int(cy) * W + int(cx), k] = 1.0
    save_cells(ws, A, C, C, np.ones_like(C), centres, np.ones(N), np.ones(N), np.ones(N, bool), (H, W))
    ws.record_step("extraction", {}, {})
    np.save(ws.path("bg_max.npy"), rng.random((H, W)).astype(np.float32))
    np.savez_compressed(ws.path(SIGNALS_FILE), times=t, fps=FPS, dff=dff, denoised=C, spikes=spikes,
                        noise=np.ones(N), taus=np.ones(N))
    ws.record_step("signals", {}, {})
    cfg = None
    if behavior:
        cfg = BehaviorConfig(fps=FPS, source="ethovision")
        from miniscope_gui.core.behavior import Shape
        cfg.zones = [Shape("rect", [0, 0, 15, 30], "left").to_dict()]
        save_track(ws, t=t, x=x, y=y, freezing=fz)
        cfg.save(ws)
        ws.record_step("behavior", {"source": "ethovision"}, {})
        save_sync(ws, t * 1000.0, "manual", {})
    return ws, cfg, dict(fz=fz, x=x, y=y, speed=speed)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return make_workspace(tmp_path_factory.mktemp("model") / "ws")


def test_dataset_and_helpers(data):
    ws, cfg, truth = data
    ds = load_dataset(ws, cfg, 0.2)
    assert ds.n_cells == 40 and ds.n_bins == 1800 and abs(ds.fps - 5.0) < 1e-6
    assert ds.has("freezing") and ds.has("speed") and "left" in ds.beh["zones"]
    assert abs(ds.beh["freezing"].mean() - truth["fz"].mean()) < 0.02
    assert load_dataset(ws, None, 0).n_bins == 3600 and not load_dataset(ws, None, 0).beh
    assert list(parse_times("1, 2；3 4.5")) == [1, 2, 3, 4.5]
    with pytest.raises(ValueError):
        parse_times("1, abc")
    q = fdr_bh(np.array([0.001, 0.01, 0.5, np.nan]))
    assert q[0] <= q[1] <= q[2] and np.isnan(q[3])
    f = block_folds(100, 5, 2.0, 5.0, np.arange(100) < 90)
    assert set(np.unique(f)) == {-1, 0, 1, 2, 3, 4}


def test_glm(data):
    ws, cfg, _ = data
    p = glm.GLMParams(event_times=", ".join(str(s) for s in SHOCKS), event_name="电击", n_shuffle=40)
    s = glm.run_glm(None, ws, {"params": p, "cfg": cfg})
    r = load_result(ws, "glm")
    names = [str(g) for g in r["groups"]]
    assert names == ["僵直", "速度", "位置", "区域", "电击"]
    fz, ev, pos = names.index("僵直"), names.index("电击"), names.index("位置")
    assert (r["p"][:6] < 0.05).all() and (r["best_group"][:6] == fz).all()
    assert (r["best_group"][16:20] == ev).all() and (r["p_group"][ev, 16:20] < 0.05).all()
    assert (r["best_group"][10:16] == pos).sum() >= 4
    assert (r["p"][28:] < 0.05).mean() < 0.35 and np.median(r["r2"][28:]) < 0.02
    k = r[f"kernel_{ev}"]
    lags = r[f"kernel_lags_{ev}"]
    assert k[16:20][:, (lags > 0.3) & (lags < 3)].mean() > 5 * abs(k[28:][:, lags < -0.5].mean())
    assert "显著" in describe("glm", s)
    # the cross-validated R^2 must stay sane when a regressor is absent from a training fold
    X = np.zeros((500, 2)); X[10:14, 0] = 1; X[:, 1] = np.sin(np.arange(500) / 10)
    Y = np.random.default_rng(0).normal(size=(500, 3))
    assert glm.RidgeCV(X, block_folds(500, 5, 20, 5)).r2(Y).min() > -0.5


def test_glm_events_only(tmp_path):
    ws, _, _ = make_workspace(tmp_path / "ws", T=3000, behavior=False)
    p = glm.GLMParams(event_times="60, 150, 240", n_shuffle=30)
    glm.run_glm(None, ws, {"params": p, "cfg": None})
    r = load_result(ws, "glm")
    assert (r["p"][16:20] < 0.05).all()
    with pytest.raises(ValueError):
        glm.run_glm(None, ws, {"params": glm.GLMParams(), "cfg": None})


def test_decoder(data):
    ws, cfg, _ = data
    s = decoder.run_decoder(None, ws, {"params": decoder.DecoderParams(target="僵直", n_shuffle=60), "cfg": cfg})
    assert s["metric"] > 0.8 and s["auc"] > 0.9 and s["p"] < 0.05
    r = load_result(ws, "decoder")
    assert r["weights"][:6].mean() > np.abs(r["weights"][28:]).mean()
    assert "平衡准确率" in describe("decoder", s)
    s = decoder.run_decoder(None, ws, {"params": decoder.DecoderParams(target="位置", n_shuffle=40), "cfg": cfg})
    assert s["metric"] > 0.2 and s["p"] < 0.05 and "median_error" in s
    s = decoder.run_decoder(None, ws, {"params": decoder.DecoderParams(target="区域 left", n_shuffle=40), "cfg": cfg})
    assert s["metric"] > 0.55
    # no information -> chance
    rng = np.random.default_rng(0)
    y = np.zeros(2000, bool)
    for a in range(100, 2000, 400):
        y[a:a + 130] = True
    out = decoder.decode(rng.normal(size=(2000, 30)), y, block_folds(2000, 5, 20, 5), True, n_shuffle=40)
    assert abs(out["metric"] - 0.5) < 0.1 and out["p"] > 0.05
    assert decoder.auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == 0.75


def test_ensembles(data):
    ws, cfg, _ = data
    for method in ("ica", "nmf"):
        s = ensembles.run_ensembles(None, ws, {"params": ensembles.EnsembleParams(method=method, n_shuffle=40),
                                               "cfg": cfg})
        r = load_result(ws, "ensembles")
        assert 1 <= s["n_assemblies"] <= 6
        best = max(r["members"], key=lambda m: m[20:28].sum())
        assert best[20:28].sum() >= 6 and best[28:].sum() <= 2, method
        assert r["activation"].shape == (s["n_assemblies"], len(r["times"]))
    # the freezing cells form an assembly that is stronger during freezing
    assert s.get("freezing_up")
    assert "细胞组" in describe("ensembles", s)
    # pure noise: nothing above the shuffle threshold
    rng = np.random.default_rng(1)
    out = ensembles.detect(rng.normal(size=(30, 1500)), ensembles.EnsembleParams(n_shuffle=40))
    assert len(out["patterns"]) <= 1


def test_hmm_recovers_states():
    rng = np.random.default_rng(0)
    T, K, D = 1500, 3, 4
    z = np.zeros(T, int)
    for t in range(1, T):
        z[t] = z[t - 1] if rng.random() < 0.98 else rng.integers(K)
    x = rng.normal(0, 2, (K, D))[z] + rng.normal(0, 1, (T, D))

    def agree(a, b, k):
        from itertools import permutations
        return max(np.mean(np.array(p)[a] == b) for p in permutations(range(k)))
    for kind in hmm.KINDS:
        m, _ = hmm.fit([x], kind, K, n_init=2)
        assert agree(m.viterbi(x), z, K) > 0.95, kind
    # states that differ only in their dynamics: needs the autoregressive model
    A = [0.95 * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) for a in (0.3, -0.3)]
    z = np.zeros(T, int)
    for t in range(1, T):
        z[t] = z[t - 1] if rng.random() < 0.98 else 1 - z[t - 1]
    x = np.zeros((T, 2))
    for t in range(1, T):
        x[t] = A[z[t]] @ x[t - 1] + rng.normal(0, 0.3, 2)
    m, _ = hmm.fit([x], "arhmm", 2, n_init=2)
    assert agree(m.viterbi(x), z, 2) > 0.9
    train, test = hmm.split_blocks(x)
    assert sum(map(len, train)) + sum(map(len, test)) == T and m.score(test) > hmm.fit([x], "hmm", 2, n_init=1)[0].score(test)
    assert list(hmm.dwell_times(np.array([0, 0, 1, 1, 1, 0]), 2)) == [1.5, 3.0]


def test_hmm_run(data):
    ws, cfg, _ = data
    for kind in hmm.KINDS:
        p = hmm.HMMParams(kind=kind, n_states=3, n_iter=25, n_init=1, compare_k=kind == "hmm", k_max=4)
        s = hmm.run_hmm(None, ws, {"params": p, "cfg": cfg})
        r = load_result(ws, "hmm")
        assert r["states"].max() <= 2 and abs(r["posterior"].sum(1) - 1).max() < 1e-6
        assert np.allclose(r["trans"].sum(1), 1) and abs(sum(s["occupancy"]) - 1) < 1e-9
        assert "freeze_frac" in s and "状态" in describe("hmm", s)
    assert "cv_ll" not in r
    # a hidden state should line up with freezing (6 of 40 cells are strongly freezing-modulated)
    hmm.run_hmm(None, ws, {"params": hmm.HMMParams(n_states=3, compare_k=True, k_max=3, n_init=1), "cfg": cfg})
    r = load_result(ws, "hmm")
    assert len(r["cv_ll"]) == 2 and r["freeze_frac"].max() > 2 * r["summary"]["freeze_overall"]


def test_network_measures():
    N = 20
    A = np.zeros((N, N), bool)
    for i in range(N):
        for d in (1, 2):
            A[i, (i + d) % N] = A[(i + d) % N, i] = True
    assert np.allclose(network.clustering(A), 0.5)
    L, eff, giant = network.path_measures(A)
    assert abs(L - 55 / 19) < 1e-9 and giant == 1.0
    B = np.zeros((10, 10), bool)
    B[:5, :5] = True; B[5:, 5:] = True
    np.fill_diagonal(B, False)
    B[4, 5] = B[5, 4] = True
    lab = network.communities(B)
    assert len(set(lab[:5])) == 1 and len(set(lab[5:])) == 1 and lab[0] != lab[9]
    assert abs(network.modularity(B, lab) - (2 * (20 / 42 - (21 / 42) ** 2))) < 1e-9
    C = np.zeros((4, 4), bool); C[0, 1] = C[1, 0] = True
    assert network.path_measures(C)[2] == 0.5 and (network.communities(C)[2:] == -1).all()


def test_network_run(data):
    ws, cfg, _ = data
    s = network.run_network(None, ws, {"params": network.NetworkParams(n_shuffle=20), "cfg": cfg})
    r = load_result(ws, "network")
    assert r["adj"][20:28, 20:28].sum() >= 40           # the assembly is densely connected
    assert r["adj"][28:, 28:].mean() < 0.02              # noise cells are not
    assert len(set(r["community"][20:28])) == 1 and r["community"][20] >= 0
    assert (r["adj"] == r["adj"].T).all() and s["n_edges"] == r["adj"].sum() // 2
    assert "by_state" in s and "枢纽" in describe("network", s)
    s2 = network.run_network(None, ws, {"params": network.NetworkParams(threshold="density", density=0.1),
                                        "cfg": None})
    assert abs(s2["density"] - 0.1) < 0.01 and "by_state" not in s2


def test_info(data):
    ws, cfg, _ = data
    s = info.run_info(None, ws, {"params": info.InfoParams(n_shuffle=200), "cfg": cfg})
    r = load_result(ws, "info")
    assert r["place"][10:16].sum() >= 5 and r["place"][28:].mean() < 0.25
    assert r["si"][10:16].mean() > 2 * r["si"][28:].mean()
    names = [str(n) for n in r["mi_names"]]
    assert names == ["僵直", "区域"]
    assert (r["mi_p_0"][:6] < 0.05).all() and (r["mi_p_0"][28:] < 0.05).mean() < 0.25
    assert r["mi_0"][:6].mean() > 10 * r["mi_0"][28:].mean()
    assert r["rate_maps"].shape == (40, 8, 8) and "空间信息" in describe("info", s)
    # independent variables carry no information after bias correction
    rng = np.random.default_rng(0)
    out = info.mutual_information((rng.random((5, 4000)) < 0.1).astype(float), rng.integers(0, 2, 4000), n_shuffle=100)
    assert out["mi"].max() < 0.002
    assert list(info.discretize(np.array([[0, 1, 2, 3, 0, 5.0]]), 3)[0]) == [0, 1, 1, 2, 0, 2]


def test_embedding_gradients_and_training():
    rng = np.random.default_rng(0)
    enc = embedding.Encoder(6, 5, 3, rng)
    enc.p = [a.astype(np.float64) for a in enc.p]
    X = rng.normal(size=(12, 6))

    def loss():
        F = enc.forward(X, cache=True)
        return embedding.infonce(F[:4], F[4:8], F[8:], 0.7)
    _, gR, gP, gN = loss()
    grads = enc.backward(np.concatenate([gR, gP, gN]))
    for i, par in enumerate(enc.p):
        ix = tuple(rng.integers(s) for s in par.shape)
        old = par[ix]
        par[ix] = old + 1e-5; lp = loss()[0]
        par[ix] = old - 1e-5; lm = loss()[0]
        par[ix] = old
        assert abs((lp - lm) / 2e-5 - grads[i][ix]) < 1e-6
    # a hidden circular variable drives the population: the embedding must recover it
    T, N = 2500, 60
    th = np.cumsum(rng.normal(0, 0.15, T))
    pos = np.column_stack([np.cos(th), np.sin(th)])
    Z = pos @ rng.normal(0, 1, (N, 2)).T + rng.normal(0, 2, (T, N))
    from miniscope_gui.core.modeling.common import pca
    sc, _, _ = pca(Z, 30)
    Xw = embedding.windows((sc / sc[:, 0].std()).astype(np.float32), 5)
    fold = block_folds(T, 5, 20, 5)
    tr, te = np.where(fold > 0)[0], np.where(fold == 0)[0]
    p = embedding.EmbedParams(steps=600)
    E, losses = embedding.train(Xw, tr, "behavior", None, pos, p)
    assert np.allclose(np.linalg.norm(E, axis=1), 1, atol=1e-4) and losses[-50:].mean() < losses[:50].mean()
    good = embedding._decode_score(E, tr, te, None, pos)["continuous_r"]
    shifted = pos.copy(); shifted[tr] = np.roll(pos[tr], 900, axis=0)
    Ec, _ = embedding.train(Xw, tr, "behavior", None, shifted, p)
    assert good > 0.85 and good > embedding._decode_score(Ec, tr, te, None, pos)["continuous_r"] + 0.2


def test_embedding_run_and_export(data, tmp_path):
    ws, cfg, _ = data
    s = embedding.run_embedding(None, ws, {"params": embedding.EmbedParams(steps=400), "cfg": cfg})
    assert s["labels"] == ["僵直", "位置"] and s["decode"]["freezing_bacc"] > 0.7
    assert "decode_control" in s and "嵌入" in describe("embedding", s)
    r = load_result(ws, "embedding")
    assert r["embedding"].shape == (1800, 3) and 0.15 < r["test"].mean() < 0.25
    s = embedding.run_embedding(None, ws, {"params": embedding.EmbedParams(steps=200, mode="time"), "cfg": None})
    assert s["mode"] == "time" and "decode" not in s
    embedding.export_for_cebra(ws, cfg, tmp_path / "cebra")
    z = np.load(tmp_path / "cebra" / "cebra_input.npz")
    assert z["neural"].shape == (1800, 40) and z["continuous"].shape[1] == 3 and z["discrete"].shape == (1800,)
    assert (tmp_path / "cebra" / "run_cebra.py").exists()


def test_kinetics_on_generic_workspace(data):
    from miniscope_gui.core.modeling import kinetics
    ws, _, _ = data
    s = kinetics.run_kinetics(None, ws, {"params": kinetics.KineticsParams(n_boot=10, n_null=3)})
    assert s["n_events"] > 100 and "【细胞层面】" in describe("kinetics", s)
    # one indicator for every cell here: measured decays sit near the simulated 0.8 s
    assert 0.5 < s["median"]["decay_tau_s"] < 1.3


def test_export_all_and_registry(data, tmp_path):
    ws, cfg, _ = data
    for key in ANALYSES:        # every analysis ran in the tests above
        assert ws.step(f"model_{key}"), key
        assert 2 <= len(VIEWS[key]) <= 5
    done = export_all(ws, tmp_path / "out")
    assert set(done) == set(ANALYSES)
    for f in ("glm_encoding.csv", "decoder_timecourse.csv", "ensemble_weights.csv", "embedding.csv",
              "hmm_states.csv", "network_edges.csv", "information.csv", "kinetics_events.csv",
              "kinetics_cells.csv", "modeling_summary.txt"):
        assert (tmp_path / "out" / f).stat().st_size > 0, f
    head = (tmp_path / "out" / "glm_encoding.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    assert head.startswith("cell_id,cv_r2,p,q_fdr,best_variable")

def test_position_decoder_axes_and_export(tmp_path):
    rng = np.random.default_rng(35)
    x = rng.normal(size=(400, 2))
    fold = np.arange(400) // 80
    result = decoder.decode(x, x.copy(), fold, False, n_pc=2, n_shuffle=3)
    weights = result["weights"]
    assert weights.shape == (2, 2)
    assert np.all(np.diag(weights) > 10 * np.abs(weights[[0, 1], [1, 0]]))
    result.update(ids=np.arange(2), summary={"target": "位置"}, times=np.arange(400),
                  valid=fold >= 0, truth=x)
    decoder.export_decoder(None, result, tmp_path)
    path = tmp_path / "decoder_cell_weights.csv"
    assert path.read_text(encoding="utf-8-sig").splitlines()[0] == "cell_id,weight_x,weight_y"
    result["weights"] = weights[:, 0]
    decoder.export_decoder(None, result, tmp_path)
    assert path.read_text(encoding="utf-8-sig").splitlines()[0] == "cell_id,weight_x"


def test_decoder_fold_training_independent_of_test_variance():
    x = np.random.default_rng(5).normal(size=(100, 4))
    fold = np.arange(100) // 20
    base = decoder._Folds(x, fold, 2).items[0]
    x[fold == 0, 0] *= 1000
    changed = decoder._Folds(x, fold, 2).items[0]
    # Training PCA must also be invariant to a change of units in one cell.
    x[:, 1] *= 100
    rescaled = decoder._Folds(x, fold, 2).items[0]
    np.testing.assert_allclose(np.abs(base[4]), np.abs(rescaled[4]))
    for i in (2, 4, 5):
        np.testing.assert_array_equal(base[i], changed[i])
