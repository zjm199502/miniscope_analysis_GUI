"""CEBRA-style contrastive embedding of the population activity.

A small neural network maps a short window of population activity to a point
on a low-dimensional sphere. It is trained with the InfoNCE loss so that
"positive" pairs end up close together and random pairs far apart
(Schneider, Lee & Mathis 2023, CEBRA):

  time       positives are samples a few bins apart (CEBRA-Time)
  behaviour  positives are samples with similar behaviour (CEBRA-Behaviour):
             same discrete state (freezing) and / or nearby continuous
             variables (position, speed)

This is a compact numpy re-implementation of the idea so that the app needs
no deep-learning library. It is not the official `cebra` package; for
publication-grade embeddings export the data and run the official package
(see export_for_cebra).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..workspace import Workspace
from .common import (block_folds, check_stop, fill_nan, load_dataset, pca, save_result,
                     write_csv, zscore)


@dataclass
class EmbedParams:
    bin_s: float = 0.2
    mode: str = "behavior"         # behavior | time
    use_freezing: bool = True
    use_position: bool = True
    use_speed: bool = False
    out_dim: int = 3
    n_input_pc: int = 50
    window: int = 5                # bins of activity seen by the network
    time_offset: int = 5           # bins between a sample and its positive
    hidden: int = 64
    steps: int = 2000
    batch: int = 512
    lr: float = 3e-3
    temperature: float = 1.0
    holdout: bool = True
    seed: int = 0


class Encoder:
    """in -> tanh(hidden) -> tanh(hidden) -> out, L2-normalised. Manual backprop + Adam."""

    def __init__(self, d_in, hidden, d_out, rng):
        def lin(a, b):
            return (rng.normal(0, 1 / np.sqrt(a), (a, b)).astype(np.float32), np.zeros(b, np.float32))
        self.p = [*lin(d_in, hidden), *lin(hidden, hidden), *lin(hidden, d_out)]
        self.m = [np.zeros_like(a) for a in self.p]
        self.v = [np.zeros_like(a) for a in self.p]
        self.t = 0

    def forward(self, X, cache=False):
        W1, b1, W2, b2, W3, b3 = self.p
        h1 = np.tanh(X @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        h = h2 @ W3 + b3
        n = np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-8)
        f = h / n
        if cache:
            self._c = (X, h1, h2, n, f)
        return f

    def backward(self, gf):
        """gf = dLoss/d(normalised output). Returns gradients for all parameters."""
        X, h1, h2, n, f = self._c
        W1, b1, W2, b2, W3, b3 = self.p
        gh = (gf - f * (f * gf).sum(1, keepdims=True)) / n
        g2 = (gh @ W3.T) * (1 - h2 ** 2)
        g1 = (g2 @ W2.T) * (1 - h1 ** 2)
        return [X.T @ g1, g1.sum(0), h1.T @ g2, g2.sum(0), h2.T @ gh, gh.sum(0)]

    def step(self, grads, lr):
        self.t += 1
        for i, g in enumerate(grads):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            self.p[i] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)


def infonce(R, P, Ng, tau=1.0):
    """Loss and its gradients with respect to the three (normalised) embeddings."""
    B = len(R)
    S = R @ Ng.T / tau
    m = S.max(1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(S - m).sum(1))
    pos = (R * P).sum(1) / tau
    loss = float(np.mean(lse - pos))
    Q = np.exp(S - lse[:, None])
    gR = (Q @ Ng - P) / (tau * B)
    gP = -R / (tau * B)
    gN = Q.T @ R / (tau * B)
    return loss, gR, gP, gN


def windows(Z, window):
    """Z (T, D) -> (T, D * window): each row is the activity around that bin."""
    T = len(Z)
    half = window // 2
    idx = np.clip(np.arange(T)[:, None] + np.arange(-half, window - half)[None, :], 0, T - 1)
    return Z[idx].reshape(T, -1)


def _positive_sampler(mode, T_idx, discrete, cont, offset, rng):
    """Returns f(ref indices) -> positive indices (both index into the full time axis)."""
    n = len(T_idx)
    pos_of = {int(t): i for i, t in enumerate(T_idx)}
    if mode == "time" or (discrete is None and cont is None):
        allowed = np.zeros(int(T_idx.max()) + 1, bool)
        allowed[T_idx] = True

        def sample(ref):
            d = rng.integers(1, offset + 1, len(ref)) * rng.choice([-1, 1], len(ref))
            p = np.clip(ref + d, 0, len(allowed) - 1)
            bad = ~allowed[p]
            p[bad] = ref[bad]
            return p
        return sample
    d = discrete[T_idx] if discrete is not None else None
    c = cont[T_idx] if cont is not None else None
    if c is not None:
        j = np.arange(n - offset)
        consecutive = T_idx[j + offset] - T_idx[j] == offset
        j = j[consecutive] if consecutive.any() else j
        diffs = c[j + offset] - c[j]
        c2 = (c ** 2).sum(1)

    def sample(ref):
        r = np.array([pos_of[int(t)] for t in ref])
        if c is None:
            out = np.empty(len(r), int)
            for i, k in enumerate(r):
                same = np.flatnonzero(d == d[k])
                out[i] = same[rng.integers(len(same))]
            return T_idx[out]
        target = c[r] + diffs[rng.integers(len(diffs), size=len(r))]
        dist = (target ** 2).sum(1)[:, None] - 2 * target @ c.T + c2[None, :]
        if d is not None:
            dist = np.where(d[r][:, None] == d[None, :], dist, np.inf)
        dist[np.arange(len(r)), r] = np.inf            # not itself
        return T_idx[dist.argmin(1)]
    return sample


def train(X, train_idx, mode="time", discrete=None, cont=None, p: EmbedParams = None,
          progress=None, should_stop=None):
    """X (T, D_in) network inputs. Returns embedding of all rows (T, out_dim) and the loss curve."""
    p = p or EmbedParams()
    rng = np.random.default_rng(p.seed)
    enc = Encoder(X.shape[1], p.hidden, p.out_dim, rng)
    sampler = _positive_sampler(mode, train_idx, discrete, cont, p.time_offset, rng)
    B = int(min(p.batch, len(train_idx)))
    losses = np.empty(p.steps, np.float32)
    for it in range(p.steps):
        if it % 50 == 0:
            check_stop(should_stop)
            if progress:
                progress(it / p.steps)
        ref = train_idx[rng.integers(len(train_idx), size=B)]
        pos = sampler(ref)
        neg = train_idx[rng.integers(len(train_idx), size=B)]
        F = enc.forward(X[np.concatenate([ref, pos, neg])], cache=True)
        loss, gR, gP, gN = infonce(F[:B], F[B:2 * B], F[2 * B:], p.temperature)
        enc.step(enc.backward(np.concatenate([gR, gP, gN]).astype(np.float32)), p.lr)
        losses[it] = loss
    return enc.forward(X), losses


def knn_decode(E, train_idx, test_idx, y, k=15):
    """Predict y on the test rows from the k nearest training rows (cosine similarity)."""
    S = E[test_idx] @ E[train_idx].T
    k = int(min(k, len(train_idx)))
    nn = np.argpartition(-S, k - 1, axis=1)[:, :k]
    return y[train_idx][nn].mean(1)


def _decode_score(E, tr, te, discrete, cont):
    from .decoder import balanced_accuracy
    out = {}
    if discrete is not None and 0 < discrete[te].mean() < 1 and 0 < discrete[tr].mean() < 1:
        pr = knn_decode(E, tr, te, discrete.astype(float))
        out["freezing_bacc"] = balanced_accuracy(pr > discrete[tr].mean(), discrete[te] > 0)
    if cont is not None:
        pr = knn_decode(E, tr, te, cont)
        rs = [np.corrcoef(pr[:, j], cont[te, j])[0, 1] for j in range(cont.shape[1]) if cont[te, j].std() > 0]
        if rs:
            out["continuous_r"] = float(np.mean(rs))
    return out


def run_embedding(session, ws: Workspace, args: dict, progress=None, should_stop=None) -> dict:
    progress = progress or (lambda f, m: None)
    p: EmbedParams = args["params"]
    ds = load_dataset(ws, args.get("cfg"), p.bin_s)
    T = ds.n_bins
    Z = zscore(ds.dff, axis=1).T
    sc, _, _ = pca(Z, min(p.n_input_pc, ds.n_cells))
    X = windows((sc / max(sc[:, 0].std(), 1e-12)).astype(np.float32), p.window)
    b = ds.beh
    discrete = cont = None
    names = []
    valid = np.ones(T, bool)
    if b:
        valid = ds.valid.copy()
        if p.use_freezing and ds.has("freezing") and 0 < b["freezing"][valid].mean() < 1:
            discrete = b["freezing"].astype(int)
            names.append("僵直")
        cs = []
        if p.use_position and ds.has("x_u") and np.nanstd(b["x_u"][valid]) > 0:
            cs += [fill_nan(np.where(valid, b["x_u"], np.nan)), fill_nan(np.where(valid, b["y_u"], np.nan))]
            names.append("位置")
        if p.use_speed and ds.has("speed"):
            cs.append(fill_nan(np.where(valid, b["speed"], np.nan)))
            names.append("速度")
        if cs:
            cont = zscore(np.column_stack(cs), axis=0)
    mode = p.mode
    if mode == "behavior" and discrete is None and cont is None:
        raise ValueError("行为模式需要行为数据（僵直、位置或速度）。请先完成行为和同步，或改用时间模式。")
    use = valid if mode == "behavior" else np.ones(T, bool)
    fold = block_folds(T, 5, 20.0, ds.fps, use)
    te = np.where(fold == 0)[0] if p.holdout else np.array([], int)
    tr = np.where((fold > 0) if p.holdout else (fold >= 0))[0]
    if len(tr) < 100:
        raise ValueError("可用的数据太短，无法训练嵌入")
    has_ctrl = p.holdout and mode == "behavior"
    span = 0.45 if has_ctrl else 0.9
    E, losses = train(X, tr, mode, discrete, cont, p, lambda f: progress(0.03 + span * f, "训练嵌入…"), should_stop)
    arrays = dict(times=ds.times, embedding=E, loss=losses, valid=use, test=np.isin(np.arange(T), te))
    summary = {"mode": mode, "labels": names, "out_dim": p.out_dim, "final_loss": float(losses[-50:].mean()),
               "first_loss": float(losses[:50].mean()), "n_cells": int(ds.n_cells)}
    if discrete is not None:
        arrays["freezing"] = discrete
    if b:
        for k in ("x_u", "y_u", "speed"):
            if ds.has(k):
                arrays[k] = b[k]
    if p.holdout and len(te) and b:
        vt = te[valid[te]]
        vtr = tr[valid[tr]]
        summary["decode"] = _decode_score(E, vtr, vt, discrete, cont)
        if has_ctrl:
            # control: same training with the behaviour labels shifted in time
            rng = np.random.default_rng(p.seed + 1)
            k = int(rng.integers(len(tr) // 4, 3 * len(tr) // 4))
            d2, c2 = discrete, cont
            if discrete is not None:
                d2 = discrete.copy(); d2[tr] = np.roll(discrete[tr], k)
            if cont is not None:
                c2 = cont.copy(); c2[tr] = np.roll(cont[tr], k, axis=0)
            Ec, _ = train(X, tr, mode, d2, c2, p, lambda f: progress(0.5 + 0.45 * f, "训练对照（打乱标签）…"),
                          should_stop)
            summary["decode_control"] = _decode_score(Ec, vtr, vt, discrete, cont)
    save_result(ws, "embedding", arrays, summary, asdict(p))
    progress(1.0, "嵌入完成")
    return summary


def export_embedding(ws: Workspace, res: dict, folder):
    E = res["embedding"]
    cols = {"time_s": res["times"]}
    for j in range(E.shape[1]):
        cols[f"dim_{j + 1}"] = E[:, j]
    cols["held_out"] = res["test"].astype(int)
    for k in ("freezing", "x_u", "y_u", "speed"):
        if k in res:
            cols[k] = res[k].astype(float)
    return write_csv(folder / "embedding.csv", cols)


CEBRA_SCRIPT = '''"""Run the official CEBRA package on the data exported by Miniscope Analysis.

    pip install cebra
    python run_cebra.py
"""
import numpy as np
import cebra

d = np.load("cebra_input.npz")
neural = d["neural"]                      # (time bins, cells), z-scored dF/F
model = cebra.CEBRA(model_architecture="offset10-model", batch_size=512, learning_rate=3e-4,
                    temperature=1, output_dimension=3, max_iterations=5000,
                    distance="cosine", conditional="time_delta", time_offsets=10, verbose=True)
labels = [d[k] for k in ("continuous", "discrete") if k in d.files]
model.fit(neural, *labels) if labels else model.fit(neural)
emb = model.transform(neural)
np.save("cebra_embedding.npy", emb)
print("saved cebra_embedding.npy", emb.shape)
'''


def export_for_cebra(ws: Workspace, cfg, folder, bin_s=0.2):
    """Write cebra_input.npz + run_cebra.py for the official package."""
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(ws, cfg, bin_s)
    v = ds.valid
    out = {"neural": zscore(ds.dff, axis=1).T[v].astype(np.float32), "times": ds.times[v],
           "cell_ids": ds.ids}
    cs = [fill_nan(ds.beh[k][v]) for k in ("x_u", "y_u", "speed") if ds.has(k)]
    if cs:
        out["continuous"] = np.column_stack(cs).astype(np.float32)
    if ds.has("freezing"):
        out["discrete"] = ds.beh["freezing"][v].astype(np.int64)
    np.savez_compressed(folder / "cebra_input.npz", **out)
    (folder / "run_cebra.py").write_text(CEBRA_SCRIPT, encoding="utf-8")
    return folder / "cebra_input.npz"
