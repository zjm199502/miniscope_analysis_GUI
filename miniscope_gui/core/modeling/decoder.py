"""Decoder: predict behaviour from the population activity.

Binary targets (freezing, being inside a zone) use L2-regularised logistic
regression on the principal components of the training data; continuous
targets (speed, position) use ridge regression. Performance is measured on
held-out blocks of time and compared with circularly shifted labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..workspace import Workspace
from .common import (block_folds, check_stop, fill_nan, load_dataset, save_result,
                     shift_amounts, write_csv, zscore)


@dataclass
class DecoderParams:
    bin_s: float = 0.2
    target: str = "僵直"           # 僵直 | 速度 | 位置 | 区域 <name>
    signal: str = "dff"            # dff | events
    n_pc: int = 50
    l2: float = 1.0
    n_folds: int = 5
    n_shuffle: int = 200
    seed: int = 0


def logistic_fit(X, y, l2=1.0, w=None, n_iter=30):
    """Newton / IRLS for class-weighted L2 logistic regression. Returns (coef, intercept)."""
    n, d = X.shape
    w = np.ones(n) if w is None else w
    Xa = np.column_stack([X, np.ones(n)])
    beta = np.zeros(d + 1)
    R = np.eye(d + 1) * l2
    R[-1, -1] = 1e-8
    for _ in range(n_iter):
        z = np.clip(Xa @ beta, -30, 30)
        pr = 1.0 / (1.0 + np.exp(-z))
        g = Xa.T @ (w * (pr - y)) + R @ beta
        H = (Xa * (w * pr * (1 - pr))[:, None]).T @ Xa + R
        step = np.linalg.solve(H, g)
        beta -= step
        if np.abs(step).max() < 1e-6:
            break
    return beta[:-1], beta[-1]


def auc(score, y) -> float:
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = np.asarray(score)[order]
    ranks = np.empty(len(s))
    # average ranks for ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def balanced_accuracy(pred, y) -> float:
    y = np.asarray(y, bool)
    pred = np.asarray(pred, bool)
    if y.all() or not y.any():
        return float("nan")
    return float(0.5 * (pred[y].mean() + (~pred[~y]).mean()))


class _Folds:
    """Per-fold PCA of the features: computed once, reused for every shuffle."""

    def __init__(self, X, fold, n_pc):
        self.items = []
        for f in np.unique(fold[fold >= 0]):
            tr, te = np.where((fold >= 0) & (fold != f))[0], np.where(fold == f)[0]
            mu = X[tr].mean(0)
            U, s, Vt = np.linalg.svd(X[tr] - mu, full_matrices=False)
            k = int(min(n_pc, (s > 1e-9 * s[0]).sum()))
            scale = s[:k] / np.sqrt(len(tr))
            Ptr = U[:, :k] * np.sqrt(len(tr))                  # unit-variance scores
            Pte = (X[te] - mu) @ Vt[:k].T / scale
            self.items.append((tr, te, Ptr, Pte, Vt[:k], scale))


def _cv_binary(folds: _Folds, y, l2):
    prob = np.full(len(y), np.nan)
    for tr, te, Ptr, Pte, _, _ in folds.items:
        yt = y[tr].astype(float)
        if yt.min() == yt.max():
            prob[te] = yt.mean()
            continue
        f1 = yt.mean()
        w = np.where(yt > 0, 0.5 / f1, 0.5 / (1 - f1))
        c, b = logistic_fit(Ptr, yt, l2, w)
        prob[te] = 1.0 / (1.0 + np.exp(-np.clip(Pte @ c + b, -30, 30)))
    return prob


def _cv_continuous(folds: _Folds, Y, l2):
    pred = np.full(Y.shape, np.nan)
    for tr, te, Ptr, Pte, _, _ in folds.items:
        mu = Y[tr].mean(0)
        G = Ptr.T @ Ptr + l2 * np.eye(Ptr.shape[1])
        pred[te] = Pte @ np.linalg.solve(G, Ptr.T @ (Y[tr] - mu)) + mu
    return pred


def _r(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def decode(X, y, fold, binary: bool, n_pc=30, l2=1.0, n_shuffle=200, seed=0,
           progress=None, should_stop=None):
    """X (T, N) features; y (T,) bool or (T, k) float. Only rows with fold >= 0 are used."""
    progress = progress or (lambda f, m: None)
    rng = np.random.default_rng(seed)
    use = fold >= 0
    folds = _Folds(X, fold, n_pc)
    T = int(use.sum())
    idx = np.where(use)[0]

    if binary:
        def score(lab):
            pr = _cv_binary(folds, lab, l2)
            return balanced_accuracy(pr[use] > 0.5, lab[use]), auc(pr[use], lab[use]), pr
        acc, a, pred = score(y)
        metric = acc
    else:
        Y = y if y.ndim == 2 else y[:, None]

        def score(lab):
            pr = _cv_continuous(folds, lab, l2)
            return float(np.mean([_r(pr[use, j], lab[use, j]) for j in range(lab.shape[1])])), None, pr
        metric, a, pred = score(Y)
    null = np.empty(n_shuffle)
    for i, k in enumerate(shift_amounts(rng, T, n_shuffle)):
        check_stop(should_stop)
        ys = y.copy() if binary else Y.copy()
        ys[idx] = np.roll(ys[idx], k, axis=0)
        null[i] = score(ys)[0]
        if i % 10 == 9:
            progress(0.2 + 0.7 * (i + 1) / n_shuffle, f"打乱检验 {i + 1}/{n_shuffle}")
    pval = float((1 + (null >= metric).sum()) / (n_shuffle + 1))
    # weights of every cell from a fit on all usable data
    mu = X[use].mean(0)
    U, s, Vt = np.linalg.svd(X[use] - mu, full_matrices=False)
    k = int(min(n_pc, (s > 1e-9 * s[0]).sum()))
    P = U[:, :k] * np.sqrt(T)
    scale = s[:k] / np.sqrt(T)
    if binary:
        yt = y[use].astype(float)
        f1 = yt.mean()
        c, _ = logistic_fit(P, yt, l2, np.where(yt > 0, 0.5 / f1, 0.5 / (1 - f1)))
        weights = Vt[:k].T @ (c / scale)
    else:
        Yc = Y[use] - Y[use].mean(0)
        C = np.linalg.solve(P.T @ P + l2 * np.eye(k), P.T @ Yc)
        weights = (Vt[:k].T @ (C / scale[:, None]))[:, 0]
    return dict(metric=metric, auc=a, pred=pred, null=null, p=pval, weights=weights)


def targets(ds) -> list[str]:
    out = []
    if ds.has("freezing"):
        out.append("僵直")
    if ds.has("speed"):
        out.append("速度")
    if ds.has("x_u"):
        out.append("位置")
    out += [f"区域 {k}" for k in ds.beh.get("zones", {})]
    return out


def run_decoder(session, ws: Workspace, args: dict, progress=None, should_stop=None) -> dict:
    progress = progress or (lambda f, m: None)
    p: DecoderParams = args["params"]
    progress(0.02, "读取数据…")
    ds = load_dataset(ws, args.get("cfg"), p.bin_s)
    if not ds.beh:
        raise ValueError("解码需要行为数据：请先完成第 7 步（行为）和第 8 步（同步）")
    b = ds.beh
    valid = ds.valid.copy()
    if p.target == "僵直":
        if not ds.has("freezing"):
            raise ValueError("行为数据里没有僵直信息")
        y, binary = b["freezing"], True
    elif p.target.startswith("区域 "):
        y, binary = b["zones"][p.target[3:]], True
    elif p.target == "速度":
        valid &= np.isfinite(b["speed"])
        y, binary = fill_nan(b["speed"]), False
    elif p.target == "位置":
        valid &= np.isfinite(b["x_u"]) & np.isfinite(b["y_u"])
        y, binary = np.column_stack([fill_nan(b["x_u"]), fill_nan(b["y_u"])]), False
    else:
        raise ValueError(f"未知的解码目标：{p.target}")
    if binary:
        frac = float(y[valid].mean()) if valid.any() else 0.0
        if not 0.02 <= frac <= 0.98:
            raise ValueError(f"「{p.target}」在记录中占 {100 * frac:.1f}%，两种状态都需要有足够的时间才能解码")
    src = ds.dff if p.signal == "dff" else ds.events
    X = zscore(src, axis=1).T
    fold = block_folds(ds.n_bins, p.n_folds, 20.0, ds.fps, valid)
    progress(0.1, "训练解码器…")
    res = decode(X, y, fold, binary, p.n_pc, p.l2, p.n_shuffle, p.seed, progress, should_stop)
    pred = res["pred"]
    arrays = dict(ids=ds.ids, centres=ds.centres, times=ds.times, valid=valid, null=res["null"],
                  weights=res["weights"], truth=np.asarray(y, float), pred=pred)
    summary = {"target": p.target, "binary": binary, "metric": res["metric"], "auc": res["auc"],
               "p": res["p"], "null_mean": float(np.mean(res["null"])),
               "null_95": float(np.percentile(res["null"], 95)), "n_cells": int(ds.n_cells),
               "n_bins": int(valid.sum()), "unit": b.get("unit", "")}
    if p.target == "位置":
        err = np.hypot(pred[valid, 0] - y[valid, 0], pred[valid, 1] - y[valid, 1])
        summary["median_error"] = float(np.median(err))
    save_result(ws, "decoder", arrays, summary, asdict(p))
    progress(1.0, "解码完成")
    return summary


def export_decoder(ws: Workspace, res: dict, folder):
    write_csv(folder / "decoder_cell_weights.csv", {"cell_id": res["ids"], "weight": res["weights"]})
    cols = {"time_s": res["times"], "used": res["valid"].astype(int)}
    truth, pred = res["truth"], res["pred"]
    if truth.ndim == 1:
        cols["truth"], cols["decoded"] = truth, pred if pred.ndim == 1 else pred[:, 0]
    else:
        for j, n in enumerate("xy"):
            cols[f"truth_{n}"], cols[f"decoded_{n}"] = truth[:, j], pred[:, j]
    return write_csv(folder / "decoder_timecourse.csv", cols)
