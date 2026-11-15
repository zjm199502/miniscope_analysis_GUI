"""Encoding model (GLM): which behaviour variables explain each cell's activity.

Each cell's z-scored dF/F is predicted from behaviour regressors with ridge
regression. Performance is the cross-validated R^2 on held-out blocks of time.
The unique contribution of a variable is the drop in cross-validated R^2 when
that variable is left out of the model (Musall et al. 2019; Engelhard et al.
2019). Significance comes from circular time shifts, which keep the slow
autocorrelation of calcium signals and behaviour intact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..workspace import Workspace
from .common import (Dataset, block_folds, check_stop, fdr_bh, fill_nan, load_dataset,
                     parse_times, save_result, shift_amounts, write_csv, zscore)

LAMBDAS = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])


@dataclass
class GLMParams:
    bin_s: float = 0.2
    use_freezing: bool = True
    use_speed: bool = True
    use_position: bool = True
    use_zones: bool = True
    event_times: str = ""          # e.g. shock times in miniscope seconds
    event_name: str = "事件"
    kernel_pre_s: float = 2.0
    kernel_post_s: float = 6.0
    n_basis: int = 6
    position_grid: int = 3
    n_folds: int = 5
    n_shuffle: int = 100
    seed: int = 0


# ------------------------------------------------------------ design matrix
def raised_cosine_basis(n_pre: int, n_post: int, n_basis: int) -> np.ndarray:
    """(n_pre + n_post + 1, n_basis) smooth bumps tiling the lags -n_pre..n_post."""
    lags = np.arange(-n_pre, n_post + 1, dtype=float)
    n_basis = int(max(1, min(n_basis, len(lags))))
    if n_basis == 1:
        return np.ones((len(lags), 1))
    centres = np.linspace(lags[0], lags[-1], n_basis)
    d = centres[1] - centres[0]
    arg = np.clip((lags[:, None] - centres[None, :]) * np.pi / (2 * d), -np.pi, np.pi)
    return 0.5 * (1 + np.cos(arg))


def event_regressors(T: int, onsets: np.ndarray, basis: np.ndarray, n_pre: int) -> np.ndarray:
    """Convolve an impulse train at the onset bins with each basis function."""
    imp = np.zeros(T)
    on = np.asarray(onsets, int)
    on = on[(on >= 0) & (on < T)]
    np.add.at(imp, on, 1.0)
    out = np.empty((T, basis.shape[1]))
    for j in range(basis.shape[1]):
        full = np.convolve(imp, basis[:, j])            # index 0 = lag -n_pre
        out[:, j] = full[n_pre:n_pre + T]
    return out


def _onsets(mask):
    m = np.concatenate([[False], np.asarray(mask, bool)])
    d = np.diff(m.astype(int))
    return np.where(d == 1)[0], np.where(d == -1)[0]


def build_design(ds: Dataset, p: GLMParams):
    """Returns X (T, P), groups [(name, column indices)], kernels {name: (cols, basis, lags_s)}."""
    T = ds.n_bins
    fps = ds.fps
    n_pre = int(round(p.kernel_pre_s * fps))
    n_post = int(round(p.kernel_post_s * fps))
    basis = raised_cosine_basis(n_pre, n_post, p.n_basis)
    lags_s = np.arange(-n_pre, n_post + 1) / fps
    cols, groups, kernels = [], [], {}

    def add(name, M, kernel=None):
        M = np.asarray(M, float)
        if M.ndim == 1:
            M = M[:, None]
        start = sum(c.shape[1] for c in cols)
        cols.append(M)
        idx = np.arange(start, start + M.shape[1])
        groups.append((name, idx))
        if kernel is not None:
            kernels[name] = (idx[kernel], basis, lags_s)

    b = ds.beh
    if p.use_freezing and ds.has("freezing") and b["freezing"].any():
        fz = b["freezing"].astype(float)
        on, off = _onsets(b["freezing"])
        M = np.column_stack([fz, event_regressors(T, on, basis, n_pre)])
        add("僵直", M, kernel=slice(1, 1 + basis.shape[1]))
    if p.use_speed and ds.has("speed"):
        sp = fill_nan(np.where(ds.valid, b["speed"], np.nan))
        k = np.exp(-np.arange(int(5 * fps) + 1) / max(1.0, fps))      # ~1 s calcium-like lag
        add("速度", np.column_stack([sp, np.convolve(sp, k / k.sum())[:T]]))
    if p.use_position and ds.has("x_u") and np.nanstd(b["x_u"][ds.valid]) > 0:
        x = fill_nan(np.where(ds.valid, b["x_u"], np.nan))
        y = fill_nan(np.where(ds.valid, b["y_u"], np.nan))
        g = int(p.position_grid)
        cx = np.linspace(np.percentile(x, 2), np.percentile(x, 98), g)
        cy = np.linspace(np.percentile(y, 2), np.percentile(y, 98), g)
        sx = max((cx[-1] - cx[0]) / max(1, g - 1), 1e-6)
        sy = max((cy[-1] - cy[0]) / max(1, g - 1), 1e-6)
        M = [np.exp(-0.5 * (((x - a) / sx) ** 2 + ((y - c) / sy) ** 2)) for a in cx for c in cy]
        add("位置", np.column_stack(M))
    if p.use_zones and b.get("zones"):
        Z = [v.astype(float) for v in b["zones"].values() if 0 < v[ds.valid].mean() < 1]
        if Z:
            add("区域", np.column_stack(Z))
    ev = parse_times(p.event_times)
    if len(ev):
        bins = np.searchsorted(ds.times, ev)
        add(p.event_name or "事件", event_regressors(T, bins, basis, n_pre), kernel=slice(0, basis.shape[1]))
    if not cols:
        raise ValueError("没有可用的行为变量。请先完成行为分析和同步，或填写事件时间。")
    return np.column_stack(cols), groups, kernels


# ---------------------------------------------------------------- ridge CV
class RidgeCV:
    """Cross-validated ridge for many responses at once; the SVDs are cached per fold."""

    def __init__(self, X, fold, lambdas=LAMBDAS):
        self.lambdas = np.asarray(lambdas, float)
        self.folds = []
        # scale with the statistics of the whole recording: a regressor that is
        # almost absent from one training fold must not be blown up in its test fold
        mu, sd = X[fold >= 0].mean(0), X[fold >= 0].std(0)
        keep = sd > 1e-9
        Xs = (X[:, keep] - mu[keep]) / sd[keep]
        for f in np.unique(fold[fold >= 0]):
            tr, te = (fold >= 0) & (fold != f), fold == f
            m = Xs[tr].mean(0)
            U, s, Vt = np.linalg.svd(Xs[tr] - m, full_matrices=False)
            self.folds.append((tr, te, U, s, (Xs[te] - m) @ Vt.T))

    def r2(self, Y) -> np.ndarray:
        """Y (T, N) -> cross-validated R^2 (n_lambda, N), pooled over folds."""
        L, N = len(self.lambdas), Y.shape[1]
        sse = np.zeros((L, N))
        sst = np.zeros(N)
        for tr, te, U, s, Z in self.folds:
            mu = Y[tr].mean(0)
            UtY = U.T @ (Y[tr] - mu)
            Yte = Y[te] - mu
            sst += (Yte ** 2).sum(0)
            for i, lam in enumerate(self.lambdas):
                pred = Z @ ((s / (s ** 2 + lam))[:, None] * UtY)
                sse[i] += ((Yte - pred) ** 2).sum(0)
        return 1.0 - sse / np.maximum(sst, 1e-12)

    def best(self, Y):
        r = self.r2(Y)
        return r.max(0), self.lambdas[r.argmax(0)]


def ridge_fit(X, Y, lam):
    """Standardised coefficients (P, N) with one lambda per response; returns B, column sd."""
    mu, sd = X.mean(0), X.std(0)
    keep = sd > 1e-9
    Xs = (X[:, keep] - mu[keep]) / sd[keep]
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    UtY = U.T @ (Y - Y.mean(0))
    D = s[:, None] / (s[:, None] ** 2 + np.asarray(lam, float)[None, :])
    B = np.zeros((X.shape[1], Y.shape[1]))
    B[keep] = Vt.T @ (D * UtY)
    return B, np.where(keep, sd, 1.0)


def fit_encoding(X, Y, groups, fold, n_shuffle=100, seed=0, progress=None, should_stop=None):
    """Core computation on valid bins only. Y (T, N) z-scored."""
    progress = progress or (lambda f, m: None)
    rng = np.random.default_rng(seed)
    T = len(Y)
    full = RidgeCV(X, fold)
    r2, lam = full.best(Y)
    G = len(groups)
    reduced = np.zeros((G, Y.shape[1]))
    for g, (_, idx) in enumerate(groups):
        check_stop(should_stop)
        rest = np.setdiff1d(np.arange(X.shape[1]), idx)
        reduced[g] = RidgeCV(X[:, rest], fold).best(Y)[0] if len(rest) else 0.0
    dr2 = r2[None, :] - reduced
    # null distributions
    ge_full = np.ones(Y.shape[1])
    ge_grp = np.ones((G, Y.shape[1]))
    # the whole-model null only needs the cached decompositions, so it can use many
    # more shifts than the per-variable null: finer p-values for the FDR correction
    n_full = int(max(n_shuffle, min(1000, 5 * n_shuffle)))
    for j, k in enumerate(shift_amounts(rng, T, n_full - n_shuffle)):
        if j % 20 == 0:
            check_stop(should_stop)
            progress(0.1 + 0.05 * j / max(1, n_full - n_shuffle), "打乱检验（整体模型）…")
        ge_full += full.best(np.roll(Y, k, axis=0))[0] >= r2
    shifts = shift_amounts(rng, T, n_shuffle)
    for i, k in enumerate(shifts):
        check_stop(should_stop)
        ge_full += full.best(np.roll(Y, k, axis=0))[0] >= r2
        if G > 1:
            for g, (_, idx) in enumerate(groups):
                Xs = X.copy()
                Xs[:, idx] = np.roll(X[:, idx], k, axis=0)
                ge_grp[g] += (RidgeCV(Xs, fold).best(Y)[0] - reduced[g]) >= dr2[g]
        progress(0.15 + 0.8 * (i + 1) / max(1, n_shuffle), f"打乱检验 {i + 1}/{n_shuffle}")
    p_full = ge_full / (n_full + 1)
    p_grp = ge_grp / (n_shuffle + 1) if G > 1 else np.tile(p_full, (1, 1))
    return dict(r2=r2, lam=lam, dr2=dr2, p=p_full, p_group=p_grp)


def run_glm(session, ws: Workspace, args: dict, progress=None, should_stop=None) -> dict:
    """args: {"params": GLMParams, "cfg": BehaviorConfig | None}"""
    progress = progress or (lambda f, m: None)
    p: GLMParams = args["params"]
    progress(0.02, "读取数据…")
    ds = load_dataset(ws, args.get("cfg"), p.bin_s)
    X, groups, kernels = build_design(ds, p)
    v = ds.valid
    if v.sum() < 50:
        raise ValueError("行为和钙成像重叠的时间太短，无法拟合模型")
    Xv = X[v]
    Y = zscore(ds.dff[:, v], axis=1).T
    fold = block_folds(len(Y), p.n_folds, 20.0, ds.fps)
    progress(0.1, "拟合编码模型…")
    res = fit_encoding(Xv, Y, groups, fold, p.n_shuffle, p.seed, progress, should_stop)
    B, sd = ridge_fit(Xv, Y, res["lam"])
    names = [g[0] for g in groups]
    arrays = dict(ids=ds.ids, centres=ds.centres, r2=res["r2"], p=res["p"], q=fdr_bh(res["p"]),
                  dr2=res["dr2"], p_group=res["p_group"], groups=np.array(names),
                  best_group=res["dr2"].argmax(0), beta=B)
    for name, (idx, basis, lags) in kernels.items():
        j = names.index(name)
        arrays[f"kernel_{j}"] = (basis @ (B[idx] / sd[idx, None])).T      # cells x lags
        arrays[f"kernel_lags_{j}"] = lags
    sig = res["p"] < 0.05
    summary = {"n_cells": int(ds.n_cells), "n_sig": int(sig.sum()), "n_sig_fdr": int((arrays["q"] < 0.05).sum()),
               "median_r2": float(np.median(res["r2"])), "median_r2_sig": float(np.median(res["r2"][sig])) if sig.any() else 0.0,
               "groups": names,
               "n_sig_group": [int(((res["p_group"][g] < 0.05) & (res["dr2"][g] > 0)).sum()) for g in range(len(names))],
               "n_bins": int(v.sum()), "bin_s": 1.0 / ds.fps}
    save_result(ws, "glm", arrays, summary, asdict(p))
    progress(1.0, "编码模型完成")
    return summary


def export_glm(ws: Workspace, res: dict, folder):
    cols = {"cell_id": res["ids"], "cv_r2": res["r2"], "p": res["p"], "q_fdr": res["q"],
            "best_variable": [str(res["groups"][i]) for i in res["best_group"]]}
    for g, name in enumerate(res["groups"]):
        cols[f"unique_r2[{name}]"] = res["dr2"][g]
        cols[f"p[{name}]"] = res["p_group"][g]
    return write_csv(folder / "glm_encoding.csv", cols)
