"""Cell assemblies (ensembles): groups of cells that are repeatedly active together.

PCA/ICA method (Lopes-dos-Santos, Ribeiro & Tort 2013): the number of
assemblies is the number of eigenvalues of the correlation matrix above what
is expected by chance; ICA on that subspace gives one weight vector per
assembly; the activation strength over time is z(t)' P z(t) with the diagonal
of P = w w' set to zero, so that a single cell cannot drive it alone.
NMF is offered as an alternative with non-negative, easier to read weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..workspace import Workspace
from .common import check_stop, load_dataset, save_result, shift_amounts, write_csv, zscore


@dataclass
class EnsembleParams:
    bin_s: float = 0.5
    method: str = "ica"            # ica | nmf
    signal: str = "dff"            # dff | events
    null: str = "shuffle"          # shuffle | mp (Marchenko-Pastur)
    n_shuffle: int = 100
    n_assemblies: int = 0          # 0 = automatic
    member_sd: float = 2.0
    seed: int = 0


def significant_components(Z, null="shuffle", n_shuffle=100, rng=None, should_stop=None):
    """Z (N, T) z-scored. Returns eigenvalues (descending), eigenvectors, threshold."""
    N, T = Z.shape
    C = Z @ Z.T / T
    w, V = np.linalg.eigh(C)
    w, V = w[::-1], V[:, ::-1]
    if null == "mp":
        thr = (1 + np.sqrt(N / T)) ** 2
    else:
        rng = rng or np.random.default_rng(0)
        top = np.empty(n_shuffle)
        for i in range(n_shuffle):
            check_stop(should_stop)
            Zs = np.stack([np.roll(z, k) for z, k in zip(Z, shift_amounts(rng, T, N, 0.02))])
            top[i] = np.linalg.eigvalsh(Zs @ Zs.T / T)[-1]
        thr = float(np.percentile(top, 95))
    return w, V, thr


def fast_ica(X, seed=0, n_iter=500, tol=1e-6):
    """Symmetric FastICA (logcosh) on whitened X (k, T). Returns unmixing W (k, k)."""
    k, T = X.shape
    rng = np.random.default_rng(seed)
    W = np.linalg.qr(rng.normal(size=(k, k)))[0]
    for _ in range(n_iter):
        Y = W @ X
        g = np.tanh(Y)
        Wn = g @ X.T / T - (1 - g ** 2).mean(1)[:, None] * W
        u, _, vt = np.linalg.svd(Wn)
        Wn = u @ vt
        lim = np.abs(np.abs(np.einsum("ij,ij->i", Wn, W)) - 1).max()
        W = Wn
        if lim < tol:
            break
    return W


def ica_assemblies(Z, n, V, w, seed=0):
    """Assembly weight vectors (n, N), unit norm, largest weights positive."""
    Vk = V[:, :n]
    X = (Vk / np.sqrt(w[:n])).T @ Z            # whitened PC scores (n, T)
    W = fast_ica(X, seed) if n > 1 else np.ones((1, 1))
    pat = (Vk @ W.T).T                         # back to cell space
    pat /= np.linalg.norm(pat, axis=1, keepdims=True)
    for k in range(n):
        if np.abs(pat[k].min()) > pat[k].max():
            pat[k] *= -1
    return pat


def activation_strength(Z, patterns):
    """(K, T): z' P z with zero diagonal of P = w w'."""
    proj = patterns @ Z
    return proj ** 2 - (patterns ** 2) @ (Z ** 2)


def nmf(X, k, seed=0, n_iter=300, tol=1e-5):
    """HALS non-negative factorisation X (N, T) ~ W (N, k) H (k, T)."""
    rng = np.random.default_rng(seed)
    N, T = X.shape
    W = np.abs(rng.normal(size=(N, k))) * np.sqrt(X.mean() / k + 1e-12)
    H = np.abs(rng.normal(size=(k, T))) * np.sqrt(X.mean() / k + 1e-12)
    prev = np.inf
    for it in range(n_iter):
        WtW, WtX = W.T @ W, W.T @ X
        for j in range(k):
            H[j] = np.maximum(H[j] + (WtX[j] - WtW[j] @ H) / max(WtW[j, j], 1e-12), 0)
        HHt, XHt = H @ H.T, X @ H.T
        for j in range(k):
            W[:, j] = np.maximum(W[:, j] + (XHt[:, j] - W @ HHt[:, j]) / max(HHt[j, j], 1e-12), 0)
        if it % 10 == 9:
            err = np.linalg.norm(X - W @ H)
            if prev - err < tol * max(prev, 1e-12):
                break
            prev = err
    scale = np.linalg.norm(W, axis=0)
    scale[scale == 0] = 1
    return W / scale, H * scale[:, None]


def detect(X, p: EnsembleParams, progress=None, should_stop=None):
    """X (N, T) activity. Returns dict(patterns, members, activation, eig, thr)."""
    progress = progress or (lambda f, m: None)
    rng = np.random.default_rng(p.seed)
    Z = zscore(X, axis=1)
    active = X.std(1) > 1e-12
    Za = Z[active]
    progress(0.1, "估计细胞组数量…")
    w, V, thr = significant_components(Za, p.null, p.n_shuffle, rng, should_stop)
    n = int(p.n_assemblies) if p.n_assemblies > 0 else int((w > thr).sum())
    n = min(n, Za.shape[0] - 1)
    N = X.shape[0]
    if n < 1:
        return dict(patterns=np.zeros((0, N)), members=np.zeros((0, N), bool),
                    activation=np.zeros((0, X.shape[1])), eig=w, thr=thr)
    progress(0.6, f"提取 {n} 个细胞组…")
    pat = np.zeros((n, N))
    if p.method == "nmf":
        Xp = np.maximum(X[active], 0)
        Xp = Xp / np.maximum(Xp.std(1, keepdims=True), 1e-12)
        Wn, H = nmf(Xp, n, p.seed)
        pat[:, active] = Wn.T
        act = H
    else:
        pat[:, active] = ica_assemblies(Za, n, V, w, p.seed)
        act = activation_strength(Z, pat)
    # members: weight well above the bulk of the cells (robust SD, so that a large
    # assembly does not raise its own threshold) and above 1/sqrt(N), the weight
    # every cell would have if all contributed equally
    med = np.median(pat, axis=1, keepdims=True)
    rsd = 1.4826 * np.median(np.abs(pat - med), axis=1, keepdims=True)
    members = pat > np.maximum(med + p.member_sd * rsd, 1.0 / np.sqrt(N))
    order = np.argsort(-act.var(1))
    return dict(patterns=pat[order], members=members[order], activation=act[order], eig=w, thr=thr)


def run_ensembles(session, ws: Workspace, args: dict, progress=None, should_stop=None) -> dict:
    progress = progress or (lambda f, m: None)
    p: EnsembleParams = args["params"]
    ds = load_dataset(ws, args.get("cfg"), p.bin_s)
    X = ds.dff if p.signal == "dff" else ds.events
    res = detect(X, p, progress, should_stop)
    K = len(res["patterns"])
    arrays = dict(ids=ds.ids, centres=ds.centres, times=ds.times, **res)
    summary = {"n_assemblies": K, "method": p.method, "n_cells": int(ds.n_cells),
               "sizes": res["members"].sum(1).tolist(),
               "n_cells_in_any": int(res["members"].any(0).sum()) if K else 0,
               "threshold": float(res["thr"])}
    if K and ds.has("freezing") and 0 < ds.beh["freezing"][ds.valid].mean() < 1:
        from ..sync import state_modulation
        m = state_modulation(res["activation"], ds.beh["freezing"], ds.valid, n_shuffle=500, seed=p.seed)
        arrays.update(freeze_in=m["in_state"], freeze_out=m["out_state"], freeze_p_up=m["p_up"],
                      freeze_p_down=m["p_down"], freezing=ds.beh["freezing"])
        summary["freezing_up"] = [int(i + 1) for i in np.where(m["p_up"] < 0.025)[0]]
        summary["freezing_down"] = [int(i + 1) for i in np.where(m["p_down"] < 0.025)[0]]
    save_result(ws, "ensembles", arrays, summary, asdict(p))
    progress(1.0, "细胞组检测完成")
    return summary


def export_ensembles(ws: Workspace, res: dict, folder):
    K = len(res["patterns"])
    cols = {"cell_id": res["ids"]}
    for k in range(K):
        cols[f"weight_{k + 1}"] = res["patterns"][k]
        cols[f"member_{k + 1}"] = res["members"][k].astype(int)
    write_csv(folder / "ensemble_weights.csv", cols)
    cols = {"time_s": res["times"]}
    for k in range(K):
        cols[f"activation_{k + 1}"] = res["activation"][k]
    return write_csv(folder / "ensemble_activation.csv", cols)
