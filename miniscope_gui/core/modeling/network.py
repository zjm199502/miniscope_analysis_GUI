"""Functional connectivity network between cells and its graph-theory description.

Edges are pairwise Pearson correlations of the activity that exceed what
circularly shifted (independent) traces produce. The binary graph is then
summarised with standard measures (Rubinov & Sporns 2010): degree, clustering,
path length / efficiency, modularity and communities, hubs, small-worldness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..workspace import Workspace
from .common import check_stop, load_dataset, save_result, shift_amounts, write_csv, zscore


@dataclass
class NetworkParams:
    bin_s: float = 0.5
    signal: str = "dff"             # dff | events
    threshold: str = "shuffle"      # shuffle | density
    percentile: float = 99.9        # of the shuffled correlations
    density: float = 0.05           # used when threshold == "density"
    n_shuffle: int = 50
    by_state: bool = True           # separate networks for freezing / not freezing
    seed: int = 0


def corr_matrix(X):
    Z = zscore(X, axis=1)
    C = Z @ Z.T / X.shape[1]
    np.fill_diagonal(C, 0.0)
    return C


def null_threshold(X, percentile=99.0, n_shuffle=50, seed=0, should_stop=None, progress=None):
    """Correlation exceeded by (100 - percentile)% of pairs of independently shifted traces."""
    rng = np.random.default_rng(seed)
    Z = zscore(X, axis=1)
    N, T = Z.shape
    iu = np.triu_indices(N, 1)
    vals = []
    for i in range(n_shuffle):
        check_stop(should_stop)
        Zs = np.stack([np.roll(z, k) for z, k in zip(Z, shift_amounts(rng, T, N, 0.02))])
        vals.append((Zs @ Zs.T / T)[iu])
        if progress:
            progress((i + 1) / n_shuffle)
    return float(np.percentile(np.concatenate(vals), percentile))


# ------------------------------------------------------------ graph measures
def clustering(A):
    """Local clustering coefficient of a binary undirected graph."""
    A = A.astype(float)
    k = A.sum(1)
    tri = np.einsum("ij,jk,ki->i", A, A, A) / 2.0
    denom = k * (k - 1) / 2.0
    return np.where(denom > 0, tri / np.maximum(denom, 1), 0.0)


def distances(A):
    """All-pairs shortest path lengths (inf when not connected)."""
    N = len(A)
    A = A.astype(bool)
    D = np.full((N, N), np.inf)
    np.fill_diagonal(D, 0)
    reach = np.eye(N, dtype=bool)
    front = np.eye(N, dtype=bool)
    Af = A.astype(np.float32)
    d = 0
    while front.any():
        d += 1
        nxt = (front.astype(np.float32) @ Af > 0) & ~reach
        D[nxt] = d
        reach |= nxt
        front = nxt
    return D


def path_measures(A):
    D = distances(A)
    N = len(A)
    off = ~np.eye(N, dtype=bool)
    with np.errstate(divide="ignore"):
        eff = float((1.0 / D[off]).mean()) if N > 1 else 0.0
    fin = np.isfinite(D) & off
    L = float(D[fin].mean()) if fin.any() else float("nan")
    # giant component
    comp = components(A)
    giant = float(np.bincount(comp).max() / N) if N else 0.0
    return L, eff, giant


def components(A):
    N = len(A)
    lab = -np.ones(N, int)
    c = 0
    for s in range(N):
        if lab[s] >= 0:
            continue
        stack = [s]
        lab[s] = c
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(A[u]):
                if lab[v] < 0:
                    lab[v] = c
                    stack.append(v)
        c += 1
    return lab


def modularity(A, lab):
    m2 = A.sum()
    if m2 == 0:
        return 0.0
    k = A.sum(1)
    same = lab[:, None] == lab[None, :]
    return float(((A - np.outer(k, k) / m2) * same).sum() / m2)


def communities(A, seed=0, n_pass=20):
    """Louvain-style modularity optimisation (local moving + aggregation)."""
    rng = np.random.default_rng(seed)
    W = A.astype(float)
    N = len(W)
    member = np.arange(N)                      # original node -> current super node
    while True:
        n = len(W)
        m2 = W.sum()
        if m2 == 0:
            break
        k = W.sum(1)
        lab = np.arange(n)
        tot = k.copy()
        moved_any = False
        for _ in range(n_pass):
            moved = False
            for i in rng.permutation(n):
                li = lab[i]
                w_to = np.bincount(lab, weights=W[i], minlength=n)
                w_to[li] -= W[i, i]
                tot[li] -= k[i]
                gain = w_to - tot * k[i] / m2
                best = int(np.argmax(gain))
                if gain[best] <= gain[li] + 1e-12:
                    best = li
                tot[best] += k[i]
                if best != li:
                    lab[i] = best
                    moved = moved_any = True
            if not moved:
                break
        if not moved_any:
            break
        u, lab = np.unique(lab, return_inverse=True)
        member = lab[member]
        S = np.eye(len(u))[lab]                # n x n_comm
        W = S.T @ W @ S
    u, member = np.unique(member, return_inverse=True)
    # biggest community first; isolated nodes get -1
    order = np.argsort(-np.bincount(member))
    member = np.argsort(order)[member]
    member[A.sum(1) == 0] = -1
    return member


def random_reference(A, n=10, seed=0):
    """Mean clustering and path length of random graphs with the same number of edges."""
    rng = np.random.default_rng(seed)
    N = len(A)
    iu = np.triu_indices(N, 1)
    m = int(A[iu].sum())
    Cs, Ls = [], []
    for _ in range(n):
        R = np.zeros((N, N), bool)
        pick = rng.choice(len(iu[0]), m, replace=False)
        R[iu[0][pick], iu[1][pick]] = True
        R |= R.T
        Cs.append(clustering(R).mean())
        Ls.append(path_measures(R)[0])
    return float(np.mean(Cs)), float(np.nanmean(Ls))


def graph_summary(A, seed=0):
    N = len(A)
    deg = A.sum(1)
    cl = clustering(A)
    L, eff, giant = path_measures(A)
    lab = communities(A, seed)
    Q = modularity(A, np.where(lab < 0, lab.max() + 1 + np.arange(N), lab))
    out = {"n_nodes": int(N), "n_edges": int(A.sum() // 2), "density": float(A.sum() / max(1, N * (N - 1))),
           "mean_degree": float(deg.mean()), "clustering": float(cl.mean()), "path_length": L,
           "efficiency": eff, "giant_component": giant, "modularity": Q,
           "n_communities": int(len(np.unique(lab[lab >= 0])))}
    # small-worldness is only meaningful when most cells are in one connected piece
    if 0 < out["n_edges"] < N * (N - 1) // 2 and giant >= 0.8:
        Cr, Lr = random_reference(A, seed=seed)
        if Cr > 0 and Lr > 0 and np.isfinite(L):
            out["small_world"] = float((out["clustering"] / Cr) / (L / Lr))
    return out, deg, cl, lab


def build_graph(X, p: NetworkParams, thr=None, should_stop=None, progress=None):
    C = corr_matrix(X)
    if thr is None:
        if p.threshold == "density":
            iu = np.triu_indices(len(C), 1)
            thr = float(np.quantile(C[iu], 1 - p.density)) if len(iu[0]) else 0.0
        else:
            thr = null_threshold(X, p.percentile, p.n_shuffle, p.seed, should_stop, progress)
    A = C > thr
    np.fill_diagonal(A, False)
    return C, A, thr


def run_network(session, ws: Workspace, args: dict, progress=None, should_stop=None) -> dict:
    progress = progress or (lambda f, m: None)
    p: NetworkParams = args["params"]
    ds = load_dataset(ws, args.get("cfg"), p.bin_s)
    X = ds.dff if p.signal == "dff" else ds.events
    if ds.n_cells < 3:
        raise ValueError("细胞太少，无法构建网络")
    progress(0.05, "计算相关并估计阈值…")
    C, A, thr = build_graph(X, p, None, should_stop, lambda f: progress(0.05 + 0.6 * f, "打乱检验…"))
    progress(0.7, "计算图论指标…")
    summ, deg, cl, lab = graph_summary(A, p.seed)
    hub_thr = deg.mean() + 1.5 * deg.std()
    hubs = (deg > hub_thr) & (deg > 0)
    dist = np.hypot(ds.centres[:, None, 0] - ds.centres[None, :, 0],
                    ds.centres[:, None, 1] - ds.centres[None, :, 1])
    iu = np.triu_indices(ds.n_cells, 1)
    summ.update(threshold=thr, n_hubs=int(hubs.sum()), hubs=[int(i) for i in ds.ids[hubs]],
                mean_corr=float(C[iu].mean()), dist_corr_r=float(np.corrcoef(dist[iu], C[iu])[0, 1]),
                n_cells=int(ds.n_cells))
    arrays = dict(ids=ds.ids, centres=ds.centres, corr=C.astype(np.float32), adj=A, degree=deg,
                  clustering=cl, community=lab, hubs=hubs,
                  strength=np.where(A, C, 0).sum(1), shape=np.array(ds.shape))
    if p.by_state and ds.has("freezing"):
        fz, v = ds.beh["freezing"], ds.valid
        states = {}
        if (fz & v).sum() >= 30 and (~fz & v).sum() >= 30:
            for name, m in (("freezing", fz & v), ("moving", ~fz & v)):
                check_stop(should_stop)
                Cs, As, _ = build_graph(X[:, m], p, thr)
                s, d_, c_, _ = graph_summary(As, p.seed)
                s["mean_corr"] = float(Cs[iu].mean())
                states[name] = s
                arrays[f"adj_{name}"] = As
                arrays[f"degree_{name}"] = d_
            summ["by_state"] = states
    save_result(ws, "network", arrays, summ, asdict(p))
    progress(1.0, "网络分析完成")
    return summ


def export_network(ws: Workspace, res: dict, folder):
    write_csv(folder / "network_cells.csv", {
        "cell_id": res["ids"], "degree": res["degree"], "strength": res["strength"],
        "clustering": res["clustering"], "community": res["community"] + 1, "hub": res["hubs"].astype(int)})
    i, j = np.where(np.triu(res["adj"], 1))
    write_csv(folder / "network_edges.csv", {"cell_a": res["ids"][i], "cell_b": res["ids"][j],
                                              "correlation": res["corr"][i, j].astype(float)})
    np.savetxt(folder / "network_correlation_matrix.csv", res["corr"], delimiter=",", fmt="%.4f",
               header=",".join(str(i) for i in res["ids"]), comments="")
    return folder / "network_cells.csv"
