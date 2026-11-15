"""Hidden-state models of the population activity.

All three models describe the low-dimensional population trajectory x_t (the
top principal components of the z-scored dF/F) as switching between K hidden
states:

  hmm     Gaussian HMM: every state has its own mean activity pattern.
  arhmm   autoregressive HMM: every state has its own linear dynamics
          x_t = A_k x_{t-1} + b_k + noise (how activity evolves, not only where it is).
  rarhmm  recurrent AR-HMM: as arhmm, and the probability of switching depends
          on where the trajectory currently is. This is the recurrent switching
          linear dynamical system (rSLDS; Linderman et al. 2017) with the
          continuous latent taken directly from PCA instead of being inferred.

Fitted with EM (scaled forward-backward); states decoded with Viterbi.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..workspace import Workspace
from .common import check_stop, load_dataset, pca, save_result, shift_amounts, write_csv, zscore

KINDS = ("hmm", "arhmm", "rarhmm")


@dataclass
class HMMParams:
    bin_s: float = 0.5
    kind: str = "hmm"
    n_states: int = 4
    n_latent: int = 8
    sticky: float = 10.0           # extra pseudo-counts on self transitions
    n_iter: int = 60
    n_init: int = 3
    compare_k: bool = False
    k_max: int = 8
    seed: int = 0


def _log_softmax(a, axis=-1):
    m = a.max(axis, keepdims=True)
    return a - m - np.log(np.exp(a - m).sum(axis, keepdims=True))


def _gauss_logpdf(R, S):
    """Rows of R ~ N(0, S)."""
    L = np.linalg.cholesky(S)
    y = np.linalg.solve(L, R.T)
    return -0.5 * (y ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * R.shape[1] * np.log(2 * np.pi)


def kmeans(X, k, rng, n_iter=50):
    c = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d = np.min([((X - m) ** 2).sum(1) for m in c], axis=0)
        c.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
    c = np.array(c)
    lab = np.zeros(len(X), int)
    for _ in range(n_iter):
        new = ((X[:, None, :] - c[None]) ** 2).sum(2).argmin(1)
        if (new == lab).all() and _ > 0:
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                c[j] = X[lab == j].mean(0)
    return lab


class SwitchingModel:
    def __init__(self, kind: str, K: int, D: int, sticky: float = 10.0):
        assert kind in KINDS
        self.kind, self.K, self.D, self.sticky = kind, K, D, sticky
        self.ar = kind != "hmm"
        self.recurrent = kind == "rarhmm"
        self.pi0 = np.full(K, 1.0 / K)
        self.logP = np.log(np.full((K, K), 1.0 / K))
        self.R = np.zeros((K, D))
        self.mu = np.zeros((K, D))
        self.A = np.tile(np.eye(D), (K, 1, 1))
        self.b = np.zeros((K, D))
        self.S = np.tile(np.eye(D), (K, 1, 1))
        self.reg = 1e-3

    # -------------------------------------------------------- likelihoods
    def obs_loglik(self, x):
        T = len(x)
        ll = np.zeros((T, self.K))
        for k in range(self.K):
            if self.ar:
                ll[1:, k] = _gauss_logpdf(x[1:] - x[:-1] @ self.A[k].T - self.b[k], self.S[k])
            else:
                ll[:, k] = _gauss_logpdf(x - self.mu[k], self.S[k])
        return ll

    def log_trans(self, x):
        """(T-1, K, K): log p(z_{t+1} = j | z_t = i, x_t)."""
        if not self.recurrent:
            return np.broadcast_to(_log_softmax(self.logP), (len(x) - 1, self.K, self.K))
        return _log_softmax(self.logP[None] + (x[:-1] @ self.R.T)[:, None, :])

    # ---------------------------------------------------------- inference
    def _fb(self, x):
        ll = self.obs_loglik(x)
        lt = self.log_trans(x)
        T, K = ll.shape
        B = np.exp(ll - ll.max(1, keepdims=True))
        P = np.exp(lt)
        alpha = np.empty((T, K))
        c = np.empty(T)
        a = self.pi0 * B[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ P[t - 1]) * B[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        beta = np.ones((T, K))
        for t in range(T - 2, -1, -1):
            beta[t] = (P[t] @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(1, keepdims=True)
        xi = alpha[:-1, :, None] * P * (B[1:] * beta[1:])[:, None, :] / c[1:, None, None]
        return gamma, xi, float(np.log(c).sum() + ll.max(1).sum())

    def score(self, seqs) -> float:
        return sum(self._fb(x)[2] for x in seqs)

    def viterbi(self, x):
        ll = self.obs_loglik(x)
        lt = self.log_trans(x)
        T, K = ll.shape
        d = np.log(self.pi0 + 1e-300) + ll[0]
        back = np.zeros((T, K), int)
        for t in range(1, T):
            m = d[:, None] + lt[t - 1]
            back[t] = m.argmax(0)
            d = m.max(0) + ll[t]
        z = np.empty(T, int)
        z[-1] = d.argmax()
        for t in range(T - 1, 0, -1):
            z[t - 1] = back[t, z[t]]
        return z

    # ------------------------------------------------------------- M-step
    def _m_emission(self, seqs, gammas):
        D = self.D
        X = np.concatenate(seqs)
        base = np.cov(X.T).reshape(D, D) * self.reg + 1e-8 * np.eye(D)
        if self.ar:
            X1 = np.concatenate([x[1:] for x in seqs])
            X0 = np.concatenate([np.column_stack([x[:-1], np.ones(len(x) - 1)]) for x in seqs])
            G = np.concatenate([g[1:] for g in gammas])
        else:
            X1, G = X, np.concatenate(gammas)
        for k in range(self.K):
            w = G[:, k]
            n = w.sum()
            if n < D + 2:              # an empty state keeps its parameters
                continue
            if self.ar:
                XtW = X0.T * w
                ridge = 1e-3 * n * np.eye(D + 1)
                ridge[-1, -1] = 0
                W = np.linalg.solve(XtW @ X0 + ridge, XtW @ X1)
                self.A[k], self.b[k] = W[:D].T, W[D]
                r = X1 - X0 @ W
            else:
                self.mu[k] = w @ X1 / n
                r = X1 - self.mu[k]
            self.S[k] = (r.T * w) @ r / n + base

    def _m_transition(self, seqs, xis):
        K = self.K
        prior = self.sticky * np.eye(K) + 0.1
        if not self.recurrent:
            counts = sum(xi.sum(0) for xi in xis) + prior
            self.logP = np.log(counts / counts.sum(1, keepdims=True))
            return
        X = np.concatenate([x[:-1] for x in seqs])
        XI = np.concatenate(xis)
        n = len(X)
        gam = XI.sum(2)
        m = [np.zeros_like(self.logP), np.zeros_like(self.R)]
        v = [np.zeros_like(self.logP), np.zeros_like(self.R)]
        for it in range(1, 26):       # a few Adam steps = generalised EM
            q = np.exp(_log_softmax(self.logP[None] + (X @ self.R.T)[:, None, :]))
            dL = XI - gam[:, :, None] * q
            gP = dL.sum(0) + prior - prior.sum(1, keepdims=True) * np.exp(_log_softmax(self.logP))
            gR = dL.sum(1).T @ X - 1e-2 * n * self.R
            for i, (par, g) in enumerate(((self.logP, gP / n), (self.R, gR / n))):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g ** 2
                par += 0.05 * (m[i] / (1 - 0.9 ** it)) / (np.sqrt(v[i] / (1 - 0.999 ** it)) + 1e-8)

    def m_step(self, seqs, gammas, xis):
        self.pi0 = np.mean([g[0] for g in gammas], axis=0) + 1e-3
        self.pi0 /= self.pi0.sum()
        self._m_emission(seqs, gammas)
        self._m_transition(seqs, xis)

    def init_from_labels(self, seqs, labels):
        gammas = [np.eye(self.K)[z] * 0.9 + 0.1 / self.K for z in labels]
        xis = [g[:-1, :, None] * g[1:, None, :] for g in gammas]
        self.m_step(seqs, gammas, xis)

    def em(self, seqs, n_iter=60, tol=1e-4, should_stop=None):
        hist = []
        for _ in range(n_iter):
            check_stop(should_stop)
            out = [self._fb(x) for x in seqs]
            ll = sum(o[2] for o in out)
            hist.append(ll)
            self.m_step(seqs, [o[0] for o in out], [o[1] for o in out])
            if len(hist) > 1 and abs(hist[-1] - hist[-2]) < tol * abs(hist[-2]):
                break
        return hist


def fit(seqs, kind="hmm", K=4, sticky=10.0, n_iter=60, n_init=3, seed=0, should_stop=None,
        progress=None):
    """Best of n_init random starts. AR models start from a fitted Gaussian HMM."""
    rng = np.random.default_rng(seed)
    X = np.concatenate(seqs)
    D = X.shape[1]
    cuts = np.cumsum([len(x) for x in seqs])[:-1]
    best = None
    for i in range(max(1, n_init)):
        check_stop(should_stop)
        lab = np.split(kmeans(X, K, rng), cuts)
        m = SwitchingModel("hmm", K, D, sticky)
        m.init_from_labels(seqs, lab)
        m.em(seqs, n_iter if kind == "hmm" else max(10, n_iter // 3), should_stop=should_stop)
        if kind != "hmm":
            gam = [m._fb(x) for x in seqs]
            m2 = SwitchingModel(kind, K, D, sticky)
            m2.logP = m.logP.copy()
            m2.m_step(seqs, [g[0] for g in gam], [g[1] for g in gam])
            m2.em(seqs, n_iter, should_stop=should_stop)
            m = m2
        ll = m.score(seqs)
        if best is None or ll > best[0]:
            best = (ll, m)
        if progress:
            progress((i + 1) / max(1, n_init))
    return best[1], best[0]


def latents(dff, n_latent):
    """Top principal components of the z-scored activity, in units of PC1's sd."""
    Z = zscore(dff, axis=1).T
    sc, comp, ratio = pca(Z, n_latent)
    return sc / max(sc[:, 0].std(), 1e-12), comp, ratio


def split_blocks(x, n_blocks=10, test=(2, 5, 8)):
    """Train / test segments for held-out likelihood."""
    edges = np.linspace(0, len(x), n_blocks + 1).astype(int)
    train, tests, cur = [], [], []
    for b in range(n_blocks):
        seg = (edges[b], edges[b + 1])
        if b in test:
            if cur:
                train.append(x[cur[0]:cur[1]])
                cur = []
            tests.append(x[seg[0]:seg[1]])
        else:
            cur = [cur[0] if cur else seg[0], seg[1]]
    if cur:
        train.append(x[cur[0]:cur[1]])
    return train, tests


def dwell_times(z, K):
    """Mean run length (in bins) of every state."""
    out = np.zeros(K)
    change = np.flatnonzero(np.diff(z)) + 1
    starts = np.concatenate([[0], change])
    lens = np.diff(np.concatenate([starts, [len(z)]]))
    for k in range(K):
        l = lens[z[starts] == k]
        out[k] = l.mean() if len(l) else 0.0
    return out


def state_behavior(z, K, state, valid, n_shuffle=500, seed=0):
    """Fraction of each hidden state spent in a behaviour state, with circular-shift p-values."""
    rng = np.random.default_rng(seed)
    zv, s = z[valid], state[valid].astype(float)
    oh = np.eye(K)[zv]
    n = np.maximum(oh.sum(0), 1)
    frac = oh.T @ s / n
    null = np.array([oh.T @ np.roll(s, k) / n for k in shift_amounts(rng, len(s), n_shuffle)])
    p_hi = (1 + (null >= frac).sum(0)) / (n_shuffle + 1)
    p_lo = (1 + (null <= frac).sum(0)) / (n_shuffle + 1)
    return frac, p_hi, p_lo


def run_hmm(session, ws: Workspace, args: dict, progress=None, should_stop=None) -> dict:
    progress = progress or (lambda f, m: None)
    p: HMMParams = args["params"]
    ds = load_dataset(ws, args.get("cfg"), p.bin_s)
    D = int(min(p.n_latent, ds.n_cells - 1, ds.n_bins // 20))
    if D < 1 or ds.n_bins < 50:
        raise ValueError("数据太短或细胞太少，无法拟合状态模型")
    x, comp, ratio = latents(ds.dff, D)
    K = int(p.n_states)
    arrays, summary = {}, {}
    span = 0.75
    if p.compare_k:
        span = 0.4
        train, test = split_blocks(x)
        ks = np.arange(2, int(p.k_max) + 1)
        cv = np.zeros(len(ks))
        n_test = sum(len(s) for s in test)
        for i, k in enumerate(ks):
            progress(0.05 + 0.5 * i / len(ks), f"比较状态数：K = {k}")
            m, _ = fit(train, p.kind, int(k), p.sticky, max(20, p.n_iter // 2), 2, p.seed, should_stop)
            cv[i] = m.score(test) / n_test
        arrays.update(cv_k=ks, cv_ll=cv)
        summary["best_k"] = int(ks[cv.argmax()])
    progress(1 - span - 0.05, f"拟合 {K} 个状态的模型…")
    model, ll = fit([x], p.kind, K, p.sticky, p.n_iter, p.n_init, p.seed, should_stop,
                    lambda f: progress(1 - span - 0.05 + span * f, f"拟合 {K} 个状态的模型…"))
    gamma, xi, _ = model._fb(x)
    z = model.viterbi(x)
    order = np.argsort(-np.bincount(z, minlength=K))          # state 1 = most common
    rank = np.argsort(order)
    z = rank[z]
    gamma = gamma[:, order]
    trans = xi.sum(0)[np.ix_(order, order)]
    trans = trans / np.maximum(trans.sum(1, keepdims=True), 1e-12)
    Zc = zscore(ds.dff, axis=1)
    occ = np.bincount(z, minlength=K) / len(z)
    mean_act = np.stack([Zc[:, z == k].mean(1) if (z == k).any() else np.zeros(ds.n_cells) for k in range(K)])
    dwell = dwell_times(z, K) / ds.fps
    arrays.update(ids=ds.ids, centres=ds.centres, times=ds.times, states=z, posterior=gamma, trans=trans,
                  occupancy=occ, dwell_s=dwell, mean_activity=mean_act, latents=x[:, :min(D, 3)],
                  explained=ratio)
    summary.update(kind=p.kind, n_states=K, n_latent=D, loglik_per_bin=ll / len(x),
                   occupancy=occ.tolist(), dwell_s=dwell.tolist(), explained=float(ratio.sum()),
                   n_cells=int(ds.n_cells))
    if model.ar:
        tau = []
        for k in order:
            ev = np.abs(np.linalg.eigvals(model.A[k])).max()
            tau.append(float(-1.0 / (ds.fps * np.log(ev))) if 0 < ev < 1 else float("inf"))
        arrays["dyn_tau_s"] = np.array(tau)
        summary["dyn_tau_s"] = [t if np.isfinite(t) else None for t in tau]
    if ds.has("freezing") and 0 < ds.beh["freezing"][ds.valid].mean() < 1:
        frac, p_hi, p_lo = state_behavior(z, K, ds.beh["freezing"], ds.valid, seed=p.seed)
        arrays.update(freeze_frac=frac, freeze_p_up=p_hi, freeze_p_down=p_lo, freezing=ds.beh["freezing"],
                      valid=ds.valid)
        summary.update(freeze_frac=frac.tolist(), freeze_overall=float(ds.beh["freezing"][ds.valid].mean()),
                       freeze_states=[int(k + 1) for k in np.where(p_hi < 0.025)[0]])
    save_result(ws, "hmm", arrays, summary, asdict(p))
    progress(1.0, "状态模型完成")
    return summary


def export_hmm(ws: Workspace, res: dict, folder):
    K = res["posterior"].shape[1]
    cols = {"time_s": res["times"], "state": res["states"] + 1}
    for k in range(K):
        cols[f"p_state_{k + 1}"] = res["posterior"][:, k]
    write_csv(folder / "hmm_states.csv", cols)
    cols = {"cell_id": res["ids"]}
    for k in range(K):
        cols[f"mean_z_state_{k + 1}"] = res["mean_activity"][k]
    write_csv(folder / "hmm_state_activity.csv", cols)
    cols = {"from_state": np.arange(1, K + 1)}
    for k in range(K):
        cols[f"to_{k + 1}"] = res["trans"][:, k]
    return write_csv(folder / "hmm_transitions.csv", cols)
