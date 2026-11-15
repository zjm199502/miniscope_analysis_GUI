"""Calcium-event kinetics and unsupervised clustering (putative cell types).

1. Events are detected on the lightly smoothed dF/F of each accepted cell
   (not on the OASIS output: its decay is the AR(1) model's tau, not a
   measurement). For each event: amplitude, 10-90 % rise time, maximal rise
   slope, half-decay time, a single-exponential decay tau fitted to the
   measured decay, full width at half maximum, area and the interval to the
   previous event. Events whose baseline is still raised by the previous
   event, or whose decay is cut by the next one, are flagged as not isolated.
2. Cell summaries: median event kinetics, event rate, burstiness, SNR.
3. Clustering at two levels (events, cells): log-transform, robust z-score,
   Gaussian mixture models with K = 1..K_max chosen by BIC, then
   - bootstrap stability (adjusted Rand index, per-cluster Jaccard),
   - silhouette,
   - a parametric bootstrap against one Gaussian (is K > 1 needed at all?),
   - for event clusters: do a cell's events share one type more than chance
     (purity vs label permutation) and how much of each kinetic parameter's
     variance lies between cells (ICC),
   - for cell clusters: confound checks against SNR and footprint size.

1-photon kinetics depend on indicator expression, SNR and neuropil, so the
clusters are hypotheses about cell types, not identifications.

numpy + scipy.ndimage only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from ..workspace import Workspace
from .common import check_stop, save_result, write_csv

# event features: key, label shown in the app, log-transform before clustering
EVENT_FEATURES = [
    ("rise_s", "上升时间 10–90%（秒）", True),
    ("decay_tau_s", "衰减时间常数 τ（秒）", True),
    ("t_half_s", "半衰减时间（秒）", True),
    ("fwhm_s", "半高宽（秒）", True),
    ("amp_dff", "振幅 ΔF/F", True),
    ("area", "面积（ΔF/F·秒）", True),
    ("max_slope", "最大上升速度（ΔF/F/秒）", True),
    ("amp_snr", "振幅 / 噪声", True),
    ("iei_s", "与上一事件的间隔（秒）", True),
    ("decay_r2", "衰减拟合 R²", False),
]
EV_INDEX = {k: i for i, (k, _, _) in enumerate(EVENT_FEATURES)}
# features that may be chosen for clustering (event level / cell level)
CLUSTER_CHOICES = {
    "rise_s": "上升时间", "decay_tau_s": "衰减 τ", "fwhm_s": "半高宽", "amp_dff": "振幅",
    "area": "面积", "t_half_s": "半衰减时间", "rate": "事件频率", "burstiness": "爆发性",
}
CELL_FEATURES = [
    ("rise_s", "上升时间中位数（秒）", True),
    ("decay_tau_s", "衰减 τ 中位数（秒）", True),
    ("t_half_s", "半衰减时间中位数（秒）", True),
    ("fwhm_s", "半高宽中位数（秒）", True),
    ("amp_dff", "振幅中位数 ΔF/F", True),
    ("area", "面积中位数", True),
    ("rate", "事件频率（次/分钟）", True),
    ("burstiness", "爆发性 B", False),
    ("cv_iei", "间隔变异系数", True),
    ("snr", "信噪比（振幅/噪声）", True),
    ("footprint", "细胞面积（像素）", True),
    ("n_events", "事件数", False),
]
CELL_INDEX = {k: i for i, (k, _, _) in enumerate(CELL_FEATURES)}
MAX_FIT = 5000          # events used to fit the mixture (all are assigned afterwards)
MAX_WAVE = 1500         # event waveforms kept for the heat map


@dataclass
class KineticsParams:
    smooth_s: float = 0.05          # Gaussian smoothing of dF/F before measuring
    threshold_sd: float = 3.0       # event = rise above the preceding trough by this many noise SDs
    baseline_s: float = 2.0         # how far back to look for the pre-event trough
    max_decay_s: float = 6.0        # longest decay window used for the fit
    isolated_only: bool = True      # cluster only events with clean baseline and decay
    min_r2: float = 0.5             # decay fit must explain this much of the decay
    features: tuple = ("rise_s", "decay_tau_s", "fwhm_s", "rate", "burstiness")   # amplitude off: it mostly reflects expression
    min_events: int = 5             # isolated events a cell needs for the cell level
    max_k: int = 6
    force_k: int = 0                # 0 = choose K automatically
    criterion: str = "icl"          # "icl" (favours well separated groups) or "bic"
    n_boot: int = 50
    n_null: int = 19
    seed: int = 0


# ======================================================== event detection
def _cross_back(y, i, lo, level):
    """Fractional index where y last rises through `level` in [lo, i]."""
    seg = y[lo:i + 1]
    below = np.flatnonzero(seg <= level)
    if not len(below):
        return np.nan
    j = lo + below[-1]
    if j >= i:
        return float(i)
    d = y[j + 1] - y[j]
    return j + ((level - y[j]) / d if d > 0 else 0.0)


def _cross_fwd(y, i, hi, level):
    """Fractional index where y first falls through `level` in (i, hi)."""
    seg = y[i:hi]
    below = np.flatnonzero(seg <= level)
    if not len(below):
        return np.nan
    j = i + below[0]
    d = y[j - 1] - y[j]
    return j - 1 + ((y[j - 1] - level) / d if d > 0 else 1.0)


_TAU_GRID = np.exp(np.linspace(np.log(0.02), np.log(30.0), 120))


def fit_decay(seg, fps):
    """Single exponential a*exp(-t/tau) through the baseline-subtracted decay.

    Grid search over tau with the amplitude solved in closed form, refined by a
    parabola in log(tau). Returns (tau, r2); NaN when it does not converge.
    """
    L = len(seg)
    if L < 4:
        return np.nan, np.nan
    t = np.arange(L) / fps
    E = np.exp(-t[None, :] / _TAU_GRID[:, None])
    ee = (E * E).sum(1)
    a = np.maximum((E @ seg) / ee, 0)
    sse = ((seg[None, :] - a[:, None] * E) ** 2).sum(1)
    j = int(np.argmin(sse))
    if j == 0 or j == len(_TAU_GRID) - 1:
        return np.nan, np.nan
    lg = np.log(_TAU_GRID)
    s0, s1, s2 = sse[j - 1], sse[j], sse[j + 1]
    den = s0 - 2 * s1 + s2
    off = 0.5 * (s0 - s2) / den if den > 0 else 0.0
    tau = float(np.exp(lg[j] + np.clip(off, -1, 1) * (lg[1] - lg[0])))
    tot = float(((seg - seg.mean()) ** 2).sum())
    r2 = 1 - s1 / tot if tot > 0 else np.nan
    return tau, float(r2)


def trace_baseline(y, sn):
    """Resting level of a trace: median of the samples that are not part of events."""
    b0 = float(np.median(y))
    for _ in range(4):
        q = y[y < b0 + 2 * sn]
        if len(q) < 10:
            break
        b0 = float(np.median(q))
    return b0


def detect_events(y, sn, fps, p: KineticsParams, b0=None):
    """Peaks of one smoothed trace: [peak frame, preceding trough value] per event.

    A peak counts when it is threshold_sd noise SDs above the resting level and
    also rises that much above the trough since the previous event.
    """
    T = len(y)
    thr = p.threshold_sd * sn
    b0 = trace_baseline(y, sn) if b0 is None else b0
    w = max(2, int(round(p.baseline_s * fps)))
    pad = np.concatenate([np.full(w, y[0]), y])
    run_min = np.lib.stride_tricks.sliding_window_view(pad, w + 1).min(1)[:T]
    loc = np.flatnonzero((y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:])) + 1
    cand = loc[((y[loc] - run_min[loc]) >= thr) & ((y[loc] - b0) >= thr)]
    kept = []
    for i in cand:
        lo = max(i - w, kept[-1][0] if kept else 0)
        b = float(y[lo:i + 1].min())
        if y[i] - b >= thr:
            kept.append([int(i), b])
        elif kept and y[i] > y[kept[-1][0]]:
            # a higher peak without a real dip in between: the earlier one was a shoulder
            kept[-1][0] = int(i)
    return kept


def smooth_pair(x, fps, smooth_s):
    """(trace to measure on, more smoothed trace to detect events on)."""
    s = smooth_s * fps
    y = ndimage.gaussian_filter1d(x, s) if s > 0.3 else np.array(x, float)
    sd = max(s, 2.0)
    return y, ndimage.gaussian_filter1d(x, sd)


def measure_cell(y, sn, fps, p: KineticsParams, n_pre, n_post, y_det=None):
    """All events of one cell: feature matrix, peak frames, isolated flags, waveforms.

    Events are found on y_det (smoothed a bit more, so noise ripples on a decay
    are not taken for events, troughs are not pulled down by noise) and measured on y.
    """
    T = len(y)
    if y_det is None:
        y_det = ndimage.gaussian_filter1d(y, 2.0)
    b0 = trace_baseline(y_det, sn)
    peaks = detect_events(y_det, sn, fps, p, b0)
    r = 3
    for q in peaks:     # exact peak on the measuring trace
        a = max(0, q[0] - r)
        q[0] = a + int(np.argmax(y[a:min(T, q[0] + r + 1)]))
    for e in range(1, len(peaks)):
        if peaks[e][0] <= peaks[e - 1][0]:
            peaks[e][0] = peaks[e - 1][0] + 1
    peaks = [q for q in peaks if q[0] < T]
    n = len(peaks)
    F = np.full((n, len(EVENT_FEATURES)), np.nan)
    iso = np.zeros(n, bool)
    wav = np.full((n, n_pre + n_post + 1), np.nan, np.float32)
    max_dec = max(4, int(round(p.max_decay_s * fps)))
    w = max(2, int(round(p.baseline_s * fps)))
    onsets = []
    for e, (pk, b) in enumerate(peaks):
        amp = y[pk] - b
        lo = max(pk - w, peaks[e - 1][0] if e else 0)
        on = _cross_back(y, pk, lo, b + 0.1 * amp)
        onsets.append(on)
    for e, (pk, b) in enumerate(peaks):
        amp = y[pk] - b
        lo = max(pk - w, peaks[e - 1][0] if e else 0)
        nxt = int(np.floor(onsets[e + 1])) if e + 1 < n and np.isfinite(onsets[e + 1]) else T
        nxt = peaks[e + 1][0] if e + 1 < n and nxt <= pk else nxt
        hi = min(T, pk + max_dec + 1, max(pk + 1, nxt + 1))
        t10 = _cross_back(y, pk, lo, b + 0.1 * amp)
        t90 = _cross_back(y, pk, lo, b + 0.9 * amp)
        l50 = _cross_back(y, pk, lo, b + 0.5 * amp)
        r50 = _cross_fwd(y, pk, hi, b + 0.5 * amp)
        r20 = _cross_fwd(y, pk, hi, b + 0.2 * amp)
        f = F[e]
        f[EV_INDEX["amp_dff"]] = amp
        f[EV_INDEX["amp_snr"]] = amp / sn if sn > 0 else np.nan
        if np.isfinite(t10) and np.isfinite(t90) and t90 > t10:
            f[EV_INDEX["rise_s"]] = (t90 - t10) / fps
        elif np.isfinite(t10):
            f[EV_INDEX["rise_s"]] = 0.5 / fps     # faster than one frame
        a0 = int(np.floor(t10)) if np.isfinite(t10) else lo
        if pk > a0:
            f[EV_INDEX["max_slope"]] = float(np.max(np.diff(y[a0:pk + 1]))) * fps
        if np.isfinite(r50):
            f[EV_INDEX["t_half_s"]] = (r50 - pk) / fps
            if np.isfinite(l50):
                f[EV_INDEX["fwhm_s"]] = (r50 - l50) / fps
        tau, r2 = fit_decay(np.asarray(y[pk:hi] - b, float), fps)
        f[EV_INDEX["decay_tau_s"]] = tau
        f[EV_INDEX["decay_r2"]] = r2
        on = int(np.floor(onsets[e])) if np.isfinite(onsets[e]) else lo
        f[EV_INDEX["area"]] = float(np.sum(y[on:hi] - b)) / fps
        if e:
            f[EV_INDEX["iei_s"]] = (pk - peaks[e - 1][0]) / fps
        # isolated: the trough before it is near the cell's baseline and the decay
        # reaches 20 % before the next event starts
        clean_pre = b - b0 < max(0.25 * amp, 2.0 * sn)
        clean_post = np.isfinite(r20) and r20 < nxt
        iso[e] = bool(clean_pre and clean_post)
        a, z = pk - n_pre, pk + n_post + 1
        s0, s1 = max(0, a), min(T, z)
        if amp > 0:
            wav[e, s0 - a:s1 - a] = (y[s0:s1] - b) / amp
    pk_frames = np.array([q[0] for q in peaks], int)
    return F, pk_frames, iso, wav


def burstiness(iei):
    """Goh & Barabasi burstiness B = (sd - mean) / (sd + mean): -1 regular, 0 Poisson, 1 bursty."""
    iei = np.asarray(iei, float)
    iei = iei[np.isfinite(iei)]
    if len(iei) < 3:
        return np.nan
    m, s = iei.mean(), iei.std()
    return float((s - m) / (s + m)) if s + m > 0 else np.nan


# ============================================================ clustering
def standardize(F, logs):
    X = np.array(F, float)
    for j, lg in enumerate(logs):
        if lg:
            X[:, j] = np.log10(np.maximum(X[:, j], 1e-6))
    med = np.median(X, 0)
    mad = 1.4826 * np.median(np.abs(X - med), 0)
    sd = X.std(0)
    scale = np.where(mad > 1e-9, mad, np.where(sd > 1e-9, sd, 1.0))
    return (X - med) / scale


def _log_gauss(X, mu, cov):
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (z * z).sum(0) - np.log(np.diag(L)).sum() - 0.5 * d * np.log(2 * np.pi)


def _logsumexp(a, axis=1):
    m = a.max(axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis, keepdims=True))).squeeze(axis)


def _kmeanspp(X, K, rng):
    n = len(X)
    c = [X[rng.integers(n)]]
    for _ in range(1, K):
        d = np.min([((X - ci) ** 2).sum(1) for ci in c], 0)
        pr = d / d.sum() if d.sum() > 0 else np.full(n, 1.0 / n)
        c.append(X[rng.choice(n, p=pr)])
    return np.array(c)


def gmm_fit(X, K, rng, n_init=4, max_iter=200, tol=1e-5, reg=1e-3):
    """Full-covariance Gaussian mixture by EM, best of n_init k-means++ starts."""
    n, d = X.shape
    best = None
    for _ in range(n_init if K > 1 else 1):
        if K == 1:
            lab = np.zeros(n, int)
        else:
            C = _kmeanspp(X, K, rng)
            for _ in range(10):
                lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(2), 1)
                C = np.array([X[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(K)])
        R = np.zeros((n, K))
        R[np.arange(n), lab] = 1
        prev = -np.inf
        for _ in range(max_iter):
            Nk = R.sum(0) + 1e-10
            w = Nk / n
            mu = (R.T @ X) / Nk[:, None]
            cov = np.empty((K, d, d))
            for k in range(K):
                D = X - mu[k]
                cov[k] = (R[:, k, None] * D).T @ D / Nk[k] + reg * np.eye(d)
            lp = np.column_stack([np.log(w[k]) + _log_gauss(X, mu[k], cov[k]) for k in range(K)])
            ll_i = _logsumexp(lp)
            ll = float(ll_i.sum())
            R = np.exp(lp - ll_i[:, None])
            if ll - prev < tol * abs(ll):
                break
            prev = ll
        if best is None or ll > best["ll"]:
            best = dict(w=w, mu=mu, cov=cov, ll=ll, ent=float(-(R * np.log(np.maximum(R, 1e-300))).sum()))
    p = (K - 1) + K * d + K * d * (d + 1) / 2
    best["bic"] = -2 * best["ll"] + p * np.log(n)
    # ICL (Biernacki et al. 2000): BIC plus twice the assignment entropy, so that
    # overlapping components used only to bend a skewed distribution are penalised
    best["icl"] = best["bic"] + 2 * best["ent"]
    best["K"] = K
    return best


def gmm_predict(model, X):
    lp = np.column_stack([np.log(model["w"][k]) + _log_gauss(X, model["mu"][k], model["cov"][k])
                          for k in range(model["K"])])
    R = np.exp(lp - _logsumexp(lp)[:, None])
    return R.argmax(1), R.max(1)


def ari(a, b):
    """Adjusted Rand index of two labelings."""
    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    M = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(M, (a, b), 1)
    c2 = lambda x: x * (x - 1) / 2
    s = c2(M).sum()
    sa, sb = c2(M.sum(1)).sum(), c2(M.sum(0)).sum()
    tot = c2(len(a))
    exp = sa * sb / tot if tot else 0
    mx = 0.5 * (sa + sb)
    return float((s - exp) / (mx - exp)) if mx != exp else 1.0


def silhouette(X, lab, rng, max_n=2000):
    if len(np.unique(lab)) < 2:
        return np.nan
    if len(X) > max_n:
        idx = rng.choice(len(X), max_n, replace=False)
        X, lab = X[idx], lab[idx]
    D = np.sqrt(np.maximum(((X[:, None, :] - X[None]) ** 2).sum(2), 0))
    ks = np.unique(lab)
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = lab == lab[i]
        same[i] = False
        if not same.any():
            continue
        a = D[i, same].mean()
        b = min(D[i, lab == k].mean() for k in ks if k != lab[i])
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0
    return float(s.mean())


def cluster_level(Z, p: KineticsParams, rng, progress=None, should_stop=None, frac=(0.0, 1.0), name=""):
    """GMM model selection + stability + null test on the standardized features Z."""
    progress = progress or (lambda f, m: None)
    n, d = Z.shape
    fit_idx = rng.choice(n, MAX_FIT, replace=False) if n > MAX_FIT else np.arange(n)
    Zf = Z[fit_idx]
    nf = len(Zf)
    k_max = int(max(1, min(p.max_k, nf // max(10, 2 * d))))
    ks = np.arange(1, k_max + 1)
    models = []
    for k in ks:
        check_stop(should_stop)
        progress(frac[0] + 0.3 * (frac[1] - frac[0]) * k / k_max, f"{name}：拟合 {k} 类模型…")
        models.append(gmm_fit(Zf, int(k), rng))
    bic = np.array([m["bic"] for m in models])
    icl = np.array([m["icl"] for m in models])
    crit = icl if p.criterion == "icl" else bic
    K = int(min(p.force_k, k_max)) if p.force_k else int(ks[np.argmin(crit)])
    model = models[K - 1]
    lab, prob = gmm_predict(model, Z)
    # order clusters by the first feature (decay or rise), fastest first
    order = np.argsort(model["mu"][:, 0])
    remap = np.empty(K, int)
    remap[order] = np.arange(K)
    lab = remap[lab]
    out = dict(K=K, ks=ks, bic=bic, icl=icl, crit=crit, labels=lab, prob=prob, means=model["mu"][order])
    # bootstrap stability
    aris, jac = [], np.zeros((0, K))
    if K > 1:
        for b in range(p.n_boot):
            check_stop(should_stop)
            if b % 5 == 0:
                progress(frac[0] + (0.3 + 0.5 * b / max(1, p.n_boot)) * (frac[1] - frac[0]),
                         f"{name}：稳定性检验 {b}/{p.n_boot}…")
            bi = rng.choice(nf, nf, replace=True)
            m = gmm_fit(Zf[bi], K, rng, n_init=2, max_iter=100)
            lb, _ = gmm_predict(m, Zf)
            ref = lab[fit_idx]
            aris.append(ari(ref, lb))
            row = []
            for c in range(K):
                A = ref == c
                row.append(max((A & (lb == k)).sum() / max(1, (A | (lb == k)).sum()) for k in range(K)))
            jac = np.vstack([jac, row])
    out["boot_ari"] = np.array(aris)
    out["jaccard"] = jac.mean(0) if len(jac) else np.full(K, np.nan)
    out["silhouette"] = silhouette(Z[fit_idx], lab[fit_idx], rng)
    # parametric bootstrap: how much does the best K > 1 beat K = 1 in one Gaussian?
    gain = float(crit[0] - crit.min())
    null = []
    if k_max > 1 and p.n_null > 0:
        mu, cov = Zf.mean(0), np.cov(Zf.T).reshape(d, d) + 1e-6 * np.eye(d)
        nn = min(nf, 2000)
        for b in range(p.n_null):
            check_stop(should_stop)
            progress(frac[0] + (0.8 + 0.2 * b / p.n_null) * (frac[1] - frac[0]), f"{name}：与单峰分布比较…")
            Y = rng.multivariate_normal(mu, cov, nn)
            bb = [gmm_fit(Y, int(k), rng, n_init=2, max_iter=100)[p.criterion] for k in ks]
            null.append((bb[0] - min(bb)) * nf / nn)
    out["gain"] = gain
    out["null_gain"] = np.array(null)
    out["null_p"] = float((1 + np.sum(np.array(null) >= gain)) / (1 + len(null))) if null else np.nan
    return out


def icc1(values, groups):
    """One-way random-effects ICC(1): share of variance that lies between groups."""
    v = np.asarray(values, float)
    g = np.asarray(groups)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    u, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= 2
    v, inv = v[keep], np.unique(inv[keep], return_inverse=True)[1]
    k = inv.max() + 1 if len(inv) else 0
    N = len(v)
    if k < 2 or N <= k:
        return np.nan
    n_i = np.bincount(inv)
    m_i = np.bincount(inv, v) / n_i
    gm = v.mean()
    msb = (n_i * (m_i - gm) ** 2).sum() / (k - 1)
    msw = ((v - m_i[inv]) ** 2).sum() / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    den = msb + (n0 - 1) * msw
    return float((msb - msw) / den) if den > 0 else np.nan


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return np.nan
    ra, rb = _rank(a[ok]), _rank(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _rank(v):
    """Ranks with ties given their average rank."""
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    r[order] = np.arange(len(v), dtype=float)
    u, inv = np.unique(v, return_inverse=True)
    return (np.bincount(inv, r) / np.bincount(inv))[inv]


def eta_squared(v, lab, rng, n_perm=500):
    """Share of the variance of v explained by the labels, with a permutation p."""
    v = np.asarray(v, float)
    ok = np.isfinite(v) & (lab >= 0)
    v, lab = v[ok], lab[ok]
    if len(np.unique(lab)) < 2:
        return np.nan, np.nan

    def e2(l):
        m = v.mean()
        between = sum(((l == k).sum() * (v[l == k].mean() - m) ** 2) for k in np.unique(l))
        tot = ((v - m) ** 2).sum()
        return between / tot if tot > 0 else 0.0
    obs = e2(lab)
    null = np.array([e2(rng.permutation(lab)) for _ in range(n_perm)])
    return float(obs), float((1 + (null >= obs).sum()) / (1 + n_perm))


def purity(ev_cell, ev_lab, K, min_events):
    """Event-weighted mean fraction of each cell's events that fall in its main cluster."""
    cells = np.unique(ev_cell)
    num = den = 0.0
    for c in cells:
        l = ev_lab[ev_cell == c]
        if len(l) < min_events:
            continue
        num += np.bincount(l, minlength=K).max()
        den += len(l)
    return num / den if den else np.nan


# ================================================================== run
def run_kinetics(session, ws: Workspace, args: dict, progress=None, should_stop=None) -> dict:
    from ..extraction import load_cells
    from ..signals import load_signals, noise_std
    progress = progress or (lambda f, m: None)
    p: KineticsParams = args["params"]
    if not ws.step("signals"):
        raise ValueError("请先完成第 6 步「信号与 spike」")
    rng = np.random.default_rng(p.seed)
    sig = load_signals(ws)
    cells = load_cells(ws)
    acc = cells["accepted"].astype(bool)
    if acc.sum() < 2:
        raise ValueError("保留的细胞太少（至少需要 2 个）")
    t = np.asarray(sig["times"], float)
    fps = float(1.0 / np.median(np.diff(t)))
    dff = np.asarray(sig["dff"], float)[acc]
    ids = np.where(acc)[0]
    centres = np.asarray(cells["centres"], float)[acc]
    A = cells["A"]
    footprint = np.diff(A.indptr)[acc].astype(float) if hasattr(A, "indptr") else np.full(len(ids), np.nan)
    N, T = dff.shape
    duration_min = T / fps / 60.0
    n_pre, n_post = int(round(1.0 * fps)), int(round(p.max_decay_s * fps))
    wave_t = np.arange(-n_pre, n_post + 1) / fps

    feats, cell_of, pk_all, iso_all, waves = [], [], [], [], []
    for k in range(N):
        if k % 10 == 0:
            check_stop(should_stop)
            progress(0.3 * k / N, f"检测钙事件并测量动力学参数（{k}/{N}）…")
        x = dff[k]
        y, y_det = smooth_pair(x, fps, p.smooth_s)
        sn = noise_std(x) or _mad_noise(x)
        F, pk, iso, wav = measure_cell(y, sn, fps, p, n_pre, n_post, y_det)
        feats.append(F); pk_all.append(pk); iso_all.append(iso); waves.append(wav)
        cell_of.append(np.full(len(pk), k, int))
    F = np.vstack(feats) if feats else np.zeros((0, len(EVENT_FEATURES)))
    ev_cell = np.concatenate(cell_of)
    ev_peak = np.concatenate(pk_all)
    ev_iso = np.concatenate(iso_all)
    W = np.vstack(waves) if len(ev_peak) else np.zeros((0, len(wave_t)), np.float32)
    n_ev = len(ev_peak)
    if n_ev < 10:
        raise ValueError(f"只检测到 {n_ev} 个钙事件，太少，无法聚类。可以降低「事件阈值」或检查第 6 步的信号。")
    ev_good = ev_iso.copy() if p.isolated_only else np.ones(n_ev, bool)
    r2 = F[:, EV_INDEX["decay_r2"]]
    ev_good &= np.isfinite(r2) & (r2 >= p.min_r2)

    # ------------------------------------------------ cell summaries
    C = np.full((N, len(CELL_FEATURES)), np.nan)
    for k in range(N):
        m = ev_cell == k
        g = m & ev_good
        for key in ("rise_s", "decay_tau_s", "t_half_s", "fwhm_s", "amp_dff", "area"):
            v = F[g, EV_INDEX[key]]
            v = v[np.isfinite(v)]
            if len(v):
                C[k, CELL_INDEX[key]] = np.median(v)
        C[k, CELL_INDEX["rate"]] = m.sum() / duration_min if duration_min > 0 else np.nan
        iei = np.diff(ev_peak[m]) / fps
        C[k, CELL_INDEX["burstiness"]] = burstiness(iei)
        if len(iei) >= 3 and iei.mean() > 0:
            C[k, CELL_INDEX["cv_iei"]] = iei.std() / iei.mean()
        if m.any():
            C[k, CELL_INDEX["snr"]] = np.median(F[m, EV_INDEX["amp_snr"]])
        C[k, CELL_INDEX["footprint"]] = footprint[k]
        C[k, CELL_INDEX["n_events"]] = g.sum()
    cell_wave = np.full((N, len(wave_t)), np.nan, np.float32)
    for k in range(N):
        g = (ev_cell == k) & ev_good
        if g.any():
            cell_wave[k] = np.nanmean(W[g], 0)

    summary = {"n_cells": int(N), "n_events": int(n_ev), "n_isolated": int(ev_iso.sum()),
               "n_used_events": int(ev_good.sum()), "fps": fps, "duration_min": duration_min,
               "median": {k: _nanmed(F[ev_good, EV_INDEX[k]]) for k in
                          ("rise_s", "decay_tau_s", "t_half_s", "fwhm_s", "amp_dff")},
               "features": list(p.features)}
    arrays = dict(ids=ids, centres=centres, shape=np.array(cells["shape"]), times=t,
                  ev_cell=ev_cell, ev_time=t[ev_peak], ev_peak=ev_peak, ev_feat=F, ev_iso=ev_iso,
                  ev_good=ev_good, ev_feat_names=np.array([k for k, _, _ in EVENT_FEATURES]),
                  cell_feat=C, cell_feat_names=np.array([k for k, _, _ in CELL_FEATURES]),
                  wave_t=wave_t, cell_wave=cell_wave)

    # ------------------------------------------------ event level
    ev_keys = _decay_first([k for k in p.features if k in EV_INDEX] or ["rise_s", "decay_tau_s", "fwhm_s"])
    idx = [EV_INDEX[k] for k in ev_keys]
    Xe = F[:, idx]
    use = ev_good & np.isfinite(Xe).all(1)
    ev_lab = np.full(n_ev, -1)
    s_ev = {"used": int(use.sum()), "features": ev_keys}
    if use.sum() >= 30:
        Z = standardize(Xe[use], [EVENT_FEATURES[i][2] for i in idx])
        r = cluster_level(Z, p, rng, progress, should_stop, (0.3, 0.65), "事件聚类")
        ev_lab[use] = r["labels"]
        K = r["K"]
        pc = _pca2(Z)
        ev_pc = np.full((n_ev, 2), np.nan)
        ev_pc[use] = pc
        ev_prob = np.full(n_ev, np.nan)
        ev_prob[use] = r["prob"]
        mean_w = np.array([np.nanmean(W[ev_lab == c], 0) for c in range(K)])
        sem_w = np.array([np.nanstd(W[ev_lab == c], 0) / np.sqrt(max(1, (ev_lab == c).sum())) for c in range(K)])
        sub = np.flatnonzero(use)
        if len(sub) > MAX_WAVE:
            sub = np.sort(rng.choice(sub, MAX_WAVE, replace=False))
        sub = sub[np.argsort(ev_lab[sub], kind="stable")]
        # composition of each cell's events and whether a cell's events share a type
        comp = np.full((N, K), np.nan)
        for k in range(N):
            l = ev_lab[(ev_cell == k) & use]
            if len(l):
                comp[k] = np.bincount(l, minlength=K) / len(l)
        pur = purity(ev_cell[use], ev_lab[use], K, p.min_events) if K > 1 else np.nan
        null = []
        if K > 1 and np.isfinite(pur):
            lu, cu = ev_lab[use], ev_cell[use]
            for _ in range(200):
                null.append(purity(cu, rng.permutation(lu), K, p.min_events))
        null = np.array(null)
        e2_snr_ev, p_snr_ev = eta_squared(np.log10(np.maximum(F[use, EV_INDEX["amp_snr"]], 1e-6)),
                                          r["labels"], rng, n_perm=200)
        icc = {k: icc1(np.log10(np.maximum(F[use, EV_INDEX[k]], 1e-6)), ev_cell[use]) for k in ev_keys}
        arrays.update(ev_label=ev_lab, ev_prob=ev_prob, ev_pc=ev_pc, ev_ks=r["ks"], ev_bic=r["bic"], ev_icl=r["icl"],
                      ev_boot_ari=r["boot_ari"], ev_jaccard=r["jaccard"], ev_means=r["means"],
                      ev_wave_mean=mean_w, ev_wave_sem=sem_w, ev_wave_sub=W[sub], ev_wave_sub_label=ev_lab[sub],
                      ev_null_gain=r["null_gain"], cell_comp=comp,
                      cell_dominant=np.where(np.isfinite(comp).all(1), np.nan_to_num(comp).argmax(1), -1),
                      ev_profile=_profile(Z, r["labels"], K))
        s_ev.update(K=K, sizes=np.bincount(r["labels"], minlength=K).tolist(), gain=r["gain"], criterion=p.criterion,
                    null_p=r["null_p"], silhouette=r["silhouette"],
                    ari=_nanmean(r["boot_ari"]), jaccard=r["jaccard"].tolist(),
                    purity=pur, purity_null=_nanmean(null),
                    purity_p=float((1 + (null >= pur).sum()) / (1 + len(null))) if len(null) else np.nan,
                    icc=icc, forced=bool(p.force_k), eta2_snr=e2_snr_ev, p_snr=p_snr_ev,
                    cluster_median={k: [_nanmed(F[ev_lab == c, EV_INDEX[k]]) for c in range(K)] for k in ev_keys})
    else:
        s_ev["skipped"] = f"可用于聚类的事件只有 {int(use.sum())} 个（至少需要 30 个）"
    arrays["ev_label"] = ev_lab
    summary["event"] = s_ev

    # ------------------------------------------------ cell level
    c_keys = _decay_first([k for k in p.features if k in CELL_INDEX] or ["rise_s", "decay_tau_s", "rate"])
    cidx = [CELL_INDEX[k] for k in c_keys]
    Xc = C[:, cidx]
    cuse = (C[:, CELL_INDEX["n_events"]] >= p.min_events) & np.isfinite(Xc).all(1)
    cell_lab = np.full(N, -1)
    s_c = {"used": int(cuse.sum()), "features": c_keys}
    if cuse.sum() >= 10:
        Z = standardize(Xc[cuse], [CELL_FEATURES[i][2] for i in cidx])
        r = cluster_level(Z, p, rng, progress, should_stop, (0.65, 0.97), "细胞聚类")
        cell_lab[cuse] = r["labels"]
        K = r["K"]
        cpc = np.full((N, 2), np.nan)
        cpc[cuse] = _pca2(Z)
        cprob = np.full(N, np.nan)
        cprob[cuse] = r["prob"]
        cw = np.array([np.nanmean(cell_wave[cell_lab == c], 0) for c in range(K)])
        cs = np.array([np.nanstd(cell_wave[cell_lab == c], 0) / np.sqrt(max(1, (cell_lab == c).sum()))
                       for c in range(K)])
        snr, area = C[:, CELL_INDEX["snr"]], C[:, CELL_INDEX["footprint"]]
        e2_snr, p_snr = eta_squared(np.log10(np.maximum(snr, 1e-6)), cell_lab, rng)
        e2_area, p_area = eta_squared(area, cell_lab, rng)
        arrays.update(cell_label=cell_lab, cell_prob=cprob, cell_pc=cpc, cell_ks=r["ks"], cell_bic=r["bic"], cell_icl=r["icl"],
                      cell_boot_ari=r["boot_ari"], cell_jaccard=r["jaccard"], cell_means=r["means"],
                      cell_wave_mean=cw, cell_wave_sem=cs, cell_null_gain=r["null_gain"],
                      cell_profile=_profile(Z, r["labels"], K))
        s_c.update(K=K, sizes=np.bincount(r["labels"], minlength=K).tolist(), gain=r["gain"], criterion=p.criterion,
                   null_p=r["null_p"], silhouette=r["silhouette"], ari=_nanmean(r["boot_ari"]),
                   jaccard=r["jaccard"].tolist(), forced=bool(p.force_k),
                   eta2_snr=e2_snr, p_snr=p_snr, eta2_area=e2_area, p_area=p_area,
                   cluster_median={k: [_nanmed(C[cell_lab == c, CELL_INDEX[k]]) for c in range(K)] for k in c_keys})
    else:
        s_c["skipped"] = f"事件数达到 {p.min_events} 个的细胞只有 {int(cuse.sum())} 个（至少需要 10 个）"
    arrays["cell_label"] = cell_lab
    tau_c, snr_c = C[:, CELL_INDEX["decay_tau_s"]], C[:, CELL_INDEX["snr"]]
    s_c["r_tau_snr"] = spearman(tau_c, snr_c)
    s_c["r_tau_area"] = spearman(tau_c, C[:, CELL_INDEX["footprint"]])
    s_c["r_rise_snr"] = spearman(C[:, CELL_INDEX["rise_s"]], snr_c)
    summary["cell"] = s_c
    params = asdict(p)
    params["features"] = list(p.features)
    save_result(ws, "kinetics", arrays, summary, params)
    progress(1.0, "钙事件聚类完成")
    return summary


def _decay_first(keys):
    """Clusters are numbered by the first feature, so put the decay first (cluster 1 = fastest)."""
    for k in ("decay_tau_s", "t_half_s", "fwhm_s", "rise_s"):
        if k in keys:
            return [k] + [x for x in keys if x != k]
    return list(keys)


def _mad_noise(x):
    d = np.diff(x)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) or float(np.std(x)) or 1e-6


def _nanmed(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else np.nan


def _nanmean(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else np.nan


def _pca2(Z):
    Zc = Z - Z.mean(0)
    if Z.shape[1] == 1:
        return np.column_stack([Zc[:, 0], np.zeros(len(Z))])
    U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
    return U[:, :2] * s[:2]


def _profile(Z, lab, K):
    return np.array([Z[lab == c].mean(0) if (lab == c).any() else np.full(Z.shape[1], np.nan) for c in range(K)])


# ================================================================ export
def export_kinetics(ws: Workspace, res: dict, folder):
    from pathlib import Path
    folder = Path(folder)
    F = res["ev_feat"]
    names = [str(n) for n in res["ev_feat_names"]]
    ids = res["ids"]
    cols = {"cell_id": ids[res["ev_cell"]], "time_s": res["ev_time"], "peak_frame": res["ev_peak"],
            "isolated": res["ev_iso"].astype(int), "used_for_clustering": res["ev_good"].astype(int)}
    for j, n in enumerate(names):
        cols[n] = F[:, j]
    cols["event_cluster"] = np.where(res["ev_label"] >= 0, res["ev_label"] + 1, 0)
    if "ev_prob" in res:
        cols["cluster_probability"] = res["ev_prob"]
    write_csv(folder / "kinetics_events.csv", cols)

    C = res["cell_feat"]
    cols = {"cell_id": ids, "y": res["centres"][:, 0], "x": res["centres"][:, 1]}
    for j, n in enumerate(str(v) for v in res["cell_feat_names"]):
        cols[n] = C[:, j]
    cols["cell_cluster"] = np.where(res["cell_label"] >= 0, res["cell_label"] + 1, 0)
    if "cell_prob" in res:
        cols["cell_cluster_probability"] = res["cell_prob"]
    if "cell_comp" in res:
        for c in range(res["cell_comp"].shape[1]):
            cols[f"frac_event_cluster_{c + 1}"] = res["cell_comp"][:, c]
        cols["main_event_cluster"] = res["cell_dominant"] + 1
    write_csv(folder / "kinetics_cells.csv", cols)

    s = res.get("summary", {})
    rows = {"level": [], "cluster": [], "n": [], "stability_jaccard": []}
    for lvl in ("event", "cell"):
        sl = s.get(lvl, {})
        if "K" not in sl:
            continue
        for c in range(sl["K"]):
            rows["level"].append(lvl)
            rows["cluster"].append(c + 1)
            rows["n"].append(int(sl["sizes"][c]))
            j = sl["jaccard"][c]
            rows["stability_jaccard"].append(float(j) if j is not None else np.nan)
    # medians per feature (blank where a feature was not used at that level)
    all_keys = sorted({k for lvl in ("event", "cell") for k in s.get(lvl, {}).get("cluster_median", {})})
    for k in all_keys:
        col = []
        for lvl, c in zip(rows["level"], rows["cluster"]):
            v = s.get(lvl, {}).get("cluster_median", {}).get(k)
            col.append(float(v[c - 1]) if v is not None and v[c - 1] is not None else np.nan)
        rows[f"median_{k}"] = col
    if rows["level"]:
        write_csv(folder / "kinetics_clusters.csv", rows)

    cols = {"time_s": res["wave_t"]}
    for pre, lab in (("ev_wave_mean", "event_cluster"), ("cell_wave_mean", "cell_cluster")):
        if pre in res:
            for c, w in enumerate(res[pre]):
                cols[f"{lab}_{c + 1}"] = w
    return write_csv(folder / "kinetics_cluster_waveforms.csv", cols)
