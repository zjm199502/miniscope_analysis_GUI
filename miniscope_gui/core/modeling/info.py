"""Information carried by single cells about position and behaviour state.

Spatial information (Skaggs et al. 1993), in bits per event:
    SI = sum_i p_i (r_i / r) log2(r_i / r)
with p_i the occupancy probability of spatial bin i, r_i the mean activity
there and r the overall mean activity.

Mutual information between the (discretised) activity of a cell and a
discrete behaviour variable such as freezing or the zone the animal is in.

Both are tested against circular time shifts of the activity relative to the
behaviour, which keeps the temporal structure of both.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import scipy.sparse as sp
from scipy import ndimage

from ..workspace import Workspace
from .common import check_stop, fdr_bh, load_dataset, save_result, shift_amounts, write_csv


@dataclass
class InfoParams:
    bin_s: float = 0.2
    signal: str = "events"          # events | dff
    n_spatial_bins: int = 8
    smooth_bins: float = 1.0
    min_occupancy_s: float = 0.5
    min_speed: float = 0.0          # only bins where the animal moves faster than this
    n_levels: int = 2               # activity levels for the mutual information
    n_shuffle: int = 500
    seed: int = 0


# ------------------------------------------------------- spatial information
def _spatial_bins(x, y, n):
    ex = np.linspace(np.nanmin(x), np.nanmax(x) + 1e-9, n + 1)
    ey = np.linspace(np.nanmin(y), np.nanmax(y) + 1e-9, n + 1)
    ix = np.clip(np.digitize(x, ex) - 1, 0, n - 1)
    iy = np.clip(np.digitize(y, ey) - 1, 0, n - 1)
    return iy * n + ix, ex, ey


def _smooth_maps(M, n, sigma):
    """M (..., n*n) -> smoothed, same shape."""
    if sigma <= 0:
        return M
    shp = M.shape
    out = ndimage.gaussian_filter(M.reshape(-1, n, n), (0, sigma, sigma), mode="constant")
    return out.reshape(shp)


def _si(rate, occ_p):
    """rate (N, B), occ_p (B,) -> bits per event."""
    mean = rate @ occ_p
    ratio = rate / np.maximum(mean[:, None], 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(ratio > 0, ratio * np.log2(ratio), 0.0)
    return term @ occ_p


def spatial_information(act, x, y, fps, n_bins=8, smooth=1.0, min_occ_s=0.5, n_shuffle=500, seed=0,
                        should_stop=None, progress=None):
    """act (N, T) non-negative activity; x, y (T,) position. All T bins are used."""
    rng = np.random.default_rng(seed)
    N, T = act.shape
    b, ex, ey = _spatial_bins(x, y, n_bins)
    B = n_bins * n_bins
    onehot = sp.csr_matrix((np.ones(T), (np.arange(T), b)), shape=(T, B))
    occ = np.asarray(onehot.sum(0)).ravel() / fps                    # seconds
    occ_s = _smooth_maps(occ[None], n_bins, smooth)[0]
    ok = occ >= min_occ_s
    occ_p = np.where(ok, occ_s, 0.0)
    occ_p = occ_p / max(occ_p.sum(), 1e-12)

    def maps(a):
        s = _smooth_maps(np.asarray(a @ onehot) / fps, n_bins, smooth)
        return np.where(ok, s / np.maximum(occ_s, 1e-12), 0.0)
    rate = maps(act)
    si = _si(rate, occ_p)
    null = np.empty((n_shuffle, N))
    for i, k in enumerate(shift_amounts(rng, T, n_shuffle)):
        check_stop(should_stop)
        null[i] = _si(maps(np.roll(act, k, axis=1)), occ_p)
        if progress and i % 20 == 19:
            progress((i + 1) / n_shuffle)
    p = (1 + (null >= si).sum(0)) / (n_shuffle + 1)
    z = (si - null.mean(0)) / np.maximum(null.std(0), 1e-12)
    return dict(si=si, p=p, z=z, rate=rate.reshape(N, n_bins, n_bins),
                occupancy=np.where(ok, occ, 0).reshape(n_bins, n_bins), x_edges=ex, y_edges=ey)


# -------------------------------------------------------- mutual information
def discretize(act, n_levels=2):
    """(N, T) -> integer levels. Level 0 = silent; the active bins are split by quantile."""
    lev = np.zeros(act.shape, np.int8)
    for i, a in enumerate(act):
        pos = a > 0
        if n_levels <= 2 or pos.sum() < n_levels:
            lev[i] = pos
            continue
        q = np.quantile(a[pos], np.linspace(0, 1, n_levels)[1:-1])
        lev[i, pos] = 1 + np.searchsorted(q, a[pos], side="right")
    return lev


def _mi(lev, state_oh, n_levels):
    """lev (N, T) ints, state_oh (T, S) one-hot -> MI in bits per cell."""
    T = lev.shape[1]
    ps = state_oh.sum(0) / T
    mi = np.zeros(lev.shape[0])
    for l in range(n_levels):
        ind = (lev == l).astype(np.float32)
        joint = ind @ state_oh / T                      # (N, S)
        pa = ind.sum(1) / T
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(joint > 0, joint * np.log2(joint / (pa[:, None] * ps[None, :])), 0.0)
        mi += t.sum(1)
    return mi


def mutual_information(act, state, n_levels=2, n_shuffle=500, seed=0, should_stop=None, progress=None):
    """state (T,) integer labels. Returns bias-corrected MI (bits), raw MI, p."""
    rng = np.random.default_rng(seed)
    lev = discretize(act, n_levels)
    u, s = np.unique(state, return_inverse=True)
    oh = np.eye(len(u), dtype=np.float32)[s]
    T = len(state)
    mi = _mi(lev, oh, n_levels)
    null = np.empty((n_shuffle, lev.shape[0]))
    for i, k in enumerate(shift_amounts(rng, T, n_shuffle)):
        check_stop(should_stop)
        null[i] = _mi(lev, np.roll(oh, k, axis=0), n_levels)
        if progress and i % 20 == 19:
            progress((i + 1) / n_shuffle)
    p = (1 + (null >= mi).sum(0)) / (n_shuffle + 1)
    return dict(mi=np.maximum(mi - null.mean(0), 0.0), mi_raw=mi, p=p)


def run_info(session, ws: Workspace, args: dict, progress=None, should_stop=None) -> dict:
    progress = progress or (lambda f, m: None)
    p: InfoParams = args["params"]
    ds = load_dataset(ws, args.get("cfg"), p.bin_s)
    if not ds.beh:
        raise ValueError("信息量分析需要行为数据：请先完成第 7 步（行为）和第 8 步（同步）")
    act = np.maximum(ds.events if p.signal == "events" else ds.dff, 0)
    b, v = ds.beh, ds.valid
    arrays = dict(ids=ds.ids, centres=ds.centres, n_events=(ds.events[:, v] > 0).sum(1))
    summary = {"n_cells": int(ds.n_cells), "unit": b.get("unit", ""), "variables": []}
    if ds.has("x_u"):
        m = v & np.isfinite(b["x_u"]) & np.isfinite(b["y_u"])
        if p.min_speed > 0 and ds.has("speed"):
            m &= np.nan_to_num(b["speed"]) >= p.min_speed
        if m.sum() >= 100 and np.nanstd(b["x_u"][m]) > 0:
            progress(0.05, "空间信息量…")
            s = spatial_information(act[:, m], b["x_u"][m], b["y_u"][m], ds.fps, p.n_spatial_bins,
                                    p.smooth_bins, p.min_occupancy_s, p.n_shuffle, p.seed, should_stop,
                                    lambda f: progress(0.05 + 0.45 * f, "空间信息量：打乱检验…"))
            enough = arrays["n_events"] >= 5
            place = (s["p"] < 0.05) & enough
            arrays.update(si=s["si"], si_p=s["p"], si_q=fdr_bh(s["p"]), si_z=s["z"], rate_maps=s["rate"].astype(np.float32),
                          occupancy=s["occupancy"], x_edges=s["x_edges"], y_edges=s["y_edges"], place=place)
            summary.update(n_place=int(place.sum()), median_si=float(np.median(s["si"])),
                           used_s=float(m.sum() / ds.fps))
    states = {}
    if ds.has("freezing") and 0 < b["freezing"][v].mean() < 1:
        states["僵直"] = b["freezing"][v].astype(int)
    zones = {k: z for k, z in b.get("zones", {}).items() if 0 < z[v].mean() < 1}
    if zones:
        lab = np.zeros(int(v.sum()), int)
        for i, z in enumerate(zones.values()):
            lab[z[v]] = i + 1
        states["区域"] = lab
    for j, (name, lab) in enumerate(states.items()):
        progress(0.5 + 0.45 * j / len(states), f"互信息：{name}…")
        r = mutual_information(act[:, v], lab, p.n_levels, p.n_shuffle, p.seed, should_stop)
        arrays.update({f"mi_{j}": r["mi"], f"mi_raw_{j}": r["mi_raw"], f"mi_p_{j}": r["p"],
                       f"mi_q_{j}": fdr_bh(r["p"])})
        summary["variables"].append({"name": name, "n_sig": int((r["p"] < 0.05).sum()),
                                     "n_sig_fdr": int((fdr_bh(r["p"]) < 0.05).sum()),
                                     "median_mi": float(np.median(r["mi"]))})
    if "si" not in arrays and not states:
        raise ValueError("没有可用的位置或行为状态（僵直/区域）数据")
    arrays["mi_names"] = np.array(list(states.keys()))
    save_result(ws, "info", arrays, summary, asdict(p))
    progress(1.0, "信息量分析完成")
    return summary


def export_info(ws: Workspace, res: dict, folder):
    cols = {"cell_id": res["ids"], "n_events": res["n_events"]}
    if "si" in res:
        cols.update(spatial_info_bits_per_event=res["si"], spatial_info_z=res["si_z"], spatial_info_p=res["si_p"],
                    spatial_info_q_fdr=res["si_q"], place_cell=res["place"].astype(int))
    for j, name in enumerate(res["mi_names"]):
        cols[f"MI_bits[{name}]"] = res[f"mi_{j}"]
        cols[f"MI_p[{name}]"] = res[f"mi_p_{j}"]
        cols[f"MI_q_fdr[{name}]"] = res[f"mi_q_{j}"]
    return write_csv(folder / "information.csv", cols)
