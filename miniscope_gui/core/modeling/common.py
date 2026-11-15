"""Shared helpers: data loading, binning, cross-validation folds, shuffles."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..workspace import Workspace

MODEL_DIR = "modeling"


class Stop(InterruptedError):
    pass


def check_stop(should_stop):
    if should_stop is not None and should_stop():
        raise Stop("已取消")


@dataclass
class Dataset:
    times: np.ndarray            # (T,) seconds, bin centres
    fps: float                   # bins per second
    dff: np.ndarray              # (N, T) dF/F
    events: np.ndarray           # (N, T) deconvolved event amplitude per bin
    ids: np.ndarray              # (N,) cell numbers as shown in the app
    centres: np.ndarray          # (N, 2) row (y), column (x) in pixels
    shape: tuple                 # image shape (H, W)
    beh: dict = field(default_factory=dict)   # behaviour on the same bins ({} = none)

    @property
    def n_cells(self):
        return self.dff.shape[0]

    @property
    def n_bins(self):
        return self.dff.shape[1]

    def has(self, key):
        return key in self.beh and self.beh[key] is not None

    @property
    def valid(self):
        """Bins that lie inside the behaviour recording (all bins if no behaviour)."""
        return self.beh.get("valid", np.ones(self.n_bins, bool))


def _bin_mean(x, n, axis=-1):
    if n <= 1:
        return np.asarray(x, float)
    x = np.asarray(x, float)
    T = x.shape[axis] // n * n
    x = np.take(x, np.arange(T), axis=axis)
    new = list(x.shape)
    new[axis] = T // n
    new.insert(axis % x.ndim + 1, n)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(x.reshape(new), axis=axis % x.ndim + 1)


def has_behavior(ws: Workspace) -> bool:
    return bool(ws.step("behavior") and ws.step("sync"))


def load_dataset(ws: Workspace, cfg=None, bin_s: float = 0.2) -> Dataset:
    """Accepted cells, optionally time-binned, with behaviour on the same bins.

    cfg is the BehaviorConfig of the behaviour page; behaviour is attached only
    when both the behaviour and the sync step are done.
    """
    from ..extraction import load_cells
    from ..signals import load_signals
    if not ws.step("signals"):
        raise ValueError("请先完成第 6 步「信号与 spike」")
    sig = load_signals(ws)
    cells = load_cells(ws)
    acc = cells["accepted"].astype(bool)
    if acc.sum() < 2:
        raise ValueError("保留的细胞太少（至少需要 2 个）")
    t = np.asarray(sig["times"], float)
    fps = 1.0 / np.median(np.diff(t))
    n = max(1, int(round(bin_s * fps))) if bin_s and bin_s > 0 else 1
    dff = _bin_mean(sig["dff"][acc], n)
    ev = _bin_mean(sig["spikes"][acc], n) * n
    ds = Dataset(times=_bin_mean(t, n), fps=fps / n, dff=dff, events=ev, ids=np.where(acc)[0],
                 centres=np.asarray(cells["centres"], float)[acc], shape=cells["shape"])
    if cfg is not None and has_behavior(ws):
        from ..sync import behavior_on_calcium_frames
        b = behavior_on_calcium_frames(ws, cfg)
        valid = _bin_mean(b["valid"].astype(float), n) > 0.999
        beh = {"valid": valid, "unit": b["unit"]}
        for k in ("speed", "x_u", "y_u", "x", "y"):
            if k in b:
                v = _bin_mean(b[k], n)
                beh[k] = v
        if "freezing" in b:
            beh["freezing"] = (_bin_mean(b["freezing"].astype(float), n) >= 0.5) & valid
        beh["zones"] = {k: (_bin_mean(v.astype(float), n) >= 0.5) & valid for k, v in b["zones"].items()}
        ds.beh = beh
    return ds


def zscore(X, axis=-1):
    X = np.asarray(X, float)
    m = X.mean(axis, keepdims=True)
    s = X.std(axis, keepdims=True)
    return (X - m) / np.where(s > 1e-12, s, 1.0)


def block_folds(T: int, n_folds: int = 5, block_s: float = 20.0, fps: float = 5.0,
                valid=None) -> np.ndarray:
    """Fold number per time bin; -1 = not used.

    Time is cut into contiguous blocks (so neighbouring, autocorrelated bins
    stay in the same fold) and the blocks are dealt to the folds in turn.
    """
    n_block = max(n_folds, int(np.ceil(T / max(1.0, block_s * fps))))
    edges = np.linspace(0, T, n_block + 1).astype(int)
    fold = np.empty(T, int)
    for b in range(n_block):
        fold[edges[b]:edges[b + 1]] = b % n_folds
    if valid is not None:
        fold[~np.asarray(valid, bool)] = -1
    return fold


def shift_amounts(rng, T: int, n: int, min_frac: float = 0.1) -> np.ndarray:
    lo = max(1, int(T * min_frac))
    hi = max(lo + 1, T - lo)
    return rng.integers(lo, hi, size=n)


def pca(X, n: int):
    """PCA of X (samples x features). Returns scores, components (n x features), explained ratio."""
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    n = int(min(n, len(s)))
    var = s ** 2
    return U[:, :n] * s[:n], Vt[:n], var[:n] / max(var.sum(), 1e-30)


def parse_times(text: str) -> np.ndarray:
    txt = (text or "").replace("，", ",").replace("；", ",").replace(";", ",").replace(" ", ",")
    vals = []
    for v in txt.split(","):
        v = v.strip()
        if v:
            try:
                vals.append(float(v))
            except ValueError:
                raise ValueError(f"事件时间里有无法识别的内容：{v}")
    return np.array(vals, float)


# --------------------------------------------------------------- storage
def model_dir(ws: Workspace) -> Path:
    d = ws.path(MODEL_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _plain(v):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def save_result(ws: Workspace, key: str, arrays: dict, summary: dict, params: dict):
    """arrays -> modeling/<key>.npz; summary + params -> project.json step 'model_<key>'."""
    d = model_dir(ws)
    np.savez_compressed(d / f"{key}.npz", **{k: np.asarray(v) for k, v in arrays.items() if v is not None})
    with open(d / f"{key}.json", "w", encoding="utf-8") as f:
        json.dump(_plain(summary), f, ensure_ascii=False, indent=2)
    ws.record_step(f"model_{key}", _plain(params), {"result": f"{MODEL_DIR}/{key}.npz"})


def load_result(ws: Workspace, key: str) -> dict:
    d = model_dir(ws)
    z = np.load(d / f"{key}.npz", allow_pickle=False)
    out = {k: z[k] for k in z.files}
    p = d / f"{key}.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            out["summary"] = json.load(f)
    return out


def write_csv(path: Path, columns: dict):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(columns.keys())
        for row in zip(*columns.values()):
            w.writerow([("" if not np.isfinite(v) else f"{v:.5g}") if isinstance(v, (float, np.floating))
                        else (int(v) if isinstance(v, (np.integer, np.bool_, bool)) else v) for v in row])
    return path


def fdr_bh(p) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(len(pv))
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def fill_nan(v):
    v = np.asarray(v, float).copy()
    bad = ~np.isfinite(v)
    if bad.all():
        return np.zeros_like(v)
    if bad.any():
        idx = np.arange(len(v))
        v[bad] = np.interp(idx[bad], idx[~bad], v[~bad])
    return v
