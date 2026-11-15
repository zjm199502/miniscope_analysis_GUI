"""Manual curation of extracted cells: accept/reject and merge."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .extraction import _centroids, load_cells, save_cells
from .workspace import Workspace


def set_accepted(ws: Workspace, ids, value: bool | None = None) -> np.ndarray:
    """Set (or toggle when value is None) the accepted flag of the given cells."""
    cells = load_cells(ws)
    acc = cells["accepted"].astype(bool).copy()
    ids = np.atleast_1d(np.asarray(ids, int))
    acc[ids] = (~acc[ids]) if value is None else bool(value)
    cells["accepted"] = acc
    save_cells(ws, **cells)
    return acc


def merge_cells(ws: Workspace, ids) -> int:
    """Merge cells into one; returns the new cell's index. Signals are recomputed if present."""
    ids = sorted(set(int(i) for i in ids))
    if len(ids) < 2:
        raise ValueError("请至少选择两个细胞")
    c = load_cells(ws)
    A = c["A"].tocsc()
    w = np.asarray(A[:, ids].sum(0)).ravel()
    w = w / max(w.sum(), 1e-12)
    merged = {k: (w[:, None] * c[k][ids]).sum(0) for k in ("C", "raw", "F_local")}
    a_new = sp.csc_matrix(A[:, ids].sum(1))
    keep = [k for k in range(A.shape[1]) if k not in ids]
    new_index = ids[0] - sum(1 for k in ids[1:] if k < ids[0])  # == ids[0]
    cols = [A[:, k] for k in keep]
    cols.insert(new_index, a_new)
    A2 = sp.hstack(cols).tocsc().astype(np.float32)
    out = {}
    for k in ("C", "raw", "F_local"):
        arr = [c[k][j] for j in keep]
        arr.insert(new_index, merged[k].astype(np.float32))
        out[k] = np.stack(arr)
    for k in ("snr", "area", "accepted"):
        arr = list(c[k][keep])
        if k == "snr":
            arr.insert(new_index, float(c[k][ids].max()))
        elif k == "area":
            arr.insert(new_index, int((a_new > 0).sum()))
        else:
            arr.insert(new_index, True)
        out[k] = np.array(arr)
    H, W = c["shape"]
    save_cells(ws, A=A2, centres=_centroids(A2, H, W), shape=(H, W), **out)
    st = ws.step("signals")
    if st:
        from .signals import SignalParams, run_signals
        prm = {k: v for k, v in st["params"].items() if k in SignalParams.__dataclass_fields__}
        run_signals(None, ws, SignalParams(**prm))
    return new_index
