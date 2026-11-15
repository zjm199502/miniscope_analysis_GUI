"""Export results to formats that Excel, MATLAB, Python and R can read."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import io as sio

from .extraction import footprint_image, load_cells
from .signals import load_signals
from .workspace import Workspace


def _write_trace_csv(path: Path, times, data, ids):
    header = "time_s," + ",".join(f"cell_{i}" for i in ids)
    arr = np.column_stack([times, data.T])
    np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt="%.5g")


def export_results(ws: Workspace, folder: str | Path | None = None, accepted_only=True) -> Path:
    out = Path(folder) if folder else ws.path("export")
    out.mkdir(parents=True, exist_ok=True)
    cells = load_cells(ws)
    sig = load_signals(ws)
    keep = cells["accepted"].astype(bool) if accepted_only else np.ones(len(cells["C"]), bool)
    ids = np.where(keep)[0]
    t = sig["times"]
    _write_trace_csv(out / "dff.csv", t, sig["dff"][keep], ids)
    _write_trace_csv(out / "spikes.csv", t, sig["spikes"][keep], ids)
    _write_trace_csv(out / "denoised.csv", t, sig["denoised"][keep], ids)
    cen = cells["centres"][keep]
    info = np.column_stack([ids, cen[:, 1], cen[:, 0], cells["area"][keep], cells["snr"][keep],
                            (sig["spikes"][keep] > 0).sum(1) / max(t[-1] - t[0], 1e-9)])
    np.savetxt(out / "cells.csv", info, delimiter=",", comments="",
               header="cell_id,x,y,area_px,snr,event_rate_hz", fmt="%.5g")
    H, W = cells["shape"]
    A = cells["A"][:, keep]
    footprints = np.stack([footprint_image(A, k, (H, W)) for k in range(A.shape[1])]) \
        if A.shape[1] else np.zeros((0, H, W))
    sio.savemat(out / "results.mat", {
        "time_s": t, "dff": sig["dff"][keep], "spikes": sig["spikes"][keep],
        "denoised": sig["denoised"][keep], "cell_id": ids, "centres_xy": cen[:, ::-1],
        "footprints": footprints.astype(np.float32), "fps": float(sig["fps"]),
    }, do_compression=True)
    return out
