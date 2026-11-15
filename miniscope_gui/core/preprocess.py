"""Step 1 preprocessing: time range, crop, spatial and temporal downsampling.

The result is written chunk by chunk to ``movie_pre.npy`` (a memory-mappable
array of shape (T, H, W), uint8), so memory use stays at a few hundred MB no
matter how long the recording is.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from ..io.ucla_v4 import MiniscopeSession
from .workspace import Workspace

MOVIE_FILE = "movie_pre.npy"
TIMES_FILE = "frame_times_pre.npy"
MEAN_FILE = "mean_projection.npy"
MAX_FILE = "max_projection.npy"


@dataclass
class PreprocessParams:
    start_s: float = 0.0
    end_s: float | None = None          # None = until the end
    crop: tuple[int, int, int, int] | None = None   # x, y, w, h in raw pixels
    spatial_bin: int = 2                # 2 = average 2x2 pixels
    temporal_bin: int = 1               # 3 = average every 3 frames (30 -> 10 Hz)
    chunk_frames: int = 500


def bin_spatial(block: np.ndarray, b: int) -> np.ndarray:
    """Average b x b pixel blocks. block: (T, H, W)."""
    if b <= 1:
        return block.astype(np.float32)
    t, h, w = block.shape
    h2, w2 = h // b, w // b
    v = block[:, :h2 * b, :w2 * b].astype(np.float32)
    return v.reshape(t, h2, b, w2, b).mean(axis=(2, 4))


def output_shape(session: MiniscopeSession, p: PreprocessParams):
    x, y, w, h = p.crop if p.crop else (0, 0, session.width, session.height)
    b = max(1, p.spatial_bin)
    return h // b, w // b


def frame_range(session: MiniscopeSession, p: PreprocessParams) -> tuple[int, int]:
    t = session.frame_times_s()
    start = int(np.searchsorted(t, p.start_s, side="left"))
    stop = session.n_frames if p.end_s is None else int(np.searchsorted(t, p.end_s, side="right"))
    start = max(0, min(start, session.n_frames - 1))
    stop = max(start + 1, min(stop, session.n_frames))
    return start, stop


def run_preprocess(session: MiniscopeSession, ws: Workspace, p: PreprocessParams,
                   progress: Callable[[float, str], None] | None = None,
                   should_stop: Callable[[], bool] | None = None) -> dict:
    progress = progress or (lambda f, m: None)
    should_stop = should_stop or (lambda: False)

    start, stop = frame_range(session, p)
    tb = max(1, p.temporal_bin)
    n_in = stop - start
    n_out = n_in // tb
    if n_out < 1:
        raise ValueError("所选时间段太短")
    stop = start + n_out * tb
    H, W = output_shape(session, p)
    x, y, w, h = p.crop if p.crop else (0, 0, session.width, session.height)

    from numpy.lib.format import open_memmap
    movie = open_memmap(ws.path(MOVIE_FILE), mode="w+", dtype=np.uint8, shape=(n_out, H, W))
    times_all = session.frame_times_s()
    times = times_all[start:stop].reshape(n_out, tb).mean(axis=1)

    sum_img = np.zeros((H, W), np.float64)
    max_img = np.zeros((H, W), np.float32)

    chunk = max(tb, (p.chunk_frames // tb) * tb)
    out_i = 0
    buf = []
    for idx, frame in session.iter_frames(start, stop):
        if should_stop():
            raise InterruptedError("已取消")
        buf.append(frame[y:y + h, x:x + w])
        if len(buf) == chunk or idx == stop - 1:
            block = bin_spatial(np.stack(buf), p.spatial_bin)
            nb = block.shape[0] // tb
            block = block[:nb * tb].reshape(nb, tb, H, W).mean(axis=1)
            block8 = np.clip(np.rint(block), 0, 255).astype(np.uint8)
            movie[out_i:out_i + nb] = block8
            sum_img += block.sum(axis=0)
            np.maximum(max_img, block.max(axis=0), out=max_img)
            out_i += nb
            buf = []
            progress(out_i / n_out, f"预处理 {out_i}/{n_out} 帧")
    movie.flush()
    del movie

    np.save(ws.path(TIMES_FILE), times)
    np.save(ws.path(MEAN_FILE), (sum_img / max(out_i, 1)).astype(np.float32))
    np.save(ws.path(MAX_FILE), max_img)

    params = asdict(p)
    params["frame_range"] = [start, stop]
    params["fps_out"] = session.fps / tb
    outputs = {"movie": MOVIE_FILE, "times": TIMES_FILE, "mean": MEAN_FILE,
               "max": MAX_FILE, "shape": [out_i, H, W]}
    ws.record_step("preprocess", params, outputs)
    progress(1.0, "预处理完成")
    return outputs


def load_movie(ws: Workspace) -> np.ndarray:
    """Open the preprocessed movie read-only without loading it into RAM."""
    return np.load(ws.path(MOVIE_FILE), mmap_mode="r")
