"""Step 4: background removal.

1-photon movies contain bright, blurry out-of-focus fluorescence. Each frame
is denoised with a small median filter and the background is removed with a
morphological white top-hat (frame minus its opening with a disk larger than
a cell). Bright objects smaller than the disk (cell bodies) are kept; larger
structures and the smooth glow are removed. This is the approach used by
MiniAn.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import cv2
import numpy as np

from .preprocess import MOVIE_DTYPE
from .storage import MovieReader, MovieWriter
from .workspace import Workspace

BG_FILE = "movie_bg.npy"


@dataclass
class BackgroundParams:
    median_ksize: int = 3        # odd; 1 = off
    cell_diameter: int = 8      # pixels in the processed movie; disk radius ~ diameter
    source: str = "motion"       # "motion" (corrected) or "preprocess"
    backend: str = "auto"
    device_id: int | None = None


def disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def remove_background(frame: np.ndarray, p: BackgroundParams, kernel=None) -> np.ndarray:
    """Float32 in, float32 out (no rounding: cell signals are only a few counts)."""
    f = np.asarray(frame, np.float32)
    if p.median_ksize > 1:
        f = cv2.medianBlur(f, min(5, int(p.median_ksize) | 1))  # float32 supports 3 and 5
    if kernel is None:
        kernel = disk(max(1, int(p.cell_diameter)))
    return cv2.morphologyEx(f, cv2.MORPH_TOPHAT, kernel)


def source_movie(ws: Workspace, source: str):
    from .motion import MC_FILE
    from .preprocess import MOVIE_FILE
    if source == "motion" and ws.path(MC_FILE).exists():
        return MovieReader(ws.path(MC_FILE))
    from .cleanup import require_file
    require_file(ws, MOVIE_FILE, "2. 预处理")
    return MovieReader(ws.path(MOVIE_FILE))


def run_background(session, ws: Workspace, p: BackgroundParams,
                   progress: Callable[[float, str], None] | None = None,
                   should_stop: Callable[[], bool] | None = None) -> dict:
    progress = progress or (lambda f, m: None)
    should_stop = should_stop or (lambda: False)
    movie = source_movie(ws, p.source)
    from .gpu import is_gpu_error, open_backend

    backend = open_backend(p.backend, p.device_id)
    reason = backend.reason if not backend.is_gpu and p.backend != "cpu" else ""
    try:
        with backend.activate():
            return _filter_movie(ws, movie, p, backend, reason, progress, should_stop)
    except Exception as exc:
        if not backend.is_gpu or not is_gpu_error(exc):
            raise
        reason = f"GPU 去背景失败：{exc}"
    finally:
        backend.release()
    if should_stop():
        raise InterruptedError("已取消")
    progress(0.0, reason + "。使用 CPU 从头重算…")
    return _filter_movie(ws, movie, p, open_backend("cpu"), reason, progress, should_stop)


def _filter_movie(ws, movie, p, backend, reason, progress, should_stop):
    T, H, W = movie.shape
    kernel = disk(max(1, int(p.cell_diameter)))
    gpu_filter = None
    if backend.is_gpu:
        from .background_cuda import BackgroundFilter
        gpu_filter = BackgroundFilter(backend, p.median_ksize, kernel)
    # Three FP32 device buffers and host transfers; keep each batch bounded.
    batch_size = max(1, min(32, (32 * 1024**2) // (H * W * 4)))
    outputs = dict(movie=BG_FILE, compute_backend="cuda" if backend.is_gpu else "cpu",
                   compute_device=backend.name, fallback_reason=reason)
    max_img = np.zeros((H, W), np.float32)
    sum_img = np.zeros((H, W), np.float64)
    with ws.stage_outputs() as staged:
        with MovieWriter(staged.path(BG_FILE), (T, H, W), MOVIE_DTYPE) as out:
            for start in range(0, T, batch_size):
                if should_stop():
                    raise InterruptedError("已取消")
                frames = movie[start:start + batch_size]
                if gpu_filter is None:
                    result = np.empty(frames.shape, dtype=np.float32)
                    for i, frame in enumerate(frames):
                        if should_stop():
                            raise InterruptedError("已取消")
                        result[i] = remove_background(frame, p, kernel)
                else:
                    result = gpu_filter.apply(frames)
                out.write(result.astype(MOVIE_DTYPE))
                np.maximum(max_img, result.max(axis=0), out=max_img)
                sum_img += result.sum(axis=0, dtype=np.float64)
                end = start + len(frames)
                progress(end / T, f"{backend.name} 去除背景 {end}/{T} 帧")
            if should_stop():
                raise InterruptedError("已取消")
        np.save(staged.path("bg_max.npy"), max_img)
        np.save(staged.path("bg_mean.npy"), (sum_img / T).astype(np.float32))
        staged.record_step("background", asdict(p), outputs)
    progress(1.0, "去除背景完成")
    return outputs


def load_bg_movie(ws: Workspace):
    return MovieReader(ws.path(BG_FILE))
