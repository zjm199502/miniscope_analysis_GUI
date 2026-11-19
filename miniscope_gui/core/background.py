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
    T, H, W = movie.shape
    kernel = disk(max(1, int(p.cell_diameter)))
    max_img = np.zeros((H, W), np.float32)
    sum_img = np.zeros((H, W), np.float64)
    with ws.stage_outputs() as staged:
        with MovieWriter(staged.path(BG_FILE), (T, H, W), MOVIE_DTYPE) as out:
            blk, blk_start = None, -1
            for t in range(T):
                if should_stop():
                    raise InterruptedError("已取消")
                if blk is None or t >= blk_start + len(blk):
                    blk_start = t
                    blk = movie[t:t + 100]
                f = remove_background(blk[t - blk_start], p, kernel)
                out.write(f[None].astype(MOVIE_DTYPE))
                np.maximum(max_img, f, out=max_img)
                sum_img += f
                if t % 200 == 0 or t == T - 1:
                    progress((t + 1) / T, f"去除背景 {t + 1}/{T} 帧")
        np.save(staged.path("bg_max.npy"), max_img)
        np.save(staged.path("bg_mean.npy"), (sum_img / T).astype(np.float32))
        staged.record_step("background", asdict(p), {"movie": BG_FILE})
    progress(1.0, "去除背景完成")
    return {"movie": BG_FILE}


def load_bg_movie(ws: Workspace):
    return MovieReader(ws.path(BG_FILE))
