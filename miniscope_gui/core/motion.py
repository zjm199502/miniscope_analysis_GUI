"""Step 3: motion correction.

Shifts are estimated on a spatially band-passed copy of each frame (blood
vessels and cell bodies stand out, smooth 1-photon background is removed),
by FFT phase correlation against a template. The shifts are then applied to
the preprocessed movie. Optional piecewise (non-rigid) correction splits the
frame into overlapping patches, estimates one shift per patch and warps the
frame with a smooth shift field.

Everything is streamed frame by frame; memory does not depend on movie length.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import cv2
import numpy as np

from .preprocess import MOVIE_DTYPE, load_movie
from .storage import MovieReader, MovieWriter
from .workspace import Workspace

MC_FILE = "movie_mc.npy"
SHIFTS_FILE = "mc_shifts.npy"
TEMPLATE_FILE = "mc_template.npy"
MC_QUALITY_FILE = "mc_quality.npy"


@dataclass
class MotionParams:
    max_shift: int = 20            # pixels (after spatial downsampling)
    filter_sigma: float = 1.5      # small Gaussian: denoise for registration
    bg_sigma: float = 10.0         # large Gaussian: removes smooth background
    template_frames: int = 300
    iterations: int = 2            # template refinement passes on the template frames
    piecewise: bool = False
    patch_size: int = 96
    patch_overlap: int = 32
    max_deviation: int = 5         # how far a patch may deviate from the rigid shift


def bandpass(frame: np.ndarray, sigma_small: float, sigma_large: float) -> np.ndarray:
    f = frame.astype(np.float32)
    lo = cv2.GaussianBlur(f, (0, 0), sigma_small) if sigma_small > 0 else f
    bg = cv2.GaussianBlur(f, (0, 0), sigma_large)
    return lo - bg


class _Registrar:
    """Phase correlation with the peak search restricted to +-max_shift.

    Restricting the search window (instead of taking the global peak, as
    cv2.phaseCorrelate does) prevents jumps to spurious far-away peaks in
    noisy frames. The template spectrum is cached.
    """

    def __init__(self, shape, p: MotionParams):
        self.p = p
        h, w = shape
        self.window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        self._template = None
        self._tpl_fft = None

    def _fft(self, img, win):
        return np.fft.rfft2(img * win)

    def shift(self, template: np.ndarray, img: np.ndarray, window=None) -> tuple[float, float, float]:
        """Return (dx, dy, response): the shift that moves img onto template."""
        win = self.window if window is None else window
        if window is None and self._template is not template:
            self._template, self._tpl_fft = template, self._fft(template, win)
        F1 = self._tpl_fft if window is None else self._fft(template, win)
        F2 = self._fft(img, win)
        R = F1 * np.conj(F2)
        R /= np.abs(R) + 1e-9
        r = np.fft.irfft2(R, s=img.shape)
        m = int(self.p.max_shift)
        h, w = img.shape
        m = min(m, h // 2 - 1, w // 2 - 1)
        # gather the (2m+1)^2 neighbourhood of zero shift (with wrap-around)
        idx_y = np.arange(-m, m + 1) % h
        idx_x = np.arange(-m, m + 1) % w
        sub = r[np.ix_(idx_y, idx_x)]
        iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
        peak = float(sub[iy, ix])
        dy, dx = float(iy - m), float(ix - m)
        # sub-pixel refinement: parabola through the peak and its neighbours
        if 0 < iy < 2 * m:
            a_, b_, c_ = sub[iy - 1, ix], sub[iy, ix], sub[iy + 1, ix]
            d = a_ - 2 * b_ + c_
            if d < 0:
                dy += 0.5 * (a_ - c_) / d
        if 0 < ix < 2 * m:
            a_, b_, c_ = sub[iy, ix - 1], sub[iy, ix], sub[iy, ix + 1]
            d = a_ - 2 * b_ + c_
            if d < 0:
                dx += 0.5 * (a_ - c_) / d
        return dx, dy, peak

    def shift_limited(self, template, img, window, max_shift):
        old = self.p.max_shift
        try:
            self.p.max_shift = max_shift
            return self.shift(template, img, window)
        finally:
            self.p.max_shift = old


def apply_shift(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = frame.shape
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _patch_grid(h, w, size, overlap):
    step = max(8, size - overlap)
    ys = list(range(0, max(1, h - size) + 1, step))
    xs = list(range(0, max(1, w - size) + 1, step))
    if ys[-1] + size < h:
        ys.append(max(0, h - size))
    if xs[-1] + size < w:
        xs.append(max(0, w - size))
    return ys, xs


def _piecewise_field(reg: _Registrar, template_f, img_f, dx0, dy0, p: MotionParams, windows):
    """Per-patch shifts around the rigid shift, upsampled to a smooth field."""
    h, w = img_f.shape
    size = min(p.patch_size, h, w)
    ys, xs = _patch_grid(h, w, size, p.patch_overlap)
    pre = apply_shift(img_f, dx0, dy0)
    gx = np.zeros((len(ys), len(xs)), np.float32)
    gy = np.zeros_like(gx)
    win = windows.setdefault(size, cv2.createHanningWindow((size, size), cv2.CV_32F))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            t = template_f[y:y + size, x:x + size]
            s = pre[y:y + size, x:x + size]
            ddx, ddy, _ = reg.shift_limited(t, s, win, p.max_deviation)
            gx[i, j], gy[i, j] = ddx, ddy
    if gx.size > 1:  # damp outlier patches
        gx = cv2.medianBlur(gx, 3) if min(gx.shape) >= 3 else gx
        gy = cv2.medianBlur(gy, 3) if min(gy.shape) >= 3 else gy
    fx = cv2.resize(gx, (w, h), interpolation=cv2.INTER_CUBIC) + dx0
    fy = cv2.resize(gy, (w, h), interpolation=cv2.INTER_CUBIC) + dy0
    return fx, fy


def apply_field(frame: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    h, w = frame.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(frame, gx - fx, gy - fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def build_template(movie, p: MotionParams, reg: _Registrar, should_stop=None) -> np.ndarray:
    n = min(p.template_frames, movie.shape[0])
    # spread template frames over the first part of the recording
    idx = np.linspace(0, movie.shape[0] - 1, num=min(movie.shape[0], max(n, 1) * 3)).astype(int)[:n]
    frames = np.empty((len(idx), *movie.shape[1:]), np.float32)
    for j, i in enumerate(idx):
        if should_stop and should_stop():
            raise InterruptedError("已取消")
        frames[j] = bandpass(np.asarray(movie[i], np.float32), p.filter_sigma, p.bg_sigma)
    aligned = frames.copy()
    template = np.median(aligned, axis=0, overwrite_input=True)
    for _ in range(max(1, p.iterations)):
        for j, f in enumerate(frames):
            if should_stop and should_stop():
                raise InterruptedError("已取消")
            dx, dy, _ = reg.shift(template, f)
            aligned[j] = apply_shift(f, dx, dy)
        template = np.median(aligned, axis=0, overwrite_input=True)
    return template


def run_motion_correction(session, ws: Workspace, p: MotionParams,
                          progress: Callable[[float, str], None] | None = None,
                          should_stop: Callable[[], bool] | None = None) -> dict:
    progress = progress or (lambda f, m: None)
    should_stop = should_stop or (lambda: False)
    from .cleanup import require_file
    require_file(ws, "movie_pre.npy", "2. 预处理")
    movie = load_movie(ws)
    T, H, W = movie.shape
    reg = _Registrar((H, W), p)
    progress(0.0, "生成参考模板…")
    template = build_template(movie, p, reg, should_stop)

    shifts = np.zeros((T, 2), np.float32)
    q_before = np.zeros(T, np.float32)
    q_after = np.zeros(T, np.float32)
    windows: dict = {}
    sum_img = np.zeros((H, W), np.float64)
    max_img = np.zeros((H, W), np.float32)
    with ws.stage_outputs() as staged:
        with MovieWriter(staged.path(MC_FILE), (T, H, W), MOVIE_DTYPE) as out:
            blk, blk_start = None, -1
            for t in range(T):
                if should_stop():
                    raise InterruptedError("已取消")
                if blk is None or t >= blk_start + len(blk):
                    blk_start = t
                    blk = movie[t:t + 100]
                raw = np.asarray(blk[t - blk_start], dtype=np.float32)
                f = bandpass(raw, p.filter_sigma, p.bg_sigma)
                dx, dy, _ = reg.shift(template, f)
                if p.piecewise:
                    fx, fy = _piecewise_field(reg, template, f, dx, dy, p, windows)
                    corrected = apply_field(raw.astype(np.float32), fx, fy)
                    fc = apply_field(f, fx, fy)
                    shifts[t] = (float(fx.mean()), float(fy.mean()))
                else:
                    corrected = apply_shift(raw.astype(np.float32), dx, dy)
                    fc = apply_shift(f, dx, dy)
                    shifts[t] = (dx, dy)
                if t % 10 == 0:  # quality on a subsample: cheap and enough for a plot
                    q_before[t] = _ncc(template, f)
                    q_after[t] = _ncc(template, fc)
                out.write(corrected[None].astype(MOVIE_DTYPE))
                sum_img += corrected
                np.maximum(max_img, corrected, out=max_img)
                if t % 100 == 0 or t == T - 1:
                    progress((t + 1) / T, f"运动校正 {t + 1}/{T} 帧")

        sub = np.arange(0, T, 10)
        np.save(staged.path(SHIFTS_FILE), shifts)
        np.save(staged.path(TEMPLATE_FILE), template)
        np.save(staged.path(MC_QUALITY_FILE), np.stack([sub, q_before[sub], q_after[sub]], axis=1))
        np.save(staged.path("mc_mean.npy"), (sum_img / T).astype(np.float32))
        np.save(staged.path("mc_max.npy"), max_img)
        summary = {
            "max_abs_shift": float(np.abs(shifts).max()),
            "mean_abs_shift": float(np.abs(shifts).mean()),
            "corr_before": float(q_before[sub].mean()),
            "corr_after": float(q_after[sub].mean()),
        }
        staged.record_step("motion", asdict(p), {"movie": MC_FILE, "shifts": SHIFTS_FILE,
                                             "template": TEMPLATE_FILE, **summary})
    progress(1.0, "运动校正完成")
    return summary


def load_mc_movie(ws: Workspace):
    return MovieReader(ws.path(MC_FILE))
