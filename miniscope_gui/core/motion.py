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
    backend: str = "auto"
    device_id: int | None = None


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

    @staticmethod
    def batch_size(shape):
        return 1

    def shifts(self, template, images):
        return [self.shift(template, image) for image in images]

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
        return _peak_shift(sub, m)

    def shift_limited(self, template, img, window, max_shift):
        old = self.p.max_shift
        try:
            self.p.max_shift = max_shift
            return self.shift(template, img, window)
        finally:
            self.p.max_shift = old

    def patch_shifts(self, template, image, origins, size, window, limit):
        return [self.shift_limited(template[y:y+size, x:x+size], image[y:y+size, x:x+size],
                                   window, limit) for y, x in origins]


def _peak_shift(sub, m):
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    peak = float(sub[iy, ix])
    dy, dx = float(iy - m), float(ix - m)
    # Sub-pixel refinement: apply identical arithmetic to both backends.
    if 0 < iy < 2 * m:
        a, b, c = sub[iy - 1, ix], sub[iy, ix], sub[iy + 1, ix]
        d = a - 2 * b + c
        if d < 0:
            dy += 0.5 * (a - c) / d
    if 0 < ix < 2 * m:
        a, b, c = sub[iy, ix - 1], sub[iy, ix], sub[iy, ix + 1]
        d = a - 2 * b + c
        if d < 0:
            dx += 0.5 * (a - c) / d
    return dx, dy, peak


class _CudaRegistrar(_Registrar):
    """Compute batched spectra on the GPU; transfer only the small peak windows."""

    def __init__(self, shape, p, backend):
        super().__init__(shape, p)
        self.backend = backend
        self.xp = backend.xp
        self.device_window = self.xp.asarray(self.window)
        self.patch_windows = {}

    @staticmethod
    def batch_size(shape):
        # Powers of two avoid creating a different cuFFT plan for short I/O blocks.
        capacity = max(1, min(32, (64 * 1024**2) // (int(np.prod(shape)) * 72)))
        return 1 << (capacity.bit_length() - 1)

    def _spectrum(self, images, window):
        xp = self.xp
        # Phase-only correlation gives weak spatial frequencies the same weight
        # as strong ones. Single-precision cuFFT loses phase on smooth images.
        weighted = (xp.asarray(images) * window).astype(xp.float64)
        batch = self.batch_size(images.shape[-2:])
        if len(images) < batch:
            padded = xp.zeros((batch, *images.shape[-2:]), dtype=weighted.dtype)
            padded[:len(images)] = weighted
            weighted = padded
        return xp.fft.rfft2(weighted, axes=(-2, -1)).astype(xp.complex64)

    def _correlate(self, spectra, images, window, limit):
        xp = self.xp
        h, w = images.shape[-2:]
        cross = spectra * xp.conj(self._spectrum(images, window))
        cross /= xp.abs(cross) + 1e-9
        corr = xp.fft.irfft2(cross, s=(h, w), axes=(-2, -1))
        m = min(limit, h // 2 - 1, w // 2 - 1)
        indices = xp.arange(-m, m + 1)
        peaks = self.backend.asnumpy(corr[:len(images), (indices % h)[:, None], indices % w])
        return [_peak_shift(sub, m) for sub in peaks]

    def shifts(self, template, images):
        if self._template is not template:
            self._template = template
            self._tpl_fft = self.xp.asarray(self._fft(template, self.window))
        return self._correlate(self._tpl_fft, images, self.device_window, self.p.max_shift)

    def patch_shifts(self, template, image, origins, size, window, limit):
        if size not in self.patch_windows:
            self.patch_windows[size] = self.xp.asarray(window)
        win = self.patch_windows[size]
        shifts = []
        batch = self.batch_size((size, size))
        for start in range(0, len(origins), batch):
            chunk = origins[start:start+batch]
            refs = np.stack([template[y:y+size, x:x+size] for y, x in chunk])
            imgs = np.stack([image[y:y+size, x:x+size] for y, x in chunk])
            shifts.extend(self._correlate(self._spectrum(refs, win), imgs, win, limit))
        return shifts


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
    if size not in windows:
        windows[size] = cv2.createHanningWindow((size, size), cv2.CV_32F)
    win = windows[size]
    origins = [(y, x) for y in ys for x in xs]
    local = reg.patch_shifts(template_f, pre, origins, size, win, p.max_deviation)
    vectors = np.asarray(local, dtype=np.float32).reshape(len(ys), len(xs), 3)
    gx, gy = vectors[:, :, 0], vectors[:, :, 1]
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
        batch = reg.batch_size(template.shape)
        for start in range(0, len(frames), batch):
            if should_stop and should_stop():
                raise InterruptedError("已取消")
            group = frames[start:start+batch]
            for j, (dx, dy, _) in enumerate(reg.shifts(template, group)):
                aligned[start + j] = apply_shift(group[j], dx, dy)
        template = np.median(aligned, axis=0, overwrite_input=True)
    return template


def run_motion_correction(session, ws: Workspace, p: MotionParams,
                          progress: Callable[[float, str], None] | None = None,
                          should_stop: Callable[[], bool] | None = None) -> dict:
    progress = progress or (lambda f, m: None)
    should_stop = should_stop or (lambda: False)
    from .cleanup import require_file
    from .gpu import open_backend, is_gpu_error

    require_file(ws, "movie_pre.npy", "2. 预处理")
    movie = load_movie(ws)
    backend = open_backend(p.backend, p.device_id)
    reason = backend.reason if not backend.is_gpu and p.backend != "cpu" else ""
    try:
        with backend.activate():
            return _correct_movie(ws, movie, p, backend, reason, progress, should_stop)
    except Exception as exc:
        if backend.is_gpu:
            # A traceback can hold the registrar, FFT buffers and pending frame
            # batches after cancellation, even after the worker has stopped.
            from traceback import clear_frames
            clear_frames(exc.__traceback__)
        if not backend.is_gpu or not is_gpu_error(exc):
            raise
        reason = f"GPU 运行失败：{exc}"
    finally:
        backend.release()
    if should_stop():
        raise InterruptedError("已取消")
    progress(0.0, reason + "。从头使用 CPU 重新校正…")
    return _correct_movie(ws, movie, p, open_backend("cpu"), reason, progress, should_stop)


def _registered_frames(movie, p, reg, template, should_stop):
    T, H, W = movie.shape
    batch = reg.batch_size((H, W))
    for block_start in range(0, T, 100):
        block = movie[block_start:block_start + 100]
        for start in range(0, len(block), batch):
            if should_stop():
                raise InterruptedError("已取消")
            raw_frames = block[start:start + batch]
            filtered = np.stack([bandpass(raw, p.filter_sigma, p.bg_sigma) for raw in raw_frames])
            shifts = reg.shifts(template, filtered)
            for offset, (dx, dy, _) in enumerate(shifts):
                yield block_start + start + offset, raw_frames[offset], filtered[offset], dx, dy


def _correct_movie(ws, movie, p, backend, reason, progress, should_stop):
    T, H, W = movie.shape
    reg = _CudaRegistrar((H, W), p, backend) if backend.is_gpu else _Registrar((H, W), p)
    progress(0.0, f"生成参考模板（{backend.name}）…")
    template = build_template(movie, p, reg, should_stop)

    shifts = np.zeros((T, 2), np.float32)
    q_before = np.zeros(T, np.float32)
    q_after = np.zeros(T, np.float32)
    windows: dict = {}
    sum_img = np.zeros((H, W), np.float64)
    max_img = np.zeros((H, W), np.float32)
    with ws.stage_outputs() as staged:
        with MovieWriter(staged.path(MC_FILE), (T, H, W), MOVIE_DTYPE) as out:
            for t, raw, f, dx, dy in _registered_frames(movie, p, reg, template, should_stop):
                if should_stop():
                    raise InterruptedError("已取消")
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

        if backend.is_gpu:
            backend.synchronize()
        if should_stop():
            raise InterruptedError("已取消")
        sub = np.arange(0, T, 10)
        np.save(staged.path(SHIFTS_FILE), shifts)
        np.save(staged.path(TEMPLATE_FILE), template)
        np.save(staged.path(MC_QUALITY_FILE), np.stack([sub, q_before[sub], q_after[sub]], axis=1))
        np.save(staged.path("mc_mean.npy"), (sum_img / T).astype(np.float32))
        np.save(staged.path("mc_max.npy"), max_img)
        summary = {
            "compute_backend": "cuda" if backend.is_gpu else "cpu",
            "compute_device": backend.name,
            "fallback_reason": reason,
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
