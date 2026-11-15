"""Step 5: cell extraction ("CNMF-E lite").

Model, on the background-removed movie Y (pixels x time):

    Y  ~=  A C  +  b 1^T

A: sparse spatial footprints (one column per cell, each confined to a small
patch around the cell), C: temporal activity, b: per-pixel constant offset.

1. Summary images, one streaming pass: local correlation image (Cn) and
   peak-to-noise ratio (PNR) of the slightly smoothed movie.
2. Seeds: local maxima of Cn * PNR above thresholds (as in CNMF-E).
3. Initial footprints: regression of each patch pixel on the seed trace,
   kept where the pixel correlates with the seed and is connected to it.
4. Alternating HALS updates of C (whole frame, per chunk) and A (local
   patches only), with merging of duplicated cells.
5. Final pass also returns "raw" traces (C + residual, like CaImAn's YrA)
   and each cell's local fluorescence level from the motion-corrected movie,
   used later as F0 for dF/F.

All passes stream the movie in chunks; memory is ~ chunk x pixels x 4 bytes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import cv2
import numpy as np
import scipy.sparse as sp
from scipy import ndimage

from .workspace import Workspace

CELLS_FILE = "cells.npz"
CN_FILE = "summary_cn.npy"
PNR_FILE = "summary_pnr.npy"


@dataclass
class ExtractionParams:
    cell_diameter: int = 10       # pixels in the processed movie
    min_corr: float = 0.8
    min_pnr: float = 8.0
    max_cells: int = 1000
    merge_threshold: float = 0.8  # temporal correlation for merging overlapping cells
    n_iterations: int = 2
    min_area_fraction: float = 0.15   # of a disk with the cell diameter
    chunk_frames: int = 200


# --------------------------------------------------------------- utilities
def _chunks(T, n):
    for s in range(0, T, n):
        yield s, min(T, s + n)


def _smooth(block: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return block.astype(np.float32)
    out = np.empty(block.shape, np.float32)
    for i in range(block.shape[0]):
        out[i] = cv2.GaussianBlur(block[i].astype(np.float32), (0, 0), sigma)
    return out


_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _shift2d(img, dy, dx):
    """img shifted so out[y, x] = img[y+dy, x+dx]; edges get NaN."""
    out = np.full(img.shape, np.nan, img.dtype)
    H, W = img.shape[-2:]
    ys = slice(max(0, -dy), H - max(0, dy))
    xs = slice(max(0, -dx), W - max(0, dx))
    ys2 = slice(max(0, dy), H - max(0, -dy))
    xs2 = slice(max(0, dx), W - max(0, -dx))
    out[..., ys, xs] = img[..., ys2, xs2]
    return out


# ------------------------------------------------------- 1. summary images
def _neighbour_slices(H, W, dy, dx):
    """Slices (a, b) so that img[a] and img[b] are neighbours at offset (dy, dx)."""
    ya = slice(max(0, -dy), H - max(0, dy)); xa = slice(max(0, -dx), W - max(0, dx))
    yb = slice(max(0, dy), H - max(0, -dy)); xb = slice(max(0, dx), W - max(0, -dx))
    return (ya, xa), (yb, xb)


def summary_images(movie, sigma: float, chunk: int, progress, should_stop, mask=None,
                   noise_thr: float = 3.0):
    """Local correlation image and PNR image, as in CNMF-E.

    Pass 1: per-pixel mean, max and noise of the smoothed movie (PNR).
    Pass 2: neighbour correlation of the smoothed movie after setting values
    below noise_thr x noise to zero. Without this thresholding, smoothing
    alone makes neighbouring noise pixels correlate (~0.7) and the map
    saturates.
    """
    T, H, W = movie.shape
    s1 = np.zeros((H, W)); mx = np.full((H, W), -np.inf, np.float32)
    d2 = np.zeros((H, W)); prev = None
    for a, b in _chunks(T, chunk):
        if should_stop():
            raise InterruptedError("已取消")
        Yf = _smooth(np.asarray(movie[a:b], np.float32), sigma)
        s1 += Yf.sum(0, dtype=np.float64)
        np.maximum(mx, Yf.max(0), out=mx)
        d2 += (np.diff(Yf, axis=0) ** 2).sum(0, dtype=np.float64)
        if prev is not None:
            d2 += (Yf[0] - prev) ** 2
        prev = Yf[-1]
        progress(b / T * 0.12, f"计算 PNR 图 {b}/{T}")
    mean = (s1 / T).astype(np.float32)
    noise = np.sqrt(d2 / max(T - 1, 1) / 2.0).astype(np.float32)
    pnr = (mx - mean) / np.maximum(noise, 1e-6)

    # pass 2: thresholded correlation
    thr = mean + noise_thr * noise
    t1 = np.zeros((H, W)); t2 = np.zeros((H, W))
    cross = np.zeros((len(_OFFSETS), H, W))
    sl = [_neighbour_slices(H, W, dy, dx) for dy, dx in _OFFSETS]
    for a, b in _chunks(T, chunk):
        if should_stop():
            raise InterruptedError("已取消")
        Yf = _smooth(np.asarray(movie[a:b], np.float32), sigma) - mean
        Yf[Yf < (thr - mean)] = 0.0
        t1 += Yf.sum(0, dtype=np.float64)
        t2 += (Yf * Yf).sum(0, dtype=np.float64)
        for i, ((ya, xa), (yb, xb)) in enumerate(sl):
            cross[i][ya, xa] += (Yf[:, ya, xa] * Yf[:, yb, xb]).sum(0, dtype=np.float64)
        progress(0.12 + b / T * 0.13, f"计算相关图 {b}/{T}")
    m = t1 / T
    sd = np.sqrt(np.maximum(t2 / T - m * m, 0))
    cn_sum = np.zeros((H, W)); cnt = np.zeros((H, W))
    for i, ((ya, xa), (yb, xb)) in enumerate(sl):
        num = cross[i][ya, xa] / T - m[ya, xa] * m[yb, xb]
        den = sd[ya, xa] * sd[yb, xb]
        c = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
        cn_sum[ya, xa] += c; cnt[ya, xa] += 1
    cn = cn_sum / np.maximum(cnt, 1)
    if mask is not None:
        cn[~mask] = 0
        pnr[~mask] = 0
    return cn.astype(np.float32), pnr.astype(np.float32), mean, noise


# ------------------------------------------------------------------ 2. seeds
def find_seeds(cn, pnr, p: ExtractionParams, mask=None):
    score = cn * pnr
    ok = (cn >= p.min_corr) & (pnr >= p.min_pnr)
    if mask is not None:
        # no seeds within ~0.75 cell diameter of the field-of-view edge (vignetted tissue)
        r_edge = max(1, int(round(0.75 * p.cell_diameter)))
        ok &= ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool), iterations=r_edge)
    r = max(1, int(round(p.cell_diameter * 0.5)))
    local_max = score == ndimage.maximum_filter(score, size=2 * r + 1)
    cand = np.argwhere(ok & local_max)
    order = np.argsort(-score[cand[:, 0], cand[:, 1]])
    seeds = []
    min_d2 = (0.6 * p.cell_diameter) ** 2
    for y, x in cand[order]:
        if all((y - sy) ** 2 + (x - sx) ** 2 >= min_d2 for sy, sx in seeds):
            seeds.append((int(y), int(x)))
        if len(seeds) >= p.max_cells:
            break
    return np.array(seeds, dtype=int).reshape(-1, 2)


# ------------------------------------------------------------- patches / A
class _Patches:
    """Square patch of radius R around each cell, as flat pixel indices."""

    def __init__(self, centres, H, W, R):
        self.H, self.W, self.R = H, W, R
        self.boxes, self.idx = [], []
        for y, x in centres:
            y0, y1 = max(0, y - R), min(H, y + R + 1)
            x0, x1 = max(0, x - R), min(W, x + R + 1)
            self.boxes.append((y0, y1, x0, x1))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            self.idx.append((yy * W + xx).ravel())

    def take(self, k, block):  # block (t, H, W) -> (t, P)
        y0, y1, x0, x1 = self.boxes[k]
        return block[:, y0:y1, x0:x1].reshape(block.shape[0], -1)


def _clean_footprint(a_patch, box, centre=None, rel_thr=0.3):
    """Keep the connected component around the peak (or centre); zero the rest."""
    y0, y1, x0, x1 = box
    img = a_patch.reshape(y1 - y0, x1 - x0).copy()
    img[img < 0] = 0
    if img.max() <= 0:
        return np.zeros_like(a_patch)
    lab, n = ndimage.label(img > rel_thr * img.max())
    if n == 0:
        return np.zeros_like(a_patch)
    if centre is not None:
        cy, cx = centre[0] - y0, centre[1] - x0
        l = lab[cy, cx] if 0 <= cy < img.shape[0] and 0 <= cx < img.shape[1] else 0
    else:
        l = 0
    if l == 0:
        l = lab.flat[np.argmax(img)]
    img[lab != l] = 0
    return img.ravel()


def _build_sparse(patches: _Patches, a_list, npix):
    rows, cols, vals = [], [], []
    for k, (idx, a) in enumerate(zip(patches.idx, a_list)):
        nz = a > 0
        rows.append(idx[nz]); cols.append(np.full(nz.sum(), k)); vals.append(a[nz])
    if not rows:
        return sp.csc_matrix((npix, 0), dtype=np.float32)
    return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(npix, len(a_list)), dtype=np.float32)


# ------------------------------------------------------------------ main run
def run_extraction(session, ws: Workspace, p: ExtractionParams,
                   progress: Callable[[float, str], None] | None = None,
                   should_stop: Callable[[], bool] | None = None) -> dict:
    from .background import load_bg_movie
    progress = progress or (lambda f, m: None)
    should_stop = should_stop or (lambda: False)

    Y = load_bg_movie(ws)
    T, H, W = Y.shape
    npix = H * W
    mask = np.load(ws.path("fov_mask.npy")) if ws.path("fov_mask.npy").exists() else None
    if mask is not None and mask.shape != (H, W):
        mask = None
    sigma = max(0.5, p.cell_diameter / 6.0)
    chunk = max(50, int(p.chunk_frames))

    # 1. summary images
    cn, pnr, _, _ = summary_images(Y, sigma, chunk, progress, should_stop, mask)
    np.save(ws.path(CN_FILE), cn)
    np.save(ws.path(PNR_FILE), pnr)

    # 2. seeds
    seeds = find_seeds(cn, pnr, p, mask)
    K = len(seeds)
    progress(0.27, f"找到 {K} 个候选细胞")
    if K == 0:
        raise ValueError("没有找到符合条件的细胞。可以降低「最小相关」或「最小 PNR」再试。")
    R = int(np.ceil(1.5 * p.cell_diameter))   # patch half-width: room for the whole cell
    patches = _Patches(seeds, H, W, R)

    # 3. init: seed traces + correlation of each patch pixel with its seed trace
    C = np.zeros((K, T), np.float32)
    ys, xs = seeds[:, 0], seeds[:, 1]
    pix_s1 = np.zeros(npix); pix_s2 = np.zeros(npix)
    yc = [np.zeros(len(ix)) for ix in patches.idx]
    for a, b in _chunks(T, chunk):
        if should_stop():
            raise InterruptedError("已取消")
        blk = np.asarray(Y[a:b]).astype(np.float32)
        Ys = _smooth(blk, 1.0)
        C[:, a:b] = Ys[:, ys, xs].T
        flat = blk.reshape(b - a, -1).astype(np.float64)
        pix_s1 += flat.sum(0); pix_s2 += (flat ** 2).sum(0)
        for k in range(K):
            yc[k] += patches.take(k, blk).T.astype(np.float64) @ C[k, a:b].astype(np.float64)
        progress(0.27 + 0.13 * b / T, f"初始化细胞形状 {b}/{T}")
    ybar = pix_s1 / T
    ystd = np.sqrt(np.maximum(pix_s2 / T - ybar ** 2, 1e-12))
    a_list = []
    for k in range(K):
        c = C[k].astype(np.float64)
        cbar, cvar = c.mean(), c.var()
        idx = patches.idx[k]
        cov = yc[k] / T - ybar[idx] * cbar
        corr = cov / (ystd[idx] * np.sqrt(max(cvar, 1e-12)))
        a = np.where(corr > 0.3, cov / max(cvar, 1e-12), 0.0)
        a_list.append(_clean_footprint(a, patches.boxes[k], seeds[k]))
    keep = [k for k in range(K) if a_list[k].sum() > 0]
    seeds, a_list, C = seeds[keep], [a_list[k] for k in keep], C[keep]
    patches = _Patches(seeds, H, W, R)
    A = _build_sparse(patches, a_list, npix)
    b0 = ybar.astype(np.float32)  # offset, refined below

    # 4. alternate C and A updates
    n_it = max(1, int(p.n_iterations))
    for it in range(n_it):
        C, b0 = _update_C(Y, A, C, b0, ybar, chunk, progress, should_stop,
                          0.40 + 0.45 * it / n_it, 0.45 / n_it / 2, f"第 {it + 1} 轮 更新活动")
        A, patches, C, seeds = _update_A(Y, A, C, b0, patches, seeds, chunk, p, progress, should_stop,
                                         0.40 + 0.45 * (it + 0.5) / n_it, 0.45 / n_it / 2,
                                         f"第 {it + 1} 轮 更新形状")
        A, C, seeds, patches = _merge(A, C, seeds, patches, p, H, W)
        b0 = (ybar - A @ C.mean(1)).astype(np.float32)

    # 5. final traces + raw traces + local fluorescence for dF/F
    C, b0, raw, F_local = _final_traces(ws, Y, A, C, b0, ybar, chunk, progress, should_stop)

    # quality metrics
    area = np.asarray((A > 0).sum(0)).ravel()
    noise = np.array([_noise_std(r) for r in raw])
    snr = C.max(1) / np.maximum(noise, 1e-6)
    min_area = p.min_area_fraction * np.pi * (p.cell_diameter / 2) ** 2
    ok = (area >= min_area) & (C.max(1) > 0)
    A, C, raw, F_local, seeds, snr, area = (A[:, ok], C[ok], raw[ok], F_local[ok],
                                            seeds[ok], snr[ok], area[ok])
    centres = _centroids(A, H, W)

    save_cells(ws, A=A, C=C, raw=raw, F_local=F_local, centres=centres, snr=snr,
               area=area, accepted=np.ones(len(C), bool), shape=(H, W))
    ws.record_step("extraction", asdict(p), {"cells": CELLS_FILE, "n_cells": int(len(C))})
    progress(1.0, f"细胞提取完成，共 {len(C)} 个细胞")
    return {"n_cells": int(len(C))}


def _noise_std(x):
    d = np.diff(x)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) if len(d) else 0.0


def _hals_C(YA, AtA, Cc, n_inner=5):
    """HALS non-negative update of C for one chunk. YA: (K, t), Cc: (K, t)."""
    for _ in range(n_inner):
        for k in range(Cc.shape[0]):
            if AtA[k, k] <= 0:
                continue
            Cc[k] = np.maximum(0, Cc[k] + (YA[k] - AtA[k] @ Cc) / AtA[k, k])
    return Cc


def _update_C(Y, A, C, b0, ybar, chunk, progress, should_stop, f0, df, label):
    T = Y.shape[0]
    At = A.T.tocsr()
    AtA = (At @ A).toarray()
    Atb = At @ b0
    for a, b in _chunks(T, chunk):
        if should_stop():
            raise InterruptedError("已取消")
        flat = np.asarray(Y[a:b]).reshape(b - a, -1).astype(np.float32)
        YA = (At @ flat.T) - Atb[:, None]
        C[:, a:b] = _hals_C(YA, AtA, C[:, a:b].copy())
        progress(f0 + df * b / T, f"{label} {b}/{T}")
    b0 = (ybar - A @ C.mean(1)).astype(np.float32)
    return C, b0


def _update_A(Y, A, C, b0, patches, seeds, chunk, p, progress, should_stop, f0, df, label):
    T, H, W = Y.shape
    K = C.shape[0]
    yc = [np.zeros(len(ix)) for ix in patches.idx]
    for a, b in _chunks(T, chunk):
        if should_stop():
            raise InterruptedError("已取消")
        blk = np.asarray(Y[a:b]).astype(np.float32)
        for k in range(K):
            yc[k] += patches.take(k, blk).T @ C[k, a:b]
        progress(f0 + df * b / T, f"{label} {b}/{T}")
    CCt = C.astype(np.float64) @ C.T.astype(np.float64)
    csum = C.sum(1).astype(np.float64)
    A = A.tocsr()
    new_a = []
    for k in range(K):
        idx = patches.idx[k]
        if CCt[k, k] <= 0:
            new_a.append(np.zeros(len(idx)))
            continue
        A_loc = A[idx]  # (P, K) sparse rows of this patch
        a_k = np.asarray(A_loc[:, k].toarray()).ravel()
        grad = yc[k] - b0[idx] * csum[k] - np.asarray(A_loc @ CCt[:, k]).ravel()
        a_k = np.maximum(0, a_k + grad / CCt[k, k])
        new_a.append(_clean_footprint(a_k, patches.boxes[k], seeds[k]))
    keep = [k for k in range(K) if new_a[k].sum() > 0]
    seeds = seeds[keep]
    C = C[keep]
    patches = _Patches(seeds, H, W, patches.R)
    A = _build_sparse(patches, [new_a[k] for k in keep], H * W)
    return A, patches, C, seeds


def _merge(A, C, seeds, patches, p, H, W):
    K = C.shape[0]
    if K < 2:
        return A, C, seeds, patches
    Ab = (A > 0).astype(np.float32)
    overlap = (Ab.T @ Ab).toarray() > 0
    np.fill_diagonal(overlap, False)
    Cz = C - C.mean(1, keepdims=True)
    nrm = np.linalg.norm(Cz, axis=1)
    corr = (Cz @ Cz.T) / np.maximum(np.outer(nrm, nrm), 1e-12)
    pairs = overlap & (corr > p.merge_threshold)
    if not pairs.any():
        return A, C, seeds, patches
    lab_n, labels = sp.csgraph.connected_components(sp.csr_matrix(pairs), directed=False)
    A = A.tocsc()
    cols, newC, newS = [], [], []
    for g in range(lab_n):
        mem = np.where(labels == g)[0]
        if len(mem) == 1:
            k = mem[0]
            cols.append(A[:, k]); newC.append(C[k]); newS.append(seeds[k])
            continue
        # merged footprint/trace: rank-1 approximation of the members' sum
        Ag = A[:, mem]
        a = np.asarray(Ag @ C[mem].max(1)).ravel()
        c = (C[mem] * np.asarray(Ag.sum(0)).ravel()[:, None]).sum(0) / max(a.sum(), 1e-12)
        best = mem[np.argmax(C[mem].max(1))]
        cols.append(sp.csc_matrix(a[:, None].astype(np.float32)))
        newC.append(c.astype(np.float32)); newS.append(seeds[best])
    A = sp.hstack(cols).tocsc().astype(np.float32)
    seeds = np.array(newS, int)
    return A, np.stack(newC).astype(np.float32), seeds, _Patches(seeds, H, W, patches.R)


def _final_traces(ws, Y, A, C, b0, ybar, chunk, progress, should_stop):
    from .background import source_movie
    T = Y.shape[0]
    F = source_movie(ws, "motion")   # motion-corrected movie (before background removal)
    At = A.T.tocsr()
    AtA = (At @ A).toarray()
    Atb = At @ b0
    diag = np.maximum(np.diag(AtA), 1e-12)
    wsum = np.asarray(A.sum(0)).ravel()
    W_norm = A @ sp.diags(1.0 / np.maximum(wsum, 1e-12))  # weighted-average operator
    raw = np.zeros_like(C)
    F_local = np.zeros_like(C)
    for a, b in _chunks(T, chunk):
        if should_stop():
            raise InterruptedError("已取消")
        flat = np.asarray(Y[a:b]).reshape(b - a, -1).astype(np.float32)
        YA = (At @ flat.T) - Atb[:, None]
        Cc = _hals_C(YA, AtA, C[:, a:b].copy())
        C[:, a:b] = Cc
        raw[:, a:b] = Cc + (YA - AtA @ Cc) / diag[:, None]
        fflat = np.asarray(F[a:b]).reshape(b - a, -1).astype(np.float32)
        F_local[:, a:b] = (W_norm.T @ fflat.T)
        progress(0.85 + 0.15 * b / T, f"提取信号 {b}/{T}")
    b0 = (ybar - A @ C.mean(1)).astype(np.float32)
    # express C and raw as weighted-average fluorescence over the footprint,
    # i.e. the same units as F_local, so dF/F = dF / F0 is meaningful
    scale = (np.asarray(A.power(2).sum(0)).ravel() / np.maximum(wsum, 1e-12)).astype(np.float32)
    return C * scale[:, None], b0, raw * scale[:, None], F_local


def _centroids(A, H, W):
    yy, xx = np.mgrid[:H, :W]
    yy, xx = yy.ravel(), xx.ravel()
    s = np.asarray(A.sum(0)).ravel()
    cy = (A.T @ yy) / np.maximum(s, 1e-12)
    cx = (A.T @ xx) / np.maximum(s, 1e-12)
    return np.stack([cy, cx], 1).astype(np.float32)


# ------------------------------------------------------------------- storage
def save_cells(ws: Workspace, A, C, raw, F_local, centres, snr, area, accepted, shape, **extra):
    A = sp.csc_matrix(A)
    np.savez_compressed(
        ws.path(CELLS_FILE),
        A_data=A.data, A_indices=A.indices, A_indptr=A.indptr, A_shape=np.array(A.shape),
        C=C, raw=raw, F_local=F_local, centres=centres, snr=snr, area=area,
        accepted=accepted, shape=np.array(shape), **extra)


def load_cells(ws: Workspace) -> dict:
    z = np.load(ws.path(CELLS_FILE))
    d = {k: z[k] for k in z.files}
    d["A"] = sp.csc_matrix((d.pop("A_data"), d.pop("A_indices"), d.pop("A_indptr")),
                           shape=tuple(d.pop("A_shape")))
    d["shape"] = tuple(int(v) for v in d["shape"])
    return d


def footprint_image(A, k, shape) -> np.ndarray:
    return np.asarray(A[:, k].toarray()).reshape(shape)


def contours(A, shape, level=0.3):
    """List of (N, 2) arrays of (x, y) contour points per cell."""
    out = []
    for k in range(A.shape[1]):
        img = footprint_image(A, k, shape)
        m = (img > level * img.max()).astype(np.uint8) if img.max() > 0 else np.zeros(shape, np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cs:
            c = max(cs, key=len)[:, 0, :].astype(np.float32)
            out.append(np.vstack([c, c[:1]]))
        else:
            out.append(np.zeros((0, 2), np.float32))
    return out
