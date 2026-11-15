"""Cross-session (cross-day) cell registration.

Follows the logic of CellReg (Sheintuch et al., 2017):

1. Field-of-view alignment. Each session's cells are summed into a "footprint
   projection" image (or the mean projection is high-pass filtered to show the
   blood vessels). Every session is aligned to the reference session with a
   rigid transform (rotation grid search + phase correlation, refined with
   ECC); optionally a smooth non-rigid correction (optical flow) follows.
   The crop position recorded during preprocessing is used as the starting
   guess, so sessions cropped differently still line up.
2. Pair statistics. For every pair of sessions all cell pairs whose centroids
   lie within ``max_dist`` pixels are collected, with their centroid distance
   and the similarity (cosine) of their spatial footprints.
3. Probabilistic model. A two-component mixture is fitted to all pairs with EM:
   same cell (distance ~ Rayleigh, similarity ~ Beta) and different cells
   (distance density rising linearly with distance, similarity ~ Beta). It
   gives P(same cell) for each pair.
4. Clustering across all sessions. Pairs are merged greedily from the most to
   the least likely; two groups are merged only if they contain no two cells
   of the same session and the merge improves sum(P - 0.5) over all their
   cross pairs (correlation clustering), so a cell missing on one day can still
   be linked between the other days.

Everything is numpy / scipy.sparse / OpenCV (no other scipy modules, so the
feature ships as a small in-app update on top of existing installs).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import scipy.sparse as sp

from .workspace import PROJECT_FILE, Workspace

RESULT_FILE = "registration.npz"
META_FILE = "registration.json"
CSV_FILE = "cell_registration.csv"


class Stop(InterruptedError):
    pass


def _check(should_stop):
    if should_stop is not None and should_stop():
        raise Stop("已取消")


@dataclass
class RegParams:
    sessions: list = field(default_factory=list)   # workspace folders; first = reference
    names: list = field(default_factory=list)      # labels shown in tables (e.g. "Day1 训练")
    image: str = "footprints"      # footprints | mean (blood vessels)
    transform: str = "rigid"       # translation | rigid | nonrigid
    max_rotation: float = 10.0     # degrees searched either way
    max_dist: float = 0.0          # pixels; 0 = automatic (1.5 x cell diameter)
    use_similarity: bool = True    # combine footprint similarity with distance
    p_threshold: float = 0.5


# --------------------------------------------------------------- sessions
@dataclass
class SessionCells:
    name: str
    folder: str
    shape: tuple
    A: sp.csc_matrix               # (pixels, n) accepted footprints, own frame
    ids: np.ndarray                # cell numbers as shown in the app
    centres: np.ndarray            # (n, 2) row, col (own frame)
    diameter: float
    offset: np.ndarray             # (row, col) of pixel (0, 0) in binned sensor coordinates
    spatial_bin: int
    mean_img: np.ndarray | None

    @property
    def n(self):
        return self.A.shape[1]


def _is_workspace(folder) -> bool:
    return (Path(folder) / PROJECT_FILE).exists()


def load_session(folder, name: str = "") -> SessionCells:
    from .extraction import load_cells
    folder = Path(folder)
    if not _is_workspace(folder):
        raise ValueError(f"不是本软件的输出文件夹（没有 {PROJECT_FILE}）：\n{folder}")
    ws = Workspace.create_or_open(folder)
    if not ws.step("extraction"):
        raise ValueError(f"这个输出文件夹还没有完成第 5 步「细胞提取」：\n{folder}")
    cells = load_cells(ws)
    acc = cells["accepted"].astype(bool)
    if acc.sum() < 1:
        raise ValueError(f"没有保留的细胞：\n{folder}")
    shape = tuple(int(v) for v in cells["shape"])
    A = sp.csc_matrix(cells["A"])[:, np.where(acc)[0]].astype(np.float32)
    A.data = np.maximum(A.data, 0)
    pre = (ws.step("preprocess") or {}).get("params", {})
    b = int(pre.get("spatial_bin", 1) or 1)
    crop = pre.get("crop") or [0, 0, 0, 0]
    offset = np.array([crop[1] / b, crop[0] / b], float)
    diam = float((ws.step("extraction") or {}).get("params", {}).get("cell_diameter", 0) or 0)
    if diam <= 0:
        diam = float(2 * np.sqrt(np.median(np.diff(A.indptr)) / np.pi)) or 8.0
    mean_img = None
    for f in ("mc_mean.npy", "mean_projection.npy"):
        p = folder / f
        if p.exists():
            m = np.load(p).astype(np.float32)
            if m.shape == shape:
                mean_img = m
                break
    return SessionCells(name=name or folder.name, folder=str(folder), shape=shape, A=A,
                        ids=np.where(acc)[0], centres=_centroids(A, shape), diameter=diam,
                        offset=offset, spatial_bin=b, mean_img=mean_img)


def _centroids(A, shape) -> np.ndarray:
    H, W = shape
    A = sp.csc_matrix(A)
    rr, cc = np.divmod(np.arange(H * W), W)
    w = np.asarray(A.sum(0)).ravel()
    w[w == 0] = 1
    return np.column_stack([np.asarray(A.T @ rr).ravel() / w, np.asarray(A.T @ cc).ravel() / w])


def footprint_projection(A, shape) -> np.ndarray:
    """Sum of footprints, each scaled to a maximum of 1."""
    A = sp.csc_matrix(A, copy=True)
    mx = np.maximum.reduceat(A.data, A.indptr[:-1]) if A.nnz else np.zeros(A.shape[1])
    counts = np.diff(A.indptr)
    scale = np.repeat(1.0 / np.where(mx > 0, mx, 1), counts) if A.nnz else A.data
    A.data = A.data * scale
    return np.asarray(A.sum(1)).reshape(shape).astype(np.float32)


def _highpass(img, sigma):
    img = img.astype(np.float32)
    return img - cv2.GaussianBlur(img, (0, 0), sigma)


def _norm(img, centre=True):
    img = img.astype(np.float32)
    if centre:
        img = img - np.median(img)
    s = img.std()
    return img / s if s > 0 else img


def registration_image(s: SessionCells, kind: str) -> np.ndarray:
    if kind == "mean" and s.mean_img is not None:
        img = _highpass(cv2.GaussianBlur(s.mean_img, (0, 0), 1.0), 3 * s.diameter)
        if img.std() > 1e-6:
            return _norm(img)
    return _norm(cv2.GaussianBlur(footprint_projection(s.A, s.shape), (0, 0), max(0.5, s.diameter / 6)), False)


# --------------------------------------------------------------- alignment
def _affine(angle_deg, centre, shift):
    """2x3 matrix: rotate by angle about centre (x, y), then shift (dx, dy). Acts on (x, y)."""
    M = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), float(angle_deg), 1.0)
    M[:, 2] += shift
    return M


def _compose(M2, M1):
    """M2 after M1 (both 2x3 on (x, y))."""
    a = np.vstack([M1, [0, 0, 1]])
    b = np.vstack([M2, [0, 0, 1]])
    return (b @ a)[:2]


def _canvas(ref, mov, ref_off, mov_off):
    """Place both images on one canvas in sensor coordinates. Returns canvases and their offsets."""
    lo = np.floor(np.minimum(ref_off, mov_off)).astype(int)
    hi = np.ceil(np.maximum(ref_off + ref.shape, mov_off + mov.shape)).astype(int)
    H, W = hi - lo
    out = []
    for img, off in ((ref, ref_off), (mov, mov_off)):
        c = np.zeros((H, W), np.float32)
        r0, c0 = np.round(off - lo).astype(int)
        c[r0:r0 + img.shape[0], c0:c0 + img.shape[1]] = img
        out.append(c)
    return out[0], out[1], lo


def _ncc(a, b, mask=None):
    if mask is None:
        mask = (a != 0) | (b != 0)
    if mask.sum() < 10:
        return 0.0
    x, y = a[mask] - a[mask].mean(), b[mask] - b[mask].mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


def estimate_rigid(ref, mov, max_rotation=10.0, step=0.5, rotate=True, should_stop=None):
    """Matrix M (2x3, on (x, y)) that maps mov onto ref, both on the same canvas."""
    H, W = ref.shape
    centre = ((W - 1) / 2, (H - 1) / 2)
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    angles = np.arange(-max_rotation, max_rotation + 1e-9, step) if rotate and max_rotation > 0 else [0.0]
    best = (-np.inf, None)
    for a in angles:
        _check(should_stop)
        R = _affine(a, centre, (0, 0))
        m = cv2.warpAffine(mov, R, (W, H))
        (dx, dy), resp = cv2.phaseCorrelate(m.astype(np.float32), ref.astype(np.float32), win)
        M = _affine(a, centre, (dx, dy))
        score = _ncc(ref, cv2.warpAffine(mov, M, (W, H)))
        if score > best[0]:
            best = (score, M)
    M = best[1]
    # sub-pixel / sub-step refinement; keep it only if it helps
    try:
        mode = cv2.MOTION_EUCLIDEAN if rotate else cv2.MOTION_TRANSLATION
        warp = M.astype(np.float32).copy()
        inv = cv2.invertAffineTransform(warp)
        _, inv = cv2.findTransformECC(ref, mov, inv, mode,
                                      (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
        M2 = cv2.invertAffineTransform(inv).astype(np.float64)
        if _ncc(ref, cv2.warpAffine(mov, M2, (W, H))) > best[0]:
            M = M2
    except cv2.error:
        pass
    return M


def estimate_flow(ref, mov_aligned, diameter, max_disp):
    """Smooth displacement field (H, W, 2) so that ref(p) ~ mov_aligned(p + flow(p))."""
    def u8(x):
        lo, hi = np.percentile(x, [1, 99.8])
        return np.clip((x - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    win = int(max(15, 4 * diameter)) | 1
    flow = cv2.calcOpticalFlowFarneback(u8(ref), u8(mov_aligned), None, 0.5, 3, win, 5, 7, 1.5, 0)
    sig = max(2.0, 3 * diameter)
    for k in range(2):
        flow[..., k] = cv2.GaussianBlur(flow[..., k], (0, 0), sig)
    mag = np.hypot(flow[..., 0], flow[..., 1])
    flow *= np.minimum(1.0, max_disp / np.maximum(mag, 1e-9))[..., None]
    return flow.astype(np.float32)


@dataclass
class Alignment:
    M: np.ndarray                  # 2x3 on (x, y): session pixel -> reference pixel
    flow: np.ndarray | None        # (H_ref, W_ref, 2): extra (dx, dy) sampled in reference frame
    ncc_before: float
    ncc_after: float


def align(ref: SessionCells, mov: SessionCells, p: RegParams, should_stop=None) -> Alignment:
    a = registration_image(ref, p.image)
    b = registration_image(mov, p.image)
    ca, cb, lo = _canvas(a, b, ref.offset, mov.offset)
    before = _ncc(ca, cb)
    M_c = estimate_rigid(ca, cb, p.max_rotation, rotate=p.transform != "translation",
                         should_stop=should_stop)
    # session pixel -> canvas -> (rigid) -> canvas -> reference pixel
    to_canvas = np.array([[1, 0, mov.offset[1] - lo[1]], [0, 1, mov.offset[0] - lo[0]]], float)
    from_canvas = np.array([[1, 0, lo[1] - ref.offset[1]], [0, 1, lo[0] - ref.offset[0]]], float)
    M = _compose(from_canvas, _compose(M_c, to_canvas))
    H, W = ref.shape
    warped = cv2.warpAffine(b, M, (W, H))
    after = _ncc(a, warped)
    flow = None
    if p.transform == "nonrigid":
        f = estimate_flow(a, warped, ref.diameter, max_disp=1.5 * ref.diameter)
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        nr = cv2.remap(warped, gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR)
        s = _ncc(a, nr)
        if s > after + 0.005:
            flow, after = f, s
    return Alignment(M=M, flow=flow, ncc_before=before, ncc_after=after)


def _maps(al: Alignment, shape):
    """For each reference pixel, where to sample in the session's own frame."""
    H, W = shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    if al.flow is not None:
        gx = gx + al.flow[..., 0]
        gy = gy + al.flow[..., 1]
    inv = cv2.invertAffineTransform(al.M.astype(np.float64))
    mx = (inv[0, 0] * gx + inv[0, 1] * gy + inv[0, 2]).astype(np.float32)
    my = (inv[1, 0] * gx + inv[1, 1] * gy + inv[1, 2]).astype(np.float32)
    return mx, my


def warp_footprints(s: SessionCells, al: Alignment, ref_shape, should_stop=None) -> sp.csc_matrix:
    """Footprints of a session resampled on the reference grid (unit L2 norm per cell)."""
    H, W = ref_shape
    mx, my = _maps(al, ref_shape)
    pad = int(np.ceil(2 * s.diameter)) + 2
    # where each cell centre lands in the reference frame (forward transform)
    c = s.centres
    fx = al.M[0, 0] * c[:, 1] + al.M[0, 1] * c[:, 0] + al.M[0, 2]
    fy = al.M[1, 0] * c[:, 1] + al.M[1, 1] * c[:, 0] + al.M[1, 2]
    if al.flow is not None:
        ix = np.clip(np.round(fx).astype(int), 0, W - 1)
        iy = np.clip(np.round(fy).astype(int), 0, H - 1)
        fx = fx - al.flow[iy, ix, 0]
        fy = fy - al.flow[iy, ix, 1]
    sh, sw = s.shape
    rows, cols, vals = [], [], []
    for k in range(s.n):
        if k % 50 == 0:
            _check(should_stop)
        col = s.A[:, k]
        idx = col.indices
        if not len(idx):
            continue
        rr, cc = np.divmod(idx, sw)
        ext = int(max(rr.max() - rr.min(), cc.max() - cc.min())) // 2 + pad
        r0, r1 = max(0, int(fy[k]) - ext), min(H, int(fy[k]) + ext + 1)
        c0, c1 = max(0, int(fx[k]) - ext), min(W, int(fx[k]) + ext + 1)
        if r1 <= r0 or c1 <= c0:
            continue
        sub_x, sub_y = mx[r0:r1, c0:c1], my[r0:r1, c0:c1]
        br0, br1 = rr.min(), rr.max() + 1
        bc0, bc1 = cc.min(), cc.max() + 1
        box = np.zeros((br1 - br0, bc1 - bc0), np.float32)
        box[rr - br0, cc - bc0] = col.data
        img = cv2.remap(box, (sub_x - np.float32(bc0)).astype(np.float32), (sub_y - np.float32(br0)).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        nz = np.nonzero(img > 1e-6)
        if not len(nz[0]):
            continue
        v = img[nz]
        v = v / np.linalg.norm(v)
        rows.append((nz[0] + r0) * W + nz[1] + c0)
        cols.append(np.full(len(v), k))
        vals.append(v)
    if rows:
        out = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                            shape=(H * W, s.n))
    else:
        out = sp.csc_matrix((H * W, s.n), dtype=np.float32)
    return out


def _unit(A):
    A = sp.csc_matrix(A, dtype=np.float64, copy=True)
    nrm = np.sqrt(np.asarray(A.multiply(A).sum(0)).ravel())
    nrm[nrm == 0] = 1
    return sp.csc_matrix(A @ sp.diags(1 / nrm))


# --------------------------------------------------------------- model
def candidate_pairs(cen_a, cen_b, A_a, A_b, max_dist):
    """All cross pairs within max_dist: (i, j, distance, footprint similarity)."""
    if not len(cen_a) or not len(cen_b):
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0)
    i_l, j_l = [], []
    for s0 in range(0, len(cen_a), 500):            # blocks keep memory small for many cells
        blk = cen_a[s0:s0 + 500]
        dd = np.hypot(blk[:, None, 0] - cen_b[None, :, 0], blk[:, None, 1] - cen_b[None, :, 1])
        ii, jj = np.nonzero(dd <= max_dist)
        i_l.append(ii + s0)
        j_l.append(jj)
    i, j = np.concatenate(i_l).astype(int), np.concatenate(j_l).astype(int)
    if not len(i):
        return i, j, np.zeros(0), np.zeros(0)
    d = np.hypot(*(cen_a[i] - cen_b[j]).T)
    sim = np.asarray(A_a[:, i].multiply(A_b[:, j]).sum(0)).ravel()
    return i, j, d, np.clip(sim, 0, 1)


def _beta_fit(x, w):
    w = np.asarray(w, float)
    sw = w.sum()
    if sw <= 1e-9:
        return 1.0, 1.0
    m = (w * x).sum() / sw
    v = (w * (x - m) ** 2).sum() / sw
    v = min(max(v, 1e-5), m * (1 - m) * 0.999)
    k = m * (1 - m) / v - 1
    return max(m * k, 0.05), max((1 - m) * k, 0.05)


def _betaln(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _expit(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _beta_logpdf(x, a, b):
    return (a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - _betaln(a, b)


@dataclass
class PairModel:
    pi: float = 0.3
    sigma: float = 2.0            # Rayleigh scale of same-cell distances (pixels)
    dmax: float = 10.0
    a1: float = 5.0
    b1: float = 2.0
    a0: float = 1.0
    b0: float = 5.0
    use_similarity: bool = True

    def log_parts(self, d, r):
        d = np.maximum(d, 1e-3)
        r = np.clip(r, 1e-3, 1 - 1e-3)
        l1 = np.log(d / self.sigma ** 2) - d ** 2 / (2 * self.sigma ** 2)
        l1 -= np.log1p(-np.exp(-self.dmax ** 2 / (2 * self.sigma ** 2)) + 1e-12)   # truncated at dmax
        l0 = np.log(2 * d / self.dmax ** 2)
        if self.use_similarity:
            l1 = l1 + _beta_logpdf(r, self.a1, self.b1)
            l0 = l0 + _beta_logpdf(r, self.a0, self.b0)
        return l1, l0

    def p_same(self, d, r):
        l1, l0 = self.log_parts(d, r)
        z = np.log(self.pi) + l1 - np.log1p(-self.pi) - l0
        return _expit(z)

    def fit(self, d, r, n_iter=300):
        d = np.asarray(d, float)
        r = np.clip(np.asarray(r, float), 1e-3, 1 - 1e-3)
        if len(d) < 5:
            return self
        # start: close and similar pairs are probably the same cell
        w = np.clip(np.exp(-(d / (0.3 * self.dmax)) ** 2) * (r if self.use_similarity else 1.0), 0.01, 0.99)
        prev = None
        for _ in range(n_iter):
            self.pi = float(np.clip(w.mean(), 1e-3, 0.999))
            self.sigma = float(np.clip(np.sqrt((w * d ** 2).sum() / (2 * max(w.sum(), 1e-9))),
                                       0.05 * self.dmax, 0.5 * self.dmax))
            if self.use_similarity:
                self.a1, self.b1 = _beta_fit(r, w)
                self.a0, self.b0 = _beta_fit(r, 1 - w)
                # keep the components in order: same cells are the more similar ones
                if self.a1 / (self.a1 + self.b1) < self.a0 / (self.a0 + self.b0):
                    self.a1, self.b1, self.a0, self.b0 = self.a0, self.b0, self.a1, self.b1
            w = self.p_same(d, r)
            if prev is not None and np.abs(w - prev).max() < 1e-6:
                break
            prev = w
        return self

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------- clustering
def cluster(n_cells: list, edges, p_threshold=0.5):
    """Greedy correlation clustering with at most one cell per session per group.

    edges: list of (p, sa, ia, sb, ib). Returns assignment (K, S) of local indices (-1 = absent).
    """
    S = len(n_cells)
    base = np.concatenate([[0], np.cumsum(n_cells)])
    n = int(base[-1])
    P = {}
    for p, sa, ia, sb, ib in edges:
        a, b = base[sa] + ia, base[sb] + ib
        P[(min(a, b), max(a, b))] = p
    sess = np.repeat(np.arange(S), n_cells)
    groups = {k: [k] for k in range(n)}
    owner = np.arange(n)
    for p, sa, ia, sb, ib in sorted(edges, key=lambda e: -e[0]):
        if p < p_threshold:
            break
        ga, gb = owner[base[sa] + ia], owner[base[sb] + ib]
        if ga == gb:
            continue
        A, B = groups[ga], groups[gb]
        if set(sess[A]) & set(sess[B]):
            continue
        gain = sum(P.get((min(x, y), max(x, y)), 0.0) - p_threshold for x in A for y in B)
        if gain <= 0:
            continue
        groups[ga] = A + B
        owner[B] = ga
        del groups[gb]
    assign = np.full((len(groups), S), -1, int)
    score = np.full(len(groups), np.nan)
    for k, members in enumerate(sorted(groups.values(), key=lambda m: (-len(m), m[0]))):
        for m in members:
            assign[k, sess[m]] = m - base[sess[m]]
        if len(members) > 1:
            ps = [P.get((min(x, y), max(x, y)), 0.0) for i, x in enumerate(members) for y in members[i + 1:]]
            score[k] = float(np.mean(ps))
    order = np.lexsort((np.arange(len(assign)), -(assign >= 0).sum(1)))
    return assign[order], score[order]


# --------------------------------------------------------------- main
def run_registration(p: RegParams, out_folder, progress=None, should_stop=None) -> dict:
    progress = progress or (lambda f, m: None)
    if len(p.sessions) < 2:
        raise ValueError("至少需要 2 个 session（输出文件夹）")
    if len(set(map(lambda s: str(Path(s).resolve()), p.sessions))) != len(p.sessions):
        raise ValueError("同一个 session 被添加了两次")
    names = list(p.names) + [""] * (len(p.sessions) - len(p.names))
    S = len(p.sessions)
    progress(0.02, "读取各 session 的细胞…")
    ses = [load_session(f, names[k] or Path(f).name) for k, f in enumerate(p.sessions)]
    names = [s.name for s in ses]
    bins = {s.spatial_bin for s in ses}
    if len(bins) > 1:
        raise ValueError("各 session 预处理时的「空间降采样」不同，像素大小不一样，无法直接配准。"
                         "请用相同的降采样重新预处理。")
    ref = ses[0]
    diam = float(np.median([s.diameter for s in ses]))
    max_dist = float(p.max_dist) if p.max_dist > 0 else 1.5 * diam
    ref_shape = ref.shape
    H, W = ref_shape

    # 1. alignment
    aligns, footprints, centres = [], [], []
    for k, s in enumerate(ses):
        _check(should_stop)
        progress(0.05 + 0.5 * k / S, f"对齐视野：{s.name}（{k + 1}/{S}）")
        if k == 0:
            al = Alignment(M=np.array([[1.0, 0, 0], [0, 1.0, 0]]), flow=None, ncc_before=1.0, ncc_after=1.0)
        else:
            al = align(ref, s, p, should_stop)
        A = warp_footprints(s, al, ref_shape, should_stop) if k else _unit(s.A)
        aligns.append(al)
        footprints.append(_unit(A))
        centres.append(_centroids(footprints[-1].multiply(footprints[-1]), ref_shape))

    # 2. candidate pairs
    progress(0.6, "计算候选细胞对…")
    pair_list = []
    for a in range(S):
        for b in range(a + 1, S):
            i, j, d, r = candidate_pairs(centres[a], centres[b], footprints[a], footprints[b], max_dist)
            pair_list.append((a, b, i, j, d, r))
    d_all = np.concatenate([x[4] for x in pair_list]) if pair_list else np.zeros(0)
    r_all = np.concatenate([x[5] for x in pair_list]) if pair_list else np.zeros(0)
    if len(d_all) < 5:
        raise ValueError("各 session 之间几乎没有位置相近的细胞。请检查是否选对了同一只动物，"
                         "或增大「最大距离」。")

    # 3. model
    progress(0.7, "拟合匹配概率模型…")
    model = PairModel(dmax=max_dist, sigma=0.15 * max_dist, use_similarity=p.use_similarity).fit(d_all, r_all)
    edges, pair_stats = [], []
    p_all = []
    for a, b, i, j, d, r in pair_list:
        ps = model.p_same(d, r) if len(d) else np.zeros(0)
        p_all.append(ps)
        edges += [(float(q), a, int(x), b, int(y)) for q, x, y in zip(ps, i, j)]

    # 4. clustering
    progress(0.8, "跨 session 归并细胞…")
    assign, score = cluster([s.n for s in ses], edges, p.p_threshold)
    K = len(assign)
    present = assign >= 0
    ids = np.where(present, 0, -1)
    for k, s in enumerate(ses):
        ids[present[:, k], k] = s.ids[assign[present[:, k], k]]

    # pairwise summary from the final grouping
    p_all = np.concatenate(p_all) if p_all else np.zeros(0)
    pos = 0
    for a, b, i, j, d, r in pair_list:
        ps = p_all[pos:pos + len(d)]
        pos += len(d)
        both = present[:, a] & present[:, b]
        matched = set(zip(assign[both, a], assign[both, b]))
        m = np.array([(x, y) in matched for x, y in zip(i, j)], bool)
        pair_stats.append({
            "a": a, "b": b, "n_a": ses[a].n, "n_b": ses[b].n, "matched": int(both.sum()),
            "frac_of_a": float(both.sum() / max(ses[a].n, 1)), "frac_of_b": float(both.sum() / max(ses[b].n, 1)),
            "exp_false_pos": float((1 - ps[m]).sum()), "exp_false_neg": float(ps[~m & (ps >= 0.5)].sum()),
        })

    # images for the views: aligned footprint projections and mean images
    proj = np.stack([footprint_projection(f, ref_shape) for f in footprints])
    raw_proj = [footprint_projection(s.A, s.shape) for s in ses]
    before = np.zeros((S, H, W), np.float32)
    for k, s in enumerate(ses):            # crop-offset placement only (no registration)
        sh = np.round(s.offset - ref.offset).astype(int)
        M0 = np.array([[1.0, 0, sh[1]], [0, 1.0, sh[0]]])
        before[k] = cv2.warpAffine(raw_proj[k], M0, (W, H))

    out = Path(out_folder)
    out.mkdir(parents=True, exist_ok=True)
    pair_a = np.concatenate([np.full(len(x[2]), x[0]) for x in pair_list])
    pair_b = np.concatenate([np.full(len(x[2]), x[1]) for x in pair_list])
    per = np.full((S, K, 2), np.nan)
    for k in range(S):
        per[k, present[:, k]] = centres[k][assign[present[:, k], k]]
    arrays = dict(assign=assign, cell_ids=ids, score=score, n_sessions=present.sum(1),
                  centres=np.nanmean(per, 0) if K else np.zeros((0, 2)),
                  proj=proj, proj_before=before,
                  pair_a=pair_a, pair_b=pair_b, pair_d=d_all, pair_r=r_all, pair_p=p_all,
                  M=np.stack([al.M for al in aligns]))
    for k in range(S):
        arrays[f"centres_{k}"] = centres[k]
        arrays[f"ids_{k}"] = ses[k].ids
    np.savez_compressed(out / RESULT_FILE, **arrays)
    summary = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sessions": [s.folder for s in ses], "names": names, "n_cells": [s.n for s in ses],
        "reference": names[0], "shape": list(ref_shape), "max_dist": max_dist, "diameter": diam,
        "n_registered": int(K), "n_in_all": int(present.all(1).sum()),
        "n_by_sessions": [int((present.sum(1) == m).sum()) for m in range(1, S + 1)],
        "alignment": [{"ncc_before": al.ncc_before, "ncc_after": al.ncc_after,
                       "rotation_deg": float(np.degrees(np.arctan2(al.M[1, 0], al.M[0, 0]))),
                       "shift_xy": _centre_shift(al.M, s_.shape), "nonrigid": al.flow is not None}
                      for al, s_ in zip(aligns, ses)],
        "model": model.to_dict(), "pairs": pair_stats,
        "params": asdict(p),
    }
    with open(out / META_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    export_csv(out, summary, arrays)
    progress(1.0, "跨 session 配准完成")
    return summary


def _centre_shift(M, shape):
    """How far the centre of a session's field of view moved in the reference frame (dx, dy)."""
    cy, cx = (np.asarray(shape, float) - 1) / 2
    return [float(M[0, 0] * cx + M[0, 1] * cy + M[0, 2] - cx), float(M[1, 0] * cx + M[1, 1] * cy + M[1, 2] - cy)]


def export_csv(folder, summary, res):
    import csv
    folder = Path(folder)
    names = summary["names"]
    with open(folder / CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["global_id", *[f"{n}_cell" for n in names], "n_sessions", "score"])
        for k, row in enumerate(res["cell_ids"]):
            sc = res["score"][k]
            w.writerow([k + 1, *["" if v < 0 else int(v) for v in row], int((row >= 0).sum()),
                        "" if not np.isfinite(sc) else f"{sc:.3f}"])
    return folder / CSV_FILE


def load_registration(folder) -> dict:
    folder = Path(folder)
    if not (folder / META_FILE).exists():
        raise FileNotFoundError(f"这里没有配准结果：{folder}")
    z = np.load(folder / RESULT_FILE, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    with open(folder / META_FILE, encoding="utf-8") as f:
        out["summary"] = json.load(f)
    out["folder"] = str(folder)
    return out


def check_output_folder(folder) -> Path:
    """Results go to an empty folder or one that already holds a registration."""
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()) and not (folder / META_FILE).exists():
        raise ValueError("配准结果文件夹不是空的。为了不覆盖已有文件，请新建或选择一个空文件夹。")
    if (folder / PROJECT_FILE).exists():
        raise ValueError("不能把配准结果放在某个 session 的输出文件夹里，请另选一个文件夹。")
    return folder


def describe(s: dict) -> str:
    names = s["names"]
    S = len(names)
    lines = [f"{S} 个 session，参考：{names[0]}。最大距离 {s['max_dist']:.1f} 像素。",
             f"共 {s['n_registered']} 个细胞（去重后），在全部 {S} 个 session 都找到的：{s['n_in_all']} 个。"]
    for k, (n, al) in enumerate(zip(names, s["alignment"])):
        if k == 0:
            continue
        lines.append(f"{n}：旋转 {al['rotation_deg']:.1f}°，视野中心移动 ({al['shift_xy'][0]:.1f}, {al['shift_xy'][1]:.1f}) 像素"
                     f"{'，含非刚性校正' if al['nonrigid'] else ''}；"
                     f"对齐相似度 {al['ncc_before']:.2f} → {al['ncc_after']:.2f}")
    for ps in s["pairs"]:
        lines.append(f"{names[ps['a']]} ↔ {names[ps['b']]}：匹配 {ps['matched']} 个"
                     f"（占 {ps['frac_of_a']:.0%} / {ps['frac_of_b']:.0%}），"
                     f"估计误配约 {ps['exp_false_pos']:.1f} 个、漏配约 {ps['exp_false_neg']:.1f} 个")
    m = s["model"]
    lines.append(f"同一细胞的中心距离约 {m['sigma'] * 1.25:.1f} 像素（中位数）。")
    low = [al["ncc_after"] for al in s["alignment"][1:] if al["ncc_after"] < 0.3]
    if low:
        lines.append("⚠ 有 session 对齐后相似度低于 0.3，视野可能变化很大或不是同一只动物，请看「对齐叠加图」确认。")
    return "\n".join(lines)
