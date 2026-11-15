"""Step 7: behavior video analysis.

Tracking (no GPU, no training): a background image is the median of frames
sampled across the video; in each frame the animal is the largest blob that
differs from the background inside the arena. From the centroid track we get
distance, speed and zone occupancy.

Freezing (for fear conditioning): the motion index is the fraction of arena
pixels whose brightness changed between consecutive frames by more than a
noise threshold. Freezing = motion index below a threshold for at least a
minimum duration (1 s by default), the same logic as EthoVision's mobility
state / FreezeFrame.

Tracks from other software can be imported instead: EthoVision XT exports
(.txt/.csv/.xlsx) and DeepLabCut CSV files.

Results live in <workspace>/behavior/.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .workspace import Workspace

BEH_DIR = "behavior"
TRACK_FILE = "track.npz"
CONFIG_FILE = "behavior.json"


# --------------------------------------------------------------------- video
class BehaviorVideo:
    def __init__(self, path: str | Path, fps: float | None = None):
        self.path = Path(path)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {self.path}")
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.file_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        if self.n_frames <= 0:
            n = 0
            while cap.grab():
                n += 1
            self.n_frames = n
        cap.release()
        self.fps = float(fps) if fps else self.file_fps
        self._cap = None
        self._pos = -1

    @property
    def duration_s(self):
        return self.n_frames / self.fps

    def get_frame(self, i: int, gray=False) -> np.ndarray:
        if self._cap is None:
            self._cap = cv2.VideoCapture(str(self.path))
        if i != self._pos + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = self._cap.read()
        self._pos = i if ok else -1
        if not ok:
            raise IOError(f"读取第 {i} 帧失败")
        if gray:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def iter_gray(self, start=0, stop=None, scale=1.0):
        stop = self.n_frames if stop is None else min(stop, self.n_frames)
        cap = cv2.VideoCapture(str(self.path))
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        i = start
        while i < stop:
            ok, frame = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if scale != 1.0:
                g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            yield i, g
            i += 1
        cap.release()

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# --------------------------------------------------------------------- shapes
@dataclass
class Shape:
    """A region in video pixel coordinates: rect (x, y, w, h), circle (cx, cy, r) or polygon."""
    kind: str
    data: list
    name: str = ""

    def mask(self, h, w, scale=1.0) -> np.ndarray:
        m = np.zeros((h, w), np.uint8)
        d = np.asarray(self.data, float) * scale
        if self.kind == "rect":
            x, y, ww, hh = d
            cv2.rectangle(m, (int(x), int(y)), (int(x + ww), int(y + hh)), 1, -1)
        elif self.kind == "circle":
            cx, cy, r = d
            cv2.circle(m, (int(round(cx)), int(round(cy))), int(round(r)), 1, -1)
        else:
            cv2.fillPoly(m, [np.round(d.reshape(-1, 2)).astype(np.int32)], 1)
        return m.astype(bool)

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float); y = np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        out = np.zeros(x.shape, bool)
        d = np.asarray(self.data, float)
        if self.kind == "rect":
            x0, y0, ww, hh = d
            out[ok] = (x[ok] >= x0) & (x[ok] <= x0 + ww) & (y[ok] >= y0) & (y[ok] <= y0 + hh)
        elif self.kind == "circle":
            cx, cy, r = d
            out[ok] = (x[ok] - cx) ** 2 + (y[ok] - cy) ** 2 <= r ** 2
        else:
            poly = d.reshape(-1, 2).astype(np.float32)
            out[ok] = [cv2.pointPolygonTest(poly, (float(a), float(b)), False) >= 0
                       for a, b in zip(x[ok], y[ok])]
        return out

    def to_dict(self):
        return {"kind": self.kind, "data": [float(v) for v in np.ravel(self.data)], "name": self.name}

    @classmethod
    def from_dict(cls, d):
        return cls(d["kind"], d["data"], d.get("name", ""))


# ------------------------------------------------------------------- config
@dataclass
class TrackingParams:
    animal: str = "auto"          # "dark", "light" or "auto"
    threshold: float = 0.0        # 0 = automatic (Otsu on the difference image)
    min_area: int = 50            # px (at processing scale)
    scale: float = 0.5            # process at reduced resolution for speed
    n_background: int = 60
    motion_threshold: int = 15    # grey levels a pixel must change to count as motion
    freeze_threshold: float = 0.10  # % of arena pixels changing
    freeze_min_s: float = 1.0
    smooth_s: float = 0.2         # position smoothing window


@dataclass
class BehaviorConfig:
    video: str = ""
    fps: float = 30.0
    arena: dict | None = None
    zones: list = field(default_factory=list)
    px_per_cm: float | None = None
    params: dict = field(default_factory=lambda: asdict(TrackingParams()))
    source: str = "tracking"      # "tracking", "ethovision", "dlc"

    def save(self, ws: Workspace):
        d = beh_dir(ws)
        with open(d / CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, ws: Workspace) -> "BehaviorConfig | None":
        p = beh_dir(ws) / CONFIG_FILE
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            return cls(**json.load(f))

    def arena_shape(self):
        return Shape.from_dict(self.arena) if self.arena else None

    def zone_shapes(self):
        return [Shape.from_dict(z) for z in self.zones]


def beh_dir(ws: Workspace) -> Path:
    d = ws.path(BEH_DIR)
    d.mkdir(exist_ok=True)
    return d


# ------------------------------------------------------------------- tracking
def background_image(video: BehaviorVideo, n: int, scale: float) -> np.ndarray:
    idx = np.linspace(0, max(0, video.n_frames - 1), num=min(n, video.n_frames)).astype(int)
    frames = []
    cap = cv2.VideoCapture(str(video.path))
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if scale != 1.0:
            g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(g)
    cap.release()
    if not frames:
        raise IOError("无法读取视频帧")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _decide_polarity(video, bg, arena_mask, scale):
    """Is the animal darker or brighter than the background?"""
    neg = pos = 0.0
    for i, g in video.iter_gray(0, min(video.n_frames, 300), scale):
        if i % 30:
            continue
        d = g.astype(np.int16) - bg.astype(np.int16)
        d[~arena_mask] = 0
        neg += np.sum(np.clip(-d, 0, None) > 25)
        pos += np.sum(np.clip(d, 0, None) > 25)
    return "dark" if neg >= pos else "light"


def run_tracking(session, ws: Workspace, cfg: BehaviorConfig,
                 progress: Callable[[float, str], None] | None = None,
                 should_stop: Callable[[], bool] | None = None) -> dict:
    progress = progress or (lambda f, m: None)
    should_stop = should_stop or (lambda: False)
    p = TrackingParams(**cfg.params)
    video = BehaviorVideo(cfg.video, cfg.fps)
    s = float(p.scale)
    progress(0.0, "计算背景图像…")
    bg = background_image(video, p.n_background, s)
    h, w = bg.shape
    arena = cfg.arena_shape()
    amask = arena.mask(h, w, s) if arena else np.ones((h, w), bool)
    polarity = p.animal if p.animal in ("dark", "light") else _decide_polarity(video, bg, amask, s)

    N = video.n_frames
    x = np.full(N, np.nan); y = np.full(N, np.nan); area = np.zeros(N)
    motion = np.full(N, np.nan)
    prev = None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    n_arena = max(1, int(amask.sum()))
    thr_used = []
    for i, g in video.iter_gray(0, N, s):
        if should_stop():
            raise InterruptedError("已取消")
        d = cv2.subtract(bg, g) if polarity == "dark" else cv2.subtract(g, bg)
        d[~amask] = 0
        thr = p.threshold
        if thr <= 0:
            vals = d[amask]
            t_otsu, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thr = max(15.0, float(t_otsu))
            thr_used.append(thr)
        fg = (d > thr).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(fg, 8)
        if n > 1:
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if stats[k, cv2.CC_STAT_AREA] >= p.min_area:
                x[i], y[i] = cents[k] / s
                area[i] = stats[k, cv2.CC_STAT_AREA] / (s * s)
        if prev is not None:
            diff = cv2.absdiff(g, prev)
            motion[i] = 100.0 * np.count_nonzero((diff > p.motion_threshold) & amask) / n_arena
        prev = g
        if i % 200 == 0 or i == N - 1:
            progress((i + 1) / N, f"追踪 {i + 1}/{N} 帧")
    video.close()
    if N > 1 and np.isnan(motion[0]):
        motion[0] = motion[1] if N > 1 else 0
    t = np.arange(N) / video.fps
    save_track(ws, t=t, x=x, y=y, area=area, motion=motion)
    np.save(beh_dir(ws) / "background.npy", bg)
    cfg.source = "tracking"
    cfg.params = asdict(p)
    cfg.save(ws)
    found = float(np.isfinite(x).mean())
    ws.record_step("behavior", {**asdict(p), "polarity": polarity,
                                "auto_threshold": float(np.median(thr_used)) if thr_used else None},
                   {"track": f"{BEH_DIR}/{TRACK_FILE}", "detected_fraction": found})
    progress(1.0, "追踪完成")
    return {"detected_fraction": found, "polarity": polarity, "n_frames": N}


# --------------------------------------------------------------- track files
def save_track(ws: Workspace, **arrays):
    np.savez_compressed(beh_dir(ws) / TRACK_FILE, **arrays)


def load_track(ws: Workspace) -> dict:
    z = np.load(beh_dir(ws) / TRACK_FILE)
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- analysis
def _interp_nans(v):
    v = v.astype(float).copy()
    ok = np.isfinite(v)
    if ok.sum() >= 2:
        v[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), v[ok])
    return v


def _smooth(v, n):
    if n <= 1:
        return v
    k = np.ones(int(n)) / int(n)
    pad = int(n) // 2
    vp = np.pad(v, pad, mode="edge")
    return np.convolve(vp, k, mode="same")[pad:pad + len(v)]


def bouts(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """(start, stop) index pairs of runs of True at least min_len long."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(int))
    starts, stops = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(int(a), int(b)) for a, b in zip(starts, stops) if b - a >= min_len]


def analyze(track: dict, cfg: BehaviorConfig) -> dict:
    """Per-frame behaviour variables and session summary."""
    p = TrackingParams(**cfg.params)
    t = track["t"]
    fps = 1.0 / np.median(np.diff(t)) if len(t) > 1 else cfg.fps
    unit = "cm" if cfg.px_per_cm else "px"
    k = 1.0 / cfg.px_per_cm if cfg.px_per_cm else 1.0
    n_s = max(1, int(round(p.smooth_s * fps)))
    xs = _smooth(_interp_nans(track["x"]), n_s)
    ys = _smooth(_interp_nans(track["y"]), n_s)
    step = np.hypot(np.diff(xs), np.diff(ys)) * k
    step = np.concatenate([[0.0], step])
    if track.get("units") is not None and str(track["units"]) == "cm":
        unit = "cm"
    speed = step * fps

    out = {"t": t, "x": xs, "y": ys, "speed": speed, "unit": unit}
    summary = {"duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
               "distance": float(np.nansum(step)), "mean_speed": float(np.nanmean(speed)),
               "unit": unit}

    if "freezing" in track:     # imported freezing state
        freezing = track["freezing"].astype(bool)
    elif "motion" in track and np.isfinite(track["motion"]).any():
        mot = _smooth(np.nan_to_num(track["motion"]), max(1, int(round(0.2 * fps))))
        out["motion"] = mot
        below = mot < p.freeze_threshold
        freezing = np.zeros(len(t), bool)
        for a, b in bouts(below, int(round(p.freeze_min_s * fps))):
            freezing[a:b] = True
    else:
        freezing = None
    if freezing is not None:
        out["freezing"] = freezing
        fb = bouts(freezing, 1)
        summary.update(freezing_percent=100.0 * freezing.mean(), freezing_bouts=len(fb),
                       freezing_mean_bout_s=float(np.mean([(b - a) / fps for a, b in fb])) if fb else 0.0)

    zones = {}
    for z in cfg.zone_shapes():
        inside = z.contains(xs, ys)
        ent = bouts(inside, 1)
        zones[z.name] = inside
        summary[f"zone[{z.name}]_time_s"] = float(inside.sum() / fps)
        summary[f"zone[{z.name}]_entries"] = len(ent)
        summary[f"zone[{z.name}]_latency_s"] = float(t[ent[0][0]] - t[0]) if ent else float("nan")
    out["zones"] = zones
    out["summary"] = summary
    return out


def export_behavior(ws: Workspace, cfg: BehaviorConfig, folder: str | Path | None = None) -> Path:
    res = analyze(load_track(ws), cfg)
    out = Path(folder) if folder else ws.path("export")
    out.mkdir(parents=True, exist_ok=True)
    u = res["unit"]
    cols = {"time_s": res["t"], f"x_{u}": res["x"] / (cfg.px_per_cm or 1) if u == "cm" and cfg.px_per_cm else res["x"],
            f"y_{u}": res["y"] / (cfg.px_per_cm or 1) if u == "cm" and cfg.px_per_cm else res["y"],
            f"speed_{u}_per_s": res["speed"]}
    if "motion" in res:
        cols["motion_index_pct"] = res["motion"]
    if "freezing" in res:
        cols["freezing"] = res["freezing"].astype(int)
    for name, inside in res["zones"].items():
        cols[f"in_{name}"] = inside.astype(int)
    with open(out / "behavior_frames.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(cols.keys())
        for row in zip(*cols.values()):
            wr.writerow([f"{v:.5g}" if isinstance(v, (float, np.floating)) else v for v in row])
    with open(out / "behavior_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["measure", "value"])
        for k_, v in res["summary"].items():
            wr.writerow([k_, v])
    return out


# ----------------------------------------------------------------- importers
def _num(s):
    s = str(s).strip().strip('"')
    if s in ("", "-", "NaN", "nan"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _rows_from_file(path: Path) -> list[list[str]]:
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws_ = wb.worksheets[0]
        rows = [["" if v is None else str(v) for v in r] for r in ws_.iter_rows(values_only=True)]
        wb.close()
        return rows
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "gbk", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    first = text.splitlines()[0] if text else ""
    delim = ";" if first.count(";") > first.count(",") else ("\t" if "\t" in first else ",")
    return [r for r in csv.reader(text.splitlines(), delimiter=delim)]


def import_ethovision(path: str | Path) -> dict:
    """Read an EthoVision XT raw-data export (txt/csv/xlsx). Positions are in cm."""
    path = Path(path)
    rows = _rows_from_file(path)
    n_header = None
    for r in rows[:5]:
        if r and "header lines" in r[0].lower():
            n_header = int(_num(r[1]))
            break
    if n_header is None:   # fall back: find the row that has "Trial time"
        n_header = next((i + 2 for i, r in enumerate(rows) if any("trial time" in c.lower() for c in r)), None)
        if n_header is None:
            raise ValueError("无法识别 EthoVision 导出格式：找不到 'Trial time' 列")
    names = [c.strip().strip('"') for c in rows[n_header - 2]]
    data = rows[n_header:]

    def col(*keys):
        for i, n in enumerate(names):
            if all(k.lower() in n.lower() for k in keys):
                return np.array([_num(r[i]) if i < len(r) else np.nan for r in data])
        return None
    t = col("trial time")
    x = col("x center")
    y = col("y center")
    if t is None or x is None or y is None:
        raise ValueError("EthoVision 文件缺少 Trial time / X center / Y center 列")
    out = {"t": t, "x": x, "y": y, "units": np.array("cm")}
    for i, n in enumerate(names):
        nl = n.lower()
        if "freez" in nl or "immobile" in nl:
            out["freezing"] = np.array([_num(r[i]) if i < len(r) else 0 for r in data]) > 0
            break
    ok = np.isfinite(out["t"])
    return {k: (v[ok] if isinstance(v, np.ndarray) and v.shape == ok.shape else v) for k, v in out.items()}


def import_dlc(path: str | Path, fps: float, bodypart: str | None = None, min_likelihood=0.6) -> dict:
    """Read a DeepLabCut CSV. Uses one body part, or the mean of all confident ones."""
    rows = _rows_from_file(Path(path))
    parts, coords = rows[1][1:], rows[2][1:]
    data = np.array([[_num(v) for v in r[1:]] for r in rows[3:] if r])
    names = sorted(set(parts), key=parts.index)
    use = [bodypart] if bodypart else names
    xs, ys = [], []
    for bp in use:
        ix = [i for i, (p_, c) in enumerate(zip(parts, coords)) if p_ == bp]
        cx = next(i for i in ix if coords[i] == "x")
        cy = next(i for i in ix if coords[i] == "y")
        cl = next((i for i in ix if coords[i] == "likelihood"), None)
        x, y = data[:, cx].copy(), data[:, cy].copy()
        if cl is not None:
            bad = data[:, cl] < min_likelihood
            x[bad] = np.nan; y[bad] = np.nan
        xs.append(x); ys.append(y)
    with np.errstate(all="ignore"):
        x = np.nanmean(np.stack(xs), 0); y = np.nanmean(np.stack(ys), 0)
    return {"t": np.arange(len(x)) / fps, "x": x, "y": y, "bodyparts": np.array(names)}


def run_import(session, ws: Workspace, args: dict, progress=None, should_stop=None) -> dict:
    """args: {"kind": "ethovision"|"dlc", "path": ..., "cfg": BehaviorConfig, "bodypart": ...}"""
    cfg: BehaviorConfig = args["cfg"]
    if args["kind"] == "ethovision":
        tr = import_ethovision(args["path"])
        cfg.px_per_cm = None          # positions already in cm
    else:
        tr = import_dlc(args["path"], cfg.fps, args.get("bodypart"))
    save_track(ws, **tr)
    cfg.source = args["kind"]
    cfg.save(ws)
    ws.record_step("behavior", {"source": args["kind"], "file": str(args["path"])},
                   {"track": f"{BEH_DIR}/{TRACK_FILE}", "n_frames": int(len(tr["t"]))})
    return {"n_frames": int(len(tr["t"])), "detected_fraction": float(np.isfinite(tr["x"]).mean())}
