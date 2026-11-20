"""Step 8: synchronise behavior with calcium imaging, and joint analysis.

Time base: miniscope time in seconds from the first raw miniscope frame (the
same time base as the calcium traces).

Alignment methods
  led     LED in the behavior video driven by the Arduino that triggers the
          miniscope (see the LED sync guide): 1 s start mark at the trigger,
          200 ms flash every N s, 2 s end mark. Piece-wise linear mapping
          through all detected flashes (handles drift and dropped frames).
  manual  one event visible in both recordings (or a known delay): behavior
          frame F happened at miniscope time T; the behavior frame rate maps
          the rest.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .workspace import Workspace

SYNC_FILE = "behavior/sync.npz"


# ------------------------------------------------------------ LED detection
def led_trace(video_path, roi, progress=None, should_stop=None):
    """Mean brightness of the LED region (x, y, w, h) in every frame."""
    progress = progress or (lambda f, m: None)
    should_stop = should_stop or (lambda: False)
    x, y, w, h = (int(round(v)) for v in roi)
    cap = cv2.VideoCapture(str(video_path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        vals = []
        while True:
            if should_stop():
                raise InterruptedError("已取消")
            ok = cap.grab()
            if not ok:
                break
            ok, frame = cap.retrieve()
            vals.append(float(frame[y:y + h, x:x + w].mean()))
            if len(vals) % 500 == 0:
                progress(min(0.99, len(vals) / n), f"读取 LED 亮度 {len(vals)}/{n}")
    finally:
        cap.release()
    return np.asarray(vals)


def detect_pulses(trace):
    lo, hi = np.percentile(trace, 5), np.percentile(trace, 99.9)
    noise = 1.4826 * np.median(np.abs(trace - np.median(trace)))
    if hi - lo < max(5 * noise, 3.0):
        raise ValueError("LED 区域亮度变化太小。请检查框选位置是否正好在 LED 上。")
    thr = (lo + hi) / 2.0
    on = trace > thr
    d = np.diff(on.astype(int))
    rises = list(np.where(d == 1)[0] + 1)
    falls = list(np.where(d == -1)[0] + 1)
    if on[0]:
        rises = [0] + rises
    if on[-1]:
        falls = falls + [len(trace)]
    pulses = []
    for r, f in zip(rises, falls):
        if r > 0 and trace[r] != trace[r - 1]:
            frac = (thr - trace[r - 1]) / (trace[r] - trace[r - 1])
            onset = (r - 1) + float(np.clip(frac, 0.0, 1.0))
        else:
            onset = float(r)
        pulses.append({"onset": onset, "rise": r, "n_frames": f - r})
    return pulses, thr


@dataclass
class LedParams:
    interval_s: float = 30.0
    flash_ms: float = 200.0
    start_ms: float = 1000.0
    end_ms: float = 2000.0
    lag_frames: float = 1.0
    mode: str = "auto"            # auto | frame | time
    miniscope_fps_setting: float | None = None


def classify(pulses, fps_b, p: LedParams):
    for q in pulses:
        dur = q["n_frames"] / fps_b * 1000.0
        q["dur_ms"] = dur
        if dur >= 0.5 * (p.start_ms + p.end_ms):
            q["kind"] = "end"
        elif dur >= 0.5 * (p.flash_ms + p.start_ms):
            q["kind"] = "start"
        elif dur >= 0.3 * p.flash_ms:
            q["kind"] = "flash"
        else:
            q["kind"] = "noise"
    starts = [q for q in pulses if q["kind"] == "start"]
    if not starts:
        raise ValueError("没有找到开始标记（约 1 秒的长亮）。请检查 LED 框选位置，"
                         "或行为视频是否在触发之后才开始录制。")
    start = starts[0]
    flashes = [q for q in pulses if q["kind"] == "flash" and q["onset"] > start["onset"]]
    ends = [q for q in pulses if q["kind"] == "end" and q["onset"] > start["onset"]]
    end = ends[0] if ends else None
    if end is not None:
        flashes = [q for q in flashes if q["onset"] < end["onset"]]
    return start, flashes, end


def read_arduino_log(path):
    mode, flashes, stop = None, {}, None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"START,.*mode=(\w+)", line)
            if m:
                mode, flashes, stop = m.group(1).lower(), {}, None
                continue
            m = re.match(r"FLASH,(\d+),(\d+),(-?\d+)", line)
            if m:
                flashes[int(m.group(1))] = (float(m.group(2)), int(m.group(3)))
                continue
            m = re.match(r"STOP,(\d+),(-?\d+)", line)
            if m:
                stop = (float(m.group(1)), int(m.group(2)))
    if mode is None:
        raise ValueError("Arduino 记录里没有 START 行")
    return mode, flashes, stop


def _ms_of_count(c, ts, daq, lag, period):
    if daq is not None:
        good = daq >= 0
        d, t = daq[good], ts[good]
        target = d[0] - lag + c
        idx = np.where(d == target)[0]
        if len(idx):
            return float(t[idx[0]])
        if target < d[0]:
            return float(t[0] - (d[0] - target) * period)
        if target > d[-1]:
            return float(t[-1] + (target - d[-1]) * period)
        return float(np.interp(target, d, t))
    k = c - lag
    if 0 <= k < len(ts):
        return float(ts[int(k)])
    return float(ts[0] + k * period)


def led_alignment(trace, fps_b, ts_ms, daq, p: LedParams, arduino_log=None):
    """Return (behavior->miniscope ms for each behavior frame, anchors, report)."""
    pulses, thr = detect_pulses(trace)
    start, flashes, end = classify(pulses, fps_b, p)
    if daq is not None and np.sum(daq >= 0) > 1:
        g = daq >= 0
        period = float((ts_ms[g][-1] - ts_ms[g][0]) / (daq[g][-1] - daq[g][0]))
    else:
        period = float((ts_ms[-1] - ts_ms[0]) / (len(ts_ms) - 1))
    fps_m = p.miniscope_fps_setting or float(round(1000.0 / period))
    sched = {}
    stop = None
    if arduino_log:
        mode, logf, stop = read_arduino_log(arduino_log)
        for k, (ms, cnt) in logf.items():
            sched[k] = (_ms_of_count(cnt, ts_ms, daq, p.lag_frames, period) if mode == "frame" else None, ms)
    else:
        mode = p.mode if p.mode != "auto" else ("frame" if daq is not None else "time")
    t_trig = _ms_of_count(0, ts_ms, daq, p.lag_frames, period) if mode == "frame" \
        else float(ts_ms[0] - p.lag_frames * period)

    def flash_ms(k):
        if k in sched and sched[k][0] is not None:
            return sched[k][0]
        if mode == "frame":
            return _ms_of_count(k * round(p.interval_s * fps_m), ts_ms, daq, p.lag_frames, period)
        return t_trig + (sched[k][1] if k in sched else k * p.interval_s * 1000.0)

    anchors = [("start", 0, start["onset"], t_trig)]
    used = set()
    skipped = 0
    for q in flashes:
        k = int(round((q["onset"] - start["onset"]) / (p.interval_s * fps_b)))
        if k < 1 or k in used:
            skipped += 1
            continue
        used.add(k)
        anchors.append(("flash", k, q["onset"], flash_ms(k)))
    missing = sorted(set(range(1, max(used) + 1)) - used) if used else []
    if end is not None:
        anchors.append(("end", -1, end["onset"], float(ts_ms[-1] + (1 - p.lag_frames) * period)))
    clock_err = None
    if mode == "time" and end is not None and len(anchors) > 2:
        fb_ = np.array([a[2] for a in anchors[:-1]])
        fa_ = np.array([a[3] - t_trig for a in anchors[:-1]])
        if stop is not None:
            a_end = stop[0]
        else:
            sl_ = np.polyfit(fb_[-3:], fa_[-3:], 1)[0] if len(fb_) >= 3 else np.polyfit(fb_, fa_, 1)[0]
            a_end = fa_[-1] + (end["onset"] - fb_[-1]) * sl_
        scale = (anchors[-1][3] - t_trig) / a_end
        clock_err = (1 / scale - 1) * 100
        anchors = [anchors[0]] + [(a[0], a[1], a[2], t_trig + (a[3] - t_trig) * scale)
                                  for a in anchors[1:-1]] + [anchors[-1]]
    a_b = np.array([a[2] for a in anchors])
    a_m = np.array([a[3] for a in anchors])
    if len(a_b) >= 2:
        slope, icpt = np.polyfit(a_b, a_m, 1)
    else:
        slope, icpt = 1000.0 / fps_b, a_m[0] - a_b[0] * 1000.0 / fps_b
    resid = a_m - (slope * a_b + icpt)
    bf = np.arange(len(trace), dtype=float)
    bm = np.interp(bf, a_b, a_m)
    left, right = bf < a_b[0], bf > a_b[-1]
    bm[left] = a_m[0] + (bf[left] - a_b[0]) * slope
    bm[right] = a_m[-1] + (bf[right] - a_b[-1]) * slope
    report = {
        "mode": mode, "n_flashes": len(used), "missing_flashes": missing, "skipped": skipped,
        "has_end_mark": end is not None, "behavior_fps_measured": 1000.0 / slope,
        "max_resid_ms": float(np.max(np.abs(resid))) if len(resid) else 0.0,
        "arduino_clock_error_pct": clock_err, "threshold": float(thr),
    }
    return bm, anchors, report


def manual_alignment(n_behavior_frames, fps_b, behavior_frame, miniscope_time_s):
    """Behavior frame `behavior_frame` happened at miniscope time `miniscope_time_s`."""
    bf = np.arange(n_behavior_frames, dtype=float)
    return (miniscope_time_s + (bf - behavior_frame) / fps_b) * 1000.0


# --------------------------------------------------------------- storage
def save_sync(ws: Workspace, behav_ms: np.ndarray, method: str, info: dict):
    np.savez_compressed(ws.path(SYNC_FILE), behav_time_s=behav_ms / 1000.0)
    ws.record_step("sync", {"method": method, **{k: v for k, v in info.items()
                                                  if isinstance(v, (int, float, str, bool, type(None)))}},
                   {"sync": SYNC_FILE})


def load_sync(ws: Workspace) -> np.ndarray:
    return np.load(ws.path(SYNC_FILE))["behav_time_s"]


def miniscope_ts_ms(ws: Workspace):
    """Relative miniscope timestamps (ms, frame 0 = 0) and DAQ frame numbers, if any."""
    from ..io.ucla_v4 import MiniscopeSession
    src = ws.info.get("source_folder")
    s = MiniscopeSession.open(src)
    ts = s.frame_times_s() * 1000.0
    daq = None
    p = Path(src) / "timeStamps.csv"
    if p.exists():
        import csv
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        hdr = [h.strip() for h in rows[0]]
        if "DAQ Frame Number" in hdr:
            i = hdr.index("DAQ Frame Number")
            vals = np.array([float(r[i]) for r in rows[1:] if r])
            if len(vals) == len(ts):
                daq = vals
    return ts, daq


def run_led_sync(session, ws: Workspace, args: dict, progress=None, should_stop=None) -> dict:
    """args: video, fps, roi (x,y,w,h), params (LedParams), arduino_log (optional)."""
    progress = progress or (lambda f, m: None)
    trace = led_trace(args["video"], args["roi"], progress, should_stop)
    ts, daq = miniscope_ts_ms(ws)
    bm, anchors, rep = led_alignment(trace, args["fps"], ts, daq, args["params"], args.get("arduino_log"))
    np.save(ws.path("behavior/led_trace.npy"), trace)
    save_sync(ws, bm, "led", {**asdict(args["params"]), **{k: v for k, v in rep.items()
                                                            if not isinstance(v, list)}})
    progress(1.0, "LED 同步完成")
    return rep


def run_manual_sync(session, ws: Workspace, args: dict, progress=None, should_stop=None) -> dict:
    bm = manual_alignment(args["n_frames"], args["fps"], args["behavior_frame"], args["miniscope_time_s"])
    save_sync(ws, bm, "manual", {"behavior_frame": args["behavior_frame"],
                                 "miniscope_time_s": args["miniscope_time_s"], "fps": args["fps"]})
    return {"mode": "manual"}


# ----------------------------------------------------------- joint analysis
def synced_frame_indices(frame_times, requested):
    """Nearest video frames; leave times outside coverage or within a clock gap unmapped."""
    t = np.asarray(frame_times, dtype=float)
    q = np.asarray(requested, dtype=float)
    if not len(t):
        return np.full(q.shape, -1, dtype=int)
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("同步帧时间无效，请重新对齐行为视频")
    right = np.searchsorted(t, q).clip(0, len(t) - 1)
    left = np.maximum(right - 1, 0)
    idx = np.where(abs(t[right] - q) < abs(t[left] - q), right, left)
    inside = np.isfinite(q) & (q >= t[0]) & (q <= t[-1])
    if len(t) > 1:
        dt = np.median(np.diff(t))
        gap = t[right] - t[left] > 1.6 * dt
        inside &= ~gap | (abs(t[idx] - q) <= .5 * dt + 1e-9)
    return np.where(inside, idx, -1)


def behavior_on_calcium_frames(ws: Workspace, cfg) -> dict:
    """Resample behaviour variables onto the calcium frame times."""
    from .behavior import analyze, load_track
    from .signals import SIGNALS_FILE
    with np.load(ws.path(SIGNALS_FILE)) as sig:
        tc = sig["times"]
    tb = load_sync(ws)
    res = analyze(load_track(ws), cfg)
    n_b = len(res["t"])
    tb = tb[:n_b]
    idx = synced_frame_indices(tb, tc)
    j = np.clip(idx, 0, n_b - 1)
    quality = np.where(idx >= 0, res["tracking_quality"][j], 0).astype(np.uint8)
    valid = quality > 0
    out = {"times": tc, "valid": valid, "behavior_frame": idx, "tracking_quality": quality}
    for key in ("x", "y", "x_u", "y_u", "speed", "motion"):
        if key in res:
            v = res[key][j].astype(float)
            v[~valid] = np.nan
            out[key] = v
    if "freezing" in res:
        out["freezing"] = res["freezing"][j] & valid
    out["zones"] = {k: v[j] & valid for k, v in res["zones"].items()}
    out["unit"] = res["unit"]
    return out


def state_modulation(dff, state, valid, n_shuffle=500, seed=0):
    """Per cell: mean dF/F in state vs outside, and a circular-shift p-value."""
    rng = np.random.default_rng(seed)
    X = dff[:, valid]
    s = state[valid].astype(bool)
    if s.all() or not s.any():
        raise ValueError("所选状态在整个记录中全有或全无，无法比较")
    diff = X[:, s].mean(1) - X[:, ~s].mean(1)
    T = X.shape[1]
    null = np.empty((n_shuffle, X.shape[0]))
    for i in range(n_shuffle):
        sh = np.roll(s, rng.integers(T // 10, T - T // 10))
        null[i] = X[:, sh].mean(1) - X[:, ~sh].mean(1)
    p_hi = (1 + (null >= diff).sum(0)) / (n_shuffle + 1)
    p_lo = (1 + (null <= diff).sum(0)) / (n_shuffle + 1)
    return {"in_state": X[:, s].mean(1), "out_state": X[:, ~s].mean(1), "diff": diff,
            "p_up": p_hi, "p_down": p_lo}


def speed_correlation(dff, speed, valid, n_shuffle=500, seed=0):
    rng = np.random.default_rng(seed)
    ok = valid & np.isfinite(speed)
    if ok.sum() < 2 or np.ptp(speed[ok]) == 0:
        return {k: np.full(dff.shape[0], np.nan) for k in ("r", "p")}
    X = dff[:, ok]
    v = speed[ok]
    vz = (v - v.mean()) / (v.std() or 1)
    Xz = (X - X.mean(1, keepdims=True)) / np.maximum(X.std(1, keepdims=True), 1e-12)
    r = Xz @ vz / len(vz)
    T = len(vz)
    null = np.array([Xz @ np.roll(vz, rng.integers(T // 10, T - T // 10)) / T for _ in range(n_shuffle)])
    p = (1 + (np.abs(null) >= np.abs(r)).sum(0)) / (n_shuffle + 1)
    return {"r": r, "p": p}


def event_onsets(mask, times, min_gap_s=0.0):
    m = np.concatenate([[False], mask.astype(bool)])
    on = np.where(np.diff(m.astype(int)) == 1)[0]
    t = times[on]
    if min_gap_s > 0 and len(t):
        keep = [0]
        for i in range(1, len(t)):
            if t[i] - t[keep[-1]] >= min_gap_s:
                keep.append(i)
        t = t[keep]
    return t


def peri_event(dff, times, events_s, pre_s=5.0, post_s=10.0):
    """Return (time axis, array events x cells x time) of dF/F around events."""
    fps = 1.0 / np.median(np.diff(times))
    n_pre, n_post = int(round(pre_s * fps)), int(round(post_s * fps))
    axis = np.arange(-n_pre, n_post + 1) / fps
    segs = []
    for e in events_s:
        i = int(np.searchsorted(times, e))
        if i - n_pre < 0 or i + n_post >= dff.shape[1]:
            continue
        segs.append(dff[:, i - n_pre:i + n_post + 1])
    if not segs:
        return axis, np.zeros((0, dff.shape[0], len(axis)))
    return axis, np.stack(segs)


def _cell(v):
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{v:.5g}"
    if isinstance(v, (int, np.integer)) and v < 0:
        return ""          # behaviour flag outside the behaviour video
    return v


def export_joint(ws: Workspace, cfg, folder=None) -> Path:
    """Per-frame table with calcium + behaviour, and per-cell modulation stats."""
    import csv
    from .behavior import TRACKING_QUALITY_LABELS
    from .extraction import load_cells
    from .signals import load_signals
    out = Path(folder) if folder else ws.path("export")
    out.mkdir(parents=True, exist_ok=True)
    sig = load_signals(ws)
    acc = load_cells(ws)["accepted"].astype(bool)
    ids = np.where(acc)[0]
    dff = sig["dff"][acc]
    b = behavior_on_calcium_frames(ws, cfg)
    cols = {"time_s": b["times"], "behavior_frame": b["behavior_frame"],
            "tracking_quality": TRACKING_QUALITY_LABELS[b["tracking_quality"]]}
    u = b["unit"]
    cols[f"x_{u}"], cols[f"y_{u}"] = b["x_u"], b["y_u"]
    cols[f"speed_{u}_per_s"] = b["speed"]
    valid = b["valid"]

    def flag(v):   # 0/1 inside the behaviour video, empty outside it
        return np.where(valid, v.astype(int), -1)
    if "freezing" in b:
        cols["freezing"] = flag(b["freezing"])
    for name, v in b["zones"].items():
        cols[f"in_{name}"] = flag(v)
    for i, k in enumerate(ids):
        cols[f"dff_cell_{k}"] = dff[i]
    with open(out / "aligned_frames.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols.keys())
        for row in zip(*cols.values()):
            w.writerow([_cell(v) for v in row])
    rows = {"cell_id": ids}
    if "freezing" in b and b["freezing"][b["valid"]].any() and not b["freezing"][b["valid"]].all():
        m = state_modulation(dff, b["freezing"], b["valid"])
        rows.update(dff_freezing=m["in_state"], dff_moving=m["out_state"],
                    freezing_minus_moving=m["diff"], p_freezing_up=m["p_up"], p_freezing_down=m["p_down"])
    if "speed" in b:
        c = speed_correlation(dff, b["speed"], b["valid"])
        rows.update(speed_r=c["r"], speed_p=c["p"])
    with open(out / "cell_behavior_stats.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(rows.keys())
        for row in zip(*rows.values()):
            w.writerow([f"{v:.5g}" if isinstance(v, (float, np.floating)) else v for v in row])
    return out
