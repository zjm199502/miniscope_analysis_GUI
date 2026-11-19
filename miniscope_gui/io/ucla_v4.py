"""Reader for UCLA Miniscope V4 recording folders.

A V4 device folder (e.g. ``My_V4_Miniscope``) usually contains::

    0.avi, 1.avi, ...      video chunks (often 1000 frames each, FFV1 codec)
    metaData.json          device settings (ROI, frameRate, gain, led0, ewl ...)
    timeStamps.csv         "Frame Number, Time Stamp (ms), Buffer Index"
    headOrientation.csv    optional BNO055 head orientation

Frames are returned as 2-D ``uint8`` grayscale arrays. The reader never writes
anything into the source folder.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = (".avi", ".mp4", ".mkv")


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.stem)]


def _parse_fps(value) -> float | None:
    """metaData.json stores frameRate as e.g. "30FPS", "30" or 30."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"[\d.]+", str(value))
    return float(m.group()) if m else None


@dataclass
class TimestampReport:
    n_timestamps: int
    duration_s: float
    measured_fps: float
    median_interval_ms: float
    n_dropped_gaps: int  # intervals > 1.5x median, likely dropped frames
    estimated_missing_frames: int

    def summary_zh(self) -> str:
        s = (f"时间戳 {self.n_timestamps} 个，时长 {self.duration_s:.1f} 秒，"
             f"实测帧率 {self.measured_fps:.2f} Hz")
        if self.n_dropped_gaps:
            s += f"；发现 {self.n_dropped_gaps} 处丢帧，约缺 {self.estimated_missing_frames} 帧"
        else:
            s += "；未发现丢帧"
        return s


@dataclass
class MiniscopeSession:
    folder: Path
    video_files: list[Path]
    frames_per_file: list[int]
    height: int
    width: int
    metadata: dict = field(default_factory=dict)
    timestamps_ms: np.ndarray | None = None

    # ------------------------------------------------------------------ open
    @classmethod
    def open(cls, folder: str | Path) -> "MiniscopeSession":
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"文件夹不存在: {folder}")
        videos = sorted(
            (p for p in folder.iterdir()
             if p.suffix.lower() in VIDEO_EXTS and p.stem.isdigit()),
            key=_natural_key,
        )
        if not videos:
            raise FileNotFoundError(
                f"在 {folder} 中没有找到 0.avi、1.avi …… 这样的视频文件。"
                "请选择 My_V4_Miniscope 这一层文件夹。")

        metadata = {}
        meta_path = folder / "metaData.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)

        counts, height, width = [], None, None
        for v in videos:
            cap = cv2.VideoCapture(str(v))
            if not cap.isOpened():
                raise IOError(f"无法打开视频: {v}")
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            if n <= 0:  # some codecs do not report a count: count by grabbing
                n = 0
                while cap.grab():
                    n += 1
            cap.release()
            if height is None:
                height, width = h, w
            counts.append(n)

        ts = cls._read_timestamps(folder / "timeStamps.csv")
        return cls(folder, videos, counts, height, width, metadata, ts)

    @staticmethod
    def _read_timestamps(path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return None
            col = 1
            for i, name in enumerate(header):
                if "time" in name.lower():
                    col = i
                    break
            vals = []
            for row in reader:
                if len(row) > col and row[col].strip():
                    vals.append(float(row[col]))
        return np.asarray(vals, dtype=np.float64)

    # ------------------------------------------------------------ properties
    @property
    def n_frames(self) -> int:
        return int(sum(self.frames_per_file))

    @property
    def nominal_fps(self) -> float | None:
        return _parse_fps(self.metadata.get("frameRate"))

    @property
    def fps(self) -> float:
        """Real average frame rate.

        V4 hardware rarely hits the nominal rate exactly (e.g. "30FPS" is
        really ~29.8 Hz), so timestamps win when they match the frame count.
        """
        rep = self.timestamp_report()
        if rep is not None and rep.measured_fps > 0 and rep.n_timestamps == self.n_frames:
            return rep.measured_fps
        return self.nominal_fps or 30.0

    @property
    def duration_s(self) -> float:
        t = self.frame_times_s()
        return float(t[-1] + 1.0 / self.fps) if len(t) else 0.0

    def frame_times_s(self) -> np.ndarray:
        """Time of each frame in seconds from the first frame.

        Uses timeStamps.csv when it matches the frame count, otherwise the
        nominal frame rate.
        """
        ts = self.timestamps_ms
        if ts is not None and len(ts) == self.n_frames:
            t = (ts - ts[0]) / 1000.0
            # V4 sometimes writes a bogus first stamp; repair if non-monotonic
            if len(t) > 2 and (t[1] - t[0]) > 10 * np.median(np.diff(t[1:])):
                t[0] = t[1] - np.median(np.diff(t[1:]))
                t = t - t[0]
            return t
        return np.arange(self.n_frames) / self.fps

    def timestamp_report(self) -> TimestampReport | None:
        ts = self.timestamps_ms
        if ts is None or len(ts) < 3:
            return None
        d = np.diff(ts[1:])  # skip first stamp, sometimes unreliable
        d = d[d > 0]
        if len(d) == 0:
            return None
        med = float(np.median(d))
        gaps = d > 1.5 * med
        missing = int(np.sum(np.round(d[gaps] / med) - 1)) if gaps.any() else 0
        dur = float(ts[-1] - ts[1]) / 1000.0
        # mean interval (not median): intervals jitter between e.g. 33 and 34 ms
        mean_fps = (len(ts) - 2) / dur if dur > 0 else 1000.0 / med
        return TimestampReport(
            n_timestamps=len(ts), duration_s=dur,
            measured_fps=mean_fps, median_interval_ms=med,
            n_dropped_gaps=int(gaps.sum()), estimated_missing_frames=missing)

    def summary_zh(self) -> str:
        lines = [
            f"文件夹: {self.folder}",
            f"视频文件: {len(self.video_files)} 个，共 {self.n_frames} 帧",
            f"画面大小: {self.width} × {self.height} 像素",
            f"帧率: {self.fps:.2f} Hz（设定值 {self.nominal_fps or '未知'}），"
            f"时长约 {self.duration_s / 60:.1f} 分钟",
        ]
        for key, label in (("gain", "增益"), ("led0", "LED 强度"), ("ewl", "焦距 (EWL)")):
            if key in self.metadata:
                lines.append(f"{label}: {self.metadata[key]}")
        rep = self.timestamp_report()
        if rep is not None:
            lines.append(rep.summary_zh())
            if rep.n_timestamps != self.n_frames:
                lines.append(f"注意: 时间戳数量 ({rep.n_timestamps}) 与帧数 "
                             f"({self.n_frames}) 不一致，将按名义帧率计算时间")
        else:
            lines.append("没有找到 timeStamps.csv，将按名义帧率计算时间")
        return "\n".join(lines)

    # ---------------------------------------------------------------- frames
    def _locate(self, index: int) -> tuple[int, int]:
        if not 0 <= index < self.n_frames:
            raise IndexError(index)
        for fi, n in enumerate(self.frames_per_file):
            if index < n:
                return fi, index
            index -= n
        raise IndexError(index)

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            # V4 writes gray data into 3 identical channels
            return frame[:, :, 0].copy()
        return frame

    def get_frame(self, index: int) -> np.ndarray:
        fi, local = self._locate(index)
        cap = cv2.VideoCapture(str(self.video_files[fi]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, local)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise IOError(f"读取第 {index} 帧失败")
        return self._to_gray(frame)

    def iter_frames(self, start: int = 0, stop: int | None = None, step: int = 1):
        """Yield ``(index, frame)`` sequentially; efficient for long reads."""
        stop = self.n_frames if stop is None else min(stop, self.n_frames)
        if start >= stop:
            return
        fi, local = self._locate(start)
        index = start
        while index < stop and fi < len(self.video_files):
            cap = cv2.VideoCapture(str(self.video_files[fi]))
            try:
                if local:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, local)
                while local < self.frames_per_file[fi] and index < stop:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if (index - start) % step == 0:
                        yield index, self._to_gray(frame)
                    index += 1
                    local += 1
            finally:
                cap.release()
            fi += 1
            local = 0

    def read_block(self, start: int, stop: int) -> np.ndarray:
        stop = min(stop, self.n_frames)
        out = np.empty((stop - start, self.height, self.width), dtype=np.uint8)
        for i, (_, fr) in enumerate(self.iter_frames(start, stop)):
            out[i] = fr
        return out
