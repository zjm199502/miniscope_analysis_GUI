"""Write a small synthetic UCLA V4 folder for tests."""

import json
from pathlib import Path

import cv2
import numpy as np


def make_fake_v4(folder: Path, n_frames=250, frames_per_file=100, size=(64, 80), fps=30,
                 drop_at=None, seed=0):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    h, w = size
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    cells = [(20, 20), (40, 55), (15, 60)]
    frames = []
    for t in range(n_frames):
        img = 60 + 20 * np.exp(-((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * 30 ** 2))
        for k, (cy, cx) in enumerate(cells):
            amp = 80 if (t // 20 + k) % 3 == 0 else 0
            img = img + amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3 ** 2))
        img = img + rng.normal(0, 3, img.shape)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    for i in range(0, n_frames, frames_per_file):
        vw = cv2.VideoWriter(str(folder / f"{i // frames_per_file}.avi"), fourcc, fps, (w, h), True)
        for fr in frames[i:i + frames_per_file]:
            vw.write(cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR))
        vw.release()
    meta = {"deviceName": "My V4 Miniscope", "deviceType": "Miniscope_V4_BNO",
            "frameRate": f"{fps}FPS", "framesPerFile": frames_per_file,
            "gain": "Low", "led0": 10, "ewl": 70,
            "ROI": {"height": h, "width": w, "leftEdge": 0, "topEdge": 0}}
    (folder / "metaData.json").write_text(json.dumps(meta))
    dt = 1000.0 / fps
    ts = np.arange(n_frames) * dt
    if drop_at is not None:
        ts[drop_at:] += 2 * dt  # two missing frames
    with open(folder / "timeStamps.csv", "w") as f:
        f.write("Frame Number,Time Stamp (ms),Buffer Index\n")
        for i, t in enumerate(ts):
            f.write(f"{i},{int(round(t))},0\n")
    return np.stack(frames)
