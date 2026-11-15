import cv2
import numpy as np

from miniscope_gui.core.behavior import (BehaviorConfig, Shape, analyze, export_behavior,
                                         import_dlc, import_ethovision, load_track, run_tracking)
from miniscope_gui.core.workspace import Workspace


def make_behavior_video(path, n=300, fps=30, size=(240, 320), seed=0):
    """Dark 'mouse' moving in a bright arena; frozen during frames 100-160."""
    rng = np.random.default_rng(seed)
    h, w = size
    xs, ys = np.zeros(n), np.zeros(n)
    x, y = 100.0, 120.0
    for i in range(n):
        if not (100 <= i < 160):
            x = np.clip(x + 3 * np.cos(i / 15), 50, w - 50)
            y = np.clip(y + 3 * np.sin(i / 20), 50, h - 50)
        xs[i], ys[i] = x, y
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h), True)
    for i in range(n):
        img = np.full((h, w), 200, np.uint8)
        cv2.rectangle(img, (20, 20), (w - 20, h - 20), 170, 3)  # arena wall
        cv2.ellipse(img, (int(xs[i]), int(ys[i])), (14, 9), (i * 2) % 180 if not (100 <= i < 160) else 0,
                    0, 360, 40, -1)
        img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
        vw.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    vw.release()
    return xs, ys


def test_tracking_and_freezing(tmp_path):
    xs, ys = make_behavior_video(tmp_path / "b.avi")
    ws = Workspace.create_or_open(tmp_path / "ws")
    cfg = BehaviorConfig(video=str(tmp_path / "b.avi"), fps=30,
                         arena=Shape("rect", [25, 25, 270, 190]).to_dict(),
                         zones=[Shape("rect", [0, 0, 160, 240], "left").to_dict()],
                         px_per_cm=10.0)
    cfg.params.update(scale=1.0, freeze_threshold=0.05)
    out = run_tracking(None, ws, cfg)
    assert out["detected_fraction"] > 0.95 and out["polarity"] == "dark"
    tr = load_track(ws)
    err = np.hypot(tr["x"] - xs, tr["y"] - ys)
    assert np.nanmedian(err) < 1.5
    res = analyze(tr, cfg)
    fr = res["freezing"]
    assert fr[110:150].mean() > 0.9           # frozen period detected
    assert fr[:90].mean() < 0.1 and fr[170:].mean() < 0.1
    assert res["unit"] == "cm"
    left = res["zones"]["left"]
    assert abs(left.mean() - (xs <= 160).mean()) < 0.05
    exp = export_behavior(ws, cfg)
    assert (exp / "behavior_frames.csv").exists() and (exp / "behavior_summary.csv").exists()


def test_import_ethovision(tmp_path):
    p = tmp_path / "ev.txt"
    lines = ['"Number of header lines:";"5"', '"Experiment";"x"', '"Arena";"1"',
             '"Trial time";"Recording time";"X center";"Y center";"Mobility state(Immobile)"',
             '"s";"s";"cm";"cm";""']
    for i in range(50):
        x = "-" if i == 3 else f"{i * 0.5:.2f}"
        lines.append(f'"{i / 25:.3f}";"{i / 25:.3f}";"{x}";"{i * 0.1:.2f}";"{1 if 20 <= i < 30 else 0}"')
    p.write_text("\n".join(lines), encoding="utf-8")
    tr = import_ethovision(p)
    assert len(tr["t"]) == 50 and np.isnan(tr["x"][3])
    assert tr["freezing"][20:30].all() and not tr["freezing"][:20].any()
    res = analyze(tr, BehaviorConfig(fps=25))
    assert res["unit"] == "cm" and abs(res["summary"]["freezing_percent"] - 20) < 1e-6


def test_import_dlc(tmp_path):
    p = tmp_path / "dlc.csv"
    rows = ["scorer,DLC,DLC,DLC,DLC,DLC,DLC", "bodyparts,nose,nose,nose,tail,tail,tail",
            "coords,x,y,likelihood,x,y,likelihood"]
    for i in range(20):
        rows.append(f"{i},{i},{2 * i},0.9,{i + 10},{2 * i},{0.2 if i == 5 else 0.95}")
    p.write_text("\n".join(rows))
    tr = import_dlc(p, fps=20)
    assert np.allclose(tr["x"][[0, 5]], [5, 5])   # mean of parts; tail ignored when unlikely
    assert list(tr["bodyparts"]) == ["nose", "tail"]
