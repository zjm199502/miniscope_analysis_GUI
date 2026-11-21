"""Time full CPU and GPU video steps on exactly the same synthetic recording."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miniscope_gui.core.background import BG_FILE, BackgroundParams, run_background
from miniscope_gui.core.gpu import detect_gpus, open_backend
from miniscope_gui.core.motion import (
    MC_FILE,
    SHIFTS_FILE,
    MotionParams,
    run_motion_correction,
)
from miniscope_gui.core.preprocess import MOVIE_FILE
from miniscope_gui.core.workspace import Workspace


def run(frames, size, repeats, backend="gpu"):
    inventory = detect_gpus()
    device = open_backend(backend)
    if not device.is_gpu:
        raise RuntimeError(device.reason)
    rng = np.random.default_rng(51)
    base = cv2.GaussianBlur(
        rng.normal(60, 32, (size, size)).astype("float32"), (0, 0), 2
    )
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        workspaces = {
            mode: Workspace.create_or_open(Path(tmp) / mode) for mode in ("cpu", "gpu")
        }
        for ws in workspaces.values():
            movie = np.lib.format.open_memmap(
                ws.path(MOVIE_FILE),
                mode="w+",
                dtype=np.float16,
                shape=(frames, size, size),
            )
            for i in range(frames):
                shift = (round(3 * np.sin(i / 8)), round(4 * np.cos(i / 9)))
                movie[i] = np.roll(base, shift, axis=(0, 1)) + (i % 7) * 0.2
            del movie
            ws.record_step(
                "preprocess", {"fps_out": 20}, {"shape": [frames, size, size]}
            )
        for step, runner in (
            ("rigid", run_motion_correction),
            ("background", run_background),
        ):
            samples = {"cpu": [], "gpu": []}
            for attempt in range(repeats + 1):
                order = ("cpu", "gpu") if attempt % 2 == 0 else ("gpu", "cpu")
                for mode in order:
                    selected = device.kind if mode == "gpu" else "cpu"
                    params = (
                        MotionParams(backend=selected, template_frames=min(30, frames))
                        if step == "rigid"
                        else BackgroundParams(backend=selected, source="preprocess")
                    )
                    start = perf_counter()
                    out = runner(None, workspaces[mode], params)
                    elapsed = perf_counter() - start
                    if mode == "gpu" and out["compute_backend"] != device.kind:
                        raise RuntimeError(out["fallback_reason"])
                    samples[mode].append(elapsed)
            name = MC_FILE if step == "rigid" else BG_FILE
            cpu, accelerated = [
                np.load(ws.path(name), mmap_mode="r") for ws in workspaces.values()
            ]
            max_diff = 0.0
            sum_diff = 0.0
            for a, b in zip(cpu, accelerated):
                diff = abs(a.astype(np.float32) - b.astype(np.float32))
                max_diff = max(max_diff, float(diff.max()))
                sum_diff += diff.sum(dtype=np.float64)
            del cpu, accelerated, a, b
            result = {
                "backend": device.kind,
                "device": device.name,
                "seconds": samples,
                "speedup": float(
                    np.median(samples["cpu"][1:]) / np.median(samples["gpu"][1:])
                ),
                "max_abs_pixel_difference": max_diff,
                "mean_abs_pixel_difference": sum_diff / (frames * size * size),
            }
            if step == "rigid":
                a, b = [np.load(ws.path(SHIFTS_FILE)) for ws in workspaces.values()]
                result["max_abs_shift_difference_px"] = float(abs(a - b).max())
            results[step] = result
    return dict(devices=inventory, frames=frames, shape=[size, size], results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gpu", "metal", "cuda"), default="gpu")
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Timed repetitions after one warm-up per backend",
    )
    args = parser.parse_args()
    if args.frames < 2 or args.size < 32 or args.repeats < 1:
        parser.error(
            "requires at least 2 frames, 32 pixels on each side and 1 timed repetition"
        )
    print(
        json.dumps(
            run(args.frames, args.size, args.repeats, args.backend),
            indent=2,
            ensure_ascii=False,
        )
    )
