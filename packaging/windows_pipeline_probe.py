"""Exercise the three accelerated application steps using only synthetic pixels."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from miniscope_gui.core import background, extraction, motion, preprocess
from miniscope_gui.core.workspace import Workspace


def exercise_pipeline(backend: str) -> dict:
    """Check that each step really used the requested device and recovered the sources."""
    if backend not in ("auto", "cpu", "cuda", "metal"):
        raise ValueError(f"Unexpected compute backend: {backend}")
    expected = "cpu" if backend == "auto" else backend
    rng = np.random.default_rng(1205)
    shape = (121, 51, 55)
    yy, xx = np.indices(shape[1:])
    centres = np.array([[16, 20], [35, 38]])
    footprints = np.array([np.exp(-((yy-y)**2 + (xx-x)**2) / 12.0) for y, x in centres])
    frames = np.arange(shape[0])
    activities = np.exp(-((frames[None, :] + np.array([[5], [18]])) % 33) / 5.0)
    texture = rng.normal(70, 6, shape[1:])
    movie = np.empty(shape, np.float16)
    for t in frames:
        field = texture + 70 * np.tensordot(activities[:, t], footprints, axes=1)
        movie[t] = np.roll(field, (int(t // 12 % 3) - 1, 0), axis=(0, 1)) + rng.normal(0, 1, shape[1:])

    with TemporaryDirectory(prefix="calcilume-pipeline-") as tmp:
        ws = Workspace.create_or_open(Path(tmp))
        np.save(ws.path(preprocess.MOVIE_FILE), movie)
        np.save(ws.path(preprocess.TIMES_FILE), frames / 10.0)
        ws.record_step("preprocess", {"fps_out": 10.0}, {"shape": list(shape)})
        stages = {
            "motion": motion.run_motion_correction(None, ws, motion.MotionParams(
                backend=backend, template_frames=12, max_shift=4)),
            "background": background.run_background(None, ws, background.BackgroundParams(
                backend=backend, source="motion", cell_diameter=7)),
            "extraction": extraction.run_extraction(None, ws, extraction.ExtractionParams(
                backend=backend, cell_diameter=7, min_corr=.65, min_pnr=5, chunk_frames=50)),
        }
        for step, values in stages.items():
            if values["compute_backend"] != expected or bool(values["fallback_reason"]) != (backend == "auto"):
                raise RuntimeError(f"{step} expected {expected}: {values}")
        cells = extraction.load_cells(ws)
        nearest = np.linalg.norm(centres[:, None, :] - cells["centres"], axis=-1).min(axis=1)
        if nearest.max() > 3 or not np.isfinite(cells["C"]).all() or not cells["A"].nnz:
            raise RuntimeError(f"Synthetic sources were not recovered at known coordinates: {nearest}")
        return {
            "movie_shape": list(shape), "cells": len(cells["C"]),
            "known_source_errors_px": nearest.tolist(),
            "stages": {step: {k: values[k] for k in ("compute_backend", "compute_device", "fallback_reason")}
                       for step, values in stages.items()},
        }
