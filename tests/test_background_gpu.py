"""Pixel equivalence and transactional recovery of accelerated background removal."""

import numpy as np
import pytest

from miniscope_gui.core import background, gpu
from synth import workspace_from_movie


@pytest.mark.parametrize("ksize,radius", [(1, 1), (3, 8), (5, 15)])
def test_cuda_pixels_and_projections(tmp_path, ksize, radius):
    if not gpu.open_backend("gpu").is_gpu:
        pytest.skip("requires working CUDA")
    block = np.random.default_rng(21).uniform(-3, 40, (35, 31, 47)).astype(np.float16)
    ws = workspace_from_movie(tmp_path / "ws", block, fps=10)
    params = background.BackgroundParams(ksize, radius, "preprocess", "cpu")
    background.run_background(None, ws, params)
    names = [background.BG_FILE, "bg_max.npy", "bg_mean.npy"]
    expected = [np.load(ws.path(name)) for name in names]
    params.backend = "gpu"
    out = background.run_background(None, ws, params)
    assert out["compute_backend"] == "cuda" and not out["fallback_reason"]
    for name, ref in zip(names, expected):
        np.testing.assert_array_equal(np.load(ws.path(name)), ref)


def test_gpu_failure_restarts_atomically(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from miniscope_gui.core import background_cuda

    movie = np.random.default_rng(5).uniform(0, 90, (34, 16, 17)).astype(np.float16)
    ws = workspace_from_movie(tmp_path / "ws", movie, fps=10)
    cpu = gpu.open_backend("cpu")
    fake = SimpleNamespace(
        is_gpu=True,
        name="Test GPU",
        reason="",
        activate=nullcontext,
        release=lambda: None,
    )
    monkeypatch.setattr(
        gpu, "open_backend", lambda mode, *a: cpu if mode == "cpu" else fake
    )

    class FailSecondBatch:
        def __init__(self, *args):
            self.n = 0

        def apply(self, frames):
            self.n += 1
            if self.n == 2:
                raise MemoryError("test out of memory")
            return np.full(frames.shape, -90, np.float32)

    monkeypatch.setattr(background_cuda, "BackgroundFilter", FailSecondBatch)
    p = background.BackgroundParams(source="preprocess", backend="gpu")
    result = background.run_background(None, ws, p)
    assert (
        result["compute_backend"] == "cpu"
        and "out of memory" in result["fallback_reason"]
    )
    expected = [background.remove_background(frame, p) for frame in movie]
    np.testing.assert_array_equal(
        np.load(ws.path(background.BG_FILE)), np.array(expected).astype(np.float16)
    )
