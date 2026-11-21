"""Regression checks for batched phase correlation and staged CUDA fallback."""
from contextlib import nullcontext

import numpy as np
import pytest
import cv2

from miniscope_gui.core import motion
from synth import make_movie, workspace_from_movie


class ArrayBackend:
    """Exercise device batching on CI without requiring an NVIDIA driver."""

    xp = np
    is_gpu = True
    kind = "cuda"
    name = "test device"
    reason = ""
    asnumpy = staticmethod(np.asarray)
    activate = staticmethod(nullcontext)

    def release(self):
        pass

    def synchronize(self):
        pass


def test_cuda_batch_matches_individual_peaks_for_odd_frames():
    rng = np.random.default_rng(21)
    ref = rng.normal(size=(51, 65)).astype(np.float32)
    moving = np.stack([np.roll(ref, s, axis=(0, 1)) for s in ((1, 3), (-4, 0), (0, -6))])
    p = motion.MotionParams(max_shift=6)
    cpu = motion._Registrar(ref.shape, p)
    gpu = motion._CudaRegistrar(ref.shape, p, ArrayBackend())
    np.testing.assert_allclose(gpu.shifts(ref, moving), cpu.shifts(ref, moving), atol=1e-5)


def test_cuda_recovers_weak_frequency_phase_in_smooth_movie(tmp_path):
    from miniscope_gui.core.gpu import open_backend

    backend = open_backend("gpu")
    if backend.kind != "cuda":
        pytest.skip(backend.reason)
    base = cv2.GaussianBlur(np.random.default_rng(51).normal(60, 32, (256, 256)).astype(np.float32),
                            (0, 0), 2)
    movie = np.stack([np.roll(base, (round(3*np.sin(i/8)), round(4*np.cos(i/9))), axis=(0, 1))
                      + (i % 7)*.2 for i in range(64)]).astype(np.float16)
    outputs = []
    for mode in ("cpu", "gpu"):
        ws = workspace_from_movie(tmp_path/mode, movie)
        result = motion.run_motion_correction(None, ws, motion.MotionParams(backend=mode, template_frames=30))
        assert result["compute_backend"] == ("cuda" if mode == "gpu" else "cpu")
        outputs.append(np.load(ws.path(motion.SHIFTS_FILE)))
    np.testing.assert_allclose(*outputs, atol=.01, rtol=0)


@pytest.mark.parametrize("piecewise", [False, True])
def test_cuda_motion_outputs_on_hardware(tmp_path, piecewise):
    from miniscope_gui.core.gpu import open_backend

    device = open_backend("gpu")
    if device.kind != "cuda":
        pytest.skip(f"CUDA unavailable: {device.reason}")
    T = 21
    shifts = np.sin(np.arange(T)[:, None] * np.array([[.4, .6]])) * 3
    data, _ = make_movie(T=T, H=65, W=71, n_cells=4, shifts=shifts)
    params = dict(piecewise=piecewise, patch_size=32, patch_overlap=10, template_frames=8,
                  max_shift=7, max_deviation=3)
    workspaces = []
    for mode in ("cpu", "gpu"):
        ws = workspace_from_movie(tmp_path / mode, data)
        result = motion.run_motion_correction(None, ws, motion.MotionParams(backend=mode, **params))
        assert result["compute_backend"] == ("cuda" if mode == "gpu" else "cpu"), result
        workspaces.append(ws)
    def compare(file, atol):
        a, b = [np.load(ws.path(file)) for ws in workspaces]
        np.testing.assert_allclose(a, b, atol=atol, rtol=1e-4)
    compare(motion.SHIFTS_FILE, .025)
    compare(motion.MC_QUALITY_FILE, .002)
    # Fixed-point OpenCV interpolation can round nearly identical shifts into
    # neighbouring 1/32-pixel bins. Check intensity agreement in signal units.
    a, b = [np.load(ws.path(motion.MC_FILE)).astype(float) for ws in workspaces]
    assert np.mean(np.abs(a-b)) < .05 and np.max(np.abs(a-b)) < 3


def test_cuda_restart_after_device_failure_and_cancel(tmp_path, monkeypatch):
    from miniscope_gui.core import gpu

    device = ArrayBackend()
    original_open = gpu.open_backend
    monkeypatch.setattr(gpu, "open_backend", lambda mode, device_id=None:
                        original_open("cpu") if mode == "cpu" else device)
    pending = [MemoryError("test GPU allocation"), InterruptedError("已取消")]
    registered_frames = motion._registered_frames
    def fail_after_first_write(movie, p, reg, template, stop):
        source = registered_frames(movie, p, reg, template, stop)
        yield next(source)
        if isinstance(reg, motion._CudaRegistrar):
            raise pending.pop(0)
        yield from source
    monkeypatch.setattr(motion, "_registered_frames", fail_after_first_write)
    data, _ = make_movie(T=9, H=48, W=51, n_cells=2)
    ws = workspace_from_movie(tmp_path / "recording", data)
    params = motion.MotionParams(backend="gpu", template_frames=3)
    result = motion.run_motion_correction(None, ws, params)
    assert result["compute_backend"] == "cpu" and "test GPU allocation" in result["fallback_reason"]
    assert ws.step("motion")["outputs"]["fallback_reason"] == result["fallback_reason"]
    first = np.load(ws.path(motion.MC_FILE))
    reference = workspace_from_movie(tmp_path / "cpu", data)
    motion.run_motion_correction(None, reference, motion.MotionParams(backend="cpu", template_frames=3))
    np.testing.assert_array_equal(first, np.load(reference.path(motion.MC_FILE)))
    with pytest.raises(InterruptedError):
        motion.run_motion_correction(None, ws, params)
    np.testing.assert_array_equal(np.load(ws.path(motion.MC_FILE)), first)
