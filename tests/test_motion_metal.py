"""Numerical checks for Apple's mixed host/Metal phase-correlation pipeline."""
from contextlib import nullcontext
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from miniscope_gui.core import gpu, motion
from synth import workspace_from_movie


def _cpu_reference_and_metal(tmp_path, piecewise):
    # The weak high-frequency phases of this smooth movie stress precision.
    # Unequal odd dimensions catch inverse FFTs inferred from half-spectrum size.
    rng = np.random.default_rng(51)
    ref = cv2.GaussianBlur(rng.normal(60, 32, (65, 71)).astype(np.float32), (0, 0), 2)
    movie = np.stack([np.roll(ref, (round(3*np.sin(i/8)), round(4*np.cos(i/9))), axis=(0, 1))
                      + (i % 7)*.2 for i in range(32)]).astype(np.float16)
    outputs = []
    for mode in ("cpu", "metal"):
        ws = workspace_from_movie(tmp_path/mode, movie)
        params = motion.MotionParams(backend=mode, template_frames=9, max_shift=7,
                                     piecewise=piecewise, patch_size=29, patch_overlap=9)
        result = motion.run_motion_correction(None, ws, params)
        assert result["compute_backend"] == mode and not result["fallback_reason"], result
        outputs.append((np.load(ws.path(motion.SHIFTS_FILE)), np.load(ws.path(motion.MC_FILE))))
    (cpu_shift, cpu_movie), (metal_shift, metal_movie) = outputs
    np.testing.assert_allclose(cpu_shift, metal_shift, rtol=0, atol=.01)
    assert np.abs(cpu_movie.astype(float) - metal_movie).mean() < .02


@pytest.mark.parametrize("piecewise", [False, True])
def test_metal_batched_inverse_reference(tmp_path, monkeypatch, piecewise):
    calls = []

    def inverse(spectrum, s, axes):
        assert spectrum.dtype == np.complex64
        calls.append(s)
        return np.fft.irfft2(spectrum, s=s, axes=axes).astype(np.float32)

    api = SimpleNamespace(fft=SimpleNamespace(irfft2=inverse), array=np.array,
                          conj=np.conj, abs=np.abs, take=np.take)
    backend = SimpleNamespace(xp=api, is_gpu=True, kind="metal", name="Metal test",
                              reason="", activate=nullcontext, asnumpy=np.asarray,
                              synchronize=lambda: None, release=lambda: None)
    original = gpu.open_backend
    monkeypatch.setattr(gpu, "open_backend", lambda mode, device_id=None:
                        backend if mode == "metal" else original(mode, device_id))
    _cpu_reference_and_metal(tmp_path, piecewise)
    assert (65, 71) in calls
    if piecewise:
        assert (29, 29) in calls


@pytest.mark.parametrize("piecewise", [False, True])
def test_mlx_motion_on_apple_gpu(tmp_path, piecewise):
    backend = gpu.open_backend("metal")
    if backend.kind != "metal":
        pytest.skip(backend.reason)
    _cpu_reference_and_metal(tmp_path, piecewise)
