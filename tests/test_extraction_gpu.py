"""Compare complete CNMF-E runs, including residual and low-confidence passes."""
from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from miniscope_gui.core import extraction, gpu
from miniscope_gui.core.background import BG_FILE, BackgroundParams, run_background
from miniscope_gui.core.workspace import PROJECT_FILE
from synth import make_movie, workspace_from_movie


def recording(root):
    # A close pair shares pixels; the weak candidates in the residual have to
    # go through both temporal fitting and spatial footprint cleaning.
    centres = [[20, 20], [20, 24], [48, 20], [48, 50], [20, 50], [36, 36]]
    movie, _ = make_movie(T=400, H=72, W=72, n_cells=6, seed=12, noise=3,
                          fixed_centres=centres)
    ws = workspace_from_movie(root, movie)
    run_background(None, ws, BackgroundParams(backend="cpu", source="preprocess", cell_diameter=8))
    return ws, np.asarray(centres)


def params(mode):
    # Multiple input chunks with a short final block exercise shape changes.
    return extraction.ExtractionParams(backend=mode, cell_diameter=7, min_corr=.7,
                                       min_pnr=6, chunk_frames=83,
                                       residual_pass=True, extra_candidates=True)


def assert_equivalent(reference, actual):
    for key in ("area", "accepted", "source"):
        np.testing.assert_array_equal(actual[key], reference[key])
    for key in ("centres", "C", "raw", "F_local", "quality", "snr"):
        np.testing.assert_allclose(actual[key], reference[key], rtol=5e-4, atol=2e-3,
                                   err_msg=key)
    np.testing.assert_allclose(actual["A"].toarray(), reference["A"].toarray(),
                               rtol=5e-4, atol=2e-4)
    for a, b in zip(extraction.contours(actual["A"], actual["shape"]),
                    extraction.contours(reference["A"], reference["shape"])):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kind", ["cuda", "metal"])
def test_extraction_matches_cpu_on_hardware(tmp_path, monkeypatch, kind):
    backend = gpu.open_backend(kind)
    if backend.kind != kind:
        pytest.skip(backend.reason)
    ws, truth = recording(tmp_path / "session")
    stages = []
    original = extraction.find_seeds

    def capture(*args, **kwargs):
        seeds = original(*args, **kwargs)
        stages.append(seeds.copy())
        return seeds

    monkeypatch.setattr(extraction, "find_seeds", capture)
    extraction.run_extraction(None, ws, params("cpu"))
    reference = extraction.load_cells(ws)
    cpu_seeds = list(stages)
    # At least the isolated cells are recovered at their known locations.
    distance = np.linalg.norm(truth[:, None] - reference["centres"][None, :], axis=2)
    assert (distance.min(axis=1) < 3).sum() >= 4
    assert len(stages) == 3 and len(stages[1]) > 0 and np.any(reference["source"] == 2)
    stages.clear()
    p = params(kind)
    result = extraction.run_extraction(None, ws, p)
    actual = extraction.load_cells(ws)
    assert result["compute_backend"] == kind and not result["fallback_reason"], result
    assert result["compute_device"] == backend.name
    assert ws.step("extraction")["params"] == asdict(p)
    assert ws.step("extraction")["outputs"]["compute_backend"] == kind
    assert len(stages) == len(cpu_seeds)
    for reference_seeds, actual_seeds in zip(cpu_seeds, stages):
        np.testing.assert_array_equal(actual_seeds, reference_seeds)
    assert_equivalent(reference, actual)


def test_mid_extraction_failure_retries_and_cancel_preserves_results(tmp_path, monkeypatch):
    from miniscope_gui.core import extraction_gpu

    ws, _ = recording(tmp_path / "session")
    extraction.run_extraction(None, ws, params("cpu"))
    reference = extraction.load_cells(ws)
    original_open = gpu.open_backend
    fake = SimpleNamespace(kind="cuda", name="test GPU", device_id=0, reason="", xp=np,
                           activate=nullcontext, is_gpu=True, asnumpy=np.asarray,
                           synchronize=lambda: None, release=lambda: None)
    monkeypatch.setattr(gpu, "open_backend", lambda mode, device_id=None:
                        original_open("cpu") if mode == "cpu" else fake)
    calls = []
    faults = [MemoryError("test GPU allocation"), InterruptedError("已取消")]

    class FailingUpdate:
        def __init__(self, backend, AtA):
            assert backend.is_gpu
            self.AtA = AtA

        def solve(self, YA, Cc):
            calls.append(Cc.shape)
            if len(calls) == 2:
                raise faults.pop(0)
            return extraction._hals_C(YA, self.AtA, Cc)

    class ReferenceProducts:
        def __init__(self, backend, indices):
            assert backend.is_gpu
            self.indices = indices

        def products(self, block, traces):
            flat = block.reshape(len(block), -1)
            return [flat[:, ix].T.astype(np.float64) @ traces[k].astype(np.float64)
                    for k, ix in enumerate(self.indices)]

    # Keep summary_images on its built-in CPU reduction until the injected HALS failure.
    monkeypatch.setattr(extraction_gpu, "SummaryProducts", lambda backend: None)
    monkeypatch.setattr(extraction_gpu, "SpatialProducts", ReferenceProducts)
    monkeypatch.setattr(extraction_gpu, "TemporalUpdate", FailingUpdate)
    result = extraction.run_extraction(None, ws, params("cuda"))
    assert len(calls) == 2
    assert result["compute_backend"] == "cpu" and "test GPU allocation" in result["fallback_reason"]
    assert ws.step("extraction")["outputs"]["fallback_reason"] == result["fallback_reason"]
    assert_equivalent(reference, extraction.load_cells(ws))
    previous = {f: ws.path(f).read_bytes() for f in
                (PROJECT_FILE, extraction.CN_FILE, extraction.PNR_FILE, extraction.CELLS_FILE)}
    # A different input makes an accidentally published new CN/PNR detectable.
    changed = np.load(ws.path(BG_FILE))[:, :, ::-1].copy()
    np.save(ws.path(BG_FILE), changed)
    calls.clear()
    with pytest.raises(InterruptedError):
        extraction.run_extraction(None, ws, params("cuda"))
    assert len(calls) == 2 and not faults
    for f, expected in previous.items():
        assert ws.path(f).read_bytes() == expected, f
    assert not list(ws.root.glob(".stage-*"))
