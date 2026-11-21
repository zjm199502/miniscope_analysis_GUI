from types import SimpleNamespace

import numpy as np
import pytest

from miniscope_gui.core import extraction_gpu
from miniscope_gui.core.extraction import _hals_C
from miniscope_gui.core.gpu import open_backend


def test_empty_models_do_not_allocate_device_memory():
    empty = np.empty((0, 3), dtype=np.float32)
    backend = SimpleNamespace(kind="cuda")
    solver = extraction_gpu.TemporalUpdate(backend, np.empty((0, 0)))
    assert solver.solve(empty, empty).shape == (0, 3)
    with pytest.raises(ValueError):
        solver.solve(empty, np.empty((1, 3)))
    images = np.ones((3, 2, 3))
    assert extraction_gpu.SpatialProducts(backend, []).products(images, empty) == []
    patches = extraction_gpu.SpatialProducts(backend, [np.array([], dtype=int)])
    assert patches.products(images, np.ones((1, 3)))[0].size == 0


@pytest.fixture(params=["cuda", "metal"])
def backend(request):
    device = open_backend(request.param)
    if device.kind != request.param:
        pytest.skip(device.reason)
    return device


def test_sequential_hals_and_noncontiguous_chunks(backend, monkeypatch):
    rng = np.random.default_rng(85)
    # Overlapping cells produce a Gram matrix where a Jacobi update differs
    # from the reference. Include a zero diagonal and non-contiguous traces.
    spatial = rng.random((19, 7), dtype=np.float32)
    spatial[:, 3] = 0
    gram = spatial.T @ spatial
    initial = rng.normal(size=(7, 32)).astype("f")[:, 1::2]
    ya = (spatial.T @ rng.random((19, 32), dtype=np.float32))[:, 1::2]
    ref = _hals_C(ya, gram, initial.copy())
    monkeypatch.setattr(extraction_gpu, "_TARGET_BYTES", 7 * 12 * 5)
    host_copy = initial.copy()
    with backend.activate():
        solver = extraction_gpu.TemporalUpdate(backend, gram)
        actual = solver.solve(ya, initial)
        del solver
    np.testing.assert_allclose(actual, ref, atol=2e-6, rtol=1e-5)
    np.testing.assert_array_equal(initial, host_copy)
    np.testing.assert_array_equal(actual[3], initial[3])
    assert actual.dtype == np.float32


def test_spatial_products_overlaps_repeated_pixels_and_last_block(backend, monkeypatch):
    rng = np.random.default_rng(222)
    movie = (1000 + rng.normal(size=(17, 3, 5))).astype(np.float32)
    C = (3 + rng.normal(size=(4, 17))).astype(np.float32)
    indices = [
        np.array([1, 4, 6, 4]),
        np.array([6, 8]),
        np.array([], dtype=int),
        np.array([10, 11, 0]),
    ]
    monkeypatch.setattr(extraction_gpu, "_TARGET_BYTES", 4 * (7 + 4) * 5)
    with backend.activate():
        solver = extraction_gpu.SpatialProducts(backend, indices)
        actual = solver.products(movie, C)
        with pytest.raises(ValueError, match="frame size"):
            solver.products(np.ones((17, 1, 2)), C)
        del solver
    assert len(actual) == len(indices)
    for k, ix in enumerate(indices):
        expected = movie.reshape(17, -1)[:, ix].T.astype(np.float64) @ C[k].astype(
            np.float64
        )
        np.testing.assert_allclose(actual[k], expected, atol=1e-5, rtol=3e-7)


@pytest.mark.parametrize("shape", [(17, 11, 19), (5, 1, 3), (4, 3, 1)])
def test_summary_neighbour_order_fp32_products_and_edges(backend, monkeypatch, shape):
    from miniscope_gui.core.extraction import _neighbour_slices, _OFFSETS

    rng = np.random.default_rng(29)
    t, h, w = shape
    y = rng.normal(0, 50, size=shape).astype("f")
    y[abs(y) < 25] = 0
    monkeypatch.setattr(extraction_gpu, "_TARGET_BYTES", 3 * 4 * h * w)
    expected = np.zeros((8, h, w))
    for j, (dy, dx) in enumerate(_OFFSETS):
        (ay, ax), (by, bx) = _neighbour_slices(h, w, dy, dx)
        expected[j, ay, ax] = (y[:, ay, ax] * y[:, by, bx]).sum(
            axis=0, dtype=np.float64
        )
    with backend.activate():
        op = extraction_gpu.SummaryProducts(backend)
        sums, squares, cross = op.statistics(y)
    # Cross products must round in FP32 before accumulation. Multiplying in
    # FP64 (or subtracting after pooling opposite neighbours) fails this test.
    np.testing.assert_allclose(
        sums, y.sum(axis=0, dtype=np.float64), atol=1e-8, rtol=1e-10
    )
    np.testing.assert_allclose(
        squares, (y * y).sum(axis=0, dtype=np.float64), atol=1e-7, rtol=1e-10
    )
    np.testing.assert_allclose(cross, expected, atol=1e-7, rtol=1e-10)
