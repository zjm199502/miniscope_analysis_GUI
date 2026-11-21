from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from miniscope_gui.core import gpu


@pytest.fixture(autouse=True)
def reset_cache(monkeypatch):
    monkeypatch.setattr(gpu, "_cache", None)


def test_cpu_does_not_probe_and_hybrid_inventory_survives_missing_cupy(monkeypatch):
    def missing():
        raise ModuleNotFoundError("no cupy", name="cupy")

    monkeypatch.setattr(gpu, "_load_cupy", missing)
    monkeypatch.setattr(
        gpu, "_system_adapters", lambda: ["Intel Graphics", "NVIDIA RTX"]
    )
    cpu = gpu.open_backend("cpu")
    with cpu.activate():
        np.testing.assert_equal(cpu.asnumpy([1]), [1])
        cpu.synchronize()
        cpu.release()
    assert gpu._cache is None
    info = gpu.detect_gpus()
    assert info["adapters"] == ["Intel Graphics", "NVIDIA RTX"]
    assert info["cuda_devices"] == [] and "CuPy" in info["detail"]
    info["adapters"].clear()
    assert len(gpu.detect_gpus()["adapters"]) == 2
    assert not gpu.open_backend("auto").is_gpu


class Plans(dict):
    """CuPy plan cache iteration yields (key, cached node) pairs."""

    def __iter__(self):
        return iter(self.items())


class Runtime:
    is_hip = False

    def __init__(self):
        self.current = None
        self.plans = [Plans(old=1) for _ in range(3)]
        self.freed = []
        self.calls = []

    @contextmanager
    def device(self, i):
        previous, self.current = self.current, i
        try:
            yield
        finally:
            self.current = previous

    def getDeviceCount(self):
        return 3

    def getDeviceProperties(self, i):
        return {"name": f"GPU-{i}".encode()}

    def memGetInfo(self):
        return (100, 300, 200)[self.current], 400

    def fft(self, array):
        self.calls.append(self.current)
        self.plans[self.current]["new"] = 1
        return np.fft.rfft2(array) * (0 if self.current == 2 else 1)

    def module(self):
        def pool():
            return SimpleNamespace(
                malloc=None, free_all_blocks=lambda: self.freed.append(self.current)
            )

        fft = SimpleNamespace(
            rfft2=self.fft,
            irfft2=np.fft.irfft2,
            config=SimpleNamespace(get_plan_cache=lambda: self.plans[self.current]),
        )
        cuda = SimpleNamespace(
            runtime=self,
            Device=self.device,
            MemoryPool=pool,
            using_allocator=lambda _: nullcontext(),
            get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None),
        )
        return SimpleNamespace(
            fft=fft, cuda=cuda, asarray=np.asarray, asnumpy=np.asarray
        )


def test_selftest_filters_bad_device_and_restores_scope(monkeypatch):
    runtime = Runtime()
    cp = runtime.module()
    monkeypatch.setattr(gpu, "_load_cupy", lambda: cp)
    monkeypatch.setattr(gpu, "_system_adapters", lambda: ["Intel Graphics"])
    info = gpu.detect_gpus()
    assert [d["id"] for d in info["cuda_devices"]] == [0, 1]
    assert "CUDA 2 自检失败" in info["errors"][0]
    assert runtime.calls == [0, 1, 2] and runtime.current is None
    assert runtime.freed == [0, 1, 2] and all(
        "old" in p and "new" not in p for p in runtime.plans
    )
    assert gpu.open_backend().device_id == 1
    selected = gpu.open_backend("gpu", 0)
    with pytest.raises(MemoryError):
        with selected.activate(), selected.activate():
            runtime.plans[0]["new"] = 1
            assert runtime.current == 0
            raise MemoryError
    assert runtime.current is None and runtime.freed == [0, 1, 2, 0]
    assert runtime.plans[0] == {"old": 1}
    assert not gpu.open_backend("gpu", 2).is_gpu
    assert runtime.calls == [0, 1, 2]
    gpu.detect_gpus(refresh=True)
    assert runtime.calls == [0, 1, 2] * 2


def test_gpu_failure_type_distinguishes_data_errors():
    device_error = type(
        "CUDARuntimeError",
        (RuntimeError,),
        {"__module__": "cupy_backends.cuda.api.runtime"},
    )
    assert gpu.is_gpu_error(device_error("driver lost")) and gpu.is_gpu_error(
        MemoryError()
    )
    assert not gpu.is_gpu_error(ValueError("wrong shape"))
