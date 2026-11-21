from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from miniscope_gui.core import gpu


@pytest.fixture(autouse=True)
def reset_cache(monkeypatch):
    monkeypatch.setattr(gpu, "_cache", None)
    monkeypatch.setattr(gpu, "_apple_silicon", lambda: False)


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


def test_metal_detection_fft_selection_and_task_lifetime(monkeypatch):
    events = []

    @contextmanager
    def stream(device):
        assert device == "gpu"
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def rfft2(x):
        assert events[-1] == "enter"
        events.append("fft")
        return np.fft.rfft2(x)

    mx = SimpleNamespace(
        gpu="gpu",
        stream=stream,
        array=np.asarray,
        metal=SimpleNamespace(is_available=lambda: True),
        device_info=lambda d: {"device_name": "Apple M4", "memory_size": 16000},
        synchronize=lambda d: events.append("sync"),
        eval=lambda x: events.append("eval"),
        clear_cache=lambda: events.append("clear"),
        fft=SimpleNamespace(rfft2=rfft2, irfft2=np.fft.irfft2),
    )
    monkeypatch.setattr(gpu, "_apple_silicon", lambda: True)
    monkeypatch.setattr(gpu, "_system_adapters", lambda: ["Apple M4"])
    monkeypatch.setattr(
        gpu, "_load_cupy", lambda: pytest.fail("CUDA import on Apple Silicon")
    )
    monkeypatch.setattr(gpu, "_load_mlx", lambda: mx)
    inventory = gpu.detect_gpus()
    assert events == ["enter", "fft", "eval", "sync", "exit", "clear"]
    assert inventory["cuda_devices"] == []
    assert inventory["devices"] == inventory["metal_devices"]
    assert inventory["adapters"] == ["Apple M4"]
    assert inventory["devices"][0]["free_bytes"] is None
    backend = gpu.open_backend("metal", 0)
    assert backend.kind == "metal" and backend.xp is mx
    assert (
        gpu.open_backend().kind == "metal" and gpu.open_backend("gpu").kind == "metal"
    )
    assert gpu.open_backend("cuda").kind == gpu.open_backend("metal", 1).kind == "cpu"
    events.clear()
    with pytest.raises(RuntimeError, match="command buffer"):
        with backend.activate():
            with backend.activate():
                pass
            assert "clear" not in events
            raise RuntimeError("[METAL] command buffer failed")
    assert events.count("clear") == 1
    backend.release()  # after the exception traceback is no longer holding arrays
    backend.release()
    assert events.count("clear") == 2
    with backend.activate():
        pass
    assert events.count("clear") == 3
    backend.release()
    assert events.count("clear") == 3
    assert gpu.is_gpu_error(RuntimeError("[METAL] command buffer failed"))
    assert gpu.is_gpu_error(RuntimeError("kernel compilation failed"), backend)
    assert not gpu.is_gpu_error(RuntimeError("project save failed"), backend)
    from miniscope_gui.core.workspace import WorkspaceError

    assert not gpu.is_gpu_error(WorkspaceError("lock timeout"), backend)
    assert not gpu.is_gpu_error(InterruptedError("cancelled"), backend)
    assert not gpu.is_gpu_error(ValueError("bad shape"), backend)
    mx.fft.irfft2 = lambda freq, s: np.zeros(s)
    assert "FFT 自检结果不符" in gpu.detect_gpus(refresh=True)["detail"]
    assert gpu.open_backend().kind == "cpu"


def test_metal_missing_and_no_device(monkeypatch):
    monkeypatch.setattr(gpu, "_apple_silicon", lambda: True)
    monkeypatch.setattr(gpu, "_system_adapters", lambda: ["Apple M2"])

    def missing():
        raise ModuleNotFoundError("mlx.core", name="mlx")

    monkeypatch.setattr(gpu, "_load_mlx", missing)
    assert gpu.detect_gpus()["adapters"] == ["Apple M2"]
    assert "未安装 MLX" in gpu.open_backend().reason
    monkeypatch.setattr(
        gpu,
        "_load_mlx",
        lambda: SimpleNamespace(metal=SimpleNamespace(is_available=lambda: False)),
    )
    assert "未提供 Metal" in gpu.detect_gpus(refresh=True)["detail"]
