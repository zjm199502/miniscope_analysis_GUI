"""System adapter inventory and a tested, optional CUDA array backend."""

from __future__ import annotations

from contextlib import contextmanager, suppress
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import import_module
import json
import subprocess
import sys
from threading import Lock, local
from types import ModuleType

import numpy as np


def _windows_adapters():
    """Enumerate present display-class PnP instances, including render-only GPUs."""
    import ctypes
    import winreg

    cfg = ctypes.WinDLL("cfgmgr32")
    guid = "{4d36e968-e325-11ce-bfc1-08002be10318}"
    flags = 0x100 | 0x200  # CM_GETIDLIST_FILTER_PRESENT | CM_GETIDLIST_FILTER_CLASS
    size = ctypes.c_ulong()
    result = cfg.CM_Get_Device_ID_List_SizeW(ctypes.byref(size), guid, flags)
    if result:
        raise OSError(f"CM_Get_Device_ID_List_SizeW: {result}")
    buf = ctypes.create_unicode_buffer(size.value)
    result = cfg.CM_Get_Device_ID_ListW(guid, buf, size.value, flags)
    if result:
        raise OSError(f"CM_Get_Device_ID_ListW: {result}")
    names = []
    for instance in filter(None, buf[:].split("\0")):
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                "SYSTEM\\CurrentControlSet\\Enum\\" + instance,
            ) as key:
                # DeviceDesc may be an indirect resource reference followed by a display name.
                desc = winreg.QueryValueEx(key, "DeviceDesc")[0].rsplit(";", 1)[-1]
                with suppress(OSError):
                    desc = winreg.QueryValueEx(key, "FriendlyName")[0]
            names.append(desc)
        except OSError:
            continue
    return names


def _system_adapters():
    if sys.platform == "win32":
        return _windows_adapters()
    if sys.platform == "darwin":
        proc = subprocess.run(
            ["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            check=True,
            timeout=8,
        )
        return [
            gpu.get("sppci_model") or gpu["_name"]
            for gpu in json.loads(proc.stdout)["SPDisplaysDataType"]
        ]
    return []


def _load_cupy():
    return import_module("cupy")


def is_gpu_error(exc):
    return isinstance(exc, (MemoryError, ImportError)) or type(
        exc
    ).__module__.startswith(("cupy.", "cupy_backends.", "cupyx.", "cuda."))


@dataclass(frozen=True)
class Backend:
    name: str
    reason: str
    device_id: int | None = None
    xp: ModuleType = field(default=np, repr=False, compare=False)

    @property
    def is_gpu(self):
        return self.device_id is not None

    _state: local = field(default_factory=local, init=False, repr=False, compare=False)

    @contextmanager
    def activate(self):
        """Scope device allocations and cuFFT plans to the calling task/thread."""
        if not self.is_gpu:
            yield
            return
        with self.xp.cuda.Device(self.device_id):
            if getattr(self._state, "pool", None) is not None:
                yield
                return
            pool = self.xp.cuda.MemoryPool()
            plans = self.xp.fft.config.get_plan_cache()
            previous = {key for key, _ in plans}
            with self.xp.cuda.using_allocator(pool.malloc):
                self._state.pool = pool
                try:
                    yield
                    self.synchronize()
                except BaseException:
                    with suppress(Exception):
                        self.synchronize()
                    raise
                finally:
                    # The plan cache belongs to this thread and device. Preserve
                    # entries already present when the task started.
                    with suppress(Exception):
                        for key in {key for key, _ in plans} - previous:
                            del plans[key]
                    self.release()
                    del self._state.pool

    def asnumpy(self, array):
        if not self.is_gpu:
            return np.asarray(array)
        with self.xp.cuda.Device(self.device_id):
            return self.xp.asnumpy(array)

    def synchronize(self):
        if self.is_gpu:
            with self.xp.cuda.Device(self.device_id):
                self.xp.cuda.get_current_stream().synchronize()

    def release(self):
        """Return idle memory in our pool; global allocators are never drained."""
        pool = getattr(self._state, "pool", None)
        if pool is not None:
            with suppress(Exception):
                pool.free_all_blocks()


def _check_device(cp, index):
    with cp.cuda.Device(index):
        properties = cp.cuda.runtime.getDeviceProperties(index)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf8", errors="replace")
        backend = Backend(name, "", index, cp)
        with backend.activate():
            # Exercise memory transfer, an actual device kernel and the FFT used
            # by motion correction. A working driver alone is insufficient.
            reference = np.array([[1, 3, -2, 4], [0, 7, 5, -6]], dtype=np.float32)
            spectrum = cp.fft.rfft2(cp.asarray(reference) * 2)
            result = backend.asnumpy(cp.fft.irfft2(spectrum, s=reference.shape))
            del spectrum
            if not np.allclose(result, reference * 2, rtol=1e-5, atol=1e-5):
                raise RuntimeError("FFT 自检结果不符")
        free, total = cp.cuda.runtime.memGetInfo()
        return dict(id=index, name=name, free_bytes=int(free), total_bytes=int(total))


def _inspect():
    errors = []
    try:
        adapters = _system_adapters()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        adapters = []
        errors.append(f"系统显卡信息不可用：{exc}")
    devices = []
    try:
        cp = _load_cupy()
        if cp.cuda.runtime.is_hip:
            raise RuntimeError("当前 CuPy 使用 HIP，本版本加速需要 CUDA")
        for i in range(cp.cuda.runtime.getDeviceCount()):
            try:
                devices.append(_check_device(cp, i))
            except Exception as exc:
                errors.append(f"CUDA {i} 自检失败：{exc}")
    except ModuleNotFoundError as exc:
        errors.append(
            "当前运行环境未安装 CuPy"
            if exc.name == "cupy"
            else f"CuPy 依赖加载失败：{exc}"
        )
    except Exception as exc:
        errors.append(f"CUDA 初始化失败：{exc}")
    for device in devices:
        if device["name"] not in adapters:
            adapters.append(device["name"])
    if devices:
        detail = f"{len(devices)} 个 CUDA 设备通过运算自检"
    else:
        detail = errors[-1] if errors else "未检测到可用 CUDA 设备"
    return dict(adapters=adapters, cuda_devices=devices, detail=detail, errors=errors)


_cache = None
_cache_lock = Lock()


def detect_gpus(refresh=False):
    """Snapshot inventory and free memory at probe time, without holding arrays."""
    global _cache
    with _cache_lock:
        if refresh or _cache is None:
            _cache = _inspect()
        return deepcopy(_cache)


def open_backend(mode="auto", device_id=None):
    if mode == "cpu":
        return Backend("CPU", "已选择 CPU")
    if mode not in ("auto", "gpu"):
        raise ValueError("计算后端应为 auto、gpu 或 cpu")
    info = detect_gpus()
    devices = info["cuda_devices"]
    if device_id is not None:
        devices = [d for d in devices if d["id"] == device_id]
        if not devices:
            return Backend("CPU", f"所选 CUDA {device_id} 不可用。{info['detail']}")
    if not devices:
        return Backend("CPU", info["detail"])
    d = max(devices, key=lambda gpu: gpu["free_bytes"])
    try:
        cp = _load_cupy()
        with cp.cuda.Device(d["id"]):
            cp.cuda.runtime.memGetInfo()
    except Exception as exc:
        return Backend("CPU", f"工作线程无法打开 CUDA {d['id']}：{exc}")
    return Backend(d["name"], f"CUDA {d['id']} 已通过运算自检", d["id"], cp)
