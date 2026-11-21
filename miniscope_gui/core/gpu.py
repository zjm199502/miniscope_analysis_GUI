"""System adapter inventory and tested optional CUDA and Apple Metal array backends."""

from __future__ import annotations

from contextlib import contextmanager, suppress
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import import_module
import json
import platform
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


def _apple_silicon():
    return sys.platform == "darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }


def _load_mlx():
    return import_module("mlx.core")


def is_gpu_error(exc, backend=None):
    if isinstance(exc, (MemoryError, ImportError)) or type(exc).__module__.startswith(
        ("cupy.", "cupy_backends.", "cupyx.", "cuda.", "mlx.")
    ):
        return True
    if type(exc) is RuntimeError:
        # nanobind often uses built-in RuntimeError for MLX execution failures.
        # The same task also handles project I/O, which must not initiate a retry.
        message = str(exc).lower()
        if getattr(backend, "kind", None) == "metal" and any(
            marker in message
            for marker in (
                "out of memory",
                "command buffer",
                "kernel compilation",
                "failed to compile",
                "device lost",
                "device removed",
                "metallib",
                "::eval_gpu",
            )
        ):
            return True
        return any(
            marker in message
            for marker in (
                "[metal",
                "mtlcommandbuffererrordomain",
                "metal command buffer",
            )
        )
    return False


_metal_lock = Lock()
_metal_users = 0


@contextmanager
def _metal_task(mx):
    # Keep work and synchronization on the worker's GPU stream. MLX's idle
    # buffer cache is process-wide, so trim it only after our last task exits.
    global _metal_users
    with _metal_lock:
        _metal_users += 1
    try:
        with mx.stream(mx.gpu):
            try:
                yield
            except BaseException:
                with suppress(Exception):
                    mx.synchronize(mx.gpu)
                raise
            else:
                mx.synchronize(mx.gpu)
    finally:
        with _metal_lock:
            _metal_users -= 1
            if not _metal_users:
                with suppress(Exception):
                    mx.clear_cache()


@dataclass(frozen=True)
class Backend:
    name: str
    reason: str
    device_id: int | None = None
    xp: ModuleType = field(default=np, repr=False, compare=False)

    kind: str = ""

    def __post_init__(self):
        if not self.kind:
            object.__setattr__(
                self, "kind", "cuda" if self.device_id is not None else "cpu"
            )

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
        if self.kind == "metal":
            try:
                with _metal_task(self.xp):
                    yield
            except BaseException:
                # The traceback can retain MLX arrays until the caller has
                # cleared its frames. release() retries the trim once after that.
                self._state.metal_retry_cleanup = True
                raise
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
        if self.kind == "metal":
            self.xp.eval(array)
            return np.array(array, copy=True)
        with self.xp.cuda.Device(self.device_id):
            return self.xp.asnumpy(array)

    def synchronize(self):
        if self.kind == "metal":
            self.xp.synchronize(self.xp.gpu)
        elif self.is_gpu:
            with self.xp.cuda.Device(self.device_id):
                self.xp.cuda.get_current_stream().synchronize()

    def release(self):
        """Release task-owned CUDA blocks or trim MLX after exception frames were cleared."""
        if self.kind == "metal":
            if getattr(self._state, "metal_retry_cleanup", False):
                self._state.metal_retry_cleanup = False
                with _metal_lock:
                    if not _metal_users:
                        with suppress(Exception):
                            self.xp.clear_cache()
            return
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
        return dict(
            id=index,
            kind="cuda",
            name=name,
            free_bytes=int(free),
            total_bytes=int(total),
        )


def _check_metal():
    mx = _load_mlx()
    if not mx.metal.is_available():
        raise RuntimeError("当前系统未提供 Metal GPU 计算设备")
    get_info = getattr(mx, "device_info", None)
    info = get_info(mx.gpu) if get_info else mx.metal.device_info()
    name = info.get("device_name", "Apple GPU")
    backend = Backend(name, "", 0, mx, "metal")
    with backend.activate():
        reference = np.array([[1, 3, -2, 4], [0, 7, 5, -6]], dtype=np.float32)
        freq = mx.fft.rfft2(mx.array(reference) * 2)
        actual = backend.asnumpy(mx.fft.irfft2(freq, s=reference.shape))
        del freq
        if not np.allclose(actual, reference * 2, rtol=1e-5, atol=1e-5):
            raise RuntimeError("FFT 自检结果不符")
    return dict(
        id=0,
        kind="metal",
        name=name,
        total_bytes=int(info.get("memory_size") or info.get("total_memory") or 0),
        free_bytes=None,
    )


def _inspect():
    errors = []
    try:
        adapters = _system_adapters()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        adapters = []
        errors.append(f"系统显卡信息不可用：{exc}")
    metal_devices = []
    if _apple_silicon():
        try:
            metal_devices.append(_check_metal())
        except ModuleNotFoundError as exc:
            errors.append(
                "当前运行环境未安装 MLX"
                if exc.name in {"mlx", "mlx.core"}
                else f"MLX 依赖加载失败：{exc}"
            )
        except Exception as exc:
            errors.append(f"Metal 自检失败：{exc}")
        cuda_devices = []
    else:
        cuda_devices = _cuda_inventory(errors)
    devices = cuda_devices + metal_devices
    for device in devices:
        if device["name"] not in adapters:
            adapters.append(device["name"])
    if devices:
        label = "Metal" if metal_devices else "CUDA"
        detail = f"{len(devices)} 个 {label} 设备通过运算自检"
    else:
        detail = errors[-1] if errors else "未检测到可用 GPU 设备"
    return dict(
        adapters=adapters,
        devices=devices,
        cuda_devices=cuda_devices,
        metal_devices=metal_devices,
        detail=detail,
        errors=errors,
    )


def _cuda_inventory(errors):
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
    return devices


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
    if mode not in ("auto", "gpu", "metal", "cuda"):
        raise ValueError("计算后端应为 auto、gpu、metal、cuda 或 cpu")
    info = detect_gpus()
    devices = info["devices"]
    if mode in ("cuda", "metal"):
        devices = [d for d in devices if d["kind"] == mode]
        if not devices:
            return Backend("CPU", f"所选 {mode.upper()} 后端不可用。{info['detail']}")
    if device_id is not None:
        devices = [d for d in devices if d["id"] == device_id]
        if not devices:
            return Backend("CPU", f"所选 GPU {device_id} 不可用。{info['detail']}")
    if not devices:
        return Backend("CPU", info["detail"])
    d = max(devices, key=lambda gpu: gpu["free_bytes"] or 0)
    if d["kind"] == "metal":
        try:
            mx = _load_mlx()
            if not mx.metal.is_available():
                raise RuntimeError("Metal GPU 当前不可用")
        except Exception as exc:
            return Backend("CPU", f"工作线程无法打开 Metal：{exc}")
        return Backend(d["name"], "Metal 已通过运算自检", d["id"], mx, "metal")
    try:
        cp = _load_cupy()
        with cp.cuda.Device(d["id"]):
            cp.cuda.runtime.memGetInfo()
    except Exception as exc:
        return Backend("CPU", f"工作线程无法打开 CUDA {d['id']}：{exc}")
    return Backend(d["name"], f"CUDA {d['id']} 已通过运算自检", d["id"], cp)
