"""Exercise the installed Apple Silicon bundle on a physical or virtual Mac."""
import ctypes
import json
from pathlib import Path
import sys


def main():
    import mlx.core as mx
    from windows_pipeline_probe import exercise_pipeline

    runtime = Path(sys._MEIPASS).resolve()
    if not Path(mx.__file__).resolve().is_relative_to(runtime):
        raise RuntimeError(f"MLX was loaded outside the app: {mx.__file__}")
    metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    has_device = bool(metal.MTLCreateSystemDefaultDevice())
    result = {"system_metal_device": has_device, "cpu": exercise_pipeline("cpu")}
    if has_device:
        result["metal"] = exercise_pipeline("metal")
    else:
        result["auto"] = exercise_pipeline("auto")
    print(json.dumps(result, ensure_ascii=False, indent=2))
