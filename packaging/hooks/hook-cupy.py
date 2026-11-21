"""Collect the CuPy extensions and the CUDA components used by CalciLume."""
from importlib.metadata import distribution
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


def runtime_module(name):
    return not any(part in {"tests", "testing"} for part in name.split("."))


hiddenimports = [
    *collect_submodules("cupy", filter=runtime_module),
    *collect_submodules("cupy_backends", filter=runtime_module),
    *collect_submodules("cuda.pathfinder", filter=runtime_module),
]
datas = collect_data_files("cupy", includes=["_core/include/**/*", ".data/*.json"])
binaries = []
for wheel in ("cupy-cuda12x", "cuda-pathfinder", "nvidia-cuda-runtime-cu12",
              "nvidia-cuda-nvrtc-cu12", "nvidia-cufft-cu12", "nvidia-nvjitlink-cu12"):
    datas.extend(copy_metadata(wheel))
    if not wheel.startswith("nvidia-"):
        continue
    dist = distribution(wheel)
    for entry in dist.files or ():
        path = Path(entry)
        if path.parts[0] != "nvidia":
            continue
        # Keep the wheels' layout. NVIDIA's pathfinder resolves this tree at runtime.
        if path.suffix.lower() == ".dll":
            binaries.append((str(dist.locate_file(entry)), str(path.parent)))
        elif "include" in path.parts and path.suffix in {".h", ".hpp", ".cuh"}:
            datas.append((str(dist.locate_file(entry)), str(path.parent)))
