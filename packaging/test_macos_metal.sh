#!/usr/bin/env bash
# Run the hardware comparisons locally from an extracted source checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
import platform
import sys

if not (sys.platform == "darwin" and platform.machine() == "arm64"
        and int(platform.mac_ver()[0].split(".")[0]) >= 14 and sys.version_info >= (3, 10)):
    sys.exit("需要 Apple M 系列、macOS 14 或更新版本及原生 arm64 Python 3.10+。")
PY
python3 -m venv .venv
python=.venv/bin/python
"$python" -m pip install -e '.[metal,dev]'
"$python" - <<'PY'
from miniscope_gui.core.gpu import open_backend

device = open_backend()
if device.kind != "metal":
    raise SystemExit(f"Metal 自检失败：{device.reason}")
print(f"通过自检：{device.name} · Metal")
PY
QT_QPA_PLATFORM=offscreen "$python" -m pytest -q \
    tests/test_motion_metal.py::test_mlx_motion_on_apple_gpu \
    tests/test_background_gpu.py::test_gpu_pixels_and_projections \
    tests/test_extraction_ops.py \
    tests/test_extraction_gpu.py::test_extraction_matches_cpu_on_hardware \
    -k 'metal or mlx_motion'
"$python" benchmarks/video_gpu.py --backend metal
