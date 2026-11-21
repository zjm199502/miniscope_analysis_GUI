#!/bin/bash
# Resolve the application and its freezer against the oldest supported arm64 OS.
set -euo pipefail
test "$(uname -m)" = arm64
stage=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/calcilume-wheels.XXXXXX")
trap 'rm -rf "$stage"' EXIT
python -m pip wheel --no-deps --wheel-dir "$stage/project" .
app=("$stage"/project/miniscope_gui-*.whl)
python -m pip download --only-binary=:all: --platform macosx_14_0_arm64 \
  --dest "$stage/runtime" "${app[0]}[metal]" pyinstaller
# Only the previously resolved wheels can enter the frozen runtime.
python -m pip install --no-index --find-links "$stage/runtime" --force-reinstall \
  "${app[0]}[metal]" pyinstaller
python -m pip check
python -c 'import cv2, mlx.core, PySide6; from importlib.metadata import version; print({p: version(p) for p in ("miniscope-gui", "opencv-python", "PySide6", "mlx", "mlx-metal")})'
