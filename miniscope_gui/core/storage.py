"""Streaming writer for large (T, H, W) movies stored as .npy files.

The .npy header is written once and frames are appended with plain file
writes. Unlike a writable memmap this keeps no growing set of dirty pages
attributed to the process (Windows counts those as process memory).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


class MovieWriter:
    def __init__(self, path: str | Path, shape: tuple[int, int, int], dtype=np.uint8):
        self.path = Path(path)
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        mm = open_memmap(self.path, mode="w+", dtype=self.dtype, shape=self.shape)
        self._offset = mm.offset
        del mm
        self._fh = None
        self.written = 0

    def __enter__(self):
        self._fh = open(self.path, "r+b")
        self._fh.seek(self._offset)
        return self

    def write(self, frames: np.ndarray):
        frames = np.ascontiguousarray(frames, dtype=self.dtype)
        if frames.shape[1:] != self.shape[1:]:
            raise ValueError(f"frame shape {frames.shape[1:]} != {self.shape[1:]}")
        if self.written + frames.shape[0] > self.shape[0]:
            raise ValueError("too many frames")
        self._fh.write(frames.tobytes())
        self.written += frames.shape[0]

    def __exit__(self, exc_type, exc, tb):
        self._fh.close()
        self._fh = None
        if exc_type is None and self.written != self.shape[0]:
            raise IOError(f"只写入了 {self.written}/{self.shape[0]} 帧")
        return False
