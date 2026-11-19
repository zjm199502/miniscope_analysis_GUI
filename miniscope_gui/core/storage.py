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


class MovieReader:
    """Read frames of a (T, H, W) .npy movie with plain file reads.

    Behaves like a read-only array for ``r[i]``, ``r[a:b]`` and ``r.shape``.
    Unlike np.load(mmap_mode="r") no file pages stay mapped into the process,
    so the reported memory use of long runs stays small (Windows counts
    mapped pages in the working set).
    """

    def __init__(self, path: str | Path):
        from numpy.lib import format as fmt
        self.path = Path(path)
        with open(self.path, "rb") as f:
            version = fmt.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = fmt.read_array_header_1_0(f)
            else:
                shape, fortran, dtype = fmt.read_array_header_2_0(f)
            self._offset = f.tell()
        if fortran or len(shape) != 3:
            raise ValueError(f"不支持的视频文件格式: {self.path}")
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self._frame_bytes = int(np.prod(self.shape[1:])) * self.dtype.itemsize
        expected = self._offset + self.shape[0] * self._frame_bytes
        missing = expected - self.path.stat().st_size
        if missing > 0:
            raise IOError(f"{self.path} 缺少 {missing} 字节")
        self.ndim = 3

    def __len__(self):
        return self.shape[0]

    def _read(self, a: int, b: int) -> np.ndarray:
        n = max(0, b - a)
        out = np.empty((n,) + self.shape[1:], self.dtype)
        if n:
            with open(self.path, "rb") as f:
                f.seek(self._offset + a * self._frame_bytes)
                received = f.readinto(memoryview(out).cast("B"))
                if received != out.nbytes:
                    raise IOError(f"{self.path} 缺少 {out.nbytes - received} 字节")
        return out

    def __getitem__(self, key):
        T = self.shape[0]
        if isinstance(key, (int, np.integer)):
            i = int(key) + (T if key < 0 else 0)
            if not 0 <= i < T:
                raise IndexError(key)
            return self._read(i, i + 1)[0]
        if isinstance(key, slice):
            a, b, step = key.indices(T)
            block = self._read(a, b) if b > a else self._read(0, 0)
            return block[::step] if step != 1 else block
        if isinstance(key, tuple):
            first = self[key[0]]
            return first[(slice(None),) + key[1:]] if not isinstance(key[0], (int, np.integer)) \
                else first[key[1:]]
        idx = np.asarray(key)
        return np.stack([self[int(i)] for i in idx]) if idx.size else self._read(0, 0)

    def __array__(self, dtype=None, copy=None):
        arr = self._read(0, self.shape[0])
        return arr.astype(dtype) if dtype is not None else arr
