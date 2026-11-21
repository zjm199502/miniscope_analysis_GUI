"""Frame-independent median and elliptical white top-hat kernels for CuPy.

Each CUDA block covers one tile in one frame. The opening uses the row
widths of OpenCV's actual structuring element, including its edge rounding.
"""

import numpy as np

from .background_kernels import DISK, MEDIAN


_MEDIAN = r"""
extern "C" __global__ void median(const float *src, float *dst, int h, int w) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int base = blockIdx.z * w * h;
""" + MEDIAN + r"""
}
"""

_DISK = r"""
extern "C" __global__ void disk_filter(const float *src, float *dst,
                                      const int *widths, int r, int h, int w) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int base = blockIdx.z*w*h;
""" + DISK + r"""
}
"""


class BackgroundFilter:
    def __init__(self, backend, median_size, kernel):
        self.backend = backend
        cp = backend.xp
        self.widths = cp.asarray((kernel.sum(axis=1, dtype=np.int32) - 1) // 2)
        self.radius = np.int32(kernel.shape[0] // 2)
        self.median = None
        if median_size > 1:
            size = min(5, int(median_size) | 1)
            self.median = cp.RawKernel("#define K %d\n" % size + _MEDIAN, "median")
        self.erode = cp.RawKernel("#define REDUCE fminf\n" + _DISK, "disk_filter")
        self.dilate = cp.RawKernel("#define REDUCE fmaxf\n" + _DISK, "disk_filter")

    def apply(self, frames):
        cp = self.backend.xp
        count, height, width = frames.shape
        grid = ((width + 31) // 32, (height + 7) // 8, count)
        block = (32, 8, 1)
        # Upload FP32 directly to avoid requiring an FP16 CUDA conversion kernel.
        src = cp.asarray(np.ascontiguousarray(frames, dtype=np.float32))
        buf = cp.empty_like(src)
        dims = (np.int32(height), np.int32(width))
        if self.median is not None:
            self.median(grid, block, (src, buf, *dims))
            src, buf = buf, src
        opened = cp.empty_like(src)
        self.erode(grid, block, (src, buf, self.widths, self.radius, *dims))
        self.dilate(grid, block, (buf, opened, self.widths, self.radius, *dims))
        cp.subtract(src, opened, out=buf)
        return self.backend.asnumpy(buf)
