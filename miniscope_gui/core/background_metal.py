"""Batched median and white top-hat filtering on Apple GPUs using MLX."""

import numpy as np

from .background_kernels import DISK, MEDIAN


_POSITION = """
    using metal::min;
    using metal::max;
    const int w = src_shape[2], h = src_shape[1];
    const int x = thread_position_in_grid.x, y = thread_position_in_grid.y;
    const int base = thread_position_in_grid.z * w * h;
"""


class BackgroundFilter:
    def __init__(self, backend, median_size, kernel):
        self.backend = backend
        mx = backend.xp
        self.size = min(5, int(median_size) | 1)
        self.widths = mx.array((kernel.sum(axis=1, dtype=np.int32) - 1) // 2)
        self.median = mx.fast.metal_kernel(
            name="calcilume_median", input_names=["src"], output_names=["dst"],
            source=_POSITION + MEDIAN,
        )
        self.disk = mx.fast.metal_kernel(
            name="calcilume_disk", input_names=["src", "widths"], output_names=["dst"],
            source=_POSITION + "const int r = widths_shape[0] / 2;\n"
            + "#define REDUCE(a,b) (Erode ? metal::min(a,b) : metal::max(a,b))\n"
            + DISK + "\n#undef REDUCE\n",
        )

    def apply(self, frames):
        mx = self.backend.xp
        src = mx.array(np.ascontiguousarray(frames, dtype=np.float32))
        count, height, width = src.shape
        dispatch = dict(
            grid=(width, height, count), threadgroup=(min(32, width), min(8, height), 1),
            output_shapes=[src.shape], output_dtypes=[mx.float32],
        )
        if self.size > 1:
            src, = self.median(inputs=[src], template=[("K", self.size)], **dispatch)
        eroded, = self.disk(inputs=[src, self.widths], template=[("Erode", True)], **dispatch)
        opened, = self.disk(inputs=[eroded, self.widths], template=[("Erode", False)], **dispatch)
        return self.backend.asnumpy(src - opened)
