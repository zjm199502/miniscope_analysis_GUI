"""Chunked CUDA and Metal operations for the existing CNMF-E alternating updates."""

import numpy as np
from scipy.sparse import csr_matrix


# One lane owns one time sample. Its updates to earlier cells are visible to
# later cells, exactly as in the CPU Gauss-Seidel/HALS sweep. Overlap between
# cells is usually sparse; only nonzero coefficients enter each dot product.
_HALS = """
    if (t >= frames) return;
    for (int k = 0; k < cells; ++k)
        out[k * frames + t] = initial[k * frames + t];
    for (int sweep = 0; sweep < 5; ++sweep) {
        for (int k = 0; k < cells; ++k) {
            if (diagonal[k] <= 0) continue;
            float predicted = 0;
            for (int j = offsets[k]; j < offsets[k+1]; ++j)
                predicted += weights[j] * out[columns[j] * frames + t];
            int i = k * frames + t;
            float updated = out[i] + (ya[i] - predicted) / diagonal[k];
            out[i] = updated < 0 ? 0.0f : updated;
        }
    }
"""

# One lane owns one (cell, footprint pixel) entry, including when several
# cells cover the same movie pixel. No scatter or global atomic is needed.
_PRODUCTS = """
    if (i >= entries) return;
    const int p = pixels[i], k = owners[i];
"""

_TARGET_BYTES = 32 * 1024 * 1024


def _upload(backend, array, dtype):
    value = np.ascontiguousarray(array, dtype=dtype)
    return (
        backend.xp.array(value)
        if backend.kind == "metal"
        else backend.xp.asarray(value)
    )


class TemporalUpdate:
    """Upload the spatial Gram matrix once per update and solve (cells, time)."""

    def __init__(self, backend, AtA):
        self.backend = backend
        gram = csr_matrix(AtA, dtype=np.float32)
        self.cells = gram.shape[0]
        if gram.shape != (self.cells, self.cells):
            raise ValueError("AtA must be square")
        self.active = bool(gram.nnz)
        if not self.active:
            return
        self.matrix = [
            _upload(backend, gram.indptr, np.int32),
            _upload(backend, gram.indices, np.int32),
            _upload(backend, gram.data, np.float32),
            _upload(backend, gram.diagonal(), np.float32),
        ]
        if backend.kind == "cuda":
            self.kernel = backend.xp.RawKernel(
                'extern "C" __global__ void hals(const int* offsets, const int* columns, '
                "const float* weights, const float* diagonal, const float* ya, "
                "const float* initial, float* out, int cells, int frames) {"
                "int t = blockIdx.x * blockDim.x + threadIdx.x;" + _HALS + "}",
                "hals",
            )
        elif backend.kind == "metal":
            self.kernel = backend.xp.fast.metal_kernel(
                name="calcilume_hals",
                input_names=[
                    "offsets",
                    "columns",
                    "weights",
                    "diagonal",
                    "ya",
                    "initial",
                ],
                output_names=["out"],
                source="int t = thread_position_in_grid.x; "
                "int cells = ya_shape[0], frames = ya_shape[1];" + _HALS,
            )
        else:
            raise ValueError("HALS accelerator requires CUDA or Metal")

    def solve(self, YA, Cc):
        if Cc.ndim != 2 or Cc.shape != YA.shape or Cc.shape[0] != self.cells:
            raise ValueError("YA and C must have shape (cells, time)")
        if not Cc.size or not self.active:
            return np.array(Cc, dtype=np.float32, copy=True)
        xp = self.backend.xp
        solved = np.empty(Cc.shape, dtype=np.float32)
        # Two inputs plus one output buffer on the device.
        batch = max(1, _TARGET_BYTES // (12 * self.cells))
        for start in range(0, Cc.shape[1], batch):
            part = slice(start, start + batch)
            ya = _upload(self.backend, YA[:, part], np.float32)
            initial = _upload(self.backend, Cc[:, part], np.float32)
            frames = ya.shape[1]
            args = [*self.matrix, ya, initial]
            if self.backend.kind == "cuda":
                out = xp.empty_like(ya)
                self.kernel(
                    ((frames + 31) // 32,),
                    (32,),
                    (*args, out, np.int32(self.cells), np.int32(frames)),
                )
            else:
                out = self.kernel(
                    inputs=args,
                    grid=(frames, 1, 1),
                    threadgroup=(32, 1, 1),
                    output_shapes=[ya.shape],
                    output_dtypes=[xp.float32],
                )[0]
            solved[:, part] = self.backend.asnumpy(out)
        return solved


class SpatialProducts:
    """Compute per-footprint Y.T @ C using the union of requested movie pixels."""

    def __init__(self, backend, indices):
        self.backend = backend
        sizes = [len(ix) for ix in indices]
        self.ends = np.cumsum(sizes, dtype=np.int64)
        self.entries = int(self.ends[-1]) if len(self.ends) else 0
        if not self.entries:
            return
        flattened = np.concatenate(indices)
        if not np.issubdtype(flattened.dtype, np.integer) or flattened.min() < 0:
            raise ValueError("Footprint pixel indices must be nonnegative integers")
        self.unique, inverse = np.unique(flattened, return_inverse=True)
        self.pixels = _upload(backend, inverse, np.int32)
        self.owners = _upload(
            backend, np.repeat(np.arange(len(sizes)), sizes), np.int32
        )
        if backend.kind == "cuda":
            # FP64 accumulation preserves the precision used during seed initialization.
            self.kernel = backend.xp.RawKernel(
                'extern "C" __global__ void products(const int* pixels, const int* owners, '
                "const float* movie, const float* traces, double* out, int entries, int frames, int width) {"
                "int i = blockIdx.x * blockDim.x + threadIdx.x;"
                + _PRODUCTS
                + "double sum = 0; for (int t = 0; t < frames; ++t) "
                "sum += (double)movie[t * width + p] * (double)traces[k * frames + t]; out[i] = sum;}",
                "products",
            )
        elif backend.kind == "metal":
            # Metal has no FP64. Preserve the residual of each product and the
            # running sum and combine both components on the host.
            self.kernel = backend.xp.fast.metal_kernel(
                name="calcilume_spatial_products",
                input_names=["pixels", "owners", "movie", "traces"],
                output_names=["out"],
                source="int i = thread_position_in_grid.x, entries = pixels_shape[0]; "
                "int frames = movie_shape[0], width = movie_shape[1];"
                + _PRODUCTS
                + """
                float sum = 0, residual = 0;
                for (int t = 0; t < frames; ++t) {
                    float x = movie[t * width + p], y = traces[k * frames + t];
                    float product = x * y;
                    float next = sum + product;
                    float z = next - sum;
                    residual += ((sum - (next - z)) + (product - z)) + metal::fma(x, y, -product);
                    sum = next;
                }
                out[i] = sum;
                out[entries + i] = residual;""",
            )
        else:
            raise ValueError("Spatial accelerator requires CUDA or Metal")

    def products(self, blk, Cchunk):
        frames = blk.shape[0]
        if blk.ndim < 2 or Cchunk.shape != (len(self.ends), frames):
            raise ValueError("Movie and traces must have matching time and cell counts")
        values = np.zeros(self.entries, dtype=np.float64)
        if frames and self.entries:
            frame_size = int(np.prod(blk.shape[1:]))
            if self.unique[-1] >= frame_size:
                raise ValueError("Footprint index exceeds frame size")
            packed_width = len(self.unique)
            batch = max(1, _TARGET_BYTES // (4 * (packed_width + len(self.ends))))
            # A sorted, complete union uses the original frame layout directly.
            for start in range(0, frames, batch):
                part = slice(start, start + batch)
                block = blk[part].reshape(-1, frame_size)
                if packed_width != frame_size:
                    block = np.take(block, self.unique, axis=1)
                values += self._product_block(block, Cchunk[:, part])
        return np.split(values, self.ends[:-1]) if len(self.ends) else []

    def _product_block(self, block, trace):
        xp = self.backend.xp
        args = [
            self.pixels,
            self.owners,
            _upload(self.backend, block, np.float32),
            _upload(self.backend, trace, np.float32),
        ]
        if self.backend.kind == "cuda":
            out = xp.empty(self.entries, dtype=xp.float64)
            dims = (
                np.int32(self.entries),
                np.int32(block.shape[0]),
                np.int32(block.shape[1]),
            )
            self.kernel(((self.entries + 127) // 128,), (128,), (*args, out, *dims))
            return self.backend.asnumpy(out)
        out = self.kernel(
            inputs=args,
            grid=(self.entries, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(2, self.entries)],
            output_dtypes=[xp.float32],
        )[0]
        return self.backend.asnumpy(out).sum(axis=0, dtype=np.float64)


# For each undirected edge, compute its product once. The host shifts the same
# accumulated value into the other endpoint's opposite direction.
_NEIGHBOURS = """
    for (int t = 0; t < frames; ++t) {
        int base = t * area + p;
        float v = movie[base];
        float terms[6] = {
            v, v * v,
            x+1 < w ? v * movie[base+1] : 0.0f,
            y+1 < h && x > 0 ? v * movie[base+w-1] : 0.0f,
            y+1 < h ? v * movie[base+w] : 0.0f,
            y+1 < h && x+1 < w ? v * movie[base+w+1] : 0.0f
        };
"""


class SummaryProducts:
    """Accumulate the two moments and eight directed neighbour sums for a time block."""

    def __init__(self, backend):
        self.backend = backend
        if backend.kind == "cuda":
            self.kernel = backend.xp.RawKernel(
                'extern "C" __global__ void summary(const float* movie, double* out, int frames, int h, int w) {'
                "int p = blockIdx.x * blockDim.x + threadIdx.x; int area = h*w; if (p>=area) return; "
                "int x = p%w, y = p/w; double sums[6]; "
                "for (int j=0; j<6; ++j) sums[j] = out[j*area+p];"
                + _NEIGHBOURS
                + "for (int j=0; j<6; ++j) sums[j] += (double)terms[j]; } "
                "for (int j=0; j<6; ++j) out[j*area+p] = sums[j]; }",
                "summary",
            )
        elif backend.kind == "metal":
            self.kernel = backend.xp.fast.metal_kernel(
                name="calcilume_summary_products",
                input_names=["movie", "previous"],
                output_names=["out"],
                source="int p = thread_position_in_grid.x; int frames = movie_shape[0]; "
                "int h = movie_shape[1], w = movie_shape[2], area = h*w; "
                "int x = p%w, y = p/w; float sums[6], low[6]; "
                "for (int j=0; j<6; ++j) { sums[j]=previous[j*area+p]; low[j]=previous[(6+j)*area+p]; } "
                + _NEIGHBOURS
                + """
                for (int j=0; j<6; ++j) {
                    float a = sums[j], b = terms[j], next = a+b;
                    float z = next-a;
                    low[j] += (a-(next-z)) + (b-z);
                    sums[j] = next;
                }}
                for (int j=0; j<6; ++j) {
                    out[j*area+p] = sums[j];
                    out[(6+j)*area+p] = low[j];
                }""",
            )
        else:
            raise ValueError("Summary accelerator requires CUDA or Metal")

    def statistics(self, Yf):
        frames, h, w = Yf.shape
        if not frames or not h or not w:
            return np.zeros((h, w)), np.zeros((h, w)), np.zeros((8, h, w))
        xp = self.backend.xp
        area = h * w
        batch = max(1, _TARGET_BYTES // (4 * area))
        cuda = self.backend.kind == "cuda"
        totals = xp.zeros(
            (6 if cuda else 12, h, w), dtype=xp.float64 if cuda else xp.float32
        )
        for start in range(0, frames, batch):
            block = _upload(self.backend, Yf[start : start + batch], np.float32)
            if cuda:
                dims = (np.int32(block.shape[0]), np.int32(h), np.int32(w))
                self.kernel(((area + 127) // 128,), (128,), (block, totals, *dims))
            else:
                totals = self.kernel(
                    inputs=[block, totals],
                    grid=(area, 1, 1),
                    threadgroup=(128, 1, 1),
                    output_shapes=[totals.shape],
                    output_dtypes=[xp.float32],
                )[0]
                xp.eval(totals)
        sums = self.backend.asnumpy(totals)
        if not cuda:
            sums = np.add(sums[:6], sums[6:], dtype=np.float64)
        cross = np.zeros((8, h, w), dtype=np.float64)
        cross[4:] = sums[2:]  # right, down-left, down, down-right
        cross[3, :, 1:] = cross[4, :, :-1]
        cross[2, 1:, :-1] = cross[5, :-1, 1:]
        cross[1, 1:, :] = cross[6, :-1, :]
        cross[0, 1:, 1:] = cross[7, :-1, :-1]
        return sums[0], sums[1], cross
