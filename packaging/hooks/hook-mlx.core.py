"""Keep the split MLX Metal wheel's kernels next to its native library."""
from importlib.metadata import distribution


# Both imports are made by the compiled mlx.core module during initialization.
hiddenimports = ["mlx._reprlib_fix", "mlx.__array_api_info"]

lib = distribution("mlx-metal").locate_file("mlx/lib")
binaries = [(str(path), "mlx/lib") for path in lib.glob("*.dylib")]
datas = [(str(path), "mlx/lib") for path in lib.glob("*.metallib")]
