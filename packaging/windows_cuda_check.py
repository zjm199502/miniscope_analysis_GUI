"""Verify CUDA redistributables and run the application's packaged analysis code."""
import argparse
import json
from pathlib import Path
import sys
import traceback


def check(report, require_gpu):
    import cupy
    from cuda.pathfinder import find_nvidia_header_directory, load_nvidia_dynamic_lib
    from cupy_backends.cuda.libs import nvrtc
    from miniscope_gui.core.gpu import open_backend
    from windows_pipeline_probe import exercise_pipeline

    bundle = Path(sys._MEIPASS).resolve()

    def bundled(path):
        # A system Toolkit or an activated Python environment must not mask omissions.
        return Path(path).resolve(strict=True).relative_to(bundle).as_posix()

    report["cupy"] = cupy.__version__
    report["libraries"] = {
        name: bundled(load_nvidia_dynamic_lib(name).abs_path)
        for name in ("cudart", "nvrtc", "cufft", "nvJitLink")
    }
    include = find_nvidia_header_directory("cudart")
    report["headers"] = bundled(Path(include) / "cuda_runtime.h")
    source = '#include <vector_types.h>\nextern "C" __global__ void probe(float2* v){v[0].x=1;}'
    program = nvrtc.createProgram(source, "bundle_probe.cu", (), ())
    try:
        try:
            nvrtc.compileProgram(program, ("--gpu-architecture=compute_80", "-I" + str(include)))
        except Exception as exc:
            raise RuntimeError(nvrtc.getProgramLog(program)) from exc
        ptx = nvrtc.getPTX(program)
        if b".entry probe" not in ptx:
            raise RuntimeError("NVRTC did not emit the probe kernel")
        report["nvrtc_ptx_bytes"] = len(ptx)
    finally:
        nvrtc.destroyProgram(program)

    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        # cudaErrorInsufficientDriver or cudaErrorNoDevice on an ordinary CPU runner.
        if exc.status not in {35, 100}:
            raise
        count = 0
        report["driver"] = str(exc)
    report["visible_devices"] = count
    backend = open_backend("cuda" if count else "auto")
    report.update(backend=backend.kind, device=backend.name, detail=backend.reason)
    if (count or require_gpu) and backend.kind != "cuda":
        raise RuntimeError("CUDA execution required: " + backend.reason)
    report["pipeline"] = exercise_pipeline("cuda" if count else "auto")


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(arguments)
    result = {}
    try:
        check(result, args.require_gpu)
        result["ok"] = True
    except Exception:
        result.update(ok=False, error=traceback.format_exc())
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if result["ok"] else 1
