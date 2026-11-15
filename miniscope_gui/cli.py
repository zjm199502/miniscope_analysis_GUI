"""Command line access, mainly for testing without the GUI.

    python -m miniscope_gui.cli info  <miniscope_folder>
    python -m miniscope_gui.cli preprocess <miniscope_folder> <empty_output_folder> [--bin 2] [--tbin 1]
        [--start 0] [--end 60] [--crop x,y,w,h]
"""

from __future__ import annotations

import argparse
import sys
import time

from .core.preprocess import PreprocessParams, run_preprocess
from .core.workspace import Workspace
from .io.ucla_v4 import MiniscopeSession


def main(argv=None):
    ap = argparse.ArgumentParser(prog="miniscope_gui.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("info")
    a.add_argument("folder")
    b = sub.add_parser("preprocess")
    b.add_argument("folder")
    b.add_argument("out")
    b.add_argument("--bin", type=int, default=2)
    b.add_argument("--tbin", type=int, default=1)
    b.add_argument("--start", type=float, default=0.0)
    b.add_argument("--end", type=float, default=None)
    b.add_argument("--crop", type=str, default=None)
    args = ap.parse_args(argv)

    s = MiniscopeSession.open(args.folder)
    print(s.summary_zh())
    if args.cmd == "info":
        return 0
    crop = tuple(int(v) for v in args.crop.split(",")) if args.crop else None
    p = PreprocessParams(start_s=args.start, end_s=args.end, crop=crop,
                         spatial_bin=args.bin, temporal_bin=args.tbin)
    ws = Workspace.create_or_open(args.out, source_folder=args.folder)
    t0 = time.time()
    last = [0.0]

    def prog(f, msg):
        if f - last[0] >= 0.1 or f >= 1.0:
            last[0] = f
            print(f"  {f * 100:5.1f}%  {msg}", flush=True)

    out = run_preprocess(s, ws, p, prog)
    print(f"完成，用时 {time.time() - t0:.1f} 秒，输出形状 {out['shape']}，保存在 {ws.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
