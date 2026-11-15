"""Delete large intermediate movies once cells have been extracted.

Kept: the motion-corrected movie (needed for dF/F and for viewing), all
summary images, cells and signals. Removed: the preprocessed movie and the
background-removed movie. Re-running step 3 or 5 later needs step 2 or 4 to
be run again first.
"""

from __future__ import annotations

from .workspace import Workspace

INTERMEDIATE = {"movie_pre.npy": "预处理后的视频", "movie_bg.npy": "去背景后的视频"}


def intermediate_size(ws: Workspace) -> int:
    return sum(ws.path(f).stat().st_size for f in INTERMEDIATE if ws.path(f).exists())


def cleanup_intermediate(ws: Workspace) -> int:
    """Delete the intermediate movies; returns bytes freed."""
    freed = 0
    for f in INTERMEDIATE:
        path = ws.path(f)
        if path.exists():
            freed += path.stat().st_size
            path.unlink()
    return freed


def require_file(ws: Workspace, fname: str, step_name: str):
    if not ws.path(fname).exists():
        raise FileNotFoundError(f"{INTERMEDIATE.get(fname, fname)}已被清理。请先重新运行「{step_name}」这一步。")
