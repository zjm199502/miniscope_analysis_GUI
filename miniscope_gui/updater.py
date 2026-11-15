"""Check GitHub releases for a newer version and install it.

Release assets (built by .github/workflows/release.yml):
    MiniscopeAnalysis-Windows.zip        the app folder "MiniscopeAnalysis/"
    MiniscopeAnalysis-macOS-arm64.zip    "MiniscopeAnalysis.app" (made with ditto)
    MiniscopeAnalysis-macOS-arm64.dmg    for first-time installs

Installing works only for the packaged app: the new build is downloaded and
unpacked next to the old one, then a small helper script waits for this
process to exit, swaps the folders and starts the new version.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import __version__

REPO = "zjm199502/miniscope_analysis_GUI"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
UA = {"User-Agent": "MiniscopeAnalysis-updater", "Accept": "application/vnd.github+json"}


@dataclass
class Release:
    version: str
    notes: str
    url: str                 # release page
    asset_url: str | None    # update package for this platform (None = not available)
    asset_size: int = 0


def parse_version(v: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", v.split("+")[0])
    return tuple(int(n) for n in nums[:3]) + (0,) * (3 - len(nums[:3]))


def is_newer(remote: str, local: str = __version__) -> bool:
    return parse_version(remote) > parse_version(local)


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def asset_name() -> str | None:
    if sys.platform == "win32":
        return "MiniscopeAnalysis-Windows.zip"
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "MiniscopeAnalysis-macOS-arm64.zip"
    return None


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def install_root() -> Path | None:
    """Folder (Windows) or .app bundle (macOS) of the running packaged app."""
    if not is_frozen():
        return None
    exe = Path(sys.executable).resolve()
    if sys.platform == "darwin":
        for p in exe.parents:
            if p.suffix == ".app":
                return p
        return None
    return exe.parent


def check_latest(timeout: float = 8.0) -> Release | None:
    """Latest release, or None when it is not newer than the running version."""
    req = urllib.request.Request(API_LATEST, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
        data = json.load(r)
    tag = str(data.get("tag_name", "")).lstrip("vV")
    if not tag or not is_newer(tag):
        return None
    want = asset_name()
    url, size = None, 0
    for a in data.get("assets", []):
        if a.get("name") == want:
            url, size = a.get("browser_download_url"), int(a.get("size", 0))
    return Release(tag, data.get("body") or "", data.get("html_url") or RELEASES_PAGE, url, size)


def download(url: str, dest: Path, progress: Callable[[float], None] | None = None,
             should_stop: Callable[[], bool] | None = None) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": UA["User-Agent"]})
    with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            if should_stop and should_stop():
                raise InterruptedError("已取消")
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress and total:
                progress(done / total)
    return dest


def prepare_update(rel: Release, progress=None, should_stop=None) -> Path:
    """Download and unpack the update; returns the folder holding the new app."""
    root = install_root()
    if root is None or rel.asset_url is None:
        raise RuntimeError("当前版本不支持自动更新，请到发布页面下载新版本。")
    if _writable_dir(root.parent) != root.parent:
        raise RuntimeError(f"没有权限写入 {root.parent}，请到发布页面手动下载新版本。")
    work = Path(tempfile.mkdtemp(prefix="miniscope_update_", dir=root.parent))
    zpath = download(rel.asset_url, work / "update.zip",
                     (lambda f: progress(0.9 * f)) if progress else None, should_stop)
    new_dir = work / "new"
    if sys.platform == "darwin":
        subprocess.run(["ditto", "-x", "-k", str(zpath), str(new_dir)], check=True)
    else:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(new_dir)
    zpath.unlink(missing_ok=True)
    if progress:
        progress(1.0)
    return new_dir


def _writable_dir(preferred: Path) -> Path:
    try:
        test = preferred / f".miniscope_write_test_{os.getpid()}"
        test.write_text("x")
        test.unlink()
        return preferred
    except OSError:
        return Path(tempfile.gettempdir())


def launch_installer(new_dir: Path) -> None:
    """Start the helper that swaps in the new version after this process exits.
    The caller must quit the application right after this returns."""
    root = install_root()
    pid = os.getpid()
    if sys.platform == "win32":
        bat = new_dir.parent / "install_update.bat"
        # paths travel as environment variables: the script itself stays ASCII
        bat.write_text(
            "@echo off\r\n"
            ":wait\r\n"
            "tasklist /FI \"PID eq %MS_PID%\" | find \"%MS_PID%\" >nul && (ping -n 2 127.0.0.1 >nul & goto wait)\r\n"
            "robocopy \"%MS_SRC%\" \"%MS_DST%\" /MIR /R:5 /W:1 /NFL /NDL /NJH /NJS >nul\r\n"
            "start \"\" \"%MS_DST%\\MiniscopeAnalysis.exe\"\r\n", encoding="ascii")
        env = dict(os.environ, MS_PID=str(pid), MS_SRC=str(new_dir / "MiniscopeAnalysis"), MS_DST=str(root))
        subprocess.Popen(["cmd", "/c", str(bat)], env=env, close_fds=True,
                         creationflags=0x00000008 | 0x00000200 | 0x08000000)  # detached, new group, no window
    elif sys.platform == "darwin":
        src = new_dir / "MiniscopeAnalysis.app"
        sh = new_dir.parent / "install_update.sh"
        sh.write_text(
            "#!/bin/sh\n"
            f"while kill -0 {pid} 2>/dev/null; do sleep 1; done\n"
            f"rm -rf \"{root}.old\" && mv \"{root}\" \"{root}.old\" && mv \"{src}\" \"{root}\" "
            f"&& rm -rf \"{root}.old\"\n"
            f"xattr -dr com.apple.quarantine \"{root}\" 2>/dev/null\n"
            f"open \"{root}\"\n"
            f"rm -rf \"{new_dir.parent}\"\n")
        sh.chmod(0o755)
        subprocess.Popen(["/bin/sh", str(sh)], start_new_session=True, close_fds=True)
    else:
        raise RuntimeError("不支持的系统")


def cleanup_stale(root: Path | None = None) -> None:
    """Remove leftovers of an interrupted update."""
    root = root or install_root()
    if root is None:
        return
    for p in root.parent.glob("miniscope_update_*"):
        shutil.rmtree(p, ignore_errors=True)
