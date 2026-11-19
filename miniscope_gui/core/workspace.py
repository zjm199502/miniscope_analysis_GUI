"""Analysis workspace: one output folder per recording.

All results are written to a folder the user chose. The folder must be empty
when a new workspace is created (or already be a workspace), so raw data and
unrelated files are never touched.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from tempfile import TemporaryDirectory
import time
from pathlib import Path

PROJECT_FILE = "project.json"
PUBLISH_DIR = ".miniscope-publishing"


class WorkspaceError(Exception):
    pass


@contextmanager
def _publish_lock(root):
    from PySide6.QtCore import QLockFile
    lock = QLockFile(str(root / ".miniscope-publish.lock"))
    lock.setStaleLockTime(0)
    if not lock.tryLock(5000):
        raise WorkspaceError("另一个进程正在保存此项目，请稍后重试。")
    try:
        yield
    finally:
        lock.unlock()


def _recover_outputs(root):
    """Undo an interrupted publication. Repeating recovery is safe."""
    pending = root / PUBLISH_DIR
    if not pending.exists():
        return
    previous, created = pending / "previous", pending / "created"
    if not previous.is_dir() or not created.is_dir():
        raise WorkspaceError(f"无法识别项目恢复目录：{pending}")
    for marker in created.iterdir():
        (root / marker.name).unlink(missing_ok=True)
    for original in previous.iterdir():
        original.replace(root / original.name)
    # Move the journal aside before cleanup so interruption during cleanup
    # cannot leave a partially removed journal visible on the next open.
    with TemporaryDirectory(dir=root) as directory:
        pending.rename(Path(directory) / "recovered")


class Workspace:
    def __init__(self, root: Path, info: dict):
        self.root = Path(root)
        self.info = info

    @classmethod
    def create_or_open(cls, root: str | Path, source_folder: str | Path | None = None):
        root = Path(root)
        if (root / PUBLISH_DIR).exists():
            with _publish_lock(root):
                _recover_outputs(root)
        pj = root / PROJECT_FILE
        if pj.exists():
            with open(pj, encoding="utf-8") as f:
                ws = cls(root, json.load(f))
            if source_folder is not None:
                old = ws.info.get("source_folder")
                if old and Path(old) != Path(source_folder):
                    raise WorkspaceError(
                        f"这个输出文件夹属于另一段录制:\n{old}\n请换一个空文件夹。")
            return ws
        if root.exists() and any(root.iterdir()):
            raise WorkspaceError(
                "输出文件夹不是空的。为了不覆盖已有文件，请新建或选择一个空文件夹。")
        if source_folder is not None:
            src = Path(source_folder).resolve()
            dst = root.resolve()
            if dst == src or src in dst.parents:
                raise WorkspaceError("输出文件夹不能放在原始数据文件夹里面，请另选位置。")
        root.mkdir(parents=True, exist_ok=True)
        ws = cls(root, {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_folder": str(source_folder) if source_folder else None,
            "steps": {},
        })
        ws.save()
        return ws

    @contextmanager
    def stage_outputs(self):
        """Prepare a complete movie step before replacing any existing results."""
        with TemporaryDirectory(dir=self.root, prefix=".stage-") as directory:
            baseline = json.dumps(self.info, sort_keys=True)
            staged = Workspace(Path(directory), deepcopy(self.info))
            yield staged
            files = sorted(staged.root.iterdir(), key=lambda p: p.name == PROJECT_FILE)
            with _publish_lock(self.root):
                _recover_outputs(self.root)
                with self.path(PROJECT_FILE).open(encoding="utf-8") as f:
                    current = json.load(f)
                if json.dumps(current, sort_keys=True) != baseline:
                    raise WorkspaceError("项目已被其他任务修改，请重新打开后重试。")
                previous = staged.root / "previous"
                previous.mkdir()
                created = staged.root / "created"
                created.mkdir()
                for source in files:
                    if not self.path(source.name).exists():
                        (created / source.name).touch()
                pending = self.root / PUBLISH_DIR
                staged.root.rename(pending)
                try:
                    for source in files:
                        dest = self.path(source.name)
                        if dest.exists():
                            dest.replace(pending / "previous" / dest.name)
                        (pending / dest.name).replace(dest)
                    pending.rename(staged.root)
                except BaseException:
                    _recover_outputs(self.root)
                    raise
            self.info = staged.info

    def save(self):
        with open(self.root / PROJECT_FILE, "w", encoding="utf-8") as f:
            json.dump(self.info, f, ensure_ascii=False, indent=2)

    def record_step(self, name: str, params: dict, outputs: dict):
        self.info.setdefault("steps", {})[name] = {
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": params,
            "outputs": outputs,
        }
        self.save()

    def path(self, name: str) -> Path:
        return self.root / name

    def step(self, name: str) -> dict | None:
        return self.info.get("steps", {}).get(name)
