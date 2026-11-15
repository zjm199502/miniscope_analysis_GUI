"""Analysis workspace: one output folder per recording.

All results are written to a folder the user chose. The folder must be empty
when a new workspace is created (or already be a workspace), so raw data and
unrelated files are never touched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

PROJECT_FILE = "project.json"


class WorkspaceError(Exception):
    pass


class Workspace:
    def __init__(self, root: Path, info: dict):
        self.root = Path(root)
        self.info = info

    @classmethod
    def create_or_open(cls, root: str | Path, source_folder: str | Path | None = None):
        root = Path(root)
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
