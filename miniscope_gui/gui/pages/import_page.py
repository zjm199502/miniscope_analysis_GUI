from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QFileDialog, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QVBoxLayout, QWidget)

from ...core.workspace import Workspace, WorkspaceError
from ...io.ucla_v4 import MiniscopeSession
from .common import heading, hint, wrappable


class ImportPage(QWidget):
    sessionReady = Signal(object, object)   # MiniscopeSession, Workspace

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session = None
        lay = QVBoxLayout(self)
        lay.addWidget(heading("1. 导入数据"))
        lay.addWidget(hint("选择 UCLA Miniscope V4 录制得到的设备文件夹，"
                           "通常叫 My_V4_Miniscope，里面有 0.avi、1.avi、"
                           "metaData.json 和 timeStamps.csv。软件只会读取，不会修改原始数据。"))
        self.src_edit = QLineEdit()
        self.src_edit.setReadOnly(True)
        b1 = QPushButton("选择 Miniscope 文件夹…")
        b1.clicked.connect(self.pick_source)
        lay.addWidget(b1)
        lay.addWidget(self.src_edit)

        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("background:#f4f4f4; padding:6px; border-radius:4px;")
        lay.addWidget(self.info)

        lay.addWidget(heading("输出文件夹"))
        lay.addWidget(hint("所有分析结果都保存在这里。请新建一个空文件夹，"
                           "不要放在原始数据文件夹里面。以后选同一个文件夹即可继续上次的分析。"
                           "预处理后的视频可能有几 GB，建议不要放在 OneDrive 等同步盘里。"))
        self.out_edit = QLineEdit()
        self.out_edit.setReadOnly(True)
        b2 = QPushButton("选择输出文件夹…")
        b2.clicked.connect(self.pick_output)
        lay.addWidget(b2)
        lay.addWidget(self.out_edit)
        lay.addStretch(1)

    def pick_source(self):
        d = QFileDialog.getExistingDirectory(self, "选择 Miniscope 文件夹")
        if d:
            self.load_source(d)

    def load_source(self, d):
        try:
            self.session = MiniscopeSession.open(d)
        except Exception as e:
            QMessageBox.warning(self, "无法读取", str(e))
            return
        self.src_edit.setText(d)
        self.info.setText(wrappable(self.session.summary_zh()))
        self._maybe_ready()

    def pick_output(self):
        start = str(Path(self.src_edit.text()).parent) if self.src_edit.text() else ""
        d = QFileDialog.getExistingDirectory(self, "选择或新建一个空文件夹", start)
        if d:
            self.set_output(d)

    def set_output(self, d):
        self.out_edit.setText(d)
        self._maybe_ready()

    def _maybe_ready(self):
        if self.session is None or not self.out_edit.text():
            return
        try:
            ws = Workspace.create_or_open(self.out_edit.text(), source_folder=self.session.folder)
        except WorkspaceError as e:
            QMessageBox.warning(self, "输出文件夹不可用", str(e))
            self.out_edit.clear()
            return
        self.sessionReady.emit(self.session, ws)
