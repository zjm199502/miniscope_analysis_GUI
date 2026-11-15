from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSplitter, QStackedWidget)

from .. import __version__
from ..core.preprocess import MAX_FILE, MEAN_FILE, load_movie
from .pages.import_page import ImportPage
from .pages.placeholder import PlaceholderPage
from .pages.preprocess_page import PreprocessPage
from .video_view import VideoView
from .worker import start_worker

STEPS = [
    ("导入数据", None),
    ("预处理", None),
    ("运动校正", "校正动物运动造成的画面抖动，让同一个细胞始终在同一位置。"),
    ("去除背景", "去掉单光子成像中大片的模糊背景荧光，让细胞轮廓更清楚。"),
    ("细胞提取", "自动找出视野中的细胞，并可以人工检查、删除或合并。"),
    ("信号与 spike", "计算每个细胞的 ΔF/F 曲线，并推断放电事件。"),
    ("行为分析", "导入行为视频，追踪动物位置，划定区域并计算行为指标。"),
    ("同步分析", "把钙信号和行为在时间上对齐，做联合分析。"),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Miniscope 分析 v{__version__}")
        self.resize(1400, 850)
        self.session = None
        self.ws = None
        self.thread = None
        self.worker = None

        self.steps = QListWidget()
        self.steps.setFixedWidth(150)
        self.steps.setStyleSheet("QListWidget{font-size:14px;} QListWidget::item{padding:8px;}")
        for i, (name, _) in enumerate(STEPS):
            it = QListWidgetItem(f"{i + 1}. {name}")
            if i > 0:
                it.setFlags(it.flags() & ~Qt.ItemIsEnabled)
            self.steps.addItem(it)

        self.view = VideoView()
        self.pages = QStackedWidget()
        self.import_page = ImportPage()
        self.pre_page = PreprocessPage(self.view)
        self.pages.addWidget(self.import_page)
        self.pages.addWidget(self.pre_page)
        for name, text in STEPS[2:]:
            self.pages.addWidget(PlaceholderPage(name, text))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.pages)
        scroll.setMinimumWidth(340)

        split = QSplitter()
        split.addWidget(self.steps)
        split.addWidget(self.view)
        split.addWidget(scroll)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(300)
        self.progress.setVisible(False)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_btn)
        self.statusBar().showMessage("第一步：选择 Miniscope 文件夹和一个空的输出文件夹")

        self.steps.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.steps.setCurrentRow(0)
        self.import_page.sessionReady.connect(self._on_session)
        self.pre_page.runRequested.connect(self._run_step)
        self.pre_page.showResult.connect(self._show_pre_result)

    # ------------------------------------------------------------ session
    def _on_session(self, session, ws):
        self.session, self.ws = session, ws
        self.view.set_movie(session.get_frame, session.n_frames, session.fps,
                            f"原始视频 · {session.folder.name}")
        self.pre_page.set_session(session)
        self.steps.item(1).setFlags(self.steps.item(1).flags() | Qt.ItemIsEnabled)
        self.statusBar().showMessage(f"已载入，结果保存在 {ws.root}")
        if ws.step("preprocess"):
            self.statusBar().showMessage(f"已载入，检测到上次的预处理结果 · {ws.root}")

    # -------------------------------------------------------------- tasks
    def _run_step(self, fn, params):
        if self.thread is not None:
            return
        self.view.stop()
        self.thread, self.worker = start_worker(self, fn, self.session, self.ws, params)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.thread.finished.connect(self._clear_thread)
        self.pre_page.set_running(True)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.cancel_btn.setVisible(True)
        self.thread.start()

    def _on_progress(self, frac, msg):
        self.progress.setValue(int(frac * 100))
        self.statusBar().showMessage(msg)

    def _on_done(self, _):
        self.statusBar().showMessage("预处理完成，可以在右侧查看结果")
        self._show_pre_result("max")

    def _on_failed(self, msg):
        self.statusBar().showMessage("任务未完成")
        QMessageBox.warning(self, "出错了", msg)

    def _clear_thread(self):
        self.thread = None
        self.worker = None
        self.pre_page.set_running(False)
        self.progress.setVisible(False)
        self.cancel_btn.setVisible(False)

    def _cancel(self):
        if self.worker is not None:
            self.worker.stop()

    # ------------------------------------------------------------ results
    def _show_pre_result(self, which):
        if self.ws is None or not self.ws.step("preprocess"):
            QMessageBox.information(self, "提示", "请先运行预处理")
            return
        if which == "movie":
            mov = load_movie(self.ws)
            fps = self.ws.step("preprocess")["params"]["fps_out"]
            self.view.set_movie(lambda i: np.asarray(mov[i]), mov.shape[0], fps, "预处理后的视频")
        else:
            f = MAX_FILE if which == "max" else MEAN_FILE
            title = "最大投影" if which == "max" else "平均投影"
            self.view.set_image(np.load(self.ws.path(f)), title)

    def closeEvent(self, e):
        if self.worker is not None:
            self.worker.stop()
            self.thread.wait(3000)
        super().closeEvent(e)
