from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton,
                               QSplitter, QStackedWidget)

from .. import __version__
from ..core.preprocess import MAX_FILE, MEAN_FILE, load_movie
from .cell_review import CellReview
from .pages.calcium_pages import BackgroundPage, ExtractionPage, MotionPage, SignalsPage
from .pages.common import FitScrollArea
from .pages.import_page import ImportPage
from .pages.placeholder import PlaceholderPage
from .pages.preprocess_page import PreprocessPage
from .video_view import VideoView
from .worker import start_worker

STEP_NAMES = ["导入数据", "预处理", "运动校正", "去除背景", "细胞提取", "信号与 spike",
              "行为分析", "同步分析"]
# step index -> workspace steps that must exist before it can be opened
REQUIRES = {1: [], 2: ["preprocess"], 3: ["preprocess"], 4: ["background"],
            5: ["extraction"], 6: [], 7: []}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Miniscope 分析 v{__version__}")
        self.resize(1400, 900)
        self.session = None
        self.ws = None
        self.thread = None
        self.worker = None
        self.running_page = None

        self.steps = QListWidget()
        self.steps.setFixedWidth(150)
        self.steps.setStyleSheet("QListWidget{font-size:14px;} QListWidget::item{padding:8px;}")
        for i, name in enumerate(STEP_NAMES):
            it = QListWidgetItem(f"{i + 1}. {name}")
            if i > 0:
                it.setFlags(it.flags() & ~Qt.ItemIsEnabled)
            self.steps.addItem(it)

        # centre: viewer on top, trace panel below
        self.view = VideoView()
        self.trace = pg.PlotWidget()
        self.trace.setBackground("w")
        self.trace.showGrid(x=True, y=True, alpha=0.2)
        self.trace.setMinimumHeight(150)
        self.trace.hide()
        centre = QSplitter(Qt.Vertical)
        centre.addWidget(self.view)
        centre.addWidget(self.trace)
        centre.setSizes([600, 220])

        self.review = CellReview(self.view, self.trace, self)

        self.import_page = ImportPage()
        self.pre_page = PreprocessPage(self.view)
        self.motion_page = MotionPage()
        self.bg_page = BackgroundPage()
        self.ext_page = ExtractionPage()
        self.sig_page = SignalsPage()
        self.step_pages = [self.motion_page, self.bg_page, self.ext_page, self.sig_page]
        self.pages = QStackedWidget()
        for w in [self.import_page, self.pre_page, *self.step_pages]:
            self.pages.addWidget(w)
        self.extra_pages = self._make_extra_pages()
        for w in self.extra_pages:
            self.pages.addWidget(w)
        self.pages.currentChanged.connect(self._fit_page_height)

        scroll = FitScrollArea()
        scroll.setWidget(self.pages)
        scroll.setMinimumWidth(320)

        split = QSplitter()
        split.addWidget(self.steps)
        split.addWidget(centre)
        split.addWidget(scroll)
        split.setStretchFactor(1, 1)
        split.setCollapsible(2, False)
        split.setSizes([150, 850, 400])
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

        self.steps.currentRowChanged.connect(self._on_step_changed)
        self.steps.setCurrentRow(0)
        self.import_page.sessionReady.connect(self._on_session)
        self.pre_page.runRequested.connect(lambda fn, p: self._run_step(self.pre_page, fn, p))
        self.pre_page.showResult.connect(self.show_result)
        for page in self.step_pages:
            page.runRequested.connect(lambda fn, p, pg_=page: self._run_step(pg_, fn, p))
            page.showRequested.connect(self.show_result)
        self.ext_page.toggle_btn.clicked.connect(self._toggle_cells)
        self.ext_page.merge_btn.clicked.connect(self._merge_cells)
        self.ext_page.prev_btn.clicked.connect(lambda: self.review.step(-1))
        self.ext_page.next_btn.clicked.connect(lambda: self.review.step(1))

    def _make_extra_pages(self):
        return [PlaceholderPage("7. 行为分析", "导入行为视频，追踪动物位置，划定区域并计算行为指标。"),
                PlaceholderPage("8. 同步分析", "把钙信号和行为在时间上对齐，做联合分析。")]

    def _fit_page_height(self, _):
        # QStackedWidget sizes to its largest page; let only the current one count
        for i in range(self.pages.count()):
            w = self.pages.widget(i)
            w.setSizePolicy(w.sizePolicy().horizontalPolicy(),
                            pg.QtWidgets.QSizePolicy.Preferred if i == self.pages.currentIndex()
                            else pg.QtWidgets.QSizePolicy.Ignored)
        self.pages.adjustSize()

    # ------------------------------------------------------------ session
    def _on_session(self, session, ws):
        self.session, self.ws = session, ws
        self.review.close()
        self.trace.hide()
        self.view.set_movie(session.get_frame, session.n_frames, session.fps,
                            f"原始视频 · {session.folder.name}")
        self.pre_page.set_session(session)
        for page in self.step_pages + self.extra_pages:
            if hasattr(page, "set_context"):
                page.set_context(session, ws)
        self._refresh_steps()
        done = [k for k in ("preprocess", "motion", "background", "extraction", "signals") if ws.step(k)]
        msg = f"已载入，结果保存在 {ws.root}"
        if done:
            msg += f"（已完成: {', '.join(done)}，可以继续上次的分析）"
        self.statusBar().showMessage(msg)

    def _refresh_steps(self):
        for i in range(1, len(STEP_NAMES)):
            req = REQUIRES.get(i, [])
            ok = self.ws is not None and all(self.ws.step(r) for r in req)
            it = self.steps.item(i)
            it.setFlags(it.flags() | Qt.ItemIsEnabled if ok else it.flags() & ~Qt.ItemIsEnabled)

    def _on_step_changed(self, row):
        self.pages.setCurrentIndex(row)
        page = self.pages.widget(row)
        if self.review.active and page is not self.ext_page and page is not self.sig_page:
            self.review.close()
            self.trace.hide()
        if page is self.ext_page and self.ws is not None and self.ws.step("extraction"):
            self.show_result("cells")

    # -------------------------------------------------------------- tasks
    def _run_step(self, page, fn, params):
        if self.thread is not None:
            return
        self.view.stop()
        if self.review.active:
            self.review.close()
        self.running_page = page
        self.thread, self.worker = start_worker(self, fn, self.session, self.ws, params)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.thread.finished.connect(self._clear_thread)
        page.set_running(True)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.cancel_btn.setVisible(True)
        self.thread.start()

    def _on_progress(self, frac, msg):
        self.progress.setValue(int(frac * 100))
        self.statusBar().showMessage(msg)

    def _on_done(self, result):
        page = self.running_page
        self._refresh_steps()
        if page is self.pre_page:
            self.statusBar().showMessage("预处理完成，可以在右侧查看结果")
            self.show_result("pre_max")
            for p in self.step_pages + self.extra_pages:
                if hasattr(p, "set_context"):
                    p.set_context(self.session, self.ws)
            return
        page.update_status()
        page.on_finished(result)
        for p in self.step_pages:
            p.update_status()
        self.statusBar().showMessage(f"{page.title} 完成")
        key = page.default_view()
        if key:
            self.show_result(key)

    def _on_failed(self, msg):
        self.statusBar().showMessage("任务未完成")
        QMessageBox.warning(self, "出错了", msg)

    def _clear_thread(self):
        if self.running_page is not None:
            self.running_page.set_running(False)
        self.thread = None
        self.worker = None
        self.running_page = None
        self.progress.setVisible(False)
        self.cancel_btn.setVisible(False)

    def _cancel(self):
        if self.worker is not None:
            self.worker.stop()

    # ------------------------------------------------------------ results
    def _need(self, step, label):
        if self.ws is None or not self.ws.step(step):
            QMessageBox.information(self, "提示", f"请先完成「{label}」")
            return False
        return True

    def _movie(self, fname):
        return np.load(self.ws.path(fname), mmap_mode="r")

    def _fps(self):
        st = self.ws.step("preprocess") or {}
        return float(st.get("params", {}).get("fps_out", 30.0))

    def show_result(self, which: str):
        """Show a result in the centre viewer (and the trace panel when useful)."""
        if which != "cells" and self.review.active:
            self.review.close()
        if which not in ("cells", "traces", "mc_shifts"):
            self.trace.hide()
        if which == "raw":
            s = self.session
            self.view.set_movie(s.get_frame, s.n_frames, s.fps, f"原始视频 · {s.folder.name}")
        elif which == "movie":
            if self._need("preprocess", "预处理"):
                mov = load_movie(self.ws)
                self.view.set_movie(lambda i: np.asarray(mov[i]), mov.shape[0], self._fps(), "预处理后的视频")
        elif which in ("mean", "max", "pre_mean", "pre_max"):
            if self._need("preprocess", "预处理"):
                f = MAX_FILE if which.endswith("max") else MEAN_FILE
                self.view.set_image(np.load(self.ws.path(f)), "最大投影" if f == MAX_FILE else "平均投影")
        elif which == "mc":
            if self._need("motion", "运动校正"):
                mov = self._movie("movie_mc.npy")
                self.view.set_movie(lambda i: np.asarray(mov[i]), mov.shape[0], self._fps(), "运动校正后的视频")
        elif which == "mc_compare":
            if self._need("motion", "运动校正"):
                a, b = load_movie(self.ws), self._movie("movie_mc.npy")
                gap = np.zeros((a.shape[1], 4), np.uint8)
                self.view.set_movie(lambda i: np.hstack([np.asarray(a[i]), gap, np.asarray(b[i])]),
                                    a.shape[0], self._fps(), "左：校正前    右：校正后")
        elif which == "mc_mean":
            if self._need("motion", "运动校正"):
                self.view.set_image(np.load(self.ws.path("mc_mean.npy")), "运动校正后的平均投影")
        elif which == "mc_shifts":
            if self._need("motion", "运动校正"):
                sh = np.load(self.ws.path("mc_shifts.npy"))
                t = np.arange(len(sh)) / self._fps()
                self.trace.clear()
                self.trace.addLegend()
                self.trace.plot(t, sh[:, 0], pen=pg.mkPen((40, 110, 220), width=1.2), name="x 位移")
                self.trace.plot(t, sh[:, 1], pen=pg.mkPen((220, 90, 40), width=1.2), name="y 位移")
                self.trace.setLabel("bottom", "时间 (秒)")
                self.trace.setLabel("left", "像素")
                self.trace.setTitle("每一帧的位移")
                self.trace.show()
        elif which == "bg":
            if self._need("background", "去除背景"):
                mov = self._movie("movie_bg.npy")
                self.view.set_movie(lambda i: np.asarray(mov[i]), mov.shape[0], self._fps(), "去除背景后的视频")
        elif which == "bg_max":
            if self._need("background", "去除背景"):
                self.view.set_image(np.load(self.ws.path("bg_max.npy")), "去除背景后的最大投影")
        elif which in ("cn", "pnr"):
            if self._need("extraction", "细胞提取"):
                f = "summary_cn.npy" if which == "cn" else "summary_pnr.npy"
                self.view.set_image(np.load(self.ws.path(f)), "相关图" if which == "cn" else "PNR 图")
        elif which == "cells":
            if self._need("extraction", "细胞提取"):
                bg = np.load(self.ws.path("bg_max.npy")) if self.ws.path("bg_max.npy").exists() \
                    else np.load(self.ws.path("summary_cn.npy"))
                self.trace.show()
                self.review.open(self.ws, bg, "检查细胞（点击细胞查看曲线）")
                self.ext_page.review_info.setText(self.review.summary())
        elif which == "traces":
            if self._need("signals", "信号与 spike"):
                self._show_heatmap()

    def _show_heatmap(self):
        from ..core.signals import load_signals
        from ..core.extraction import load_cells
        sig = load_signals(self.ws)
        acc = load_cells(self.ws)["accepted"].astype(bool)
        dff = sig["dff"][acc]
        if not len(dff):
            QMessageBox.information(self, "提示", "没有保留的细胞")
            return
        order = np.argsort(-dff.max(1))
        img = dff[order]
        hi = np.percentile(img, 99.5) or 1.0
        self.view.set_image(np.clip(img / hi, 0, 1), f"所有保留细胞的 ΔF/F（{len(img)} 个，按峰值排序）")
        self.view.hist.gradient.loadPreset("viridis")
        # the viewer keeps the aspect ratio; stretch time so the heat map is readable
        self.view.vb.setAspectLocked(False)
        self.view.vb.autoRange()
        self.trace.clear()
        t = sig["times"]
        self.trace.plot(t, sig["dff"][acc].mean(0), pen=pg.mkPen((40, 110, 220), width=1.2))
        self.trace.setTitle("所有保留细胞的平均 ΔF/F")
        self.trace.setLabel("bottom", "时间 (秒)")
        self.trace.show()

    # ------------------------------------------------------------- review
    def _toggle_cells(self):
        from ..core.review import set_accepted
        if not self.review.active or not self.review.selected:
            QMessageBox.information(self, "提示", "请先在「检查细胞」视图中选中细胞")
            return
        set_accepted(self.ws, self.review.selected)
        self.review.reload()
        self.ext_page.review_info.setText(self.review.summary())

    def _merge_cells(self):
        from ..core.review import merge_cells
        if not self.review.active or len(self.review.selected) < 2:
            QMessageBox.information(self, "提示", "按住 Ctrl 点击，选中至少两个细胞后再合并")
            return
        try:
            k = merge_cells(self.ws, self.review.selected)
        except Exception as e:
            QMessageBox.warning(self, "合并失败", str(e))
            return
        self.review.selected = [k]
        self.review.reload()
        self.ext_page.review_info.setText(self.review.summary())

    def closeEvent(self, e):
        if self.worker is not None:
            self.worker.stop()
            self.thread.wait(3000)
        super().closeEvent(e)
