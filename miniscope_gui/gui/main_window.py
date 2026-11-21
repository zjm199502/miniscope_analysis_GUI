from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton,
                               QSplitter, QStackedWidget)

from .. import APP_NAME, __version__
from ..core.preprocess import MAX_FILE, MEAN_FILE, load_movie
from .cell_review import CellReview
from .pages.calcium_pages import BackgroundPage, ExtractionPage, MotionPage, SignalsPage
from .pages.common import FitScrollArea
from .pages.compute_page import DeviceReport, probe_devices
from .pages.import_page import ImportPage
from .pages.behavior_pages import BehaviorPage, SyncPage
from .pages.modeling_page import ModelingPage
from .pages.preprocess_page import PreprocessPage
from .pages.registration_page import RegistrationPage
from .pages.studies_page import StudiesPage
from .video_view import VideoView
from .worker import start_worker

STEP_NAMES = ["导入数据", "预处理", "运动校正", "去除背景", "细胞提取", "信号与 spike",
              "行为分析", "同步分析", "建模分析", "跨天配准", "行为与空间专题"]
# step index -> workspace steps that must exist before it can be opened
REQUIRES = {1: [], 2: ["preprocess"], 3: ["preprocess"], 4: ["background"],
            5: ["extraction"], 6: [], 7: ["behavior"], 8: ["signals"], 9: []}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{__version__}")
        self.setWindowIcon(QIcon(str(Path(__file__).resolve().parents[1] / "assets" / "calcilume.png")))
        self.resize(1400, 900)
        self.session = None
        self.ws = None
        self.thread = None
        self.worker = None
        self.running_page = None
        self._probe_thread = self._probe_worker = None
        self._closing = False
        self._pending_session = None

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
        from .update_ui import UpdateManager
        self.updates = UpdateManager(self)

        self.steps.currentRowChanged.connect(self._on_step_changed)
        self.steps.setCurrentRow(0)
        self.import_page.sessionReady.connect(self._on_session)
        self.pre_page.runRequested.connect(lambda fn, p: self._run_step(self.pre_page, fn, p))
        self.pre_page.showResult.connect(self.show_result)
        for page in self.step_pages + self.extra_pages:
            page.runRequested.connect(lambda fn, p, pg_=page: self._run_step(pg_, fn, p))
            page.showRequested.connect(self.show_result)
        self.view.frameChanged.connect(self._on_frame)
        self._joint = None
        self.ext_page.toggle_btn.clicked.connect(self._toggle_cells)
        self.ext_page.merge_btn.clicked.connect(self._merge_cells)
        self.ext_page.prev_btn.clicked.connect(lambda: self.review.step(-1))
        self.ext_page.next_btn.clicked.connect(lambda: self.review.step(1))
        self.ext_page.reject_low_btn.clicked.connect(self._reject_low_quality)
        self.ext_page.restore_btn.clicked.connect(self._restore_cells)
        self.ext_page.q_thr.valueChanged.connect(self._quality_threshold)
        self.ext_page.by_quality.toggled.connect(lambda on: setattr(self.review, "by_quality", on))
        self.review.threshold = float(self.ext_page.q_thr.value())
        self.review.by_quality = self.ext_page.by_quality.isChecked()

    def _make_extra_pages(self):
        self.beh_page = BehaviorPage(self)
        self.sync_page = SyncPage(self)
        self.model_page = ModelingPage(self)
        self.reg_page = RegistrationPage(self)
        self.studies_page = StudiesPage(self)
        return [self.beh_page, self.sync_page, self.model_page, self.reg_page, self.studies_page]

    def refit_page(self):
        """Re-fit the right panel after a page changed its own height."""
        if hasattr(self, "pages"):
            self._fit_page_height(self.pages.currentIndex())

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
        if self._closing:
            return
        if self.thread is not None:
            self._pending_session = (session, ws)
            self._stop_for_transition()
            return
        if self.session is not None:
            self.steps.setCurrentRow(0)
            self._release_project()
        self.import_page.session = session
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
        self._joint = None
        self._clear_model_view()
        if page in (self.motion_page, self.bg_page) and self.ws:
            self._identify_devices(page)
        if page is self.model_page:
            self.model_page.on_context()
            self.model_page.update_status()
        if page is not self.beh_page and getattr(self.beh_page, "edit_roi", None) is not None:
            self.beh_page.cancel_edit()
        if page in (self.beh_page, self.sync_page) and self.beh_page.video is not None:
            self.beh_page.show_video()
        if self.review.active and page is not self.ext_page and page is not self.sig_page:
            self.review.close()
            self.trace.hide()
        if page is self.ext_page and self.ws is not None and self.ws.step("extraction"):
            self.show_result("cells")

    # -------------------------------------------------------------- tasks
    def _identify_devices(self, page):
        # Device discovery belongs to the window, not to a project or its analysis queue.
        if page.has_inventory() or self._probe_thread is not None or self._closing:
            return
        for target in (self.motion_page, self.bg_page):
            target.device_info.setText("正在识别系统显卡并检查 CUDA…")
            target.device_info.show()
        self._probe_thread, self._probe_worker = start_worker(
            self, probe_devices, None, None, {"refresh": False})
        self._probe_worker.finished.connect(self._show_device_report)
        self._probe_worker.failed.connect(self._probe_failed)
        self._probe_thread.finished.connect(self._finish_probe)
        self._probe_thread.start()

    def _show_device_report(self, result):
        if not self._closing and result.inventory:
            self.motion_page.show_devices(result.inventory)
            self.bg_page.show_devices(result.inventory)

    def _probe_failed(self, msg):
        if not self._closing:
            for page in (self.motion_page, self.bg_page):
                page.device_info.setText(f"设备识别失败：{msg}")

    def _finish_probe(self):
        self._probe_thread.wait()
        self._probe_thread.deleteLater()
        self._probe_thread = self._probe_worker = None
        if self._closing and self.thread is None:
            self.close()

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
        if self._closing or self._pending_session:
            return
        self.progress.setValue(int(frac * 100))
        self.statusBar().showMessage(msg)

    def _on_done(self, result):
        if self._closing or self._pending_session:
            return
        if isinstance(result, DeviceReport):
            if result.inventory:
                self._show_device_report(result)
                self.statusBar().showMessage("显卡检测完成")
            else:
                self.statusBar().showMessage("已取消显卡检测")
            return
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
        for p in self.step_pages + self.extra_pages:
            p.update_status()
        if page is self.beh_page:
            self.sync_page.on_context()
        if page in (self.model_page, self.reg_page):
            page.update_status()
            self.refit_page()
        self.statusBar().showMessage(f"{page.title} 完成")
        key = page.default_view()
        if key:
            self.show_result(key)

    def _on_failed(self, msg):
        if self._closing or self._pending_session:
            return
        self.statusBar().showMessage("任务未完成")
        QMessageBox.warning(self, "出错了", msg)

    def _clear_thread(self):
        if self.running_page is not None:
            self.running_page.set_running(False)
        self.thread.wait()
        self.thread.deleteLater()
        self.thread = None
        self.worker = None
        self.running_page = None
        self.progress.setVisible(False)
        self.cancel_btn.setVisible(False)
        if self._closing:
            self.close()
        elif self._pending_session:
            pending, self._pending_session = self._pending_session, None
            self.centralWidget().setEnabled(True)
            self._on_session(*pending)

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
        from ..core.storage import MovieReader
        return MovieReader(self.ws.path(fname))

    def _fps(self):
        st = self.ws.step("preprocess") or {}
        return float(st.get("params", {}).get("fps_out", 30.0))

    def show_result(self, which: str):
        """Show a result in the centre viewer (and the trace panel when useful)."""
        try:
            self._show_result(which)
        except FileNotFoundError as e:
            from ..core.cleanup import INTERMEDIATE
            name = getattr(e, "filename", None)
            msg = str(e)
            for f, label in INTERMEDIATE.items():
                if name and str(name).endswith(f):
                    msg = f"{label}已在「清理中间文件」时删除。需要查看时，重新运行对应的步骤即可。"
            QMessageBox.information(self, "文件不存在", msg)

    def _show_result(self, which: str):
        if which != "cells" and self.review.active:
            self.review.close()
        self._joint = None
        self._clear_model_view()
        from .model_views import is_model_view, show as show_model
        from .registration_views import is_reg_view
        if is_reg_view(which):
            try:
                self.reg_page.show_view(which)
            except (ValueError, FileNotFoundError) as e:
                QMessageBox.information(self, "提示", str(e))
            return
        if is_model_view(which):
            try:
                show_model(self, which)
            except ValueError as e:
                QMessageBox.information(self, "提示", str(e))
            return
        if which.startswith("beh_"):
            self.beh_page.show_view(which)
            return
        if which == "sync_joint":
            self._show_joint()
            return
        if which == "sync_qc":
            self._show_sync_qc()
            return
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

    def _clear_model_view(self):
        if getattr(self, "model_overlays", False):
            self.view.clear_overlays()
            self.trace.getAxis("bottom").setTicks(None)     # model plots use named ticks
            self.model_overlays = False

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

    def _quality_threshold(self, v):
        self.review.set_threshold(v)
        if self.review.active:
            self.ext_page.review_info.setText(self.review.summary())

    def _reject_low_quality(self):
        from ..core.review import reject_low_quality
        if not (self.ws and self.ws.step("extraction")):
            QMessageBox.information(self, "提示", "请先完成细胞提取")
            return
        thr = float(self.ext_page.q_thr.value())
        n = reject_low_quality(self.ws, thr)
        self.review.reload()
        self.ext_page.review_info.setText(self.review.summary())
        self.statusBar().showMessage(f"已删除 {n} 个质量分低于 {thr:.0f} 的候选（可点「全部恢复」撤销）", 8000)

    def _restore_cells(self):
        from ..core.review import restore_all
        if not (self.ws and self.ws.step("extraction")):
            return
        n = restore_all(self.ws)
        self.review.reload()
        self.ext_page.review_info.setText(self.review.summary())
        self.statusBar().showMessage(f"已恢复 {n} 个细胞", 5000)

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

    # -------------------------------------------------------- sync / joint
    def _on_frame(self, i):
        if self._joint is None:
            return
        tb = self._joint["tb"]
        if 0 <= i < len(tb):
            self._joint["line"].setValue(tb[i])
            tr = self._joint.get("track")
            if tr is not None and i < len(tr["x"]):
                self._joint["dot"].setData([tr["x"][i]], [tr["y"][i]])

    def _show_joint(self):
        from ..core.behavior import analyze, load_track
        from ..core.extraction import load_cells
        from ..core.signals import load_signals
        from ..core.sync import load_sync
        if not self._need("sync", "同步") or self.beh_page.video is None:
            if self.beh_page.video is None:
                QMessageBox.information(self, "提示", "请先在第 7 步导入行为视频")
            return
        tb = load_sync(self.ws)
        self.beh_page.show_video()
        self.view.title.setText("联合播放：拖动滑块或播放，下方竖线 = 当前时刻")
        tr = self.trace
        tr.clear()
        tr.addLegend()
        if self.ws.step("signals"):
            sig = load_signals(self.ws)
            acc = load_cells(self.ws)["accepted"].astype(bool)
            tr.plot(sig["times"], sig["dff"][acc].mean(0), pen=pg.mkPen((40, 110, 220), width=1.2),
                    name="平均 ΔF/F")
        track = None
        if self.ws.step("behavior"):
            res = analyze(load_track(self.ws), self.beh_page.params())
            n = min(len(tb), len(res["t"]))
            if "freezing" in res:
                tr.plot(tb[:n], res["freezing"][:n].astype(float) * 0.1, pen=pg.mkPen((220, 60, 60)),
                        fillLevel=0, brush=(220, 60, 60, 50), name="僵直")
            if self.beh_page.cfg.source == "tracking":
                track = res
        line = pg.InfiniteLine(0, angle=90, pen=pg.mkPen((0, 0, 0), width=1.5))
        tr.addItem(line)
        dot = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(255, 60, 60), pen=pg.mkPen("w"))
        self.view.add_overlay(dot)
        tr.setLabel("bottom", "miniscope 时间 (秒)")
        tr.setTitle("")
        tr.show()
        self._joint = {"tb": tb, "line": line, "dot": dot, "track": track}
        self._on_frame(self.view.current_index())

    def _show_sync_qc(self):
        from ..core.sync import load_sync
        if not self._need("sync", "同步"):
            return
        tr = self.trace
        tr.clear()
        f = self.ws.path("behavior/led_trace.npy")
        if f.exists():
            led = np.load(f)
            tr.plot(np.arange(len(led)), led, pen=pg.mkPen((40, 110, 220)))
            st = self.ws.step("sync")["params"]
            if "threshold" in st:
                tr.addItem(pg.InfiniteLine(st["threshold"], angle=0, pen=pg.mkPen((220, 60, 60), style=Qt.DashLine)))
            tr.setLabel("bottom", "行为视频帧")
            tr.setTitle("LED 亮度：应能清楚看到开始长亮、周期短闪和结束长亮，红线为识别阈值")
        else:
            tb = load_sync(self.ws)
            tr.plot(np.arange(len(tb)), tb, pen=pg.mkPen((40, 110, 220)))
            tr.setLabel("bottom", "行为视频帧")
            tr.setLabel("left", "miniscope 时间 (秒)")
            tr.setTitle("手动对齐：行为帧与 miniscope 时间的对应关系")
        tr.show()

    def joint_analysis(self, kind):
        from ..core.extraction import load_cells
        from ..core.signals import load_signals
        from ..core import sync as S
        if not (self._need("signals", "信号与 spike") and self._need("behavior", "行为分析")
                and self._need("sync", "同步")):
            return
        sp = self.sync_page
        try:
            cfg = self.beh_page.params()
            if kind == "export":
                from PySide6.QtWidgets import QFileDialog
                d = QFileDialog.getExistingDirectory(self, "选择导出文件夹", str(self.ws.root))
                if d:
                    out = S.export_joint(self.ws, cfg, d)
                    QMessageBox.information(self, "导出完成", f"已保存到:\n{out}\n"
                                            "aligned_frames.csv：逐帧对齐的钙信号与行为\n"
                                            "cell_behavior_stats.csv：每个细胞与僵直/速度的关系")
                return
            sig = load_signals(self.ws)
            acc = load_cells(self.ws)["accepted"].astype(bool)
            ids = np.where(acc)[0]
            dff = sig["dff"][acc]
            b = S.behavior_on_calcium_frames(self.ws, cfg)
            tr = self.trace
            tr.clear()
            if kind == "freezing":
                if "freezing" not in b:
                    raise ValueError("行为数据里没有僵直信息")
                m = S.state_modulation(dff, b["freezing"], b["valid"])
                up = m["p_up"] < 0.025
                dn = m["p_down"] < 0.025
                col = [(220, 60, 60) if u else (40, 110, 220) if d else (150, 150, 150) for u, d in zip(up, dn)]
                tr.addItem(pg.ScatterPlotItem(m["out_state"], m["in_state"], size=8,
                                              brush=[pg.mkBrush(*c) for c in col], pen=None))
                lo = float(min(m["out_state"].min(), m["in_state"].min()))
                hi = float(max(m["out_state"].max(), m["in_state"].max()))
                tr.plot([lo, hi], [lo, hi], pen=pg.mkPen((0, 0, 0), style=Qt.DashLine))
                tr.setLabel("bottom", "运动时平均 ΔF/F"); tr.setLabel("left", "僵直时平均 ΔF/F")
                tr.setTitle("每个点是一个细胞：红 = 僵直时更活跃，蓝 = 僵直时更安静（双侧 p < 0.05）")
                sp.result_lbl.setText(f"{len(ids)} 个细胞中：僵直时显著升高 {up.sum()} 个，"
                                      f"显著降低 {dn.sum()} 个（循环平移检验，双侧 p < 0.05）。\n"
                                      f"升高：{', '.join('#' + str(i) for i in ids[up][:30])}")
            elif kind == "speed":
                c = S.speed_correlation(dff, b["speed"], b["valid"])
                sig_ = c["p"] < 0.05
                hist, edges = np.histogram(c["r"], bins=20)
                tr.addItem(pg.BarGraphItem(x0=edges[:-1], x1=edges[1:], height=hist, brush=(40, 110, 220, 160)))
                tr.setLabel("bottom", "与速度的相关系数 r"); tr.setLabel("left", "细胞数")
                tr.setTitle("每个细胞 ΔF/F 与运动速度的相关性分布")
                sp.result_lbl.setText(f"{len(ids)} 个细胞中：与速度显著正相关 {(sig_ & (c['r'] > 0)).sum()} 个，"
                                      f"负相关 {(sig_ & (c['r'] < 0)).sum()} 个（p < 0.05）。")
            elif kind == "peth":
                src = sp.event_src.currentText()
                t = b["times"]
                if src == "僵直开始":
                    if "freezing" not in b:
                        raise ValueError("行为数据里没有僵直信息")
                    ev = S.event_onsets(b["freezing"], t, min_gap_s=sp.post.value())
                elif src.startswith("进入区域 "):
                    ev = S.event_onsets(b["zones"][src[5:]], t, min_gap_s=sp.post.value())
                else:
                    txt = sp.event_times.text().replace("，", ",").replace(" ", ",")
                    ev = np.array([float(v) for v in txt.split(",") if v.strip()])
                axis, seg = S.peri_event(dff, t, ev, sp.pre.value(), sp.post.value())
                if len(seg) == 0:
                    raise ValueError("没有可用的事件（事件太靠近记录开头或结尾，或没有事件）")
                mean_cells = seg.mean(0)               # cells x time
                pop = mean_cells.mean(0)
                sem = mean_cells.std(0) / np.sqrt(max(1, mean_cells.shape[0]))
                tr.plot(axis, pop, pen=pg.mkPen((40, 110, 220), width=2))
                tr.addItem(pg.FillBetweenItem(pg.PlotDataItem(axis, pop - sem), pg.PlotDataItem(axis, pop + sem),
                                              brush=(40, 110, 220, 50)))
                tr.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen((220, 60, 60), style=Qt.DashLine)))
                tr.setLabel("bottom", "相对事件的时间 (秒)"); tr.setLabel("left", "ΔF/F")
                tr.setTitle(f"所有细胞平均（± SEM），{len(seg)} 个事件")
                order = np.argsort(-mean_cells[:, axis >= 0].mean(1))
                img = mean_cells[order]
                hi = np.percentile(np.abs(img), 99) or 1
                self.view.set_image(np.clip(img / hi, -1, 1), f"每个细胞的事件前后平均（{src}，按事件后反应排序）")
                self.view.hist.gradient.loadPreset("bipolar")
                self.view.vb.setAspectLocked(False)
                self.view.img.setRect(pg.QtCore.QRectF(axis[0], 0, axis[-1] - axis[0], len(img)))
                self.view.vb.autoRange()
                sp.result_lbl.setText(f"事件：{src}，共 {len(ev)} 个，可用 {len(seg)} 个（窗口内完整）。")
            tr.show()
        except Exception as e:
            QMessageBox.warning(self, "分析失败", str(e))

    def _stop_for_transition(self):
        self.view.stop()
        self.centralWidget().setEnabled(False)
        self.statusBar().showMessage("正在结束当前项目的任务…")
        self._cancel()

    def _release_project(self):
        self.view.stop()
        self._joint = None
        self._clear_model_view()
        self.review.release_project()
        self.trace.clear()
        self.trace.hide()
        self.view.release_project()
        self.beh_page.release_project()
        self.studies_page.release_project()
        if self.sync_page.led_roi is not None:
            self.view.vb.removeItem(self.sync_page.led_roi)
            self.sync_page.led_roi = None
        self.sync_page.arduino_log = None
        self.sync_page.log_btn.setText("Arduino 记录（可选）…")
        self.pre_page.crop_cb.setChecked(False)
        self.pre_page.roi = None
        self.pre_page.session = self.import_page.session = None
        self.reg_page.folder = None
        self.reg_page.table.setRowCount(0)
        self.reg_page.out_edit.clear()
        for page in self.step_pages + self.extra_pages:
            page.session = page.ws = None
        self.session = self.ws = None

    def closeEvent(self, e):
        self._closing = True
        self._pending_session = None
        if self.thread is not None:
            self._stop_for_transition()
            e.ignore()
            return
        if self._probe_thread is not None:
            e.ignore()
            return
        self.updates.shutdown()
        self._release_project()
        super().closeEvent(e)
