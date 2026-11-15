import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFormLayout,
                               QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from ...core.preprocess import PreprocessParams, output_shape, run_preprocess
from .common import heading, hint


class PreprocessPage(QWidget):
    runRequested = Signal(object, object)      # fn, PreprocessParams
    showResult = Signal(str)                   # "movie" | "mean" | "max"

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.view = view
        self.session = None
        self.roi = None

        lay = QVBoxLayout(self)
        lay.addWidget(heading("2. 预处理"))
        lay.addWidget(hint("截取需要分析的时间段，裁掉视野外的黑边，并降低分辨率和帧率。"
                           "这一步决定了后续所有步骤的速度和硬盘占用。"))
        form = QFormLayout()

        self.start = QDoubleSpinBox(); self.start.setDecimals(1); self.start.setSuffix(" 秒")
        self.end = QDoubleSpinBox(); self.end.setDecimals(1); self.end.setSuffix(" 秒")
        form.addRow("开始时间", self.start)
        form.addRow("结束时间", self.end)
        form.addRow(hint("例如去掉开头动物刚放进去的几秒。"))

        self.crop_cb = QCheckBox("在画面上圈选分析区域")
        self.crop_cb.toggled.connect(self._toggle_roi)
        form.addRow(self.crop_cb)
        self.diam = QSpinBox(); self.diam.setSuffix(" 像素"); self.diam.setEnabled(False)
        self.diam.valueChanged.connect(self._diam_changed)
        form.addRow("圆圈直径", self.diam)
        form.addRow(hint("拖动黄色圆圈可以移动；拖动圆圈上的小方块，或修改上面的直径，可以放大缩小。"
                         "只保留圆圈内镜头视野的区域，圆圈外的像素会被置为 0。"))

        self.sbin = QSpinBox(); self.sbin.setRange(1, 4); self.sbin.setValue(2)
        form.addRow("空间降采样", self.sbin)
        form.addRow(hint("2 = 每 2×2 个像素合并成 1 个，数据量变为 1/4。"
                         "V4 的 608×608 画面推荐 2；细胞很小时用 1。"))
        self.tbin = QSpinBox(); self.tbin.setRange(1, 6); self.tbin.setValue(1)
        form.addRow("时间降采样", self.tbin)
        form.addRow(hint("3 = 每 3 帧平均成 1 帧，30 Hz 变为 10 Hz。钙信号变化较慢，"
                         "30 Hz 录制时可以用 2–3 来减少计算量；10 Hz 录制请保持 1。"))
        lay.addLayout(form)

        self.estimate = QLabel("")
        self.estimate.setWordWrap(True)
        lay.addWidget(self.estimate)
        for w in (self.start, self.end, self.sbin, self.tbin):
            w.valueChanged.connect(self._update_estimate)

        self.run_btn = QPushButton("开始预处理")
        self.run_btn.setStyleSheet("font-weight:bold; padding:6px;")
        self.run_btn.clicked.connect(self._run)
        lay.addWidget(self.run_btn)

        lay.addWidget(heading("查看结果"))
        row = QHBoxLayout()
        for key, txt in (("movie", "预处理视频"), ("mean", "平均投影"), ("max", "最大投影")):
            b = QPushButton(txt)
            b.clicked.connect(lambda _=False, k=key: self.showResult.emit(k))
            row.addWidget(b)
        lay.addLayout(row)
        lay.addWidget(hint("最大投影上亮的小圆点通常就是活跃的细胞，可以用来判断视野和焦距是否合适。"))
        lay.addStretch(1)
        self.setEnabled(False)

    def set_session(self, session):
        self.session = session
        dur = session.duration_s
        for w in (self.start, self.end):
            w.setRange(0, dur)
        self.start.setValue(0)
        self.end.setValue(dur)
        if self.crop_cb.isChecked():
            self.crop_cb.setChecked(False)
        self.setEnabled(True)
        self._update_estimate()

    def params(self) -> PreprocessParams:
        crop = None
        circle = False
        if self.roi is not None and self.crop_cb.isChecked():
            x, y = (int(round(v)) for v in self.roi.pos())
            d = int(round(self.roi.size()[0]))
            x, y = max(0, x), max(0, y)
            w = max(1, min(d, self.session.width - x))
            h = max(1, min(d, self.session.height - y))
            crop = (x, y, w, h)
            circle = True
        end = self.end.value()
        return PreprocessParams(
            start_s=self.start.value(),
            end_s=None if end >= self.session.duration_s - 1e-6 else end,
            crop=crop, circle=circle, spatial_bin=self.sbin.value(), temporal_bin=self.tbin.value())

    def _toggle_roi(self, on):
        if on and self.session is not None:
            w, h = self.session.width, self.session.height
            if self.roi is None:
                d = 0.8 * min(w, h)
                self.roi = pg.CircleROI([(w - d) / 2, (h - d) / 2], [d, d],
                                        pen=pg.mkPen("y", width=2),
                                        hoverPen=pg.mkPen("c", width=3),
                                        handlePen=pg.mkPen("c", width=2),
                                        maxBounds=pg.QtCore.QRectF(0, 0, w, h))
                self.roi.sigRegionChanged.connect(self._roi_changed)
            self.diam.blockSignals(True)
            self.diam.setRange(8, min(w, h))
            self.diam.setValue(int(round(self.roi.size()[0])))
            self.diam.blockSignals(False)
            self.view.vb.addItem(self.roi)
        elif self.roi is not None:
            self.view.vb.removeItem(self.roi)
        self.diam.setEnabled(bool(on and self.roi is not None))
        self._update_estimate()

    def _roi_changed(self):
        self.diam.blockSignals(True)
        self.diam.setValue(int(round(self.roi.size()[0])))
        self.diam.blockSignals(False)
        self._update_estimate()

    def _diam_changed(self, d):
        """Resize the circle about its centre, keeping it inside the image."""
        if self.roi is None or self.session is None:
            return
        w, h = self.session.width, self.session.height
        d = float(min(d, w, h))
        (x0, y0), s0 = self.roi.pos(), self.roi.size()[0]
        cx, cy = x0 + s0 / 2, y0 + s0 / 2
        x = min(max(cx - d / 2, 0.0), w - d)
        y = min(max(cy - d / 2, 0.0), h - d)
        self.roi.setPos([x, y], update=False)
        self.roi.setSize([d, d])

    def _update_estimate(self):
        if self.session is None:
            return
        p = self.params()
        H, W = output_shape(self.session, p)
        dur = (p.end_s if p.end_s is not None else self.session.duration_s) - p.start_s
        n = int(round(max(0, dur) * self.session.fps)) // p.temporal_bin
        gb = n * H * W / 1e9
        self.estimate.setText(f"输出: {W}×{H} 像素，{n} 帧，"
                              f"{self.session.fps / p.temporal_bin:g} Hz，约 {gb:.2f} GB 硬盘空间")

    def _run(self):
        if self.session is not None:
            self.runRequested.emit(run_preprocess, self.params())

    def set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
