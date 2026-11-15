"""Central viewer: shows a movie (with slider and play) or a still image."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QSlider,
                               QVBoxLayout, QWidget)

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)


class VideoView(QWidget):
    frameChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("#1e1e1e")
        self.vb = self.glw.addViewBox(lockAspect=True, invertY=True)
        self.img = pg.ImageItem()
        self.vb.addItem(self.img)
        self.hist = pg.HistogramLUTItem(image=self.img)
        self.hist.gradient.loadPreset("grey")
        self.glw.addItem(self.hist)

        self.title = QLabel("请先在左侧「导入数据」中选择 Miniscope 文件夹")
        self.title.setAlignment(Qt.AlignCenter)
        self.play_btn = QPushButton("▶ 播放")
        self.play_btn.setFixedWidth(80)
        self.slider = QSlider(Qt.Horizontal)
        self.time_lbl = QLabel("")
        self.time_lbl.setMinimumWidth(170)

        ctl = QHBoxLayout()
        ctl.addWidget(self.play_btn)
        ctl.addWidget(self.slider, 1)
        ctl.addWidget(self.time_lbl)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.title)
        lay.addWidget(self.glw, 1)
        lay.addLayout(ctl)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.play_btn.clicked.connect(self.toggle_play)
        self.slider.valueChanged.connect(self._show)

        self._overlays = []
        self._get = None
        self._n = 0
        self._fps = 30.0
        self._levels_set = False
        self._set_controls(False)

    # ----------------------------------------------------------------- API
    def _reset_look(self):
        self.img.resetTransform()
        self.vb.setAspectLocked(True)
        self.hist.gradient.loadPreset("grey")

    def set_movie(self, get_frame, n_frames: int, fps: float, title: str = ""):
        self.stop()
        self._reset_look()
        self._get, self._n, self._fps = get_frame, n_frames, fps
        self._levels_set = False
        self.title.setText(title)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, n_frames - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._set_controls(True)
        self._show(0)
        self.vb.autoRange()

    def set_image(self, image: np.ndarray, title: str = ""):
        self.stop()
        self._reset_look()
        self._get = None
        self._set_controls(False)
        self.title.setText(title)
        self.img.setImage(image, autoLevels=True)
        self.time_lbl.setText("")
        self.vb.autoRange()

    def add_overlay(self, item):
        self.vb.addItem(item)
        self._overlays.append(item)

    def clear_overlays(self):
        for it in self._overlays:
            self.vb.removeItem(it)
        self._overlays = []

    def current_index(self) -> int:
        return self.slider.value()

    def image_shape(self):
        im = self.img.image
        return None if im is None else im.shape[:2]

    def stop(self):
        self.timer.stop()
        self.play_btn.setText("▶ 播放")

    def toggle_play(self):
        if self.timer.isActive():
            self.stop()
        elif self._get is not None:
            self.timer.start(int(1000 / min(self._fps, 30)))
            self.play_btn.setText("⏸ 暂停")

    # ------------------------------------------------------------- private
    def _set_controls(self, on: bool):
        self.play_btn.setEnabled(on)
        self.slider.setEnabled(on)

    def _tick(self):
        i = self.slider.value() + 1
        if i >= self._n:
            i = 0
        self.slider.setValue(i)

    def _show(self, i: int):
        if self._get is None:
            return
        frame = self._get(i)
        self.img.setImage(frame, autoLevels=not self._levels_set)
        self._levels_set = True
        t = i / self._fps
        self.time_lbl.setText(f"第 {i + 1}/{self._n} 帧   {int(t // 60):02d}:{t % 60:05.2f}")
        self.frameChanged.emit(i)
