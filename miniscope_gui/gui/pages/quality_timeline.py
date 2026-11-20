"""Clickable behavioural tracking quality over recording time."""

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget


QUALITY_LABELS = ("缺失", "短段插值", "原始观测")
QUALITY_COLORS = ("#ae4a65", "#ac7414", "#237967")


class QualityTimeline(QWidget):
    sampleSelected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(44)
        self.setCursor(Qt.PointingHandCursor)
        self.setAccessibleName("追踪质量时间轴")
        self.setToolTip("点击色带查看该时刻，并回放对应的视频帧。")
        self.clear()

    def clear(self):
        self.times = np.array([], dtype=float)
        self.quality = np.array([], dtype=np.uint8)
        self._pixels = np.array([], dtype=int)
        self.update()

    def set_samples(self, times, quality):
        self.times = times
        self.quality = quality
        self._pixels = np.array([], dtype=int)
        self.update()

    def _pixel_samples(self):
        width = max(1, self.width())
        if len(self._pixels) != width:
            # Use the worst observed quality in each pixel so a brief loss remains
            # both visible and selectable even for very long recordings.
            edges = np.linspace(self.times[0], self.times[-1], width + 1)
            offsets = np.searchsorted(self.times, edges)
            offsets[-1] = len(self.times)
            pixels = np.maximum(offsets[1:] - 1, 0)
            for i in np.flatnonzero(np.diff(offsets) > 1):
                start, end = offsets[i : i + 2]
                pixels[i] = start + self.quality[start:end].argmin()
            self._pixels = pixels
        return self._pixels

    def paintEvent(self, event):
        if not len(self.times):
            return
        painter = QPainter(self)
        levels = self.quality[self._pixel_samples()]
        edges = np.r_[0, np.flatnonzero(np.diff(levels)) + 1, len(levels)]
        for start, end in zip(edges, edges[1:]):
            painter.fillRect(int(start), 4, int(end-start), 16, QColor(QUALITY_COLORS[levels[start]]))
        painter.drawText(0, 38, f"{self.times[0]:g} 秒")
        last = f"{self.times[-1]:g} 秒"
        painter.drawText(self.width() - painter.fontMetrics().horizontalAdvance(last), 38, last)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and len(self.times):
            x = int(np.clip(event.position().x(), 0, self.width() - 1))
            self.sampleSelected.emit(int(self._pixel_samples()[x]))
            event.accept()
        else:
            super().mousePressEvent(event)
