"""Per-trial rows and replay from the study page."""

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ActivityDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option, index):
        super().paint(painter, option, index)
        image = index.data(Qt.UserRole)
        if image is None:
            return
        rect = option.rect.adjusted(1, 2, -1, -2)
        painter.drawImage(rect, image)
        fraction = index.data(Qt.UserRole + 1)
        painter.save()
        painter.setPen(QPen(QColor("#ffffff"), 1, Qt.DashLine))
        x = rect.left() + int(rect.width() * fraction)
        painter.drawLine(x, rect.top(), x, rect.bottom())
        painter.restore()


class TrialView(QWidget):
    replayRequested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["试次", "结果", "ΔF/F", "观测率"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().hide()
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setItemDelegateForColumn(2, ActivityDelegate(self.table))
        self.table.cellClicked.connect(self._activate)
        self.table.cellActivated.connect(self._activate)
        self.table.setMinimumHeight(190)
        self.details = QLabel()
        self.details.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in (self.summary, self.table, self.details):
            layout.addWidget(widget)
        self.rows = []

    def clear(self):
        self.rows = []
        self.table.setRowCount(0)
        self.summary.clear()
        self.details.clear()

    @staticmethod
    def percent(v):
        return "—" if v in (None, "") else f"{float(v):.0%}"

    def load(self, folder: Path):
        self.clear()
        params = json.loads((folder / "parameters.json").read_text("utf8"))
        cell = params["cell_id"]
        data = json.loads((folder / f"cell_{cell}.json").read_text("utf8"))
        self.rows = data["rows"]
        matrix = np.asarray(data["dff"], dtype=float)
        times = data["offset_s"]
        finite = matrix[np.isfinite(matrix)]
        low, high = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
        t = np.nan_to_num((matrix - low) / (high - low or 1.0)).clip(0, 1)
        # Common continuous scale across every row. Gray identifies missing bins.
        rgb = np.stack((33 + 208 * t, 54 + 151 * t, 112 - 58 * t), axis=-1).astype(
            np.uint8
        )
        rgb[~np.isfinite(matrix)] = (215, 218, 219)
        self.summary.setText(
            f"细胞 {cell} · {times[0]:g} 至 {times[-1]:g} 秒 · ΔF/F 色阶 {low:.3g} 至 {high:.3g}。点选一行可回放，虚线为对齐事件。"
        )
        zero_fraction = (np.argmin(np.abs(times)) + 0.5) / len(times)
        self.table.setRowCount(len(self.rows))
        for i, row in enumerate(self.rows):
            labels = [
                row["trial_id"],
                row["outcome"],
                "",
                self.percent(row["observed_fraction"]),
            ]
            for j, label in enumerate(labels):
                item = QTableWidgetItem(str(label))
                if j == 2:
                    pixels = rgb[i : i + 1]
                    image = QImage(
                        pixels.data,
                        pixels.shape[1],
                        1,
                        pixels.strides[0],
                        QImage.Format_RGB888,
                    )
                    item.setData(Qt.UserRole, image.copy())
                    item.setData(Qt.UserRole + 1, zero_fraction)
                self.table.setItem(i, j, item)

    def _activate(self, i, col):
        row = self.rows[i]
        self.details.setText(
            f"{row['animal_id']} · 试次 {row['trial_id']} · 神经覆盖 {self.percent(row['coverage'])} · 原始追踪 {self.percent(row['observed_fraction'])} · 插值 {self.percent(row['interpolated_fraction'])} · 无效 {self.percent(row['invalid_fraction'])} · 同步范围外 {self.percent(row['outside_tracking_fraction'])}"
        )
        event = row.get("event_s")
        if event not in (None, "") and np.isfinite(float(event)):
            self.replayRequested.emit(float(event))
