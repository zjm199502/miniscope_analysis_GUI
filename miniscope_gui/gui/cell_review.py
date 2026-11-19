"""Interactive cell review on the central image + trace panel."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt

from ..core.extraction import contours, load_cells, suspect_cells

GREEN = (60, 220, 90)
RED = (235, 70, 70)
SEL = (255, 230, 0)
ORANGE = (255, 150, 30)
CYAN = (60, 200, 255)     # low-confidence candidate


class CellReview(QObject):
    def __init__(self, view, trace_plot, parent=None):
        super().__init__(parent)
        self.view = view
        self.plot = trace_plot
        self.ws = None
        self.cells = None
        self.suspect = np.zeros(0, bool)
        self.quality = np.zeros(0)
        self.threshold = 60.0      # quality score (0-100) below which a candidate is "low confidence"
        self.by_quality = False    # previous/next walk from the lowest score up
        self.signals = None
        self.curves = []
        self.selected: list[int] = []
        self.active = False
        view.vb.scene().sigMouseClicked.connect(self._clicked)

    def open(self, ws, background: np.ndarray, title: str):
        self.ws = ws
        self.cells = load_cells(ws)
        st = ws.step("extraction") or {}
        diam = float(st.get("params", {}).get("cell_diameter", 10))
        self.suspect = suspect_cells(self.cells["A"], self.cells["shape"], diam)
        from ..core.review import quality_of
        self.quality = quality_of(self.cells)
        from ..core.signals import load_signals
        self.signals = load_signals(ws) if ws.path("signals.npz").exists() else None
        self.view.set_image(background, title)
        self.view.clear_overlays()
        self.curves = []
        for c in contours(self.cells["A"], self.cells["shape"]):
            item = pg.PlotCurveItem(c[:, 0], c[:, 1])
            self.view.add_overlay(item)
            self.curves.append(item)
        self.active = True
        if self.selected and max(self.selected) >= len(self.curves):
            self.selected = []
        if not self.selected and len(self.curves):
            self.selected = [0]
        self._redraw()

    def close(self):
        self.active = False
        self.view.clear_overlays()
        self.curves = []

    def release_project(self):
        self.close()
        self.ws = self.cells = self.signals = None
        self.selected.clear()
        self.quality = np.empty(0)
        self.suspect = np.empty(0, bool)

    def reload(self):
        if self.active and self.ws is not None:
            img = self.view.img.image
            self.open(self.ws, img, self.view.title.text())

    # ----------------------------------------------------------- selection
    def _clicked(self, ev):
        if not self.active or self.cells is None or ev.button() != Qt.LeftButton:
            return
        p = self.view.vb.mapSceneToView(ev.scenePos())
        cen = self.cells["centres"]
        if len(cen) == 0:
            return
        d = np.hypot(cen[:, 0] - p.y(), cen[:, 1] - p.x())
        k = int(np.argmin(d))
        if d[k] > 15:
            return
        if ev.modifiers() & Qt.ControlModifier:
            if k in self.selected:
                self.selected.remove(k)
            else:
                self.selected.append(k)
        else:
            self.selected = [k]
        self._redraw()

    def order(self) -> np.ndarray:
        n = len(self.cells["C"])
        return np.argsort(self.quality, kind="stable") if self.by_quality else np.arange(n)

    def step(self, delta):
        if self.cells is None or not len(self.cells["C"]):
            return
        order = self.order()
        if self.selected:
            pos = int(np.flatnonzero(order == self.selected[-1])[0]) + delta
        else:
            pos = 0 if delta > 0 else -1
        self.selected = [int(order[pos % len(order)])]
        self._redraw()

    def set_threshold(self, value: float):
        self.threshold = float(value)
        if self.active:
            self._redraw()

    def low_quality(self) -> np.ndarray:
        return self.quality < self.threshold

    # ------------------------------------------------------------- drawing
    def _redraw(self):
        acc = self.cells["accepted"].astype(bool)
        low = self.low_quality()
        for k, item in enumerate(self.curves):
            if k in self.selected:
                item.setPen(pg.mkPen(SEL, width=3))
            elif not acc[k]:
                item.setPen(pg.mkPen(RED, width=1.5))
            elif self.suspect[k]:
                item.setPen(pg.mkPen(ORANGE, width=1.5))
            elif low[k]:
                item.setPen(pg.mkPen(CYAN, width=1.5, style=Qt.DashLine))
            else:
                item.setPen(pg.mkPen(GREEN, width=1.5))
        self._plot_traces()

    def _plot_traces(self):
        self.plot.clear()
        if not self.selected:
            return
        fps = float(self.signals["fps"]) if self.signals is not None else None
        T = self.cells["C"].shape[1]
        t = self.signals["times"] if self.signals is not None else np.arange(T)
        self.plot.setLabel("bottom", "时间 (秒)" if self.signals is not None else "帧")
        offset = 0.0
        for k in self.selected:
            if self.signals is not None:
                y = self.signals["dff"][k]
                d = self.signals["denoised"][k]
                s = self.signals["spikes"][k]
                self.plot.plot(t, y + offset, pen=pg.mkPen((150, 150, 150), width=1))
                self.plot.plot(t, d + offset, pen=pg.mkPen((40, 110, 220), width=1.5))
                ev = np.where(s > 0)[0]
                if len(ev):
                    top = offset + float(np.nanmax(y)) * 1.05 if np.nanmax(y) > 0 else offset + 0.1
                    self.plot.plot(t[ev], np.full(len(ev), top), pen=None, symbol="|",
                                   symbolSize=8, symbolPen=pg.mkPen((220, 60, 60)))
                span = float(np.nanmax(y) - np.nanmin(y)) or 1.0
            else:
                y = self.cells["raw"][k]
                c = self.cells["C"][k]
                self.plot.plot(t, y + offset, pen=pg.mkPen((150, 150, 150), width=1))
                self.plot.plot(t, c + offset, pen=pg.mkPen((40, 110, 220), width=1.5))
                span = float(np.nanmax(y) - np.nanmin(y)) or 1.0
            offset += span * 1.15
        acc = self.cells["accepted"]
        names = ", ".join(f"#{k}（质量 {self.quality[k]:.0f}{'' if acc[k] else '，已删除'}）"
                          for k in self.selected)
        unit = "ΔF/F" if self.signals is not None else "荧光"
        self.plot.setTitle(f"细胞 {names}  ·  灰色 = {unit}，蓝色 = 去噪"
                           + ("，红色竖线 = spike" if self.signals is not None else ""))

    def summary(self) -> str:
        if self.cells is None:
            return ""
        acc = self.cells["accepted"].astype(bool)
        txt = f"共 {len(acc)} 个细胞，保留 {acc.sum()} 个，删除 {(~acc).sum()} 个。"
        n_s = int((self.suspect & acc).sum())
        if n_s:
            txt += f"\n橙色 {n_s} 个形状可疑（过大或过长），建议逐个看一下。"
        n_low = int((self.low_quality() & acc & ~self.suspect).sum())
        if n_low:
            txt += (f"\n蓝色虚线 {n_low} 个是低置信候选（质量分 < {self.threshold:.0f}），"
                    "可以逐个看，也可以一键删除。")
        return txt
