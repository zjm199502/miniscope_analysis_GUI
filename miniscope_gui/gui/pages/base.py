"""Common layout for a processing step: explanation, parameters, run button, results."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QGridLayout, QLabel, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from .common import heading, hint

RUN_STYLE = ("QPushButton{font-weight:bold; padding:8px; color:white; background:#2b6cb0;"
             " border:none; border-radius:4px;} QPushButton:disabled{background:#a0aec0;}")


class StepPage(QWidget):
    runRequested = Signal(object, object)   # fn, params
    showRequested = Signal(str)             # view key, handled by the main window

    step_key = ""          # name used in project.json
    title = ""
    intro = ""
    run_label = "运行"
    results: tuple = ()     # ((view key, button label), ...)
    results_hint = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session = None
        self.ws = None
        self.lay = QVBoxLayout(self)
        self.lay.addWidget(heading(self.title))
        if self.intro:
            self.lay.addWidget(hint(self.intro))
        self.form = QFormLayout()
        self.form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.lay.addLayout(self.form)
        self.build_params()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.lay.addWidget(self.status)
        self.run_btn = QPushButton(self.run_label)
        self.run_btn.setStyleSheet(RUN_STYLE)
        self.run_btn.clicked.connect(self._run)
        self.lay.addWidget(self.run_btn)
        if self.results:
            self.lay.addWidget(heading("查看结果"))
            grid = QGridLayout()
            self.result_buttons = []
            for i, (key, label) in enumerate(self.results):
                b = QPushButton(label)
                b.clicked.connect(lambda _=False, k=key: self.showRequested.emit(k))
                grid.addWidget(b, i // 2, i % 2)
                self.result_buttons.append(b)
            self.lay.addLayout(grid)
            if self.results_hint:
                self.lay.addWidget(hint(self.results_hint))
        self.build_extra()
        self.lay.addStretch(1)

    # ---------------------------------------------------- for subclasses
    def build_params(self):
        pass

    def build_extra(self):
        pass

    def params(self):
        raise NotImplementedError

    def run_function(self):
        raise NotImplementedError

    def default_view(self) -> str | None:
        return self.results[0][0] if self.results else None

    def on_context(self):
        """Called when session/workspace changes."""

    def on_finished(self, result):
        """Called after a successful run."""

    # ------------------------------------------------------------ helpers
    def add_param(self, label, widget, help_text=None):
        self.form.addRow(label, widget)
        if help_text:
            self.form.addRow(hint(help_text))
        return widget

    @staticmethod
    def spin(lo, hi, val, step=1, suffix=""):
        w = QSpinBox()
        w.setRange(lo, hi); w.setValue(val); w.setSingleStep(step)
        if suffix:
            w.setSuffix(suffix)
        return w

    @staticmethod
    def dspin(lo, hi, val, step=0.1, decimals=2, suffix=""):
        w = QDoubleSpinBox()
        w.setRange(lo, hi); w.setDecimals(decimals); w.setValue(val); w.setSingleStep(step)
        if suffix:
            w.setSuffix(suffix)
        return w

    @staticmethod
    def check(text, val=False):
        w = QCheckBox(text)
        w.setChecked(val)
        return w

    @staticmethod
    def combo(items, current=None):
        w = QComboBox()
        w.addItems(items)
        if current:
            w.setCurrentText(current)
        return w

    # ------------------------------------------------------------- plumbing
    def set_context(self, session, ws):
        self.session, self.ws = session, ws
        self.update_status()
        self.on_context()

    def update_status(self):
        st = self.ws.step(self.step_key) if (self.ws and self.step_key) else None
        self.status.setText(f"✓ 已完成（{st['finished']}）" if st else "")
        if self.results:
            for b in self.result_buttons:
                b.setEnabled(bool(st))

    def set_running(self, running: bool):
        self.run_btn.setEnabled(not running)

    def _run(self):
        if self.ws is not None:
            self.runRequested.emit(self.run_function(), self.params())
