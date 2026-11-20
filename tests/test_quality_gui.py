"""Review the tracking mask on a real video, change the analysis limit and reopen."""
import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from miniscope_gui.core.behavior import BehaviorConfig, save_track
from miniscope_gui.core.workspace import Workspace
from miniscope_gui.gui.main_window import MainWindow
from test_behavior import make_behavior_video


def test_quality_click_setting_and_release(tmp_path):
    app = QApplication.instance() or QApplication([])
    path = tmp_path / "quality.avi"
    make_behavior_video(path, n=30, fps=10, size=(128, 160))
    ws = Workspace.create_or_open(tmp_path / "project")
    cfg = BehaviorConfig(video=str(path), fps=10)
    cfg.save(ws)
    x = np.arange(30, dtype=float)
    x[2] = np.nan
    x[8:18] = np.nan
    save_track(ws, t=np.arange(30) / 10, x=x, y=np.ones(30))
    ws.record_step("behavior", {}, {})
    w = MainWindow()
    page = w.beh_page
    page.set_context(None, ws)
    assert page.timeline.quality[2] == 1 and page.timeline.quality[8] == 0
    assert "33.3%" in page.quality_stats.text()
    point = int(np.flatnonzero(page.timeline._pixel_samples() == 9)[0])
    QTest.mouseClick(page.timeline, Qt.LeftButton, pos=QPoint(point, 10))
    assert w.view.slider.value() == 9 and w.view.timer.isActive()
    page.max_interp.setValue(0)
    assert page.timeline.quality[2] == 0
    assert BehaviorConfig.load(ws).params["max_interp_s"] == 0
    page.release_project()
    assert len(page.timeline.times) == 0 and page.timeline.isHidden()
    page.set_context(None, ws)
    assert page.max_interp.value() == 0 and page.timeline.quality[2] == 0
    page.release_project()
    other = Workspace.create_or_open(tmp_path / "empty")
    page.set_context(None, other)
    assert page.max_interp.value() == .5 and page.timeline.isHidden()
    w.close()
    app.processEvents()
