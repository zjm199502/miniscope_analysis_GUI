"""Smoke test: build the window, load a fake session, run preprocessing."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from fake_data import make_fake_v4


def test_gui_flow(tmp_path, monkeypatch):
    from miniscope_gui.gui.main_window import MainWindow
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail(str(a)))
    app = QApplication.instance() or QApplication([])
    make_fake_v4(tmp_path / "My_V4_Miniscope")
    w = MainWindow()
    w.import_page.load_source(str(tmp_path / "My_V4_Miniscope"))
    w.import_page.set_output(str(tmp_path / "out"))
    assert w.session is not None and w.ws is not None
    w.pre_page.crop_cb.setChecked(True)
    w.pre_page.tbin.setValue(2)
    w.pre_page._run()
    thread = w.thread
    assert thread is not None
    thread.wait(20000)
    for _ in range(50):
        app.processEvents()
    assert w.ws.step("preprocess") is not None
    w._show_pre_result("movie")
    w._show_pre_result("mean")
    w.view.slider.setValue(3)
    w.close()
