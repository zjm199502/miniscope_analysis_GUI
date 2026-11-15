"""Smoke tests: build the window and drive every calcium step through the GUI."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox

from fake_data import make_fake_v4


def _wait(app, w):
    th = w.thread
    if th is not None:
        th.wait(60000)
    for _ in range(50):
        app.processEvents()


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: pytest.fail(str(a)))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    return QApplication.instance() or QApplication([])


def test_gui_flow(tmp_path, app):
    from miniscope_gui.gui.main_window import MainWindow
    make_fake_v4(tmp_path / "My_V4_Miniscope", n_frames=600, frames_per_file=200, size=(96, 96))
    w = MainWindow()
    w.show()  # must not be shadowed
    w.import_page.load_source(str(tmp_path / "My_V4_Miniscope"))
    w.import_page.set_output(str(tmp_path / "out"))
    assert w.session is not None and w.ws is not None

    w.steps.setCurrentRow(1)
    w.pre_page.crop_cb.setChecked(True)
    w.pre_page.sbin.setValue(1)
    w.pre_page._run(); _wait(app, w)
    assert w.ws.step("preprocess") is not None
    assert np.load(w.ws.path("fov_mask.npy")).any()
    for key in ("movie", "mean", "max", "raw"):
        w.show_result(key)

    w.steps.setCurrentRow(2)
    w.motion_page._run(); _wait(app, w)
    assert w.ws.step("motion")
    for key in ("mc", "mc_compare", "mc_shifts", "mc_mean"):
        w.show_result(key)

    w.steps.setCurrentRow(3)
    w.bg_page.diam.setValue(8)
    w.bg_page._run(); _wait(app, w)
    assert w.ws.step("background")

    w.steps.setCurrentRow(4)
    w.ext_page.min_corr.setValue(0.6)
    w.ext_page.min_pnr.setValue(4)
    w.ext_page._run(); _wait(app, w)
    assert w.ws.step("extraction")
    w.show_result("cells")
    assert w.review.active and len(w.review.curves) > 0
    w.review.step(1)
    w._toggle_cells()
    assert w.review.summary()
    w.show_result("cn"); w.show_result("pnr")

    w.steps.setCurrentRow(5)
    w.sig_page._run(); _wait(app, w)
    assert w.ws.step("signals")
    w.show_result("traces")
    w.show_result("cells")
    w.view.slider.setValue(3)

    # behavior: tracking with arena, zone and scale
    from test_behavior import make_behavior_video
    from miniscope_gui.core.behavior import Shape
    make_behavior_video(tmp_path / "beh.avi", n=200)
    w.steps.setCurrentRow(6)
    bp = w.beh_page
    bp._open_video(str(tmp_path / "beh.avi"))
    bp.start_edit("arena", "rect"); bp.finish_edit()
    bp.start_edit("scale", "line"); bp.finish_edit()
    bp.cfg.zones.append(Shape("rect", [0, 0, 160, 240], "left").to_dict())
    bp._refresh_lists(); bp._draw_shapes()
    bp.proc_scale.setCurrentIndex(1)
    bp._run(); _wait(app, w)
    assert w.ws.step("behavior")
    assert bp.cfg.px_per_cm and bp.cfg.arena
    for key in ("beh_track", "beh_traces", "beh_video"):
        w.show_result(key)
    assert "僵直" in bp.summary.text()

    # sync (manual) + joint analysis
    w.steps.setCurrentRow(7)
    sp = w.sync_page
    sp.m_manual.setChecked(True)
    sp.b_frame.setValue(0); sp.m_time.setValue(0.0)
    sp._run(); _wait(app, w)
    assert w.ws.step("sync")
    w.show_result("sync_joint")
    w.view.slider.setValue(10)
    w.show_result("sync_qc")
    sp.event_src.setCurrentText("手动输入时间")
    sp.event_times.setText("5, 10")
    sp.pre.setValue(1); sp.post.setValue(2)
    for kind in ("freezing", "speed", "peth"):
        w.joint_analysis(kind)
    w.close()
