"""Compute controls share an asynchronous probe and preserve step provenance."""

import os
from threading import Event, get_ident

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QMessageBox

from fake_data import make_fake_v4
from miniscope_gui.core.workspace import Workspace
from miniscope_gui.gui.main_window import MainWindow


def test_compute_settings_restore_and_reset(tmp_path):
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    ws = Workspace.create_or_open(tmp_path / "saved")
    new_ws = Workspace.create_or_open(tmp_path / "new")
    for page in (w.motion_page, w.bg_page, w.ext_page):
        ws.record_step(page.step_key, {"backend": "gpu", "device_id": 3},
                       {"compute_backend": "cpu", "compute_device": "CPU",
                        "fallback_reason": "test fallback"})
        page.set_context(None, ws)
        assert page.backend.currentData() == "cuda" and page.device.currentData() == 3
        assert page.params().device_id == 3 and page.params().backend == "cuda"
        assert "CPU" in page.compute_result.text() and "test fallback" in page.compute_result.text()
        page.backend.setCurrentIndex(page.backend.findData("cpu"))
        assert page.params().device_id is None and not page.device.isEnabled()
        page.set_context(None, new_ws)
        assert page.params().backend == "auto" and page.params().device_id is None
        assert page.compute_result.text() == ""
    w.close()
    app.processEvents()


def test_metal_and_cuda_zero_remain_distinct_on_refresh_and_reopen(tmp_path):
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    ws = Workspace.create_or_open(tmp_path / "saved")
    cards = [{"id": 0, "name": "Apple test", "kind": "metal",
              "free_bytes": None, "total_bytes": 16 * 2**30},
             {"id": 0, "name": "NVIDIA test", "kind": "cuda",
              "free_bytes": 4 * 2**30, "total_bytes": 8 * 2**30}]
    for page in (w.motion_page, w.bg_page, w.ext_page):
        page.show_devices({"devices": cards, "adapters": [c["name"] for c in cards]})
        assert "统一内存 16.0 GiB" in page.device_info.text()
        assert "可用显存 4.0 / 8.0 GiB" in page.device_info.text()
        page.device.setCurrentIndex(page.device.findText("Metal 0 · Apple test"))
        assert page.compute_params() == {"device_id": 0, "backend": "metal"}
        page.show_devices({"devices": list(reversed(cards))})
        assert page.compute_params() == {"device_id": 0, "backend": "metal"}
        ws.record_step(page.step_key, page.compute_params(),
                       {"compute_backend": "metal", "compute_device": "Apple test"})
        page.set_context(None, ws)
        assert page.backend.currentData() == "metal"
        assert page.device.count() == 2 and page.device.currentText() == "Metal 0 · Apple test"
        assert "Metal · Apple test" in page.compute_result.text()
        ws.record_step(page.step_key, page.compute_params(),
                       {"compute_backend": "cpu", "compute_device": "CPU", "fallback_reason": "Metal 自检失败"})
        page.update_status()
        assert "Metal 自检失败" in page.compute_result.text()
    w.close()
    app.processEvents()


@pytest.mark.parametrize("switch", [False, True])
def test_device_probe_worker(tmp_path, monkeypatch, switch):
    from miniscope_gui.core import gpu

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: pytest.fail(str(a)))
    started, release = Event(), Event()
    thread_ids = []

    def probe(refresh):
        assert refresh
        thread_ids.append(get_ident())
        started.set()
        assert release.wait(5)
        return {"adapters": ["Test Card"], "cuda_devices": [
            {"id": 3, "name": "Test Card", "free_bytes": 2**30, "total_bytes": 2**31}],
                "detail": "CUDA self-check passed"}

    monkeypatch.setattr(gpu, "detect_gpus", probe)
    raw = tmp_path / "raw"
    make_fake_v4(raw, n_frames=30, size=(64, 64))
    w = MainWindow()
    w.import_page.load_source(str(raw))
    w.import_page.set_output(str(tmp_path / "one"))
    first = w.ws
    w.motion_page.detect.click()
    assert started.wait(5) and thread_ids != [get_ident()]
    assert not w.motion_page.detect.isEnabled()
    if switch:
        other = Workspace.create_or_open(tmp_path / "two", raw)
        w._on_session(w.session, other)
    release.set()
    assert w.thread.wait(5000)
    for _ in range(20):
        app.processEvents()
    assert w.thread is None and first.step("motion") is None
    for page in (w.motion_page, w.bg_page, w.ext_page):
        if switch:
            assert page.ws is other and not page.device_info.text()
        else:
            assert "Test Card" in page.device_info.text()
            assert page.device.findData(3) >= 0
    w.close()



def test_auto_detection_does_not_occupy_video_worker(tmp_path, monkeypatch):
    from miniscope_gui.core import gpu
    from synth import workspace_from_movie
    import numpy as np

    app = QApplication.instance() or QApplication([])
    called, release = Event(), Event()

    def probe(refresh):
        assert not refresh
        called.set()
        assert release.wait(10)
        return {"adapters": ["CPU-only test adapter"], "cuda_devices": [], "detail": "CPU"}

    monkeypatch.setattr(gpu, "detect_gpus", probe)
    raw = tmp_path / "raw"
    make_fake_v4(raw, n_frames=30, size=(64, 64))
    ws = workspace_from_movie(tmp_path / "out", np.ones((30, 64, 64), np.float16))
    w = MainWindow()
    try:
        w.import_page.load_source(str(raw))
        w.import_page.set_output(str(ws.root))
        w.steps.setCurrentRow(2)
        assert called.wait(5) and w.thread is None
        w.motion_page.backend.setCurrentIndex(w.motion_page.backend.findData("cpu"))
        w.motion_page.run_btn.click()
        assert w.thread and w.thread.wait(10000)
        for _ in range(30):
            app.processEvents()
        assert w.ws.step("motion")["outputs"]["compute_backend"] == "cpu"
    finally:
        release.set()
        if w._probe_thread:
            assert w._probe_thread.wait(10000)
        for _ in range(30):
            app.processEvents()
        w.close()
    assert "CPU-only test adapter" in w.bg_page.device_info.text()


def test_extraction_click_run_and_review_after_gpu_fallback(tmp_path, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from miniscope_gui.core import gpu
    from miniscope_gui.core.background import BackgroundParams, run_background
    from synth import make_movie, workspace_from_movie

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: pytest.fail(str(a)))
    monkeypatch.setattr(gpu, "detect_gpus", lambda refresh=False: {
        "adapters": [], "devices": [], "detail": "测试设备未通过自检"})
    raw = tmp_path / "raw"
    make_fake_v4(raw, n_frames=30, size=(64, 64))
    movie, _ = make_movie(T=250, H=64, W=64, n_cells=4, seed=2)
    ws = workspace_from_movie(tmp_path / "out", movie)
    run_background(None, ws, BackgroundParams(source="preprocess", backend="cpu"))
    w = MainWindow()
    w.show()
    try:
        w.import_page.load_source(str(raw))
        w.import_page.set_output(str(ws.root))
        w.steps.setCurrentRow(4)
        page = w.ext_page
        page.min_corr.setValue(.6)
        page.min_pnr.setValue(4)
        page.backend.setCurrentIndex(page.backend.findData("metal"))
        QTest.mouseClick(page.run_btn, Qt.LeftButton)
        assert w.thread and w.thread.wait(20000)
        if w._probe_thread:
            assert w._probe_thread.wait(5000)
        for _ in range(30):
            app.processEvents()
        step = w.ws.step("extraction")
        assert step["params"]["backend"] == "metal" and step["outputs"]["compute_backend"] == "cpu"
        assert page.compute_result.isVisible()
        assert "CPU" in page.compute_result.text() and "测试设备未通过自检" in page.compute_result.text()
        assert w.review.active and len(w.review.curves) >= 2
        selected = list(w.review.selected)
        QTest.mouseClick(page.next_btn, Qt.LeftButton)
        assert w.review.selected != selected
        cell = w.review.selected[0]
        accepted = bool(w.review.cells["accepted"][cell])
        QTest.mouseClick(page.toggle_btn, Qt.LeftButton)
        assert bool(w.review.cells["accepted"][cell]) != accepted
    finally:
        w.close()
