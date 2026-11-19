"""Smoke tests: build the window and drive every calcium step through the GUI."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
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
    w.bg_page._estimate()          # fake data has only 3 cells: must fail gracefully
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
    # bulk review of low-confidence candidates
    w.ext_page.q_thr.setValue(101 if w.ext_page.q_thr.maximum() > 100 else 100)
    w._reject_low_quality()
    from miniscope_gui.core.extraction import load_cells
    assert not load_cells(w.ws)["accepted"][w.review.quality < 100].any()
    w._restore_cells()
    assert load_cells(w.ws)["accepted"].all()
    w.ext_page.q_thr.setValue(60)
    w.ext_page.by_quality.setChecked(True)
    w.review.selected = []
    w.review.step(1)
    assert w.review.selected == [int(np.argsort(w.review.quality, kind="stable")[0])]
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


def test_modeling_page(tmp_path, app):
    """Step 9: run every modelling analysis through the GUI and open every result view."""
    from miniscope_gui.core.modeling.report import ANALYSES, VIEWS
    from miniscope_gui.gui.main_window import MainWindow
    from test_modeling import make_workspace
    make_fake_v4(tmp_path / "My_V4_Miniscope", n_frames=60, frames_per_file=60, size=(64, 64))
    make_workspace(tmp_path / "out", T=2400)
    w = MainWindow()
    w.show()
    w.import_page.load_source(str(tmp_path / "My_V4_Miniscope"))
    w.import_page.set_output(str(tmp_path / "out"))
    assert w.ws.step("signals") and w.steps.item(8).flags() & Qt.ItemIsEnabled
    w.steps.setCurrentRow(8)
    mp = w.model_page
    assert "区域 left" in [mp.dec_target.itemText(i) for i in range(mp.dec_target.count())]
    mp.glm_events.setText("60, 150")
    mp.glm_shuf.setValue(20); mp.dec_shuf.setValue(20); mp.info_shuf.setValue(50)
    mp.emb_steps.setValue(200); mp.hmm_cmp.setChecked(True)
    for i, key in enumerate(ANALYSES):
        mp.kind.setCurrentIndex(i)
        assert mp.key == key and not mp.view_buttons[0].isEnabled()
        assert mp.forms[key].isVisibleTo(mp) and sum(f.isVisibleTo(mp) for f in mp.forms.values()) == 1
        mp._run(); _wait(app, w)
        assert w.ws.step(f"model_{key}"), key
        assert mp.result_lbl.text() and mp.view_buttons[0].isEnabled()
        for j, (view, _) in enumerate(VIEWS[key]):
            mp._show(j)
            assert w.view.title.text()
    # results survive switching analyses and steps; other views are not left with model overlays
    mp.kind.setCurrentIndex(0)
    assert "R²" in mp.result_lbl.text()
    w.show_result("net_graph")
    assert w.model_overlays and len(w.view._overlays) == 2
    w.steps.setCurrentRow(5)
    assert not w.view._overlays
    w.show_result("traces")
    w.close()


def test_modeling_page_without_behavior(tmp_path, app, monkeypatch):
    from miniscope_gui.gui.main_window import MainWindow
    from test_modeling import make_workspace
    make_fake_v4(tmp_path / "My_V4_Miniscope", n_frames=60, frames_per_file=60, size=(64, 64))
    make_workspace(tmp_path / "out", T=1500, behavior=False)
    w = MainWindow()
    w.import_page.load_source(str(tmp_path / "My_V4_Miniscope"))
    w.import_page.set_output(str(tmp_path / "out"))
    w.steps.setCurrentRow(8)
    mp = w.model_page
    said = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: said.append(a[2]))
    mp.kind.setCurrentIndex(1)                      # decoder needs behaviour
    assert "需要行为数据" in mp.need.text()
    mp._run()
    assert w.thread is None and said and "行为" in said[-1]
    for key in ("ensembles", "hmm", "network"):     # these do not
        mp.kind.setCurrentIndex(mp.keys.index(key))
        mp._run(); _wait(app, w)
        assert w.ws.step(f"model_{key}"), key
        for j in range(3):
            mp._show(j)
    mp.kind.setCurrentIndex(mp.keys.index("embedding"))
    mp.emb_steps.setValue(200)
    mp.emb_mode.setCurrentIndex(1)
    mp._run(); _wait(app, w)
    assert w.ws.step("model_embedding")
    mp._show(0); mp._show(1)
    w.close()


def test_registration_page(tmp_path, app):
    import numpy as np
    from miniscope_gui.gui.main_window import MainWindow
    from miniscope_gui.gui.registration_views import VIEWS
    from test_registration import _activity, _world, make_session
    cells = _world(seed=5, n=60)
    n = len(cells[0])
    ws1, _ = make_session(tmp_path / "d1", cells, np.arange(n), seed=1, T=1500, activity=_activity(1, T=1500))
    ws2, _ = make_session(tmp_path / "d2", cells, np.arange(5, n), angle=2, shift=(2, 3), seed=2, T=1500,
                          activity=_activity(2, T=1500))
    make_fake_v4(tmp_path / "My_V4_Miniscope", n_frames=60, frames_per_file=60, size=(64, 64))
    w = MainWindow()
    w.show()
    w.import_page.load_source(str(tmp_path / "My_V4_Miniscope"))
    w.import_page.set_output(str(ws1.root))
    assert w.steps.item(9).flags() & Qt.ItemIsEnabled
    w.steps.setCurrentRow(9)
    rp = w.reg_page
    assert rp.sessions() == [("Day1", str(ws1.root))]       # current session is added automatically
    assert rp.add_session(ws2.root) and not rp.add_session(ws2.root)
    rp.table.item(1, 0).setText("Day2 测试")
    rp.out_edit.setText(str(tmp_path / "reg"))
    rp._run(); _wait(app, w)
    assert rp.folder and rp.reg_lbl.text() and rp.reg_buttons[0].isEnabled()
    assert w.ws.step("registration")["outputs"]["folder"] == str(tmp_path / "reg")
    for key, _ in VIEWS["reg"]:
        rp._show(key)
        assert w.view.title.text()
    for i, kind in enumerate(("react", "stab")):
        rp.xd_kind.setCurrentIndex(i)
        rp.xd_shuf.setValue(50)
        rp._run_xd(); _wait(app, w)
        assert rp.xd_lbl.text() and rp.xd_buttons[0].isEnabled(), kind
        for key, _ in VIEWS[kind]:
            rp._show(key)
            assert w.view.title.text()
    # reopening the session finds the earlier registration
    w2 = MainWindow()
    w2.import_page.load_source(str(tmp_path / "My_V4_Miniscope"))
    w2.import_page.set_output(str(ws1.root))
    assert w2.reg_page.folder == str(tmp_path / "reg") and len(w2.reg_page.sessions()) == 2
    w2.close()
    w.steps.setCurrentRow(5)
    assert not w.view._overlays
    w.close()


@pytest.mark.parametrize("close", [False, True])
def test_release_running_project(tmp_path, app, close):
    import threading
    import time
    import weakref
    from miniscope_gui.gui.main_window import MainWindow
    from miniscope_gui.core.workspace import Workspace

    w = MainWindow()
    w.show()
    raw = tmp_path / "raw"
    make_fake_v4(raw, n_frames=30, size=(64, 64))
    w.import_page.load_source(str(raw))
    w.import_page.set_output(str(tmp_path / "one"))
    video = tmp_path / "behavior.avi"
    import shutil
    shutil.copyfile(raw / "0.avi", video)
    w.beh_page._open_video(str(video))
    capture = w.beh_page.video
    assert capture._cap.isOpened()
    active = threading.Event()
    ended = threading.Event()

    def task(session, ws, params, progress, should_stop):
        active.set()
        while not should_stop():
            time.sleep(0.005)
        ended.set()
        return None

    w._run_step(w.pre_page, task, {})
    assert active.wait(5)
    w.view.toggle_play()
    w.review.signals = {"test": np.ones(10000)}
    ref = weakref.ref(w.review.signals["test"])
    if close:
        w.close()
    else:
        other = Workspace.create_or_open(tmp_path / "two", raw)
        w._on_session(w.session, other)
    _wait(app, w)
    assert ended.is_set() and w.thread is None and w.worker is None
    assert ref() is None and capture._cap is None and not w.view.timer.isActive()
    video.rename(tmp_path / "released.avi")
    if close:
        assert not w.isVisible() and w.ws is None and w.view._get is None
    else:
        assert w.ws is other and w.beh_page.video is None and w.isVisible()
        w.pre_page._run()
        _wait(app, w)
        assert w.ws.step("preprocess")
        for path in ("one", "two"):
            w.import_page.set_output(str(tmp_path / path))
        assert w.ws.root == tmp_path / "two"
        w.close()
