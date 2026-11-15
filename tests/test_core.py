import numpy as np
import pytest

from miniscope_gui.core.preprocess import PreprocessParams, load_movie, run_preprocess
from miniscope_gui.core.workspace import Workspace, WorkspaceError
from miniscope_gui.io.ucla_v4 import MiniscopeSession

from fake_data import make_fake_v4


@pytest.fixture
def session(tmp_path):
    frames = make_fake_v4(tmp_path / "My_V4_Miniscope", drop_at=120)
    return MiniscopeSession.open(tmp_path / "My_V4_Miniscope"), frames


def test_open_and_read(session):
    s, frames = session
    assert s.n_frames == 250
    assert s.frames_per_file == [100, 100, 50]
    assert (s.height, s.width) == (64, 80)
    assert s.nominal_fps == 30
    # 2 dropped frames in 250 -> real average rate is a bit under 30 Hz
    assert 29.5 < s.fps < 30
    np.testing.assert_array_equal(s.get_frame(0), frames[0])
    np.testing.assert_array_equal(s.get_frame(150), frames[150])
    blk = s.read_block(95, 105)  # crosses a file boundary
    np.testing.assert_array_equal(blk, frames[95:105])


def test_timestamp_report(session):
    s, _ = session
    rep = s.timestamp_report()
    assert rep.n_dropped_gaps == 1
    assert rep.estimated_missing_frames == 2
    assert abs(rep.measured_fps - 30) < 1.5


def test_workspace_requires_empty(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    (d / "something.txt").write_text("x")
    with pytest.raises(WorkspaceError):
        Workspace.create_or_open(d)


def test_workspace_not_inside_source(tmp_path, session):
    s, _ = session
    with pytest.raises(WorkspaceError):
        Workspace.create_or_open(s.folder / "analysis", source_folder=s.folder)


def test_preprocess(tmp_path, session):
    s, frames = session
    ws = Workspace.create_or_open(tmp_path / "out", source_folder=s.folder)
    p = PreprocessParams(crop=(8, 4, 64, 56), spatial_bin=2, temporal_bin=3, chunk_frames=64)
    out = run_preprocess(s, ws, p)
    mov = load_movie(ws)
    assert mov.shape == (250 // 3, 28, 32)
    expected = frames[:3, 4:60, 8:72].astype(np.float32).reshape(3, 28, 2, 32, 2).mean(axis=(2, 4)).mean(0)
    assert np.abs(mov[0].astype(float) - expected).max() <= 1
    assert out["shape"] == [83, 28, 32]
    assert abs(ws.step("preprocess")["params"]["fps_out"] - s.fps / 3) < 1e-9
    # re-open workspace works, a different source is refused
    Workspace.create_or_open(tmp_path / "out", source_folder=s.folder)
    with pytest.raises(WorkspaceError):
        Workspace.create_or_open(tmp_path / "out", source_folder=tmp_path / "other")


def test_fps_uses_mean_interval(session):
    s, _ = session
    # V4-like jitter: intervals 33, 34, 34 ms -> median says 29.41 Hz, real is ~29.7 Hz
    d = np.tile([33.0, 34.0, 34.0], s.n_frames)[:s.n_frames - 1]
    s.timestamps_ms = np.concatenate([[0.0], np.cumsum(d)])
    assert abs(s.fps - 1000 / d[1:].mean()) < 1e-6
    assert abs(s.duration_s - (s.timestamps_ms[-1] / 1000 + 1 / s.fps)) < 1e-6


def test_fps_falls_back_to_metadata(session):
    s, _ = session
    s.timestamps_ms = None
    assert s.fps == 30  # nominal value from metaData.json


def test_preprocess_memory_is_bounded(tmp_path):
    import tracemalloc
    make_fake_v4(tmp_path / "v4", n_frames=600, frames_per_file=600, size=(400, 400))
    s = MiniscopeSession.open(tmp_path / "v4")
    ws = Workspace.create_or_open(tmp_path / "out", source_folder=s.folder)
    tracemalloc.start()
    run_preprocess(s, ws, PreprocessParams(spatial_bin=2, chunk_frames=50))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # the raw movie is 600*400*400 = 96 MB as uint8; memory must depend only
    # on chunk size (50 binned frames = 8 MB float32), not on movie length
    assert peak < 40e6, peak  # measured ~25 MB for both 300 and 900 frames
    assert load_movie(ws).shape == (600, 200, 200)


def test_preprocess_circle(tmp_path, session):
    s, frames = session
    ws = Workspace.create_or_open(tmp_path / "out", source_folder=s.folder)
    p = PreprocessParams(crop=(10, 4, 56, 56), circle=True, spatial_bin=2, chunk_frames=64)
    out = run_preprocess(s, ws, p)
    mov = load_movie(ws)
    assert mov.shape == (250, 28, 28)
    mask = np.load(ws.path(out["mask"]))
    assert mask.shape == (28, 28) and mask[14, 14] and not mask[0, 0]
    assert (mov[:, ~mask] == 0).all()
    expected = frames[0, 4:60, 10:66].astype(np.float32).reshape(28, 2, 28, 2).mean(axis=(1, 3))
    assert np.abs(mov[0][mask].astype(float) - expected[mask]).max() <= 1
    assert ws.step("preprocess")["params"]["circle"] is True
