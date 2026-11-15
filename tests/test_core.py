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
    assert s.fps == 30
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
    assert ws.step("preprocess")["params"]["fps_out"] == 10
    # re-open workspace works, a different source is refused
    Workspace.create_or_open(tmp_path / "out", source_folder=s.folder)
    with pytest.raises(WorkspaceError):
        Workspace.create_or_open(tmp_path / "out", source_folder=tmp_path / "other")
