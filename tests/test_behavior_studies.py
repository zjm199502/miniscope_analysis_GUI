import json

import numpy as np
import pytest

from miniscope_gui.core import behavior_studies as studies
from miniscope_gui.core import study_report
from miniscope_gui.core.behavior import BehaviorConfig, Shape
from miniscope_gui.core.modeling.common import Dataset
from miniscope_gui.core.workspace import Workspace


def dataset(x, y, signals):
    signals = np.asarray(signals, float)
    t = np.arange(len(x), dtype=float)
    return Dataset(
        t,
        1,
        signals,
        signals,
        np.arange(len(signals)),
        None,
        (10, 10),
        dict(x=x, y=y, speed=np.ones(len(x)), valid=t >= 0),
    )


def arena(kind="rect", data=(0, 0, 10, 10)):
    return BehaviorConfig(arena=Shape(kind, list(data)).to_dict())


def test_freeze_alignment_and_event_holdout():
    t = np.arange(200)
    freeze = np.zeros(200, bool)
    speed = np.full(200, 4.0)
    for start in range(10, 185, 20):
        speed[start : start + 6] = 0
        if start % 40 == 10:
            freeze[start : start + 6] = True
    ds = dataset(t * 0, t * 0, [freeze, freeze])
    ds.beh.update(freezing=freeze, speed=speed)
    r = studies.transitions(ds, 1, 2)
    assert len(r["events"]) == 8
    np.testing.assert_array_equal(r["on_accuracy"], [0.5, 0.5, 1, 1, 1])
    np.testing.assert_array_equal(r["off_accuracy"], [1, 1, 0.5, 0.5, 0.5])
    assert r["on_trajectory"].shape == (2, 5, 2)


@pytest.mark.parametrize(
    "kind,data",
    [
        ("rect", [0, 0, 10, 10]),
        ("circle", [5, 5, 5]),
        ("polygon", [0, 0, 10, 0, 10, 10, 0, 10]),
    ],
)
def test_boundary_direction_and_occupancy(kind, data):
    x = np.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0, -1.0])
    ds = dataset(x, np.full_like(x, 5), [x, x])
    r = studies.boundary(ds, arena(kind, data), 3)
    assert r["all_seconds"].sum() == 6
    assert r["away_seconds"].sum() == r["toward_seconds"].sum() == 2


def test_relocation_separates_reference_frames():
    x = np.tile([2.5, 7.5], 20)
    y = np.full_like(x, 5)
    distance = abs(x - np.repeat([2.5, 7.5], 20))
    ds = dataset(x, y, [x, distance])
    r = studies.target(ds, arena(), [[0, 2.5, 5], [20, 7.5, 5]], 2)
    for epoch, cell, count, place, goal in r["prediction"]:
        assert count == 20
        assert place < goal if cell == 0 else goal < place
    assert not studies.target(ds, arena(), [[0, 2.5, 5]], 2)["prediction"]


def test_registered_order_and_common_support(tmp_path, monkeypatch):
    x = np.tile([2.5, 7.5], 20)
    y = np.full_like(x, 2.5)
    day = dataset(x, y, [x, 10 - x])
    rev = dataset(x, y, [10 - x, x])
    projects = [Workspace.create_or_open(tmp_path / name) for name in ("a", "b")]
    for ws in projects:
        arena().save(ws)
    ids = np.array([[0, 1], [1, 0]])
    monkeypatch.setattr(
        studies,
        "load_registration",
        lambda _: dict(
            cell_ids=ids,
            summary=dict(sessions=[w.root for w in projects], names=["a", "b"]),
        ),
    )
    monkeypatch.setattr(
        studies, "load_dataset", lambda ws, cfg: day if ws.root.name == "a" else rev
    )
    r = studies.crossday(tmp_path, 2)
    np.testing.assert_allclose(r["errors"], 0)
    assert all(row[-1] == 0 and row[3] == 40 for row in r["pairs"])
    ids[:, 1] = -1
    with pytest.raises(ValueError, match="共同细胞"):
        studies.crossday(tmp_path, 2)


def test_saved_report_and_project_reopen(tmp_path):
    from test_modeling import make_workspace

    ws, cfg, _ = make_workspace(tmp_path / "real", T=500, N=8)
    cfg.arena = arena("rect", [0, 0, 30, 30]).arena
    cfg.save(ws)
    args = dict(mode="wall", title="距墙调谐", bins=4)
    study_report.run_study(None, ws, args)
    reopened = Workspace.create_or_open(ws.root)
    folder = ws.path(reopened.step("study_wall")["outputs"]["folder"])
    assert (folder / "boundary.csv").is_file()
    assert len(json.loads((folder / "data.json").read_text("utf8"))["cells"]) == 8
    assert "__PAYLOAD__" not in (folder / "report.html").read_text("utf8")
