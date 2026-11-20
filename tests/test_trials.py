"""Trial identity, original calcium timestamps and tracking quality stay aligned."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from miniscope_gui.core import trials
from miniscope_gui.core.extraction import CELLS_FILE
from miniscope_gui.core.signals import SIGNALS_FILE
from miniscope_gui.core.workspace import Workspace


def row(trial_id, cue, outcome=None):
    return dict(
        animal_id="鼠A", trial_id=trial_id, cue_s=cue, outcome_s=outcome, outcome="成功"
    )


def test_window_preserves_csv_order_and_true_quality():
    rows = [row("later", 5), row("early", 1), row("middle", 3), row("absent", None)]
    times = np.arange(7.0)
    tracking = {
        "tracking_quality": np.array([2, 1, 0, 2, 1, 0, 2]),
        "behavior_frame": np.array([-1, 1, 2, 3, 4, 5, 6]),
    }
    result = trials.aligned_trials(
        rows, times, times, 4, "cue", 1, 1, tracking, order="csv"
    )
    assert result["trial_ids"] == ["later", "early", "middle", "absent"]
    np.testing.assert_equal(result["early_mean"], [0, 1, 2])
    np.testing.assert_equal(result["late_mean"], [4, 5, 6])
    a, b, c, d = result["rows"]
    assert (
        b["interpolated_fraction"]
        == b["invalid_fraction"]
        == b["outside_tracking_fraction"]
        == 1 / 3
    )
    assert (
        a["observed_fraction"]
        == a["interpolated_fraction"]
        == a["invalid_fraction"]
        == 1 / 3
    )
    assert a["delta_dff"] == 1.5 and c["phase"] == "middle"
    assert d["status"] == "missing_event" and d["observed_fraction"] is None


def test_missing_signal_and_tracking_do_not_become_measurements():
    times = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
    rows = [row("one", 0, 0), row("two", 1, 3), row("three", 2)]
    result = trials.aligned_trials(rows, times, times, 0, "outcome", 1.0, 1.0)
    assert [r["status"] for r in result["rows"]] == [
        "partial_window",
        "partial_window",
        "missing_event",
    ]
    assert np.isnan(result["dff"][1, 1:]).all()
    assert all(r["observed_fraction"] is None for r in result["rows"])
    assert result["early_n"] == result["late_n"] == 0


def test_selected_cell_exports_ids_and_reopen(tmp_path):
    ws = Workspace.create_or_open(tmp_path / "run")
    np.savez(ws.path(CELLS_FILE), accepted=[False, True])
    np.savez(
        ws.path(SIGNALS_FILE),
        times=np.arange(11.0),
        dff=[np.zeros(11), np.arange(11.0)],
    )
    ws.record_step("signals", {}, {})
    source = tmp_path / "trials.csv"
    source.write_text(
        "animal_id,trial_id,cue_s,outcome_s,outcome\n鼠A,007,1,3,成功\n鼠A,015,4,,遗漏\n",
        encoding="utf-8-sig",
    )
    args = dict(csv_path=str(source), align="outcome", pre_s=1.0, post_s=1.0, cell_id=1)
    folder = Path(trials.run_trials(None, ws, args))
    result = json.loads((folder / "cell_1.json").read_text("utf8"))
    assert result["dff"] == [[2, 3, 4], [None, None, None]]
    assert result["rows"][1]["event_s"] is None
    assert (
        ws.path(
            Workspace.create_or_open(ws.root).step("study_trials")["outputs"]["folder"]
        )
        == folder
    )
    with (folder / "cell_1.csv").open(encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))
    assert records[0]["trial_id"] == "007" and records[0]["animal_id"] == "鼠A"
    assert len(records) == 6 and not records[-1]["dff"]
    html = (folder / "report.html").read_text("utf8")
    assert "__SCRIPT__" not in html and "trial_rows" in html
    with pytest.raises(ValueError, match="保留"):
        trials.run_trials(None, ws, dict(args, cell_id=0))
    source.write_text(
        "animal_id,trial_id,cue_s,outcome_s,outcome\nA,1,1,2,X\nB,1,1,2,X\n"
    )
    with pytest.raises(ValueError, match="一只动物"):
        trials.read_trials(source)


def test_real_sync_separates_tracking_from_neural_coverage(tmp_path):
    from miniscope_gui.core.behavior import BehaviorConfig, save_track
    from miniscope_gui.core.sync import save_sync

    ws = Workspace.create_or_open(tmp_path / "session")
    times = np.arange(20) * 0.1
    np.savez(ws.path(CELLS_FILE), accepted=[True])
    np.savez(ws.path(SIGNALS_FILE), times=times, dff=[times])
    ws.record_step("signals", {}, {})
    cfg = BehaviorConfig(fps=10)
    cfg.params["max_interp_s"] = 0.11
    cfg.save(ws)
    positions = np.array([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
    save_track(ws, t=times[:6], x=positions, y=positions)
    ws.record_step("behavior", {}, {})
    save_sync(ws, 1000 + np.arange(6) * 100, "manual", {})
    source = tmp_path / "trials.csv"
    source.write_text("animal_id,trial_id,cue_s,outcome_s,outcome\nA,hit,1.2,1.5,hit\n")
    folder = Path(
        trials.run_trials(
            None, ws, dict(csv_path=str(source), align="cue", pre_s=0.3, post_s=0.3)
        )
    )
    row = json.loads((folder / "cell_0.json").read_text("utf8"))["rows"][0]
    assert row["coverage"] == 1 and row["status"] == "complete"
    np.testing.assert_allclose(
        [row[key] for key in trials.QUALITY_FIELDS], np.array([3, 1, 2, 1]) / 7
    )


def test_default_time_order_preserves_original_row_id_and_missing_rows():
    rows = [
        row("late", 5, 6),
        row("both-missing", None),
        row("cue-missing", None, 2),
        row("early", 1),
    ]
    times = np.arange(8.0)
    result = trials.aligned_trials(rows, times, times, 0, "outcome", 1.0, 1.0)
    assert result["trial_ids"] == ["early", "cue-missing", "late", "both-missing"]
    assert [r["source_row"] for r in result["rows"]] == [5, 4, 2, 3]
    assert [r["row"] for r in result["rows"]] == [1, 2, 3, 4]
    assert result["rows"][-1]["status"] == "missing_event"
    ordered = trials.aligned_trials(
        rows, times, times, 0, "outcome", 1.0, 1.0, order="csv"
    )
    assert [r["source_row"] for r in ordered["rows"]] == [2, 3, 4, 5]
    np.testing.assert_equal(ordered["early_mean"], result["early_mean"])
