import numpy as np

from miniscope_gui.core.sync import (LedParams, led_alignment, manual_alignment, peri_event,
                                     speed_correlation, state_modulation)


def _led_trace(fps_b, n, start_f, interval_s, end_f, rng):
    tr = 20 + rng.normal(0, 1, n)
    tr[start_f:start_f + int(fps_b)] = 200                      # 1 s start mark
    k = 1
    while start_f + k * interval_s * fps_b + 6 < end_f:
        f = int(round(start_f + k * interval_s * fps_b))
        tr[f:f + int(0.2 * fps_b)] = 200                         # 200 ms flash
        k += 1
    tr[end_f:end_f + int(2 * fps_b)] = 200                       # 2 s end mark
    return tr


def test_led_alignment_time_mode():
    rng = np.random.default_rng(0)
    fps_true = 29.97                       # behavior camera, slightly off nominal
    n = int(fps_true * 130)
    start_f = 90                           # trigger 3 s into the behavior video
    rec_s = 120.0
    end_f = int(round(start_f + rec_s * fps_true))
    tr = _led_trace(fps_true, n, start_f, 10, end_f, rng)
    # miniscope: 20 Hz, first frame one frame after the trigger
    period = 50.0
    ts = 1000.0 + period + np.arange(int(rec_s * 20)) * period
    bm, anchors, rep = led_alignment(tr, 30.0, ts, None, LedParams(interval_s=10, mode="time"))
    assert rep["has_end_mark"] and rep["n_flashes"] >= 10
    # behavior frame at the trigger maps to the trigger time
    assert abs(bm[start_f] - (ts[0] - period)) < 20
    # a frame 60 s after the trigger maps to ~60 s later on the miniscope clock
    f60 = start_f + 60 * fps_true
    assert abs(np.interp(f60, np.arange(n), bm) - (ts[0] - period + 60000)) < 60  # flash onsets are rounded to whole frames here
    assert abs(rep["behavior_fps_measured"] - fps_true) < 0.05


def test_manual_alignment():
    bm = manual_alignment(100, 25.0, behavior_frame=50, miniscope_time_s=3.0)
    assert bm[50] == 3000.0 and abs(bm[75] - 4000.0) < 1e-9


def test_state_modulation_and_speed():
    rng = np.random.default_rng(1)
    T = 3000
    state = np.zeros(T, bool)
    for a in range(200, T, 600):
        state[a:a + 250] = True
    dff = rng.normal(0, 0.1, (3, T))
    dff[0, state] += 0.5          # freezing-up cell
    dff[1, state] -= 0.3          # freezing-down cell
    m = state_modulation(dff, state, np.ones(T, bool), n_shuffle=200)
    assert m["p_up"][0] < 0.05 and m["p_down"][1] < 0.05 and m["p_up"][2] > 0.05
    speed = np.abs(rng.normal(0, 1, T)).cumsum() % 10
    dff[2] += 0.2 * speed
    c = speed_correlation(dff, speed, np.ones(T, bool), n_shuffle=200)
    assert c["r"][2] > 0.5 and c["p"][2] < 0.05


def test_peri_event():
    t = np.arange(1000) / 10.0
    dff = np.zeros((2, 1000))
    for e in (20.0, 50.0, 80.0):
        dff[0, int(e * 10):int(e * 10) + 20] = 1.0
    axis, segs = peri_event(dff, t, [20.0, 50.0, 80.0, 99.5], pre_s=2, post_s=5)
    assert segs.shape == (3, 2, len(axis))
    assert segs[:, 0, axis >= 0][:, :20].mean() == 1.0 and segs[:, 0, axis < 0].mean() == 0
