"""Single-cell trial-aligned calcium responses and synchronized tracking coverage."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np

from .behavior import BehaviorConfig, TrackingParams
from .extraction import CELLS_FILE
from .modeling.common import check_stop
from .signals import SIGNALS_FILE
from .study_report import Report, TEMPLATE, plain
from .sync import behavior_on_calcium_frames


REQUIRED = ("animal_id", "trial_id", "cue_s", "outcome_s", "outcome")
QUALITY_FIELDS = (
    "observed_fraction",
    "interpolated_fraction",
    "invalid_fraction",
    "outside_tracking_fraction",
)


def read_trials(path):
    """One session clock per CSV. Empty event times represent absent events."""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if not set(REQUIRED).issubset(fields) or len(set(fields)) != len(fields):
            raise ValueError(
                "试次 CSV 需要唯一列名 animal_id,trial_id,cue_s,outcome_s,outcome"
            )
        rows = []
        seen = set()
        for row in reader:
            values = {k: (row[k] or "").strip() for k in REQUIRED}
            if not values["animal_id"] or not values["trial_id"]:
                raise ValueError(f"CSV 第 {reader.line_num} 行缺少动物或试次编号")
            ident = values["animal_id"], values["trial_id"]
            if ident in seen:
                raise ValueError(f"重复的动物与试次编号：{ident}")
            seen.add(ident)
            for key in ("cue_s", "outcome_s"):
                try:
                    t = float(values[key]) if values[key] else None
                    if t is not None and not np.isfinite(t):
                        raise ValueError
                except ValueError:
                    raise ValueError(
                        f"CSV 第 {reader.line_num} 行的 {key} 需要有限秒数或留空"
                    ) from None
                values[key] = t
            if values["cue_s"] is not None and values["outcome_s"] is not None:
                if values["outcome_s"] < values["cue_s"]:
                    raise ValueError(f"CSV 第 {reader.line_num} 行的结果时间早于线索")
            values["source_row"] = reader.line_num
            rows.append(values)
    if not rows:
        raise ValueError("试次表为空")
    if len({r["animal_id"] for r in rows}) != 1:
        raise ValueError("一个记录项目只对应一只动物，请按动物和记录分别导入试次")
    return rows


def _chronological_event(row):
    return row["cue_s"] if row["cue_s"] is not None else row["outcome_s"]


def _phases(rows):
    # Assign phase before applying response/quality inclusion criteria.
    phases = ["unknown"] * len(rows)
    ordered = [
        (_chronological_event(r), i)
        for i, r in enumerate(rows)
        if r["cue_s"] is not None or r["outcome_s"] is not None
    ]
    ordered.sort()
    half = len(ordered) // 2
    for j, (_, i) in enumerate(ordered):
        phases[i] = (
            "early" if j < half else "late" if j >= len(ordered) - half else "middle"
        )
    return phases


def _nearest(times, desired, dt):
    right = np.searchsorted(times, desired).clip(0, len(times) - 1)
    left = (right - 1).clip(0)
    idx = np.where(
        abs(times[left] - desired) <= abs(times[right] - desired), left, right
    )
    eps = 1e-8 * dt
    covered = (
        (desired >= times[0] - eps)
        & (desired <= times[-1] + eps)
        & (abs(times[idx] - desired) <= dt / 2 + eps)
    )
    return idx, covered


def aligned_trials(
    rows,
    times,
    trace,
    cell_id,
    align,
    pre_s,
    post_s,
    behavior=None,
    should_stop=None,
    order="time",
):
    if order not in ("time", "csv"):
        raise ValueError("试次顺序须选择事件时间或原始 CSV 顺序")
    rows = [dict(r, source_row=r.get("source_row", i + 2)) for i, r in enumerate(rows)]
    if order == "time":
        rows.sort(
            key=lambda r: (_chronological_event(r) is None, _chronological_event(r))
        )
    if align not in ("cue", "outcome"):
        raise ValueError("对齐事件须选择线索或结果")
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("钙信号时间戳不足或未严格递增")
    if not np.isfinite([pre_s, post_s]).all() or min(pre_s, post_s) <= 0:
        raise ValueError("前后窗口必须是正数")
    dt = float(np.median(np.diff(times)))
    npre, npost = np.floor(np.array([pre_s, post_s]) / dt + 1e-8).astype(int)
    if min(npre, npost) < 1:
        raise ValueError(f"前后窗口需分别覆盖至少一个采样间隔（{dt:.3g} 秒）")
    offsets = np.arange(-npre, npost + 1) * dt
    heat = np.full((len(rows), len(offsets)), np.nan)
    analysed = []
    phases = _phases(rows)
    for i, (raw, phase) in enumerate(zip(rows, phases)):
        check_stop(should_stop)
        t = raw[align + "_s"]
        row = dict(
            row=i + 1,
            **raw,
            cell_id=cell_id,
            event_s=t,
            phase=phase,
            status="missing_event",
            coverage=0.0,
            tracking_status="unavailable",
            **dict.fromkeys(QUALITY_FIELDS),
            pre_mean_dff=None,
            post_mean_dff=None,
            delta_dff=None,
        )
        analysed.append(row)
        if t is None:
            if behavior is not None:
                row["tracking_status"] = "missing_event"
            continue
        idx, sampled = _nearest(times, t + offsets, dt)
        available = sampled & np.isfinite(trace[idx])
        heat[i, available] = trace[idx[available]]
        row["coverage"] = available.mean()
        if available.all():
            row["status"] = "complete"
            row["pre_mean_dff"] = heat[i, :npre].mean()
            row["post_mean_dff"] = heat[i, npre:].mean()
            row["delta_dff"] = row["post_mean_dff"] - row["pre_mean_dff"]
        else:
            row["status"] = "partial_window" if available.any() else "no_signal"
        if behavior is not None:
            overlap = sampled & (behavior["behavior_frame"][idx] >= 0)
            quality = behavior["tracking_quality"][idx]
            row["tracking_status"] = "available" if overlap.any() else "outside"
            for key, code in zip(QUALITY_FIELDS, (2, 1, 0)):
                row[key] = np.mean(overlap & (quality == code))
            row["outside_tracking_fraction"] = np.mean(~overlap)
    result = dict(
        cell_id=cell_id,
        animal_id=rows[0]["animal_id"],
        trial_ids=[r["trial_id"] for r in rows],
        offset_s=offsets,
        dff=heat,
        rows=analysed,
    )
    for phase in ("early", "late"):
        selected = [
            i
            for i, r in enumerate(analysed)
            if r["phase"] == phase and r["status"] == "complete"
        ]
        result[phase + "_mean"] = (
            heat[selected].mean(axis=0) if selected else np.full(len(offsets), np.nan)
        )
        result[phase + "_n"] = len(selected)
    return result


def _write_json(file, obj):
    file.write_text(
        json.dumps(plain(obj), ensure_ascii=False, allow_nan=False), encoding="utf8"
    )


def _write_report(folder, result, params, run_id, should_stop):
    cell_id = result["cell_id"]
    rows = result["rows"]
    offsets = result["offset_s"]
    with (folder / "trials.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (folder / f"cell_{cell_id}.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "animal_id",
                "trial_id",
                "source_row",
                "row",
                "cell_id",
                "outcome",
                "phase",
                "status",
                "offset_s",
                "dff",
            ]
        )
        for row, signal in zip(rows, result["dff"]):
            check_stop(should_stop)
            writer.writerows(
                (
                    row["animal_id"],
                    row["trial_id"],
                    row["source_row"],
                    row["row"],
                    cell_id,
                    row["outcome"],
                    row["phase"],
                    row["status"],
                    t,
                    v if np.isfinite(v) else "",
                )
                for t, v in zip(offsets, signal)
            )
    _write_json(folder / f"cell_{cell_id}.json", result)
    _write_json(folder / "parameters.json", params)
    report = Report()
    group = f"{result['animal_id']} · 细胞 {cell_id}"
    order_label = "事件时间顺序" if params["order"] == "time" else "原始 CSV 顺序"
    report.add(
        "逐试次响应",
        result["dff"],
        kind="heat",
        x=offsets,
        zero=True,
        panel=True,
        group=group,
        trial_rows=rows,
        labels=result["trial_ids"],
        axes=f"横轴 相对事件时间 (s) · 纵轴 {order_label} · ΔF/F",
    )
    report.add(
        "早期与晚期平均",
        [result["early_mean"], result["late_mean"]],
        x=offsets,
        zero=True,
        panel=True,
        group=group,
        axes="相对事件时间 (s) / 平均 ΔF/F",
        labels=[f"早期 n={result['early_n']}", f"晚期 n={result['late_n']}"],
    )
    label = "线索" if params["align"] == "cue" else "结果"
    note = (
        "按"
        + label
        + f"时刻对齐。每行一个试次，按{order_label}显示。灰格表示神经信号缺失。划分早晚阶段时按线索时间排序（缺线索时用结果时间），"
        "取试次数相等的前后两半。奇数时中间试次单独保留。阶段曲线仅纳入神经窗口完整的试次。"
        "追踪质量按全部时间格计算，区分原始、插值、无效与覆盖范围外；无追踪资料时留空。"
    )
    payload = dict(
        title=f"逐试次响应 · {group}",
        note=note,
        params=params,
        charts=report.charts,
        run=run_id,
    )
    raw = json.dumps(plain(payload), ensure_ascii=False, allow_nan=False).replace(
        "<", "\\u003c"
    )
    script = (
        Path(__file__).resolve().parents[1] / "assets" / "figure_export.js"
    ).read_text("utf8")
    (folder / "report.html").write_text(
        TEMPLATE.replace("__SCRIPT__", script).replace("__PAYLOAD__", raw),
        encoding="utf8",
    )


def run_trials(session, ws, args, progress=None, should_stop=None):
    check_stop(should_stop)
    if not ws.step("signals"):
        raise ValueError("请先完成「信号与 spike」")
    path = Path(args["csv_path"])
    rows = read_trials(path)
    if progress:
        progress(0.1, "读取所选细胞信号…")
    with np.load(ws.path(CELLS_FILE)) as cells:
        ids = np.flatnonzero(cells["accepted"])
    if not len(ids):
        raise ValueError("没有保留的细胞")
    cell_id = int(args["cell_id"]) if args.get("cell_id") is not None else int(ids[0])
    if cell_id not in ids:
        raise ValueError("请从当前项目保留的细胞中选择编号")
    with np.load(ws.path(SIGNALS_FILE)) as signals:
        times = np.asarray(signals["times"], float)
        trace = signals["dff"][cell_id].copy()
    if len(times) != len(trace):
        raise ValueError("钙信号时间与细胞轨迹长度不一致")
    params = dict(
        mode="trials",
        cell_id=cell_id,
        animal_id=rows[0]["animal_id"],
        csv_path=str(path),
        align=args["align"],
        order=args.get("order", "time"),
        pre_s=float(args["pre_s"]),
        post_s=float(args["post_s"]),
    )
    behavior = None
    if ws.step("sync") and ws.step("behavior"):
        cfg = BehaviorConfig.load(ws)
        if cfg is not None:
            params["max_interp_s"] = TrackingParams(**cfg.params).max_interp_s
            params["sync_method"] = ws.step("sync")["params"].get("method")
            behavior = behavior_on_calcium_frames(ws, cfg)
    if progress:
        progress(0.4, "按试次对齐并计算追踪质量…")
    result = aligned_trials(
        rows,
        times,
        trace,
        cell_id,
        params["align"],
        params["pre_s"],
        params["post_s"],
        behavior,
        should_stop,
        params["order"],
    )
    parent = ws.path("behavior_studies")
    parent.mkdir(exist_ok=True)
    final = parent / ("trials-" + uuid4().hex[:12])
    with TemporaryDirectory(dir=parent) as tmp:
        folder = Path(tmp)
        if progress:
            progress(0.75, "生成逐试次热图和导出数据…")
        _write_report(folder, result, params, final.name, should_stop)
        check_stop(should_stop)
        folder.rename(final)
    ws.record_step("study_trials", params, {"folder": str(final.relative_to(ws.root))})
    return str(final)
