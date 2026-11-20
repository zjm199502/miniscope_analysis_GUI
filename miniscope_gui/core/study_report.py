"""Portable reports and labelled CSV tables for behavioural studies."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np

from . import behavior_studies as studies
from .behavior import BehaviorConfig
from .modeling.common import check_stop, load_dataset


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def table(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(columns.split(","))
        writer.writerows(rows)


class Report:
    def __init__(self):
        self.charts = []

    def add(
        self,
        title,
        data,
        kind="curve",
        x=None,
        labels=None,
        group="",
        axes="",
        bounds=None,
        band=None,
        **style,
    ):
        self.charts.append(
            dict(
                title=title,
                data=data,
                kind=kind,
                x=x,
                labels=labels,
                group=group,
                axes=axes,
                bounds=bounds,
                band=band,
                **style,
            )
        )


def freeze_report(out, r, folder, sync):
    times = r["offset_s"]
    cols = "pair,state,onset_s,offset_s,x_px,y_px,mean_speed,pre_minus_bout_speed,match_distance"
    table(folder / "events.csv", cols, r["events"])
    note = f"匹配 {len(r['events']) // 2} 对事件。热图每行一个事件，颜色表示群体平均 ΔF/F。"
    residual = sync.get("max_resid_ms")
    if residual is not None:
        note += f" 时间轴标出 LED 锚点最大拟合残差 ±{residual:.1f} ms。"
    else:
        note += " 当前同步记录未提供锚点残差。"
    cells = r["cells"]
    lines = []
    for key, name in (("on", "起始"), ("off", "解除")):
        band = None if residual is None else residual / 1000
        heat = r[key + "_heat"]
        limits = [float(np.min(heat)), float(np.max(heat))]
        for i, label in enumerate(("僵直", "普通停顿")):
            out.add(
                name + " · " + label,
                heat[:, i],
                kind="heat",
                x=times,
                group=name,
                axes="横轴 秒 · 纵轴 匹配对",
                bounds=limits,
                panel=True, zero=True, band=band,
            )
        out.add(name + " · 群体平均响应", heat.mean(axis=0), x=times, group=name,
                axes="秒 / 平均 ΔF/F", labels=["僵直", "普通停顿"], panel=True, zero=True, band=band)
        out.add(
            name + " · 留出匹配对准确率",
            [r[key + "_accuracy"]],
            x=times,
            group=name,
            axes="秒 / 准确率",
            bounds=[0, 1],
            band=band,
        )
        trajectory = r[key + "_trajectory"]
        out.add(
            name + " · 群体轨迹",
            trajectory[:, :, 1],
            x=trajectory[:, :, 0],
            kind="path",
            labels=["僵直", "普通停顿"],
            group=name,
            axes="PC1 / PC2，点为窗口起点",
        )
        for j, cell in enumerate(cells):
            curves = r[key + "_cells"][:, j]
            out.add(
                name + " · 细胞 " + str(cell),
                curves,
                x=times,
                labels=["僵直", "普通停顿"],
                group=f"细胞 {cell}",
                axes="秒 / ΔF/F",
                band=band,
            )
            lines.extend((name, cell, t, a, b) for t, a, b in zip(times, *curves))
    table(
        folder / "cell_transitions.csv",
        "phase,cell_id,time_s,freezing_dff,pause_dff",
        lines,
    )
    return note + " 少于三对时不估计分类准确率。"


def wall_report(out, r, folder):
    labels = ["全部", "接近", "远离", "沿墙"]
    keys = ["all", "toward", "away", "along"]
    x = r["distance"]
    out.add(
        "各距离区间停留时间",
        [r[k + "_seconds"] for k in keys],
        x=x,
        labels=labels,
        axes=f"距墙 {r['unit']} / 秒",
    )
    rows = []
    for i, cell in enumerate(r["cells"]):
        out.add(
            f"细胞 {cell} 距墙调谐",
            [r[k][i] for k in keys],
            x=x,
            labels=labels,
            group=f"细胞 {cell}",
            axes=f"距墙 {r['unit']} / 平均 ΔF/F",
        )
        for k in keys:
            rows.extend(
                (cell, k, r["unit"], d, t, f)
                for d, t, f in zip(x, r[k + "_seconds"], r[k][i])
            )
    table(
        folder / "boundary.csv",
        "cell_id,direction,unit,distance,seconds,mean_dff",
        rows,
    )
    return "方向来自相邻质心位移。径向位移占比小于 −0.5 为接近，大于 0.5 为远离，中间为沿墙；静止点仅进入全部曲线。"


def goal_report(out, r, folder, n):
    rows = []
    stages = len(r["schedule"])
    limits = []
    for cell in range(len(r["cells"])):
        values = np.concatenate([r[f"place_{k}"][cell] for k in range(stages)])
        finite = values[np.isfinite(values)]
        limits.append([finite.min(), finite.max()] if finite.size else None)
    for k, (start, x, y) in enumerate(r["schedule"]):
        group = f"阶段 {k + 1}"
        out.add(
            f"{group} · 位置停留时间",
            r[f"place_seconds_{k}"].reshape(n, n),
            "heat",
            group=group,
            axes="场地网格 / 秒",
        )
        distance = (r["edges"][1:] + r["edges"][:-1]) / 2
        out.add(
            f"{group} · 目标距离停留时间",
            [r[f"distance_seconds_{k}"]],
            x=distance,
            group=group,
            axes=r["unit"] + " / 秒",
        )
        for i, cell in enumerate(r["cells"]):
            g = f"细胞 {cell}"
            out.add(
                f"{group} · 位置活动图",
                r[f"place_{k}"][i].reshape(n, n),
                "heat",
                group=g,
                axes="场地网格 / ΔF/F · ★ 目标", panel=True, bounds=limits[i],
                marker=r["markers"][k], aspect=r["aspect"],
            )
            out.add(
                f"{group} · 距目标中心",
                [r[f"distance_{k}"][i]],
                x=distance,
                group=g,
                axes=r["unit"] + " / ΔF/F",
            )
            for key in ("place", "distance"):
                rows.extend(
                    (k + 1, start, x, y, cell, key, j, r[f"{key}_seconds_{k}"][j], v)
                    for j, v in enumerate(r[f"{key}_{k}"][i])
                )
    for i, cell in enumerate(r["cells"]):
        out.add("各阶段距目标调谐", [r[f"distance_{k}"][i] for k in range(stages)],
                x=distance, group=f"细胞 {cell}", labels=[f"阶段 {k+1}" for k in range(stages)],
                axes=r["unit"]+" / ΔF/F", panel=True)
    table(
        folder / "target_tuning.csv",
        "epoch,start_s,target_x_px,target_y_px,cell_id,coordinate,bin,seconds,mean_dff",
        rows,
    )
    table(
        folder / "target_prediction.csv",
        "held_out_epoch,cell_id,common_bins,place_mse,distance_mse",
        ([k + 1, *row] for k, *row in r["prediction"]),
    )
    if r["prediction"]:
        scores = np.array(r["prediction"])
        for k in np.unique(scores[:, 0]):
            sub = scores[scores[:, 0] == k]
            out.add(
                f"留出阶段 {int(k) + 1} · 预测误差",
                sub[:, 3:5].T,
                x=sub[:, 1],
                labels=["场地位置", "目标距离"],
                axes="细胞编号 / 均方误差，越低越好",
            )
        return "在其他目标位置训练，在相同留出样本比较误差；以训练集线性回归控制速度。仅比较两种模型在训练时均至少采到两次的位置或距离区间。"
    return "可查看描述性调谐。区分场地与目标编码需要不同目标位置及足够共同采样。"


def day_report(out, r, folder, n):
    names = r["names"]
    out.add(
        "训练日 × 测试日中位误差",
        r["errors"],
        "heat",
        panel=True, axes="纵轴训练日 / 横轴测试日；天序：" + "、".join(names),
    )
    pairs = [row for row in r["pairs"] if row[0] != row[1]]
    for a,b,count,_,error,baseline in pairs:
        out.add(f"{names[a]} → {names[b]} · {count} 个细胞", [[error, baseline]],
                axes="归一化位置误差", bounds=[0,max(max(row[-2:]) for row in pairs)],
                points=True, panel=True)
    table(
        folder / "crossday.csv",
        "training_day,test_day,shared_cells,test_samples,median_error,within_test_day_error",
        ((names[a], names[b], *v) for a, b, *v in r["pairs"]),
    )
    rows = []
    for a, b, count, samples, error, baseline in r["pairs"]:
        key = f"{a}_{b}"
        group = f"{names[a]} → {names[b]}"
        out.add(
            f"{group} · 位置解码误差",
            np.array(r["spatial_" + key]).reshape(n, n),
            "heat",
            group=group,
            axes=f"场地网格 · {count} 个共同细胞 · 跨日 {error:.3f} / 同细胞日内 {baseline:.3f}",
        )
        for i, cell in enumerate(r["ids_" + key]):
            maps = r["maps_" + key][:, i]
            finite = maps[np.isfinite(maps)]
            limits = [finite.min(), finite.max()] if len(finite) else None
            for j, m in enumerate(maps):
                out.add(
                    f"{names[[a, b][j]]} · 全局细胞 {cell}",
                    m.reshape(n, n),
                    "heat",
                    group=f"{group} · 细胞 {cell}",
                    axes="场地网格 / 平均 ΔF/F",
                    bounds=limits,
                )
                rows.extend(
                    (names[a], names[b], j, cell, idx, v) for idx, v in enumerate(m)
                )
    table(
        folder / "registered_maps.csv",
        "training_day,test_day,side_0_train_1_test,global_id,bin,mean_dff",
        rows,
    )
    return "误差按场地宽、高分别归一化。每对日期使用同一批细胞、共同探索位置及相同测试帧；日内基线使用测试日的另一半记录。空格表示样本不足。"


def run_study(session, ws, args, progress=None, should_stop=None):
    mode = args["mode"]
    n = args["bins"]
    check_stop(should_stop)
    if progress:
        progress(0.05, "读取同步数据…")
    if mode == "days":
        r = studies.crossday(args["registration"], n, should_stop)
    else:
        cfg = BehaviorConfig.load(ws)
        ds = load_dataset(ws, cfg)
        if mode == "freeze":
            r = studies.transitions(ds, args["speed"], args["window"], should_stop)
        elif mode == "target":
            r = studies.target(ds, cfg, args["schedule"], n)
        else:
            r = studies.boundary(ds, cfg, n)
    check_stop(should_stop)
    if progress:
        progress(0.8, "生成图表和 CSV…")
    parent = ws.path("behavior_studies")
    parent.mkdir(exist_ok=True)
    final = parent / (mode + "-" + uuid4().hex[:12])
    with TemporaryDirectory(dir=parent) as tmp:
        folder = Path(tmp)
        out = Report()
        if mode == "freeze":
            note = freeze_report(out, r, folder, ws.step("sync")["params"])
        elif mode == "days":
            note = day_report(out, r, folder, n)
        elif mode == "target":
            note = goal_report(out, r, folder, n)
        else:
            note = wall_report(out, r, folder)
        payload = plain({"title": args["title"], "note": note, "params": args, "run": final.name, "charts": out.charts})
        raw = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).replace("<", "\\u003c")
        (folder / "report.html").write_text(
            TEMPLATE.replace("__SCRIPT__", (Path(__file__).resolve().parents[1] / "assets" / "figure_export.js").read_text("utf8")).replace("__PAYLOAD__", raw), encoding="utf-8"
        )
        (folder / "data.json").write_text(
            json.dumps(plain(r), ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        check_stop(should_stop)
        folder.rename(final)
    ws.record_step("study_" + mode, args, {"folder": str(final.relative_to(ws.root))})
    return str(final)


TEMPLATE = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:15px system-ui;background:#f3f6f5;color:#203a42;margin:24px auto;max-width:1000px;padding:0 16px}select,button{font:inherit;padding:6px;margin:4px}article{margin:14px 0;background:white;padding:12px;width:fit-content;max-width:100%}svg{display:block;width:var(--width);max-width:100%;height:auto}@media print{body{background:white;margin:0;padding:0}body>*:not(main),article:not(.printing),button{display:none}article{padding:0;margin:0}}</style>
<h1 id="title"></h1><p id="note"></p><label>查看<select id="group"></select></label>
<label>图宽<select id="width"><option>180</option><option>90</option></select> mm</label>
<label>格式<select id="format"><option value="svg">SVG · 论文</option><option value="png">PNG · 组会</option><option value="pdf">PDF · 打印到文件</option></select></label>
<main id="plots"></main><script>const r=__PAYLOAD__;__SCRIPT__</script></html>"""
