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
                band=band,
            )
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
                axes="场地网格 / ΔF/F",
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
        axes="纵轴训练日 / 横轴测试日；天序：" + "、".join(names),
    )
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
        payload = plain({"title": args["title"], "note": note, "charts": out.charts})
        raw = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).replace("<", "\\u003c")
        (folder / "report.html").write_text(
            TEMPLATE.replace("__PAYLOAD__", raw), encoding="utf-8"
        )
        (folder / "data.json").write_text(
            json.dumps(plain(r), ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        check_stop(should_stop)
        folder.rename(final)
    ws.record_step("study_" + mode, args, {"folder": str(final.relative_to(ws.root))})
    return str(final)


TEMPLATE = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:16px system-ui;color:#17363b;background:#f6f8f7;max-width:1100px;margin:32px auto;padding:0 16px}select{padding:6px;font:inherit}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr));gap:14px;margin-top:16px}article{background:white;padding:14px;border:1px solid #d9e1df}canvas{width:100%}small{display:block}h3{margin:0 0 8px}</style>
<h1></h1><p id="note"></p><label>查看 <select></select></label><main></main>
<script>const r=__PAYLOAD__, sel=document.querySelector('select'), main=document.querySelector('main');
document.title=document.querySelector('h1').textContent=r.title;note.textContent=r.note;
for(const g of new Set(r.charts.map(c=>c.group||'总览')))sel.add(new Option(g,g));
const colors=['#147e79','#cf702d','#7058a5','#4279ad'];
function extent(v){let a=Infinity,b=-Infinity;for(const x of v)if(Number.isFinite(x)){a=Math.min(a,x);b=Math.max(b,x)}return a===Infinity?[0,1]:[a,b]}
function draw(c){const card=document.createElement('article');main.append(card);const h=document.createElement('h3');h.textContent=c.title;card.append(h);
const cv=document.createElement('canvas');cv.width=600;cv.height=300;card.append(cv);const ctx=cv.getContext('2d'),L=46,T=12,W=534,H=247;
const [lo,hi]=c.bounds||extent(c.data.flat());const range=hi-lo||1;
const pos=(v,a,b)=> (v-a)/(b-a||1);
let xmin=0,xmax=c.data[0].length-1;if(c.x)[xmin,xmax]=extent(c.x.flat());
if(c.kind==='heat'){let ny=c.data.length,nx=c.data[0].length;
for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){let v=c.data[j][i],u=(v-lo)/range;ctx.fillStyle=v==null?'#ddd':'hsl('+(235-195*u)+' 63% '+(28+32*u)+'%)';ctx.fillRect(L+W*i/nx,T+H*j/ny,W/nx+.35,H/ny+.35)}
}else{c.data.forEach((ys,j)=>{const xs=c.kind==='path'?c.x[j]:c.x;ctx.strokeStyle=colors[j%4];ctx.lineWidth=2.3;ctx.beginPath();let start=true;
ys.forEach((v,i)=>{if(v==null){start=true;return}let x=L+W*pos(xs?xs[i]:i,xmin,xmax),y=T+H*(1-(v-lo)/range);if(start)ctx.moveTo(x,y);else ctx.lineTo(x,y);start=false;if(i===0&&c.kind==='path'){ctx.fillStyle=colors[j%4];ctx.fillRect(x-3,y-3,6,6)}});ctx.stroke()})}
if(c.band!=null&&xmax>xmin){let a=Math.max(xmin,-c.band),b=Math.min(xmax,c.band);ctx.fillStyle='#6663';ctx.fillRect(L+W*pos(a,xmin,xmax),T,W*(b-a)/(xmax-xmin),H);ctx.strokeStyle='#333';ctx.beginPath();ctx.moveTo(L+W*pos(0,xmin,xmax),T);ctx.lineTo(L+W*pos(0,xmin,xmax),T+H);ctx.stroke()}
ctx.strokeStyle='#8fa19c';ctx.strokeRect(L,T,W,H);ctx.font='15px sans-serif';ctx.fillStyle='#264544';ctx.fillText(xmin.toFixed(1),L,T+H+20);ctx.fillText(xmax.toFixed(1),L+W-30,T+H+20);
if(c.kind!=='heat'){ctx.fillText(hi.toFixed(2),1,T+8);ctx.fillText(lo.toFixed(2),1,T+H)}
const info=document.createElement('small');info.textContent=c.axes+(c.kind==='heat'?' · 色阶 '+lo.toPrecision(3)+' 至 '+hi.toPrecision(3)+'，灰色为缺测':'');card.append(info);
(c.labels||[]).forEach((l,i)=>{let el=document.createElement('span');el.style.color=colors[i%4];el.textContent=' ● '+l;card.append(el)})}
function show(){main.replaceChildren();r.charts.filter(c=>(c.group||'总览')===sel.value).forEach(draw)}sel.onchange=show;show();</script></html>"""
