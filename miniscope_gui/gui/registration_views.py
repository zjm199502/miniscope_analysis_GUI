"""Views of cross-session registration and cross-day analyses."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt

from ..core import crossday as X
from ..core.registration import load_registration
from .model_views import BLUE, GREY, RED, Painter, _ramp, colour

VIEWS = {
    "reg": (("reg_after", "对齐后叠加图"), ("reg_before", "对齐前叠加图"),
            ("reg_cells", "配准的细胞"), ("reg_model", "匹配模型")),
    "react": (("xd_react", "重激活强度"), ("xd_members", "细胞组成员"), ("xd_act", "激活随时间")),
    "stab": (("xd_rates", "各 session 事件率"), ("xd_corr", "session 间相关")),
}


def is_reg_view(key: str) -> bool:
    return key.startswith("reg_") or key.startswith("xd_")


def _overlay(a, b):
    """Magenta = a, green = b, white = both."""
    def n(x):
        hi = np.percentile(x, 99.5) or 1.0
        return np.clip(x / hi, 0, 1)
    a, b = n(a), n(b)
    return np.stack([a, b, a], -1)


class RegPainter(Painter):
    def __init__(self, main, folder, compare: int = 1):
        super().__init__(main)
        self.reg = load_registration(folder)
        self.folder = folder
        self.compare = compare

    def background(self):
        return self.reg["proj"][0]

    def rgb(self, img, title):
        self.view.set_image(img, title)
        self.view.clear_overlays()
        self.view.img.setLevels([0, 1])
        self.main.model_overlays = True

    # --------------------------------------------------------- registration
    def _pair(self):
        names = self.reg["summary"]["names"]
        k = min(max(1, self.compare), len(names) - 1)
        return k, names

    def reg_after(self):
        k, names = self._pair()
        p = self.reg["proj"]
        self.rgb(_overlay(p[0], p[k]), f"对齐后：品红 = {names[0]}，绿 = {names[k]}，白 = 重合")
        self._align_plot()

    def reg_before(self):
        k, names = self._pair()
        p = self.reg["proj_before"]
        self.rgb(_overlay(p[0], p[k]), f"对齐前（只按裁剪位置摆放）：品红 = {names[0]}，绿 = {names[k]}")
        self._align_plot()

    def _align_plot(self):
        s = self.reg["summary"]
        names = s["names"]
        tr = self.plot("各 session 与参考的相似度：灰 = 对齐前，蓝 = 对齐后", "", "相似度", legend=True)
        x = np.arange(len(names))
        b = [a["ncc_before"] for a in s["alignment"]]
        a = [a["ncc_after"] for a in s["alignment"]]
        tr.addItem(pg.BarGraphItem(x=x - 0.18, height=b, width=0.35, brush=GREY))
        tr.addItem(pg.BarGraphItem(x=x + 0.18, height=a, width=0.35, brush=BLUE))
        tr.getAxis("bottom").setTicks([[(i, n) for i, n in enumerate(names)]])

    def reg_cells(self):
        r = self.reg
        S = len(r["summary"]["names"])
        ns = r["n_sessions"]
        col = _ramp(ns, 1, S, "viridis")
        self.cell_map({"centres": r["centres"]}, col,
                      f"配准后的细胞：颜色 = 在几个 session 里找到（深 = 1 个，亮 = 全部 {S} 个）",
                      sizes=np.where(ns == S, 10, 7))
        tr = self.plot("在几个 session 里找到的细胞数", "session 数", "细胞数")
        for m in range(1, S + 1):
            tr.addItem(pg.BarGraphItem(x=[m], height=[int((ns == m).sum())], width=0.6,
                                       brush=_ramp([m], 1, S, "viridis")[0]))

    def reg_model(self):
        r = self.reg
        d, sim, p = r["pair_d"], r["pair_r"], r["pair_p"]
        col = [(int(255 * q), 60, int(255 * (1 - q))) for q in p]
        self.scatter_view(d, sim, col, "每个点是两天之间的一对候选细胞：横 = 中心距离（像素），纵 = 形状相似度；"
                                       "红 = 很可能是同一个细胞，蓝 = 不同细胞", size=6)
        self.view.vb.setAspectLocked(False)
        self.view.vb.autoRange()
        thr = r["summary"]["params"]["p_threshold"]
        tr = self.plot("中心距离分布：红 = 判为同一细胞，蓝 = 判为不同细胞", "距离（像素）", "细胞对数", legend=True)
        edges = np.linspace(0, max(float(d.max()) if len(d) else 1, 1e-6), 25)
        h1, _ = np.histogram(d[p >= thr], edges)
        h0, _ = np.histogram(d[p < thr], edges)
        tr.addItem(pg.BarGraphItem(x0=edges[:-1], x1=edges[1:], height=h0, brush=(*BLUE, 140)))
        tr.addItem(pg.BarGraphItem(x0=edges[:-1], x1=edges[1:], y0=h0, height=h1, brush=(*RED, 160)))

    # ---------------------------------------------------------- reactivation
    def _react(self):
        if not X.has_result(self.folder, "crossday_reactivation"):
            raise ValueError("还没有运行「细胞组重激活」")
        r = X.load_crossday(self.folder, "crossday_reactivation")
        if not len(r["patterns"]):
            raise ValueError("模板 session 里没有检测到细胞组")
        return r

    def xd_react(self):
        r = self._react()
        s = r["summary"]
        K = len(r["patterns"])
        self.xd_members(plot=False)
        tr = self.plot(f"平均激活强度：灰 = 模板「{s['template']}」，红 = 目标「{s['target']}」，"
                       "黑线 = 打乱细胞身份的 95% 范围", "细胞组", "平均激活强度")
        x = np.arange(1, K + 1)
        tr.addItem(pg.BarGraphItem(x=x - 0.18, height=r["mean_template"], width=0.35, brush=GREY))
        tr.addItem(pg.BarGraphItem(x=x + 0.18, height=r["mean_target"], width=0.35, brush=RED))
        tr.addItem(pg.ErrorBarItem(x=x + 0.18, y=r["null_mean"], top=r["null_hi"] - r["null_mean"],
                                   bottom=r["null_mean"] - r["null_lo"], beam=0.2, pen=pg.mkPen("k", width=1.5)))
        tr.getAxis("bottom").setTicks([[(int(i), f"#{int(i)}" + (" *" if r["p"][i - 1] < 0.05 else ""))
                                        for i in x]])

    def xd_members(self, plot=True):
        r = self._react()
        reg = self.reg
        cen = reg["centres"][r["global_ids"]]
        mem = r["members"]
        col = [GREY] * len(cen)
        size = np.full(len(cen), 7.0)
        for k in range(len(mem) - 1, -1, -1):
            for i in np.where(mem[k])[0]:
                col[i] = colour(k)
                size[i] = 11
        self.cell_map({"centres": cen}, col, "两天都追踪到的细胞：同一颜色 = 模板 session 里的同一个细胞组", sizes=size)
        if plot:
            tr = self.plot("各细胞组的成员数", "细胞组", "细胞数")
            for k in range(len(mem)):
                tr.addItem(pg.BarGraphItem(x=[k + 1], height=[int(mem[k].sum())], width=0.6, brush=colour(k)))

    def xd_act(self):
        r = self._react()
        s = r["summary"]
        act, t = r["act_target"], r["times_target"]
        K = len(act)
        img = act / np.maximum(np.percentile(np.abs(r["act_template"]), 99.5, axis=1, keepdims=True), 1e-12)
        self.heat(np.clip(img, 0, 1), f"目标「{s['target']}」中各细胞组的激活强度（按模板 session 的尺度归一化）",
                  "inferno", rect=(t[0], 0, t[-1] - t[0], K))
        tr = self.plot(f"目标「{s['target']}」中的激活强度（已错开）", "时间 (秒)", "", legend=True)
        if "freezing_target" in r:
            self.shade(t, r["freezing_target"].astype(bool), K)
        for k in range(K):
            tr.plot(t, np.clip(img[k], -0.2, 1.2) * 0.9 + (K - 1 - k), pen=pg.mkPen(colour(k), width=1.2),
                    name=f"细胞组 {k + 1}")

    # ------------------------------------------------------------- stability
    def _stab(self):
        if not X.has_result(self.folder, "crossday_stability"):
            raise ValueError("还没有运行「跨天活动稳定性」")
        return X.load_crossday(self.folder, "crossday_stability")

    def xd_rates(self):
        r = self._stab()
        names = self.reg["summary"]["names"]
        rate = r["rate"]
        keep = np.isfinite(rate[:, r["has_sig"]]).all(1)
        img = rate[keep][:, r["has_sig"]]
        if not len(img):
            raise ValueError("没有在所有带信号的 session 里都追踪到的细胞")
        img = img[np.argsort(-img[:, 0])]
        cols = [n for n, h in zip(names, r["has_sig"]) if h]
        self.heat(np.log10(img + 1e-3), f"每行一个在所有 session 都追踪到的细胞（{len(img)} 个），每列一个 session："
                                        f"{' | '.join(cols)}；颜色 = log10 事件率", "viridis",
                  rect=(0, 0, len(cols), len(img)))
        tr = self.plot("各 session 的事件率（每条灰线一个细胞，蓝 = 中位数）", "", "事件/秒")
        x = np.arange(len(cols))
        for row in img[:: max(1, len(img) // 200)]:
            tr.plot(x, row, pen=pg.mkPen((*GREY, 60)))
        tr.plot(x, np.median(img, 0), pen=pg.mkPen(BLUE, width=3), symbol="o", symbolBrush=BLUE)
        tr.getAxis("bottom").setTicks([[(i, n) for i, n in enumerate(cols)]])

    def xd_corr(self):
        r = self._stab()
        names = self.reg["summary"]["names"]
        S = len(names)
        self.heat(np.nan_to_num(r["pv"], nan=0), "session 间事件率的相关（Spearman，只用两天都追踪到的细胞）；"
                                               "行列顺序：" + " | ".join(names), "viridis", rect=(0, 0, S, S))
        self.view.img.setLevels([0, 1])
        tr = self.plot("每个 session 的细胞在其它 session 里也找到的比例", "", "比例", legend=True)
        ov = r["overlap"]
        w = 0.8 / S
        for b in range(S):
            x = np.arange(S) + (b - (S - 1) / 2) * w
            h = [ov[a, b] if a != b else 0 for a in range(S)]
            tr.addItem(pg.BarGraphItem(x=x, height=h, width=w * 0.9, brush=colour(b)))
            tr.plot([], [], pen=pg.mkPen(colour(b), width=6), name=f"在 {names[b]} 里")
        tr.getAxis("bottom").setTicks([[(i, n) for i, n in enumerate(names)]])


def show(main, key: str, folder, compare: int = 1):
    getattr(RegPainter(main, folder, compare), key)()
