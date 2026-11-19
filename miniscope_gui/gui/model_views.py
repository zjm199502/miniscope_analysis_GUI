"""Result views of the modelling analyses, drawn in the central viewer and the trace panel."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt

from ..core.modeling.common import load_result

PALETTE = [(228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163), (255, 127, 0),
           (0, 190, 190), (166, 86, 40), (247, 129, 191), (210, 210, 40), (120, 120, 120),
           (90, 60, 200), (0, 120, 90)]
GREY = (140, 140, 140)
BLUE = (40, 110, 220)
RED = (220, 60, 60)


def colour(i):
    return PALETTE[int(i) % len(PALETTE)]


def _ramp(v, lo=None, hi=None, preset="viridis"):
    """Values -> list of RGB tuples."""
    v = np.asarray(v, float)
    lo = np.nanmin(v) if lo is None else lo
    hi = np.nanmax(v) if hi is None else hi
    x = np.clip((v - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
    cm = pg.colormap.get(preset)
    return [tuple(int(c) for c in rgb[:3]) for rgb in cm.map(np.nan_to_num(x), mode="byte")]


def _diverging(v):
    v = np.asarray(v, float)
    m = np.nanmax(np.abs(v)) or 1.0
    out = []
    for x in np.nan_to_num(v) / m:
        a = int(255 * (1 - abs(x)))
        out.append((255, a, a) if x > 0 else (a, a, 255))
    return out


class Painter:
    def __init__(self, main):
        self.main, self.view, self.tr, self.ws = main, main.view, main.trace, main.ws

    # ------------------------------------------------------------ helpers
    def background(self):
        for f in ("bg_max.npy", "summary_cn.npy", "max.npy"):
            p = self.ws.path(f)
            if p.exists():
                return np.load(p)
        return None

    def cell_map(self, res, colours, title, sizes=None, edges=None, outline=None):
        bg = self.background()
        if bg is None:
            bg = np.zeros(tuple(int(v) for v in res.get("shape", (100, 100))))
        self.view.set_image(bg, title)
        self.view.clear_overlays()
        cen = res["centres"]
        x, y = cen[:, 1], cen[:, 0]
        if edges is not None and len(edges[0]):
            i, j = edges
            ex = np.column_stack([x[i], x[j]]).ravel()
            ey = np.column_stack([y[i], y[j]]).ravel()
            self.view.add_overlay(pg.PlotCurveItem(ex, ey, connect="pairs",
                                                   pen=pg.mkPen((255, 255, 255, 110), width=1)))
        n = len(x)
        sizes = np.full(n, 9.0) if sizes is None else np.asarray(sizes, float)
        pens = [pg.mkPen("w", width=2) if (outline is not None and outline[k]) else pg.mkPen((0, 0, 0, 120))
                for k in range(n)]
        sc = pg.ScatterPlotItem(x=x, y=y, size=sizes, brush=[pg.mkBrush(*c) for c in colours], pen=pens)
        self.view.add_overlay(sc)
        self.main.model_overlays = True

    def heat(self, img, title, preset="viridis", rect=None, symmetric=False):
        img = np.asarray(img, float)
        self.view.set_image(img, title)
        self.view.clear_overlays()
        self.view.hist.gradient.loadPreset(preset)
        if symmetric:
            m = float(np.nanpercentile(np.abs(img), 99)) or 1.0
            self.view.img.setLevels([-m, m])
        self.view.vb.setAspectLocked(False)
        if rect is not None:
            self.view.img.setRect(pg.QtCore.QRectF(*rect))
        self.view.vb.autoRange()

    def scatter_view(self, x, y, colours, title, size=5):
        """Scatter plot in the central viewer (used for embeddings and trajectories)."""
        self.view.set_image(np.zeros((2, 2)), title)
        self.view.clear_overlays()
        pad = 0.05 * max(np.ptp(x), np.ptp(y), 1e-9)
        self.view.img.setRect(pg.QtCore.QRectF(x.min() - pad, y.min() - pad,
                                               np.ptp(x) + 2 * pad, np.ptp(y) + 2 * pad))
        self.view.img.setLevels([0, 1])
        sc = pg.ScatterPlotItem(x=x, y=y, size=size, pen=None, brush=[pg.mkBrush(*c, 170) for c in colours])
        self.view.add_overlay(sc)
        self.view.vb.autoRange()
        self.main.model_overlays = True

    def plot(self, title, xlabel="", ylabel="", legend=False):
        tr = self.tr
        tr.clear()
        if tr.plotItem.legend is not None:
            tr.plotItem.legend.clear()
        if legend:
            tr.addLegend(offset=(-10, 5))
        tr.getAxis("bottom").setTicks(None)
        tr.getAxis("left").enableAutoSIPrefix(True)
        self.main.model_overlays = True
        tr.setTitle(title)
        tr.setLabel("bottom", xlabel)
        tr.setLabel("left", ylabel)
        tr.enableAutoRange()
        tr.show()
        return tr

    def hist(self, values, title, xlabel, bins=30, mark=None, colour_=BLUE):
        tr = self.plot(title, xlabel, "数量")
        v = np.asarray(values, float)
        v = v[np.isfinite(v)]
        if len(v):
            h, e = np.histogram(v, bins=bins)
            tr.addItem(pg.BarGraphItem(x0=e[:-1], x1=e[1:], height=h, brush=(*colour_, 160)))
        if mark is not None:
            tr.addItem(pg.InfiniteLine(mark, angle=90, pen=pg.mkPen(RED, width=2)))
        return tr

    def shade(self, t, mask, height, name="僵直"):
        self.tr.plot(t, mask.astype(float) * height, pen=pg.mkPen((*RED, 120)), fillLevel=0,
                     brush=(*RED, 40), name=name)

    # ---------------------------------------------------------------- GLM
    def glm_map(self, r):
        names = [str(n) for n in r["groups"]]
        sig = r["p"] < 0.05
        col = [colour(g) if s else GREY for g, s in zip(r["best_group"], sig)]
        self.cell_map(r, col, "每个细胞主要编码的变量（灰色 = 行为不能显著解释）",
                      sizes=np.where(sig, 11, 7))
        tr = self.plot("主要编码各变量的细胞数（颜色与细胞图一致）", "", "细胞数")
        for g, n in enumerate(names):
            c = int((sig & (r["best_group"] == g)).sum())
            tr.addItem(pg.BarGraphItem(x=[g], height=[c], width=0.6, brush=colour(g)))
        tr.getAxis("bottom").setTicks([[(g, n) for g, n in enumerate(names)]])

    def glm_r2(self, r):
        self.cell_map(r, _ramp(np.clip(r["r2"], 0, None)), "交叉验证 R²（越亮 = 行为解释得越多）")
        self.hist(r["r2"], "所有细胞的交叉验证 R²", "R²", mark=0)

    def glm_kernel(self, r):
        ks = sorted(int(k.split("_")[1]) for k in r if k.startswith("kernel_") and "lags" not in k)
        if not ks:
            raise ValueError("模型里没有事件类变量（僵直开始或你填写的事件）")
        j = ks[-1]                               # the user's events if given, else freezing onset
        K, lags = r[f"kernel_{j}"], r[f"kernel_lags_{j}"]
        name = str(r["groups"][j])
        label = "僵直开始" if name == "僵直" else name
        order = np.argsort(-K[:, lags >= 0].mean(1))
        self.heat(K[order], f"每个细胞对「{label}」的反应核（按事件后反应排序）", "bipolar",
                  rect=(lags[0], 0, lags[-1] - lags[0], len(K)), symmetric=True)
        tr = self.plot(f"「{label}」前后的平均反应：红 = 该变量显著的细胞，灰 = 全部细胞",
                       "相对事件的时间 (秒)", "ΔF/F (z)", legend=True)
        tr.plot(lags, K.mean(0), pen=pg.mkPen(GREY, width=1.5), name="全部细胞")
        s = (r["p_group"][j] < 0.05) & (r["dr2"][j] > 0)
        if s.any():
            tr.plot(lags, K[s].mean(0), pen=pg.mkPen(RED, width=2), name=f"显著细胞 ({int(s.sum())})")
        tr.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen((0, 0, 0), style=Qt.DashLine)))

    # ------------------------------------------------------------ decoder
    def dec_time(self, r):
        s = r["summary"]
        t, v = r["times"], r["valid"].astype(bool)
        truth, pred = r["truth"], r["pred"]
        self.dec_weights(r, plot=False)
        if s["binary"]:
            tr = self.plot(f"解码「{s['target']}」：红色填充 = 真实，蓝线 = 解码出的概率（均为未参与训练的数据）",
                           "时间 (秒)", "概率", legend=True)
            self.shade(t, truth > 0.5, 1.0, "真实")
            tr.plot(t[v], pred[v], pen=pg.mkPen(BLUE, width=1.2), name="解码概率")
        else:
            tr = self.plot(f"解码「{s['target']}」：灰 = 真实，蓝 = 解码", "时间 (秒)", s["target"], legend=True)
            a = truth if truth.ndim == 1 else truth[:, 0]
            b = pred if pred.ndim == 1 else pred[:, 0]
            tr.plot(t[v], a[v], pen=pg.mkPen(GREY, width=1.5), name="真实" + (" (x)" if truth.ndim == 2 else ""))
            tr.plot(t[v], b[v], pen=pg.mkPen(BLUE, width=1.2), name="解码")

    def dec_null(self, r):
        s = r["summary"]
        self.dec_weights(r, plot=False)
        self.hist(r["null"], f"蓝 = 打乱时间后的结果，红线 = 真实结果（p = {s['p']:.3f}）",
                  "平衡准确率" if s["binary"] else "相关系数 r", mark=s["metric"])

    def dec_weights_y(self, r):
        self.dec_weights(r, axis=1)

    def dec_weights(self, r, plot=True, axis=0):
        w = r["weights"]
        title = "解码权重"
        if w.ndim == 2:
            w = w[:, axis]
            title += " · 位置 " + "XY"[axis]
        elif r["summary"]["target"] == "位置":
            title += " · 位置 X（旧结果）"
        self.cell_map(r, _diverging(w), title,
                      sizes=6 + 8 * np.abs(w) / (np.abs(w).max() or 1))
        if plot:
            self.hist(w, "各细胞的解码权重", "权重", mark=0)

    # ---------------------------------------------------------- ensembles
    def _ens_colours(self, r):
        mem = r["members"]
        col, size = [], []
        for i in range(mem.shape[1]):
            ks = np.flatnonzero(mem[:, i])
            col.append(colour(ks[0]) if len(ks) else GREY)
            size.append(12 if len(ks) else 6)
        return col, np.array(size)

    def ens_map(self, r):
        if not len(r["patterns"]):
            raise ValueError("没有检测到细胞组")
        col, size = self._ens_colours(r)
        self.cell_map(r, col, "细胞组成员：同一颜色 = 同一个细胞组，灰色 = 不属于任何细胞组", sizes=size)
        self._ens_traces(r)

    def _ens_traces(self, r):
        tr = self.plot("各细胞组的激活强度（颜色与细胞图一致，已归一化并上下错开）", "时间 (秒)", "", legend=True)
        t, act = r["times"], r["activation"]
        K = len(act)
        if "freezing" in r:
            self.shade(t, r["freezing"].astype(bool), K)
        for k in range(K):
            a = act[k] / (np.percentile(np.abs(act[k]), 99.5) or 1.0)
            tr.plot(t, np.clip(a, -0.2, 1.2) * 0.9 + (K - 1 - k), pen=pg.mkPen(colour(k), width=1.2),
                    name=f"细胞组 {k + 1}")

    def ens_act(self, r):
        if not len(r["patterns"]):
            raise ValueError("没有检测到细胞组")
        act = r["activation"]
        img = act / np.maximum(np.percentile(np.abs(act), 99.5, axis=1, keepdims=True), 1e-12)
        t = r["times"]
        self.heat(np.clip(img, 0, 1), "细胞组激活强度（每行一个细胞组）", "inferno",
                  rect=(t[0], 0, t[-1] - t[0], len(act)))
        if "freeze_in" in r:
            tr = self.plot("各细胞组的平均激活：红 = 僵直时，蓝 = 非僵直时（* 表示差异显著）", "细胞组", "激活强度")
            K = len(act)
            tr.addItem(pg.BarGraphItem(x=np.arange(K) - 0.18, height=r["freeze_in"], width=0.34, brush=RED))
            tr.addItem(pg.BarGraphItem(x=np.arange(K) + 0.18, height=r["freeze_out"], width=0.34, brush=BLUE))
            sig = (r["freeze_p_up"] < 0.025) | (r["freeze_p_down"] < 0.025)
            tr.getAxis("bottom").setTicks([[(k, f"{k + 1}{'*' if sig[k] else ''}") for k in range(K)]])
        else:
            self._ens_traces(r)

    def ens_weights(self, r):
        if len(r["patterns"]):
            self.heat(r["patterns"], "每个细胞组里各细胞的权重（每行一个细胞组，每列一个细胞）", "bipolar", symmetric=True)
        tr = self.plot("相关矩阵的特征值：红线以上的个数 = 细胞组数量", "主成分序号", "特征值")
        e = r["eig"][:min(60, len(r["eig"]))]
        tr.plot(np.arange(1, len(e) + 1), e, pen=pg.mkPen(BLUE), symbol="o", symbolSize=5, symbolBrush=BLUE)
        tr.addItem(pg.InfiniteLine(float(r["thr"]), angle=0, pen=pg.mkPen(RED, style=Qt.DashLine)))

    # ---------------------------------------------------------- embedding
    def _emb_xy(self, r):
        E = r["embedding"]
        v = r["valid"].astype(bool)
        Ec = E - E[v].mean(0)
        if E.shape[1] > 2:
            _, _, Vt = np.linalg.svd(Ec[v], full_matrices=False)
            Ec = Ec @ Vt[:2].T
        return Ec[v, 0], Ec[v, 1], v

    def _emb_loss(self, r):
        tr = self.plot("训练损失（应逐渐下降并趋于平稳）", "训练步数", "InfoNCE 损失")
        l = r["loss"]
        n = max(1, len(l) // 300)
        tr.plot(np.arange(len(l) // n) * n, l[:len(l) // n * n].reshape(-1, n).mean(1), pen=pg.mkPen(BLUE, width=1.2))

    def emb_state(self, r):
        x, y, v = self._emb_xy(r)
        if "freezing" in r:
            fz = r["freezing"][v] > 0
            col = [RED if f else BLUE for f in fz]
            title = "嵌入（投影到最主要的两个方向）：红 = 僵直，蓝 = 非僵直"
        elif "x_u" in r:
            col = _ramp(r["x_u"][v])
            title = "嵌入：颜色 = 动物的 x 位置"
        else:
            col = _ramp(r["times"][v])
            title = "嵌入：没有行为数据，颜色 = 时间"
        self.scatter_view(x, y, col, title)
        self._emb_loss(r)

    def emb_time(self, r):
        x, y, v = self._emb_xy(r)
        self.scatter_view(x, y, _ramp(r["times"][v]), "嵌入：颜色 = 时间（深紫 = 开始，黄 = 结束）")
        self._emb_loss(r)

    # ---------------------------------------------------------------- HMM
    def hmm_states(self, r):
        t, z, post = r["times"], r["states"], r["posterior"]
        K = post.shape[1]
        img = np.zeros((1, len(z), 3), np.uint8)
        img[0] = np.array([colour(k) for k in z])
        self.view.set_image(img, "每个时刻所处的状态（一种颜色一个状态）")
        self.view.clear_overlays()
        self.view.vb.setAspectLocked(False)
        self.view.img.setLevels([0, 255])
        self.view.img.setRect(pg.QtCore.QRectF(t[0], 0, t[-1] - t[0], 1))
        self.view.vb.autoRange()
        tr = self.plot("每个状态的后验概率（上下错开）" + ("，红色填充 = 僵直" if "freezing" in r else ""),
                       "时间 (秒)", "", legend=True)
        if "freezing" in r:
            self.shade(t, r["freezing"].astype(bool), K)
        for k in range(K):
            tr.plot(t, post[:, k] * 0.9 + (K - 1 - k), pen=pg.mkPen(colour(k), width=1.2), name=f"状态 {k + 1}")

    def hmm_trans(self, r):
        x = r["latents"]
        z = r["states"]
        if x.shape[1] >= 2:
            self.scatter_view(x[:, 0], x[:, 1], [colour(k) for k in z],
                              "群体活动轨迹（主成分 1 和 2），颜色 = 状态", size=5)
            line = pg.PlotCurveItem(x[:, 0], x[:, 1], pen=pg.mkPen((255, 255, 255, 40)))
            self.view.add_overlay(line)
        P = r["trans"]
        K = len(P)
        tr = self.plot("状态转移概率：每组柱子 = 从该状态出发，下一时刻到各状态的概率", "出发状态", "概率")
        w = 0.8 / K
        for j in range(K):
            tr.addItem(pg.BarGraphItem(x=np.arange(K) - 0.4 + w * (j + 0.5), height=P[:, j], width=w,
                                       brush=colour(j)))
        tr.getAxis("bottom").setTicks([[(k, f"状态 {k + 1}") for k in range(K)]])

    def hmm_cells(self, r):
        M = r["mean_activity"]
        order = np.argsort(M.argmax(0) * 1000 - M.max(0))
        self.heat(M[:, order].T, "各状态下每个细胞的平均活动（每行一个细胞，每列一个状态；z 分数）",
                  "bipolar", symmetric=True)
        if "cv_ll" in r:
            tr = self.plot("留出数据的对数似然（越高越好，出现拐点处即合适的状态数）", "状态数 K", "每个时间点的对数似然")
            tr.plot(r["cv_k"], r["cv_ll"], pen=pg.mkPen(BLUE, width=1.5), symbol="o", symbolBrush=BLUE)
        elif "freeze_frac" in r:
            K = len(r["freeze_frac"])
            tr = self.plot("每个状态中僵直所占的比例（虚线 = 整段记录的僵直比例）", "状态", "僵直比例")
            tr.addItem(pg.BarGraphItem(x=np.arange(K), height=r["freeze_frac"], width=0.6,
                                       brushes=[colour(k) for k in range(K)]))
            tr.addItem(pg.InfiniteLine(float(r["summary"]["freeze_overall"]), angle=0,
                                       pen=pg.mkPen((0, 0, 0), style=Qt.DashLine)))
            tr.getAxis("bottom").setTicks([[(k, f"状态 {k + 1}") for k in range(K)]])
        else:
            K = len(r["occupancy"])
            tr = self.plot("每个状态占总时间的比例", "状态", "比例")
            tr.addItem(pg.BarGraphItem(x=np.arange(K), height=r["occupancy"], width=0.6,
                                       brushes=[colour(k) for k in range(K)]))
            tr.getAxis("bottom").setTicks([[(k, f"状态 {k + 1}") for k in range(K)]])

    # ------------------------------------------------------------ network
    def net_graph(self, r):
        deg = r["degree"].astype(float)
        col = [colour(c) if c >= 0 else GREY for c in r["community"]]
        i, j = np.where(np.triu(r["adj"], 1))
        if len(i) > 4000:                               # keep the drawing responsive
            keep = np.argsort(-r["corr"][i, j])[:4000]
            i, j = i[keep], j[keep]
        self.cell_map(r, col, "功能连接网络：颜色 = 模块，圆点越大连接越多，白圈 = 枢纽细胞",
                      sizes=6 + 12 * deg / (deg.max() or 1), edges=(i, j), outline=r["hubs"].astype(bool))
        self.net_degree(r, map_=False)

    def net_matrix(self, r):
        lab = r["community"]
        order = np.lexsort((-r["degree"], np.where(lab < 0, 10 ** 6, lab)))
        self.heat(r["corr"][np.ix_(order, order)], "细胞之间的相关矩阵（按模块排序）", "bipolar", symmetric=True)
        self.view.vb.setAspectLocked(True)
        cen = r["centres"]
        iu = np.triu_indices(len(cen), 1)
        d = np.hypot(cen[iu[0], 0] - cen[iu[1], 0], cen[iu[0], 1] - cen[iu[1], 1])
        c = r["corr"][iu]
        tr = self.plot("相关系数与细胞间距离的关系（每个点 = 一个距离区间的平均）", "距离 (像素)", "平均相关")
        if len(d):
            edges = np.linspace(0, np.percentile(d, 99), 21)
            idx = np.digitize(d, edges) - 1
            xs = [(edges[b] + edges[b + 1]) / 2 for b in range(20) if (idx == b).any()]
            ys = [c[idx == b].mean() for b in range(20) if (idx == b).any()]
            tr.plot(xs, ys, pen=pg.mkPen(BLUE, width=1.5), symbol="o", symbolBrush=BLUE)

    def net_degree(self, r, map_=True):
        if map_:
            self.cell_map(r, _ramp(r["degree"]), "每个细胞的度（越亮连接越多），白圈 = 枢纽细胞",
                          outline=r["hubs"].astype(bool))
        deg = r["degree"]
        tr = self.hist(deg, "度分布：每个细胞与多少个细胞相连", "度", bins=min(30, max(5, int(deg.max()) + 1)))
        if "degree_freezing" in r:
            tr.setTitle("度分布（全程）。僵直 / 非僵直时的网络指标见右侧结果说明")

    # --------------------------------------------------------------- info
    def info_si(self, r):
        if "si" not in r:
            raise ValueError("没有位置数据，未计算空间信息")
        place = r["place"].astype(bool)
        col = [c if p else GREY for c, p in zip(_ramp(r["si_z"], 0, max(4.0, float(np.nanmax(r["si_z"])))), place)]
        self.cell_map(r, col, "空间信息：彩色 = 显著（越亮越强），灰色 = 不显著", sizes=np.where(place, 11, 7))
        self.hist(r["si"], f"空间信息量分布（显著 {int(place.sum())} / {len(place)} 个）", "bits / 事件")

    def info_fields(self, r):
        if "rate_maps" not in r:
            raise ValueError("没有位置数据，未计算位置野")
        order = np.argsort(-np.where(r["place"], r["si_z"], -np.inf))[:16]
        maps = r["rate_maps"][order]
        n = maps.shape[1]
        cols = int(np.ceil(np.sqrt(len(maps))))
        rows = int(np.ceil(len(maps) / cols))
        mosaic = np.full((rows * (n + 1) - 1, cols * (n + 1) - 1), np.nan)
        for k, m in enumerate(maps):
            a, b = divmod(k, cols)
            mosaic[a * (n + 1):a * (n + 1) + n, b * (n + 1):b * (n + 1) + n] = m / (m.max() or 1.0)
        self.heat(np.nan_to_num(mosaic), "空间信息最高的 16 个细胞的位置野（每格归一化到自身最大值）：#"
                  + ", #".join(str(i) for i in r["ids"][order][:16]), "inferno")
        self.view.vb.setAspectLocked(True)
        tr = self.plot("动物在各位置停留的时间（把二维占据图按行展开）", "空间格子", "秒")
        occ = r["occupancy"].ravel()
        tr.addItem(pg.BarGraphItem(x=np.arange(len(occ)), height=occ, width=0.9, brush=(*BLUE, 160)))

    def info_mi(self, r):
        names = [str(n) for n in r["mi_names"]]
        if not names:
            raise ValueError("没有僵直或区域数据，未计算互信息")
        mi, p = r["mi_0"], r["mi_p_0"]
        sig = p < 0.05
        col = [c if s else GREY for c, s in zip(_ramp(mi), sig)]
        self.cell_map(r, col, f"与「{names[0]}」的互信息：彩色 = 显著（越亮越高），灰色 = 不显著",
                      sizes=np.where(sig, 11, 7))
        self.hist(mi, f"与「{names[0]}」的互信息分布（显著 {int(sig.sum())} / {len(sig)} 个）", "bits")


    # ----------------------------------------------------------- kinetics
    def _kin_criterion(self, r, level):
        crit = (r.get("summary", {}).get(level, {}) or {}).get("criterion", "icl")
        pre = "ev" if level == "event" else "cell"
        return ("ICL", r[f"{pre}_icl"]) if crit == "icl" and f"{pre}_icl" in r else ("BIC", r[f"{pre}_bic"])

    def _kin_scatter(self, r, level):
        pre = "ev" if level == "event" else "cell"
        sl = r["summary"][level]
        if "skipped" in sl:
            raise ValueError(f"{'事件' if level == 'event' else '细胞'}层面没有聚类：{sl['skipped']}")
        lab = r[f"{pre}_label"]
        pc = r[f"{pre}_pc"]
        ok = (lab >= 0) & np.isfinite(pc).all(1)
        idx = np.flatnonzero(ok)
        if len(idx) > 6000:
            idx = np.sort(np.random.default_rng(0).choice(idx, 6000, replace=False))
        from ..core.modeling.kinetics import CLUSTER_CHOICES
        names = "、".join(CLUSTER_CHOICES.get(str(f), str(f)) for f in sl["features"])
        what = "事件" if level == "event" else "细胞"
        self.scatter_view(pc[idx, 0], -pc[idx, 1], [colour(c) for c in lab[idx]],
                          f"每个点是一个{what}（指标 {names} 标准化后的前两个主成分），颜色 = 类别 1–{sl['K']}",
                          size=5 if level == "event" else 10)
        name, crit = self._kin_criterion(r, level)
        ks = r[f"{pre}_ks"]
        tr = self.plot(f"类别数选择：{name} 越低越好，选中 K = {sl['K']}", "类别数 K", name)
        tr.plot(ks, crit, pen=pg.mkPen(BLUE, width=2), symbol="o", symbolBrush=BLUE)
        tr.getAxis("bottom").setTicks([[(int(k), str(int(k))) for k in ks]])
        tr.addItem(pg.InfiniteLine(sl["K"], angle=90, pen=pg.mkPen(RED, width=2)))

    def kin_ev_scatter(self, r):
        self._kin_scatter(r, "event")

    def kin_cell_scatter(self, r):
        self._kin_scatter(r, "cell")

    def _waves(self, t, mean, sem, title, label):
        tr = self.plot(title, "相对峰值的时间（秒）", "ΔF/F / 峰值", legend=True)
        for c, (m, e) in enumerate(zip(mean, sem)):
            col = colour(c)
            m, e = np.nan_to_num(m), np.nan_to_num(e)
            hi = tr.plot(t, m + e, pen=pg.mkPen((*col, 0)))
            lo = tr.plot(t, m - e, pen=pg.mkPen((*col, 0)))
            tr.addItem(pg.FillBetweenItem(hi, lo, brush=(*col, 60)))
            tr.plot(t, m, pen=pg.mkPen(col, width=2), name=f"{label} {c + 1}")
        tr.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen((120, 120, 120), style=Qt.DashLine)))
        return tr

    def kin_ev_wave(self, r):
        sl = r["summary"]["event"]
        if "skipped" in sl:
            raise ValueError(f"事件层面没有聚类：{sl['skipped']}")
        W = np.nan_to_num(r["ev_wave_sub"])
        t = r["wave_t"]
        self.heat(np.clip(W, -0.3, 1.2).T, f"{len(W)} 个事件的波形（每列一个事件，按类别 1–{sl['K']} 从左到右排列，"
                  "已按峰值归一化；从上到下 = 时间，白线分隔各类）", "viridis",
                  rect=(0, t[0], len(W), t[-1] - t[0]))
        lab = r["ev_wave_sub_label"]
        edges = np.flatnonzero(np.diff(lab)) + 1
        for e in edges:
            self.view.add_overlay(pg.InfiniteLine(e, angle=90, pen=pg.mkPen("w", width=2)))
        self.main.model_overlays = True
        self._waves(t, r["ev_wave_mean"], r["ev_wave_sem"], "各类事件的平均波形（阴影 = 标准误）", "事件类")

    def kin_map(self, r):
        lab = r["cell_label"]
        s = r["summary"]
        if lab.max() >= 0:
            col = [colour(c) if c >= 0 else GREY for c in lab]
            self.cell_map(r, col, f"细胞层面的类别（颜色 = 类别 1–{s['cell']['K']}，灰色 = 事件太少未参与）",
                          sizes=np.where(lab >= 0, 11, 7))
            self._waves(r["wave_t"], r["cell_wave_mean"], r["cell_wave_sem"],
                        "各类细胞的平均事件波形（阴影 = 标准误）", "细胞类")
        elif "cell_dominant" in r:
            dom = r["cell_dominant"]
            col = [colour(c) if c >= 0 else GREY for c in dom]
            self.cell_map(r, col, "细胞层面没有聚类，按每个细胞最主要的事件类别着色（灰色 = 没有可用事件）")
            self._waves(r["wave_t"], r["ev_wave_mean"], r["ev_wave_sem"], "各类事件的平均波形", "事件类")
        else:
            raise ValueError("事件和细胞层面都没有得到聚类结果")

    def kin_check(self, r):
        C = r["cell_feat"]
        names = [str(n) for n in r["cell_feat_names"]]
        tau, snr = C[:, names.index("decay_tau_s")], C[:, names.index("snr")]
        ok = np.isfinite(tau) & np.isfinite(snr) & (tau > 0) & (snr > 0)
        if ok.sum() < 2:
            raise ValueError("可用的细胞太少，无法画混杂检查图")
        lab = r["cell_label"]
        ls = np.log10(np.where(snr > 0, snr, np.nan))
        col = [c if np.isfinite(v) else GREY for c, v in zip(_ramp(np.nan_to_num(ls, nan=np.nanmin(ls))), ls)]
        self.cell_map(r, col, "每个细胞的信噪比（事件振幅 / 噪声；越亮越强）。对照细胞图里的类别，看类别是否只是按亮暗分开")
        rr = r["summary"]["cell"].get("r_tau_snr")
        rt = "—" if rr is None else f"{rr:.2f}"
        tr = self.plot(f"混杂检查：衰减 τ 与信噪比（相关 r = {rt}，|r| 大说明测得的动力学受信号强弱影响；颜色 = 细胞类别）",
                       "log10 信噪比", "log10 衰减 τ（秒）")
        tr.getAxis("left").enableAutoSIPrefix(False)
        tr.addItem(pg.ScatterPlotItem(x=ls[ok], y=np.log10(tau[ok]), size=9, pen=pg.mkPen((0, 0, 0, 120)),
                                      brush=[pg.mkBrush(*(colour(c) if c >= 0 else GREY)) for c in lab[ok]]))

def show(main, key: str):
    """Draw the result view `key` (e.g. 'glm_map')."""
    prefix = {"glm": "glm", "dec": "decoder", "ens": "ensembles", "emb": "embedding", "hmm": "hmm",
              "net": "network", "info": "info", "kin": "kinetics"}[key.split("_")[0]]
    res = load_result(main.ws, prefix)
    getattr(Painter(main), key)(res)


def is_model_view(key: str) -> bool:
    return key.split("_")[0] in ("glm", "dec", "ens", "emb", "hmm", "net", "info", "kin") and "_" in key
