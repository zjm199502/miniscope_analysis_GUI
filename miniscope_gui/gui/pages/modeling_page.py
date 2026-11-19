"""Step 9: models linking population activity and behaviour."""

from __future__ import annotations

from PySide6.QtWidgets import (QFileDialog, QFormLayout, QGridLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QWidget)

from ...core.modeling import decoder, embedding, ensembles, glm, hmm, info, kinetics, network
from ...core.modeling.common import has_behavior, load_result
from ...core.modeling.report import ANALYSES, VIEWS, describe, export_all
from .base import StepPage
from .common import heading, hint

RUNNERS = {"glm": glm.run_glm, "decoder": decoder.run_decoder, "ensembles": ensembles.run_ensembles,
           "embedding": embedding.run_embedding, "hmm": hmm.run_hmm, "network": network.run_network,
           "info": info.run_info, "kinetics": kinetics.run_kinetics}
SIGNALS = ["ΔF/F", "spike 事件"]


class ModelingPage(StepPage):
    title = "9. 建模分析"
    intro = ("用数学模型分析神经活动和行为的关系。先在下面选一种分析，看灰色说明，再点「运行」。"
             "每种分析的结果都会保存，可以随时回来查看。")
    run_label = "运行这项分析"

    def __init__(self, main, parent=None):
        self.main = main
        self.forms: dict[str, QWidget] = {}
        self.keys = list(ANALYSES)
        super().__init__(parent)

    # ------------------------------------------------------------- layout
    def _form(self, key):
        w = QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(0, 0, 0, 0)
        f.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.forms[key] = w
        self.form.addRow(w)
        return f

    @staticmethod
    def _row(f, label, widget, help_text=None):
        f.addRow(label, widget)
        if help_text:
            f.addRow(hint(help_text))
        return widget

    def build_params(self):
        self.kind = self.combo([ANALYSES[k][0] for k in self.keys])
        self.form.addRow("选择分析", self.kind)
        self.desc = hint("")
        self.form.addRow(self.desc)
        self.need = QLabel("")
        self.need.setWordWrap(True)
        self.need.setStyleSheet("color:#b7791f;")
        self.form.addRow(self.need)
        bin_help = "先把信号按这个时间长度取平均再分析。钙信号变化慢，0.2–1 秒比较合适。"

        f = self._form("glm")
        self.glm_bin = self._row(f, "时间分箱", self.dspin(0.03, 5, 0.2, 0.1, 2, " 秒"), bin_help)
        self.glm_fz = self.check("僵直", True)
        self.glm_speed = self.check("速度", True)
        self.glm_pos = self.check("位置", True)
        self.glm_zone = self.check("区域", True)
        g = QGridLayout()
        for i, c in enumerate((self.glm_fz, self.glm_speed, self.glm_pos, self.glm_zone)):
            g.addWidget(c, i // 2, i % 2)
        f.addRow("行为变量", g)
        self.glm_events = QLineEdit()
        self.glm_events.setPlaceholderText("例如电击时间：120, 180, 240")
        self._row(f, "事件时间", self.glm_events, "可选。单位秒，miniscope 时间（0 = 第一帧）。例如电击、声音开始的时刻。")
        self.glm_post = self._row(f, "事件后窗口", self.dspin(1, 60, 6, 1, 1, " 秒"),
                                  "事件（以及僵直开始）之后多长时间内的反应算作由它引起。")
        self.glm_shuf = self._row(f, "打乱次数", self.spin(20, 2000, 100, 20),
                                  "越多，p 值越精细，但越慢。100 次时最小 p 值约 0.01。")

        f = self._form("decoder")
        self.dec_bin = self._row(f, "时间分箱", self.dspin(0.03, 5, 0.2, 0.1, 2, " 秒"), bin_help)
        self.dec_target = self._row(f, "解码目标", self.combo(["僵直", "速度", "位置"]))
        self.dec_signal = self._row(f, "使用的信号", self.combo(SIGNALS))
        self.dec_pc = self._row(f, "主成分数", self.spin(2, 500, 50),
                                "先把群体活动降到这么多维再训练，防止细胞多、数据少时过拟合。")
        self.dec_shuf = self._row(f, "打乱次数", self.spin(20, 5000, 200, 50))

        f = self._form("ensembles")
        self.ens_bin = self._row(f, "时间分箱", self.dspin(0.03, 5, 0.5, 0.1, 2, " 秒"),
                                 "「一起活动」的时间精度。0.5–1 秒适合钙信号。")
        self.ens_method = self._row(f, "方法", self.combo(["PCA + ICA（推荐）", "NMF（非负矩阵分解）"]),
                                    "ICA 是文献里最常用的方法；NMF 的权重都是正数，更容易看懂。")
        self.ens_signal = self._row(f, "使用的信号", self.combo(SIGNALS))
        self.ens_n = self._row(f, "细胞组数量", self.spin(0, 50, 0),
                               "0 = 自动（特征值超过打乱数据 95% 水平的个数）。")
        self.ens_sd = self._row(f, "成员阈值", self.dspin(0.5, 10, 2.0, 0.5, 1, " 倍标准差"),
                                "权重比大多数细胞高出这么多（并且高于 1/√细胞数）的细胞算作该细胞组的成员。")

        f = self._form("embedding")
        self.emb_bin = self._row(f, "时间分箱", self.dspin(0.03, 5, 0.2, 0.1, 2, " 秒"), bin_help)
        self.emb_mode = self._row(f, "模式", self.combo(["行为模式（需要行为数据）", "时间模式（不需要行为）"]),
                                  "行为模式：行为相近的时刻靠近。时间模式：时间相近的时刻靠近，完全不看行为。")
        self.emb_fz = self.check("僵直", True)
        self.emb_pos = self.check("位置", True)
        self.emb_speed = self.check("速度", False)
        g = QGridLayout()
        for i, c in enumerate((self.emb_fz, self.emb_pos, self.emb_speed)):
            g.addWidget(c, 0, i)
        f.addRow("行为标签", g)
        self.emb_steps = self._row(f, "训练步数", self.spin(200, 20000, 2000, 500),
                                   "2000 步在普通笔记本上大约 1 分钟。")
        self.emb_hold = self._row(f, "", self.check("留出 20% 数据做检验", True),
                                  "用没参与训练的数据检查嵌入里是否真的有行为信息，并和打乱标签的对照比较（训练时间加倍）。")
        b = QPushButton("导出给官方 CEBRA…")
        b.clicked.connect(self._export_cebra)
        f.addRow(b)

        f = self._form("hmm")
        self.hmm_bin = self._row(f, "时间分箱", self.dspin(0.03, 5, 0.5, 0.1, 2, " 秒"), bin_help)
        self.hmm_kind = self._row(f, "模型", self.combo(["HMM（从这个开始）", "AR-HMM", "循环 AR-HMM（rSLDS 简化版）"]))
        self.hmm_k = self._row(f, "状态数", self.spin(2, 12, 4), "先试 3–5 个。不确定时勾选下面的「比较状态数」。")
        self.hmm_d = self._row(f, "主成分数", self.spin(1, 30, 8), "模型在群体活动的前几个主成分上拟合。")
        self.hmm_sticky = self._row(f, "状态粘性", self.dspin(0, 1000, 10, 5, 1),
                                    "越大，状态越不容易频繁跳变。")
        self.hmm_cmp = self._row(f, "", self.check("比较状态数（2–8，较慢）", False),
                                 "用留出的数据比较不同状态数的似然，帮助选择状态数。")

        f = self._form("network")
        self.net_bin = self._row(f, "时间分箱", self.dspin(0.03, 5, 0.5, 0.1, 2, " 秒"), bin_help)
        self.net_signal = self._row(f, "使用的信号", self.combo(SIGNALS))
        self.net_thr = self._row(f, "连边标准", self.combo(["超过打乱水平（推荐）", "固定连接密度"]))
        self.net_pct = self._row(f, "打乱百分位", self.dspin(90, 99.99, 99.9, 0.1, 2, " %"),
                                 "相关系数高于打乱数据的这个百分位才连边。越高，边越少、越可靠。")
        self.net_density = self._row(f, "连接密度", self.dspin(0.5, 50, 5, 1, 1, " %"),
                                     "选「固定连接密度」时使用：保留相关最强的这部分细胞对。比较不同动物时常用。")
        self.net_state = self._row(f, "", self.check("分别计算僵直 / 非僵直时的网络", True))

        f = self._form("info")
        self.info_bin = self._row(f, "时间分箱", self.dspin(0.03, 5, 0.2, 0.1, 2, " 秒"), bin_help)
        self.info_signal = self._row(f, "使用的信号", self.combo(["spike 事件", "ΔF/F"]),
                                     "信息量通常用 spike 事件计算。")
        self.info_nbin = self._row(f, "空间格子数", self.spin(3, 40, 8), "把场地每个方向分成多少格。恐惧箱 6–10，旷场 15–20。")
        self.info_speed = self._row(f, "最低速度", self.dspin(0, 100, 0, 0.5, 1),
                                    "只用速度高于此值的时刻计算空间信息（单位同行为分析）。0 = 全部使用。")
        self.info_shuf = self._row(f, "打乱次数", self.spin(50, 5000, 500, 100))

        f = self._form("kinetics")
        self.kin_smooth = self._row(f, "平滑", self.dspin(0, 1, 0.05, 0.01, 2, " 秒"),
                                    "测量前对 ΔF/F 做轻度高斯平滑以压低噪声。越大越平滑，但上升时间会被拉长；0 = 不平滑。")
        self.kin_thr = self._row(f, "事件阈值", self.dspin(1.5, 10, 3.0, 0.5, 1, " 倍噪声"),
                                 "从事件前的低谷到峰值至少升高这么多倍噪声才算一个事件。")
        self.kin_decay = self._row(f, "衰减窗口", self.dspin(1, 30, 6, 1, 0, " 秒"),
                                   "峰值之后最多用多长时间拟合衰减。慢指示剂（GCaMP6s）可以加大。")
        self.kin_iso = self._row(f, "", self.check("只用孤立事件测量和聚类（推荐）", True),
                                 "孤立事件：前面的基线已回落，衰减在下一个事件前降到 20% 以下。叠加事件的衰减测不准。")
        self.kin_feat = {}
        g = QGridLayout()
        defaults = set(kinetics.KineticsParams().features)
        for i, (k, label) in enumerate(kinetics.CLUSTER_CHOICES.items()):
            c = self.check(label, k in defaults)
            self.kin_feat[k] = c
            g.addWidget(c, i // 3, i % 3)
        f.addRow("聚类指标", g)
        f.addRow(hint("前几项按每个事件计算；细胞层面用每个细胞的中位数，「事件频率」「爆发性」只用于细胞层面。"
                      "振幅、面积默认不用：它们主要反映指示剂表达量和信噪比。"))
        self.kin_min = self._row(f, "每个细胞最少事件", self.spin(2, 200, 5),
                                 "事件少于这个数的细胞不参与细胞层面的聚类（中位数不可靠）。")
        self.kin_kmax = self._row(f, "最多类别数", self.spin(2, 10, 6))
        self.kin_k = self._row(f, "固定类别数", self.spin(0, 10, 0),
                               "0 = 自动选择（推荐）。只在想比较某个特定类别数时填写。")
        self.kin_crit = self._row(f, "选择标准", self.combo(["ICL（推荐，偏向分得开的类别）", "BIC"]),
                                  "BIC 在事件多时容易把一个偏态分布切成好几类；ICL 额外惩罚互相重叠的类别。")
        self.kin_boot = self._row(f, "重抽样次数", self.spin(10, 500, 50, 10),
                                  "用于检验类别是否稳定，越多越可靠也越慢。")

        self.kind.currentIndexChanged.connect(self._on_kind)

    def build_extra(self):
        self.lay.addWidget(heading("查看结果"))
        self.grid = QGridLayout()
        self.view_buttons = []
        n = max(len(v) for v in VIEWS.values())
        for i in range(n):
            b = QPushButton("")
            b.clicked.connect(lambda _=False, i=i: self._show(i))
            self.grid.addWidget(b, i // 2, i % 2)
            self.view_buttons.append(b)
        self.export_btn = QPushButton("导出全部建模结果…")
        self.export_btn.clicked.connect(self._export)
        self.grid.addWidget(self.export_btn, n // 2, n % 2)
        self.lay.addLayout(self.grid)
        self.result_lbl = QLabel("")
        self.result_lbl.setWordWrap(True)
        self.result_lbl.setStyleSheet("background:#f4f4f4; padding:6px; border-radius:4px;")
        self.lay.addWidget(self.result_lbl)
        self._on_kind()

    # ------------------------------------------------------------ plumbing
    @property
    def key(self) -> str:
        return self.keys[self.kind.currentIndex()]

    def _cfg(self):
        if self.ws is not None and has_behavior(self.ws):
            return self.main.beh_page.params()
        return None

    def _on_kind(self, *_):
        key = self.key
        for k, w in self.forms.items():
            w.setVisible(k == key)
        self.desc.setText(ANALYSES[key][2])
        for b, (_, label) in zip(self.view_buttons, VIEWS[key]):
            b.setText(label)
        for b in self.view_buttons[len(VIEWS[key]):]:
            b.hide()
        for b in self.view_buttons[:len(VIEWS[key])]:
            b.show()
        self.update_status()
        if hasattr(self.main, "refit_page"):
            self.main.refit_page()

    def on_context(self):
        if self.ws is None:
            return
        cur = self.dec_target.currentText()
        items = ["僵直", "速度", "位置"]
        cfg = self._cfg()
        if cfg is not None:
            items += [f"区域 {z['name']}" for z in cfg.zones]
        self.dec_target.clear()
        self.dec_target.addItems(items)
        if cur in items:
            self.dec_target.setCurrentText(cur)

    def update_status(self):
        if self.ws is None:
            return
        key = self.key
        st = self.ws.step(f"model_{key}")
        self.status.setText(f"✓ 已完成（{st['finished']}）" if st else "")
        for b in self.view_buttons:
            b.setEnabled(bool(st))
        self.export_btn.setEnabled(any(self.ws.step(f"model_{k}") for k in self.keys))
        beh = has_behavior(self.ws)
        if ANALYSES[key][1] and not beh:
            self.need.setText("这项分析需要行为数据：请先完成第 7 步（行为分析）和第 8 步（同步分析）。")
        elif not beh:
            self.need.setText("没有同步好的行为数据也可以运行；有的话，结果里会多出与僵直的比较。")
        else:
            self.need.setText("")
        text = ""
        if key == "decoder":
            self.view_buttons[3].hide()
        if st:
            try:
                result = load_result(self.ws, key)
                if key == "decoder":
                    position = result["summary"]["target"] == "位置"
                    self.view_buttons[2].setText("位置 X 权重" if position else "细胞权重图")
                    self.view_buttons[3].setVisible(position and result["weights"].ndim == 2)
                text = describe(key, result["summary"])
            except Exception:
                text = ""
        self.result_lbl.setText(text)
        self.result_lbl.setVisible(bool(text))

    def params(self):
        key = self.key
        sig = lambda c: "dff" if c.currentText() == "ΔF/F" else "events"
        if key == "glm":
            p = glm.GLMParams(bin_s=self.glm_bin.value(), use_freezing=self.glm_fz.isChecked(),
                              use_speed=self.glm_speed.isChecked(), use_position=self.glm_pos.isChecked(),
                              use_zones=self.glm_zone.isChecked(), event_times=self.glm_events.text(),
                              kernel_post_s=self.glm_post.value(), n_shuffle=self.glm_shuf.value())
        elif key == "decoder":
            p = decoder.DecoderParams(bin_s=self.dec_bin.value(), target=self.dec_target.currentText(),
                                      signal=sig(self.dec_signal), n_pc=self.dec_pc.value(),
                                      n_shuffle=self.dec_shuf.value())
        elif key == "ensembles":
            p = ensembles.EnsembleParams(bin_s=self.ens_bin.value(),
                                         method="ica" if self.ens_method.currentIndex() == 0 else "nmf",
                                         signal=sig(self.ens_signal), n_assemblies=self.ens_n.value(),
                                         member_sd=self.ens_sd.value())
        elif key == "embedding":
            p = embedding.EmbedParams(bin_s=self.emb_bin.value(),
                                      mode="behavior" if self.emb_mode.currentIndex() == 0 else "time",
                                      use_freezing=self.emb_fz.isChecked(), use_position=self.emb_pos.isChecked(),
                                      use_speed=self.emb_speed.isChecked(), steps=self.emb_steps.value(),
                                      holdout=self.emb_hold.isChecked())
        elif key == "hmm":
            p = hmm.HMMParams(bin_s=self.hmm_bin.value(), kind=hmm.KINDS[self.hmm_kind.currentIndex()],
                              n_states=self.hmm_k.value(), n_latent=self.hmm_d.value(),
                              sticky=self.hmm_sticky.value(), compare_k=self.hmm_cmp.isChecked())
        elif key == "network":
            p = network.NetworkParams(bin_s=self.net_bin.value(), signal=sig(self.net_signal),
                                      threshold="shuffle" if self.net_thr.currentIndex() == 0 else "density",
                                      percentile=self.net_pct.value(), density=self.net_density.value() / 100.0,
                                      by_state=self.net_state.isChecked())
        elif key == "kinetics":
            p = kinetics.KineticsParams(smooth_s=self.kin_smooth.value(), threshold_sd=self.kin_thr.value(),
                                        max_decay_s=self.kin_decay.value(), isolated_only=self.kin_iso.isChecked(),
                                        features=tuple(k for k, c in self.kin_feat.items() if c.isChecked()),
                                        min_events=self.kin_min.value(), max_k=self.kin_kmax.value(),
                                        force_k=self.kin_k.value(),
                                        criterion="icl" if self.kin_crit.currentIndex() == 0 else "bic",
                                        n_boot=self.kin_boot.value())
        else:
            p = info.InfoParams(bin_s=self.info_bin.value(), signal=sig(self.info_signal),
                                n_spatial_bins=self.info_nbin.value(), min_speed=self.info_speed.value(),
                                n_shuffle=self.info_shuf.value())
        return {"params": p, "cfg": self._cfg()}

    def run_function(self):
        return RUNNERS[self.key]

    def _run(self):
        if self.ws is None:
            return
        if not self.ws.step("signals"):
            QMessageBox.information(self, "提示", "请先完成第 6 步「信号与 spike」")
            return
        if ANALYSES[self.key][1] and not has_behavior(self.ws) and not (
                self.key == "glm" and self.glm_events.text().strip()):
            QMessageBox.information(self, "提示", "这项分析需要行为数据：请先完成第 7 步（行为分析）和第 8 步（同步分析）。")
            return
        self.runRequested.emit(self.run_function(), self.params())

    def default_view(self):
        return VIEWS[self.key][0][0]

    def _show(self, i):
        self.showRequested.emit(VIEWS[self.key][i][0])

    def _export(self):
        d = QFileDialog.getExistingDirectory(self, "选择导出文件夹", str(self.ws.root))
        if not d:
            return
        try:
            done = export_all(self.ws, d)
            QMessageBox.information(self, "导出完成", f"已保存到:\n{d}\n共 {len(done)} 项分析的 CSV 表格和 modeling_summary.txt")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))

    def _export_cebra(self):
        if self.ws is None or not self.ws.step("signals"):
            QMessageBox.information(self, "提示", "请先完成第 6 步「信号与 spike」")
            return
        d = QFileDialog.getExistingDirectory(self, "选择导出文件夹", str(self.ws.root))
        if not d:
            return
        try:
            embedding.export_for_cebra(self.ws, self._cfg(), d, self.emb_bin.value())
            QMessageBox.information(self, "导出完成", f"已保存到:\n{d}\ncebra_input.npz：数据\nrun_cebra.py：在装有官方 cebra 包的 Python 里运行")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))
