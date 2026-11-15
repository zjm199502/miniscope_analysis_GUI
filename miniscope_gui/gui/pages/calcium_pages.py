"""Pages for steps 3-6: motion correction, background, cell extraction, signals."""

from __future__ import annotations

from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton

from ...core.background import BackgroundParams, run_background
from ...core.extraction import ExtractionParams, run_extraction
from ...core.motion import MotionParams, run_motion_correction
from ...core.signals import INDICATORS, SignalParams, run_signals
from .base import StepPage
from .common import heading, hint


class MotionPage(StepPage):
    step_key = "motion"
    title = "3. 运动校正"
    intro = ("动物活动时，脑组织会相对镜头轻微移动。这一步以血管等结构为参照，"
             "把每一帧对齐到同一个参考模板上。")
    run_label = "开始运动校正"
    results = (("mc", "校正后视频"), ("mc_compare", "校正前后对比"),
               ("mc_shifts", "位移曲线"), ("mc_mean", "平均投影"))
    results_hint = "对比视频里，校正后的画面应该基本不再抖动；位移曲线显示每一帧移动了多少像素。"

    def build_params(self):
        self.max_shift = self.add_param("最大位移", self.spin(2, 100, 20, suffix=" 像素"),
                                        "允许的最大移动距离（预处理后的像素）。一般不用改。")
        self.piecewise = self.check("分块非刚性校正（较慢）")
        self.form.addRow(self.piecewise)
        self.form.addRow(hint("只做整体平移通常就够了。如果画面不同部位移动不一致"
                              "（例如边缘有扭曲），再勾选这一项。"))
        self.patch = self.add_param("分块大小", self.spin(32, 256, 96, 16, " 像素"))

    def params(self):
        return MotionParams(max_shift=self.max_shift.value(), piecewise=self.piecewise.isChecked(),
                            patch_size=self.patch.value(), patch_overlap=self.patch.value() // 3)

    def run_function(self):
        return run_motion_correction

    def on_finished(self, result):
        self.status.setText(
            f"✓ 完成。平均位移 {result['mean_abs_shift']:.2f} 像素，最大 {result['max_abs_shift']:.1f} 像素；"
            f"与模板的相似度 {result['corr_before']:.2f} → {result['corr_after']:.2f}")


class BackgroundPage(StepPage):
    step_key = "background"
    title = "4. 去除背景"
    intro = ("单光子成像里，焦平面外的荧光形成大片模糊的背景。这一步先轻微去噪，"
             "再用比细胞稍大的圆盘做形态学 tophat，去掉比细胞大的结构，只留下细胞。")
    run_label = "开始去除背景"
    results = (("bg", "去背景后视频"), ("bg_max", "最大投影"))
    results_hint = "最大投影上应该能看到很多清晰、分开的亮点，每个亮点就是一个细胞。"

    def build_params(self):
        self.diam = self.add_param("细胞直径", self.spin(3, 60, 10, suffix=" 像素"),
                                   "预处理后画面里一个细胞大约多宽。可以在最大投影上量一下："
                                   "608×608 原始画面、空间降采样 2 时，通常是 6–12 像素。")
        self.median = self.add_param("去噪强度", self.combo(["关闭", "3", "5"], "3"),
                                     "中值滤波窗口，3 适合大多数数据。")

    def params(self):
        m = self.median.currentText()
        return BackgroundParams(cell_diameter=self.diam.value(),
                                median_ksize=1 if m == "关闭" else int(m))

    def run_function(self):
        return run_background


class ExtractionPage(StepPage):
    step_key = "extraction"
    title = "5. 细胞提取"
    intro = ("先计算每个像素和周围像素的相关性（相关图）以及信号强度（PNR），"
             "两者都高的位置作为候选细胞；再在每个细胞周围的小区域内反复优化细胞形状和活动曲线。")
    run_label = "开始提取细胞"
    results = (("cells", "检查细胞"), ("cn", "相关图"), ("pnr", "PNR 图"), ("bg_max", "最大投影"))

    def build_params(self):
        self.diam = self.add_param("细胞直径", self.spin(3, 60, 10, suffix=" 像素"),
                                   "和上一步保持一致。")
        self.min_corr = self.add_param("最小相关", self.dspin(0.3, 0.99, 0.8, 0.05),
                                       "候选细胞中心与周围像素的相关系数下限。漏掉细胞时调低（如 0.7），"
                                       "假细胞太多时调高。")
        self.min_pnr = self.add_param("最小 PNR", self.dspin(2, 50, 8, 0.5, 1),
                                      "峰值与噪声之比的下限。信号弱的数据可以调到 5–6。")
        self.merge = self.add_param("合并阈值", self.dspin(0.5, 0.99, 0.8, 0.05),
                                    "两个重叠的候选细胞活动相关性超过这个值，就认为是同一个细胞并合并。")

    def build_extra(self):
        self.lay.addWidget(heading("人工检查"))
        self.review_info = QLabel("")
        self.review_info.setWordWrap(True)
        self.lay.addWidget(self.review_info)
        self.lay.addWidget(hint("在「检查细胞」视图中：点击细胞查看它的活动曲线；"
                                "按住 Ctrl 点击可以多选。绿色 = 保留，红色 = 删除。"))
        row = QHBoxLayout()
        self.toggle_btn = QPushButton("删除 / 恢复所选")
        self.merge_btn = QPushButton("合并所选")
        row.addWidget(self.toggle_btn)
        row.addWidget(self.merge_btn)
        self.lay.addLayout(row)
        row2 = QHBoxLayout()
        self.prev_btn = QPushButton("◀ 上一个")
        self.next_btn = QPushButton("下一个 ▶")
        row2.addWidget(self.prev_btn)
        row2.addWidget(self.next_btn)
        self.lay.addLayout(row2)

    def params(self):
        return ExtractionParams(cell_diameter=self.diam.value(), min_corr=self.min_corr.value(),
                                min_pnr=self.min_pnr.value(), merge_threshold=self.merge.value())

    def run_function(self):
        return run_extraction

    def on_context(self):
        st = self.ws.step("background") if self.ws else None
        if st:
            self.diam.setValue(int(st["params"].get("cell_diameter", 10)))

    def on_finished(self, result):
        self.status.setText(f"✓ 找到 {result['n_cells']} 个细胞")


class SignalsPage(StepPage):
    step_key = "signals"
    title = "6. 信号与 spike"
    intro = ("ΔF/F：细胞荧光的变化量除以细胞所在位置的基线荧光（取一段时间窗内的低百分位数）。"
             "单光子数据的基线包含大量背景荧光，所以典型值在 0.02–0.3 之间，比双光子小。"
             "spike：用 OASIS 算法从钙信号反推出放电事件的时间和大小。")
    run_label = "计算信号"
    results = (("traces", "所有细胞活动（热图）"), ("cells", "单个细胞曲线"))

    def build_params(self):
        self.indicator = self.add_param(
            "钙指示剂", self.combo(list(INDICATORS) + ["自定义"], "GCaMP6f"),
            "用于设定钙信号的衰减时间。")
        self.tau = self.add_param("衰减时间", self.dspin(0.05, 5.0, 0.7, 0.05, 2, " 秒"),
                                  "选「自定义」时生效。")
        self.indicator.currentTextChanged.connect(self._sync_tau)
        self.win = self.add_param("基线窗口", self.dspin(5, 600, 60, 5, 0, " 秒"),
                                  "在这个时间窗内取低百分位数作为基线 F0。")
        self.thr = self.add_param("spike 阈值", self.dspin(0.5, 10, 3.0, 0.5, 1, " × 噪声"),
                                  "小于这个大小的事件不算 spike。数值越大越保守。")
        self._sync_tau(self.indicator.currentText())

    def build_extra(self):
        self.lay.addWidget(heading("导出"))
        self.lay.addWidget(hint("导出 ΔF/F、z-score、spike、细胞位置等 CSV 文件（Excel 可直接打开）"
                                "和一个 MATLAB 的 .mat 文件。只导出检查时保留的细胞。"))
        self.export_btn = QPushButton("导出结果…")
        self.export_btn.clicked.connect(self._export)
        self.lay.addWidget(self.export_btn)

    def _sync_tau(self, name):
        if name in INDICATORS:
            self.tau.setValue(INDICATORS[name])
        self.tau.setEnabled(name == "自定义")

    def params(self):
        name = self.indicator.currentText()
        return SignalParams(indicator=name if name in INDICATORS else "custom",
                            tau_s=self.tau.value(), baseline_window_s=self.win.value(),
                            spike_threshold=self.thr.value())

    def run_function(self):
        return run_signals

    def update_status(self):
        super().update_status()
        self.export_btn.setEnabled(bool(self.ws and self.ws.step("signals")))

    def _export(self):
        from ...core.export import export_results
        d = QFileDialog.getExistingDirectory(self, "选择导出文件夹", str(self.ws.root))
        if not d:
            return
        try:
            out = export_results(self.ws, d)
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))
            return
        QMessageBox.information(self, "导出完成", f"已保存到:\n{out}")
