"""Step 10: register cells across sessions (days) and analyse them across days."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QGridLayout, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton,
                               QTableWidget, QTableWidgetItem, QWidget)

from ...core import crossday as X
from ...core import registration as R
from ..registration_views import VIEWS
from .base import RUN_STYLE, StepPage
from .common import heading, hint

TRANSFORMS = ["translation", "rigid", "nonrigid"]
XD_KINDS = ["react", "stab"]
XD_DESC = {
    "react": ("在「模板」session（例如条件化训练那天）里找细胞组，再用配准好的细胞身份，看同一组细胞在「目标」session"
              "（例如第二天的场景测试）里是否仍然一起活动。检验方法是打乱目标 session 里的细胞身份：如果真实的激活强度"
              "超过打乱后的 95%，说明这个细胞组被重新激活了。两个 session 都需要完成第 6 步。"),
    "stab": ("比较追踪到的细胞在各 session 的活动：事件率、session 之间的相关、每个 session 的细胞在其它 session 里找到的比例。"
             "如果 session 带同步好的行为数据，还会比较两天的僵直相关细胞是否是同一批细胞（超几何检验）。"),
}


def _run_reg(session, ws, args, progress=None, should_stop=None):
    s = R.run_registration(args["params"], args["out"], progress, should_stop)
    if ws is not None:
        ws.record_step("registration", {"sessions": s["sessions"], "names": s["names"]},
                       {"folder": str(args["out"])})
    return {"kind": "reg", "summary": s}


def _run_xd(session, ws, args, progress=None, should_stop=None):
    if args["kind"] == "react":
        s = X.run_reactivation(args["folder"], args["params"], progress, should_stop)
    else:
        s = X.run_stability(args["folder"], args["params"], progress, should_stop)
    return {"kind": args["kind"], "summary": s}


class RegistrationPage(StepPage):
    title = "10. 跨天细胞配准"
    intro = ("把同一只动物不同天（不同 session）的细胞对应起来：先对齐各天的视野，再按细胞中心距离和形状相似度，"
             "用概率模型判断哪些是同一个细胞（与 CellReg 的方法相同）。每个 session 需要先在自己的输出文件夹里完成第 5 步。")
    run_label = "开始配准"

    def __init__(self, main, parent=None):
        self.main = main
        self.folder: str | None = None       # registration result folder in use
        self._last_view = "reg_after"
        super().__init__(parent)

    # ------------------------------------------------------------- layout
    def build_params(self):
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["名称（可改）", "输出文件夹"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(130)
        self.form.addRow(QLabel("Session（第一个是参考）"))
        self.form.addRow(self.table)
        row = QHBoxLayout()
        for text, fn in (("添加…", self._add), ("移除", self._remove), ("上移", lambda: self._move(-1)),
                         ("下移", lambda: self._move(1))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        w = QWidget()
        w.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        self.form.addRow(w)
        self.form.addRow(hint("「添加…」选择其它天的输出文件夹（里面有 project.json）。名称会出现在表格和图里，"
                              "例如 Day1 训练、Day2 测试。其它 session 都对齐到第一个。"))
        self.image = self.add_param("对齐依据", self.combo(["细胞形状图（推荐）", "平均投影（血管）"]),
                                    "细胞形状图：把所有细胞叠在一起的图，与 CellReg 相同。平均投影：用血管等结构对齐，"
                                    "适合两天找到的细胞差别很大时。")
        self.transform = self.add_param("变换方式", self.combo(["只平移", "刚性：平移 + 旋转（推荐）",
                                                               "非刚性：刚性 + 局部形变"]),
                                        "镜头重新安装后视野通常会平移和轻微旋转。只有对齐后仍有局部错位时才用非刚性。")
        self.transform.setCurrentIndex(1)
        self.max_rot = self.add_param("最大旋转", self.dspin(0, 30, 10, 1, 1, " 度"))
        self.max_dist = self.add_param("最大距离", self.dspin(0, 100, 0, 1, 1, " 像素"),
                                       "两天的细胞中心相距超过这个距离就不可能是同一个细胞。0 = 自动（细胞直径的 1.5 倍）。")
        self.use_sim = self.add_param("", self.check("同时使用细胞形状相似度（推荐）", True),
                                      "不勾选时只用中心距离判断。")
        self.p_thr = self.add_param("匹配阈值", self.dspin(0.05, 0.95, 0.5, 0.05, 2),
                                    "「是同一个细胞」的概率超过这个值才配对。提高会更严格（误配少、漏配多）。")
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("保存配准结果的空文件夹")
        b = QPushButton("选择…")
        b.clicked.connect(self._choose_out)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.out_edit, 1)
        row.addWidget(b)
        w = QWidget()
        w.setLayout(row)
        self.add_param("结果文件夹", w, "需要一个空文件夹（不能是某个 session 的输出文件夹）。再次配准同一个文件夹会覆盖旧结果。")
        b = QPushButton("打开已有的配准结果…")
        b.clicked.connect(self._open_existing)
        self.form.addRow(b)

    def build_extra(self):
        self.lay.addWidget(heading("查看配准结果"))
        self.compare = self.combo([])
        f = QHBoxLayout()
        f.addWidget(QLabel("叠加显示"))
        f.addWidget(self.compare, 1)
        self.lay.addLayout(f)
        self.compare.currentIndexChanged.connect(self._compare_changed)
        self.reg_buttons = self._buttons(VIEWS["reg"])
        self.reg_lbl = self._result_box()
        b = QPushButton("导出表格（CSV）…")
        b.clicked.connect(self._export)
        self.export_btn = b
        self.lay.addWidget(b)

        self.lay.addWidget(heading("跨天分析"))
        self.xd_kind = self.combo(["细胞组重激活", "跨天活动稳定性"])
        self.xd_kind.currentIndexChanged.connect(self._on_xd_kind)
        g = QGridLayout()
        g.addWidget(QLabel("选择分析"), 0, 0)
        g.addWidget(self.xd_kind, 0, 1)
        self.xd_desc = hint("")
        self.xd_template = self.combo([])
        self.xd_target = self.combo([])
        self.xd_bin = self.dspin(0.03, 5, 0.5, 0.1, 2, " 秒")
        self.xd_method = self.combo(["PCA + ICA（推荐）", "NMF"])
        self.xd_signal = self.combo(["ΔF/F", "spike 事件"])
        self.xd_shuf = self.spin(50, 5000, 500, 100)
        self.xd_rows = []
        for i, (label, w) in enumerate((("模板 session", self.xd_template), ("目标 session", self.xd_target),
                                        ("时间分箱", self.xd_bin), ("方法", self.xd_method),
                                        ("使用的信号", self.xd_signal)), start=1):
            lb = QLabel(label)
            g.addWidget(lb, i, 0)
            g.addWidget(w, i, 1)
            self.xd_rows.append((lb, w))
        g.addWidget(QLabel("打乱次数"), 6, 0)
        g.addWidget(self.xd_shuf, 6, 1)
        self.lay.addLayout(g)
        self.lay.addWidget(self.xd_desc)
        self.xd_btn = QPushButton("运行跨天分析")
        self.xd_btn.setStyleSheet(RUN_STYLE)
        self.xd_btn.clicked.connect(self._run_xd)
        self.lay.addWidget(self.xd_btn)
        self.xd_buttons = self._buttons(VIEWS["react"])
        self.xd_lbl = self._result_box()
        self._on_xd_kind()

    def _buttons(self, views):
        grid = QGridLayout()
        out = []
        for i in range(4):
            b = QPushButton("")
            b.clicked.connect(lambda _=False, b=b: self._show(b.property("view")))
            grid.addWidget(b, i // 2, i % 2)
            out.append(b)
        self.lay.addLayout(grid)
        self._label_buttons(out, views)
        return out

    @staticmethod
    def _label_buttons(buttons, views):
        for b, (key, label) in zip(buttons, views):
            b.setText(label)
            b.setProperty("view", key)
            b.show()
        for b in buttons[len(views):]:
            b.hide()

    def _result_box(self):
        lbl = QLabel("")
        lbl.setWordWrap(True)
        lbl.setStyleSheet("background:#f4f4f4; padding:6px; border-radius:4px;")
        lbl.setVisible(False)
        self.lay.addWidget(lbl)
        return lbl

    # ------------------------------------------------------------ sessions
    def sessions(self):
        out = []
        for r in range(self.table.rowCount()):
            name = (self.table.item(r, 0).text() if self.table.item(r, 0) else "").strip()
            out.append((name, self.table.item(r, 1).toolTip()))
        return out

    def add_session(self, folder, name=""):
        folder = str(Path(folder))
        if any(Path(f) == Path(folder) for _, f in self.sessions()):
            return False
        if not (Path(folder) / "project.json").exists():
            QMessageBox.information(self, "提示", "这不是本软件的输出文件夹（里面没有 project.json）。")
            return False
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(name or f"Day{r + 1}"))
        it = QTableWidgetItem(Path(folder).name)
        it.setToolTip(folder)
        it.setFlags(it.flags() & ~pg_editable())
        self.table.setItem(r, 1, it)
        return True

    def _add(self):
        start = str(Path(self.ws.root).parent) if self.ws is not None else ""
        d = QFileDialog.getExistingDirectory(self, "选择另一天的输出文件夹", start)
        if d:
            self.add_session(d)

    def _remove(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _move(self, step):
        r = self.table.currentRow()
        t = r + step
        if r < 0 or not 0 <= t < self.table.rowCount():
            return
        for c in range(2):
            a, b = self.table.takeItem(r, c), self.table.takeItem(t, c)
            self.table.setItem(r, c, b)
            self.table.setItem(t, c, a)
        self.table.selectRow(t)

    def _choose_out(self):
        start = str(Path(self.ws.root).parent) if self.ws is not None else ""
        d = QFileDialog.getExistingDirectory(self, "选择保存配准结果的空文件夹", start)
        if d:
            self.out_edit.setText(d)

    def _open_existing(self):
        start = str(Path(self.ws.root).parent) if self.ws is not None else ""
        d = QFileDialog.getExistingDirectory(self, "选择配准结果文件夹", start)
        if d:
            self.open_result(d)

    def open_result(self, folder):
        try:
            s = R.load_registration(folder)["summary"]
        except Exception as e:
            QMessageBox.information(self, "提示", str(e))
            return False
        self.folder = str(folder)
        self.out_edit.setText(str(folder))
        self.table.setRowCount(0)
        for name, f in zip(s["names"], s["sessions"]):
            if (Path(f) / "project.json").exists():
                self.add_session(f, name)
        self.update_status()
        return True

    # ------------------------------------------------------------ plumbing
    def on_context(self):
        if self.ws is None:
            return
        st = self.ws.step("registration")
        if st and self.folder is None:
            f = st["outputs"].get("folder")
            if f and (Path(f) / R.META_FILE).exists():
                self.open_result(f)
                return
        if self.table.rowCount() == 0 and self.ws.step("extraction"):
            self.add_session(self.ws.root, "Day1")

    def update_status(self):
        has = self.folder is not None and (Path(self.folder) / R.META_FILE).exists()
        names = []
        text = ""
        if has:
            try:
                s = R.load_registration(self.folder)["summary"]
                names = s["names"]
                text = R.describe(s)
                self.status.setText(f"✓ 已完成（{s['created']}），结果在 {self.folder}")
            except Exception:
                has = False
        if not has:
            self.status.setText("")
        for b in self.reg_buttons:
            b.setEnabled(has)
        self.export_btn.setEnabled(has)
        self.reg_lbl.setText(text)
        self.reg_lbl.setVisible(bool(text))
        for c, items, default in ((self.compare, names[1:], 0), (self.xd_template, names, 0),
                                  (self.xd_target, names, 1)):
            cur = c.currentIndex()
            c.blockSignals(True)
            c.clear()
            c.addItems(items)
            c.setCurrentIndex(cur if 0 <= cur < len(items) else min(default, len(items) - 1))
            c.blockSignals(False)
        self.xd_btn.setEnabled(has)
        self._update_xd_status()

    def _update_xd_status(self):
        key = XD_KINDS[self.xd_kind.currentIndex()]
        res = f"crossday_{'reactivation' if key == 'react' else 'stability'}"
        done = self.folder is not None and X.has_result(self.folder, res)
        for b in self.xd_buttons:
            b.setEnabled(done)
        text = ""
        if done:
            try:
                s = X.load_crossday(self.folder, res)["summary"]
                text = X.describe_reactivation(s) if key == "react" else X.describe_stability(s)
            except Exception:
                text = ""
        self.xd_lbl.setText(text)
        self.xd_lbl.setVisible(bool(text))

    def _on_xd_kind(self, *_):
        key = XD_KINDS[self.xd_kind.currentIndex()]
        for lb, w in self.xd_rows:
            lb.setVisible(key == "react")
            w.setVisible(key == "react")
        self.xd_desc.setText(XD_DESC[key])
        self._label_buttons(self.xd_buttons, VIEWS[key])
        self._update_xd_status()
        if hasattr(self.main, "refit_page"):
            self.main.refit_page()

    def _compare_changed(self, *_):
        if self._last_view in ("reg_after", "reg_before") and self.folder:
            self._show(self._last_view)

    def params(self):
        sessions = self.sessions()
        p = R.RegParams(sessions=[f for _, f in sessions], names=[n for n, _ in sessions],
                        image="footprints" if self.image.currentIndex() == 0 else "mean",
                        transform=TRANSFORMS[self.transform.currentIndex()],
                        max_rotation=self.max_rot.value(), max_dist=self.max_dist.value(),
                        use_similarity=self.use_sim.isChecked(), p_threshold=self.p_thr.value())
        return {"params": p, "out": self.out_edit.text().strip()}

    def run_function(self):
        return _run_reg

    def _run(self):
        if self.ws is None:
            return
        args = self.params()
        p = args["params"]
        if len(p.sessions) < 2:
            QMessageBox.information(self, "提示", "请至少添加 2 个 session（不同天的输出文件夹）。")
            return
        if len({n for n in p.names}) != len(p.names) or not all(p.names):
            QMessageBox.information(self, "提示", "每个 session 需要一个不重复的名称。")
            return
        if not args["out"]:
            QMessageBox.information(self, "提示", "请选择保存配准结果的文件夹。")
            return
        try:
            R.check_output_folder(args["out"])
        except ValueError as e:
            QMessageBox.information(self, "提示", str(e))
            return
        self._pending = args["out"]
        self._last_view = "reg_after"
        self.runRequested.emit(_run_reg, args)

    def _run_xd(self):
        if self.folder is None:
            QMessageBox.information(self, "提示", "请先完成配准（或打开已有的配准结果）。")
            return
        key = XD_KINDS[self.xd_kind.currentIndex()]
        if key == "react":
            if self.xd_template.currentIndex() == self.xd_target.currentIndex():
                QMessageBox.information(self, "提示", "模板 session 和目标 session 不能相同。")
                return
            p = X.ReactivationParams(template=self.xd_template.currentIndex(), target=self.xd_target.currentIndex(),
                                     bin_s=self.xd_bin.value(),
                                     method="ica" if self.xd_method.currentIndex() == 0 else "nmf",
                                     signal="dff" if self.xd_signal.currentIndex() == 0 else "events",
                                     n_shuffle=self.xd_shuf.value())
            self._last_view = "xd_react"
        else:
            p = X.StabilityParams(n_shuffle=self.xd_shuf.value())
            self._last_view = "xd_rates"
        self.runRequested.emit(_run_xd, {"kind": key, "folder": self.folder, "params": p})

    def set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.xd_btn.setEnabled(not running and self.folder is not None)

    def on_finished(self, result):
        if isinstance(result, dict) and result.get("kind") == "reg":
            self.folder = str(getattr(self, "_pending", self.out_edit.text()))
        self.update_status()

    def default_view(self):
        return self._last_view

    def _show(self, key):
        if key:
            self._last_view = key
            self.showRequested.emit(key)

    def show_view(self, key):
        from ..registration_views import show
        show(self.main, key, self.folder, self.compare.currentIndex() + 1)

    def _export(self):
        if self.folder is None:
            return
        d = QFileDialog.getExistingDirectory(self, "选择导出文件夹", self.folder)
        if not d:
            return
        try:
            import shutil
            done = [Path(self.folder) / R.CSV_FILE]
            done += X.export_crossday(self.folder)
            for f in done:
                if Path(d) != Path(self.folder):
                    shutil.copy(f, Path(d) / f.name)
            QMessageBox.information(self, "导出完成",
                                    f"已保存到:\n{d}\n" + "\n".join(f.name for f in done) +
                                    "\n\ncell_registration.csv：每行一个细胞，各列是它在每个 session 里的细胞编号（空 = 没找到）。")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))


def pg_editable():
    from PySide6.QtCore import Qt
    return Qt.ItemIsEditable
