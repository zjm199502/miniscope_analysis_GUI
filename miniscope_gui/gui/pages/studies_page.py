"""Behaviour study controls, saved reports and synchronized bout playback."""

import csv
from pathlib import Path

import numpy as np
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
)

from ...core.study_report import run_study
from ...core.sync import load_sync
from .base import StepPage


class StudiesPage(StepPage):
    title = "行为与空间专题"
    intro = "使用已同步的轨迹和保留细胞的 ΔF/F。每次运行保存独立报告和数据表。"
    modes = ("freeze", "days", "target", "wall")

    def __init__(self, main):
        self.main = main
        self.folder = None
        super().__init__(main)
        self.mode.currentIndexChanged.connect(self.update_status)
        self.update_status()

    def build_params(self):
        self.mode = self.add_param(
            "专题",
            self.combo(["僵直起始与解除", "跨天位置解码", "目标移位", "距墙调谐"]),
        )
        self.speed = self.add_param(
            "普通停顿速度上限（标定后 cm/s，否则 px/s）", self.dspin(0, 1000, 2)
        )
        self.window = self.add_param(
            "事件前后各取", self.dspin(0.2, 120, 2, suffix=" 秒")
        )
        self.bins = self.add_param("每轴分箱数", self.spin(2, 40, 8))
        self.schedule = self.add_param(
            "各阶段起始钙记录秒数、目标 X、Y（视频像素）", QPlainTextEdit()
        )
        self.schedule.setPlaceholderText("每行一个阶段，例如\n0, 120, 80\n300, 240, 80")
        self.schedule.setMaximumHeight(95)
        self.reg = self.add_param("配准结果目录", QLineEdit())
        self.browse = self.add_param("", QPushButton("选择配准目录…"))
        self.browse.clicked.connect(self._browse)
        self.aligned = self.add_param(
            "", self.check("已确认各天场地相同、坐标朝向一致")
        )

    def build_extra(self):
        self.report = QPushButton("打开专题报告")
        self.files = QPushButton("查看导出文件夹")
        self.events = QComboBox()
        self.replay = QPushButton("回放所选事件")
        for widget in (self.report, self.files, self.events, self.replay):
            self.lay.addWidget(widget)
        self.report.clicked.connect(lambda: self._open("report.html"))
        self.files.clicked.connect(lambda: self._open(""))
        self.replay.clicked.connect(self._replay)

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "选择跨天配准结果目录")
        if path:
            self.reg.setText(path)

    def _open(self, name):
        if self.folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.folder / name)))

    def _replay(self):
        if self.main.beh_page.video is None:
            QMessageBox.information(
                self, "事件回放", "请先在行为分析页载入与同步记录对应的行为视频。"
            )
            return
        times = load_sync(self.ws)
        indices = np.flatnonzero(np.isfinite(times))
        if not len(indices):
            return
        frame = indices[np.argmin(abs(times[indices] - self.events.currentData()))]
        self.main.beh_page.show_video()
        self.main.view.slider.setValue(int(frame))
        self.main.view.toggle_play()

    def params(self):
        mode = self.modes[self.mode.currentIndex()]
        args = dict(mode=mode, title=self.mode.currentText(), bins=self.bins.value())
        if mode == "freeze":
            args.update(window=self.window.value(), speed=self.speed.value())
        elif mode == "target":
            rows = self.schedule.toPlainText().strip().splitlines()
            try:
                schedule = [
                    [float(v) for v in row.replace("，", ",").split(",")]
                    for row in rows
                    if row.strip()
                ]
            except ValueError:
                schedule = []
            if not schedule or any(len(row) != 3 for row in schedule):
                raise ValueError("请逐行填写开始秒、X、Y，使用逗号分隔。")
            args["schedule"] = schedule
        elif mode == "days":
            path = self.reg.text().strip() or self.main.reg_page.folder
            if not path or not Path(path).is_dir() or not self.aligned.isChecked():
                raise ValueError("请选择配准结果目录，并确认各天场地和坐标朝向。")
            args["registration"] = str(path)
        return args

    def run_function(self):
        return run_study

    def _run(self):
        try:
            super()._run()
        except ValueError as exc:
            QMessageBox.information(self, self.title, str(exc))

    def release_project(self):
        self.folder = None
        self.events.clear()
        self.schedule.clear()
        self.reg.clear()
        self.aligned.setChecked(False)

    def update_status(self):
        mode = self.modes[self.mode.currentIndex()]
        for widget, visible in (
            (self.window, mode == "freeze"),
            (self.speed, mode == "freeze"),
            (self.bins, mode != "freeze"),
            (self.schedule, mode == "target"),
            (self.reg, mode == "days"),
            (self.browse, mode == "days"),
            (self.aligned, mode == "days"),
        ):
            self.form.setRowVisible(widget, visible)
        step = self.ws.step("study_" + mode) if self.ws else None
        folder = self.ws.path(step["outputs"]["folder"]) if step else None
        if folder != self.folder:
            self.events.clear()
            if mode == "freeze" and folder and (folder / "events.csv").is_file():
                with (folder / "events.csv").open(encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        for key, label in (("onset_s", "起始"), ("offset_s", "解除")):
                            time = float(row[key])
                            self.events.addItem(
                                f"第{row['pair']}对 · {row['state']} · {label} {time:.2f}s",
                                time,
                            )
        self.folder = folder
        self.status.setText("最近完成于 " + step["finished"] if step else "")
        self.report.setEnabled(bool(folder and (folder / "report.html").is_file()))
        self.files.setEnabled(bool(folder and folder.is_dir()))
        for widget in (self.events, self.replay):
            widget.setVisible(mode == "freeze")
            widget.setEnabled(bool(self.events.count()))
