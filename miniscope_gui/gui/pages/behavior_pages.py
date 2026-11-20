"""Pages for step 7 (behavior) and step 8 (sync + joint analysis)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (QButtonGroup, QFileDialog, QGridLayout, QHBoxLayout,
                               QInputDialog, QLabel, QLineEdit, QListWidget, QMessageBox,
                               QPushButton, QRadioButton)

from ...core.behavior import (BehaviorConfig, BehaviorVideo, Shape, TrackingParams, analyze,
                              export_behavior, load_track, run_import, run_tracking)
from ...core.sync import LedParams, run_led_sync, run_manual_sync
from .base import StepPage
from .common import heading, hint
from .quality_timeline import QUALITY_COLORS, QUALITY_LABELS, QualityTimeline

ROI_PEN = pg.mkPen((255, 200, 0), width=2)
ZONE_COLORS = [(0, 200, 255), (255, 120, 200), (120, 255, 120), (255, 160, 60), (180, 140, 255)]


def roi_to_shape(roi, name="") -> Shape:
    if isinstance(roi, pg.CircleROI) or isinstance(roi, pg.EllipseROI):
        d = float(roi.size()[0])
        x, y = roi.pos()
        return Shape("circle", [x + d / 2, y + d / 2, d / 2], name)
    if isinstance(roi, pg.PolyLineROI):
        pts = [roi.mapToParent(h["item"].pos()) for h in roi.handles]
        return Shape("polygon", [v for p in pts for v in (p.x(), p.y())], name)
    x, y = roi.pos()
    w, h = roi.size()
    return Shape("rect", [x, y, w, h], name)


def shape_to_item(shape: Shape, color=(255, 200, 0)):
    d = np.asarray(shape.data, float)
    pen = pg.mkPen(color, width=2)
    if shape.kind == "rect":
        x, y, w, h = d
        xs, ys = [x, x + w, x + w, x, x], [y, y, y + h, y + h, y]
    elif shape.kind == "circle":
        cx, cy, r = d
        a = np.linspace(0, 2 * np.pi, 100)
        xs, ys = cx + r * np.cos(a), cy + r * np.sin(a)
    else:
        p = d.reshape(-1, 2)
        p = np.vstack([p, p[:1]])
        xs, ys = p[:, 0], p[:, 1]
    return pg.PlotCurveItem(np.asarray(xs), np.asarray(ys), pen=pen)


class BehaviorPage(StepPage):
    step_key = "behavior"
    title = "7. 行为分析"
    intro = ("导入行为视频，划出场地范围和感兴趣区域，软件自动追踪动物位置并检测僵直（freezing）。"
             "也可以直接导入 EthoVision 或 DeepLabCut 的结果。")
    run_label = "开始追踪"
    results = (("beh_track", "轨迹"), ("beh_traces", "速度与僵直曲线"),
               ("beh_video", "行为视频"), ("beh_summary", "统计结果"))

    def __init__(self, main, parent=None):
        self.main = main
        self.video: BehaviorVideo | None = None
        self.cfg = BehaviorConfig()
        self.edit_roi = None
        self.edit_kind = None
        super().__init__(parent)

    # ------------------------------------------------------------- layout
    def build_params(self):
        self.form.addRow(heading("视频"))
        self.video_lbl = QLabel("未导入")
        self.video_lbl.setWordWrap(True)
        b = QPushButton("导入行为视频…")
        b.clicked.connect(self.pick_video)
        self.form.addRow(b)
        self.form.addRow(self.video_lbl)
        self.fps = self.add_param("帧率", self.dspin(1, 240, 30, 1, 3, " Hz"),
                                  "默认读取视频文件里的帧率。相机实际帧率不同时可以在这里改。")

        self.form.addRow(heading("场地和区域"))
        self.form.addRow(hint("先画场地（只在场地内追踪），再添加感兴趣区域（如中心区、角落）。"
                              "画好后拖动调整，点「确定」保存。"))
        row = QGridLayout()
        self.arena_btns = []
        for i, (kind, label) in enumerate((("rect", "矩形场地"), ("circle", "圆形场地"),
                                           ("polygon", "多边形场地"))):
            bb = QPushButton(label)
            bb.clicked.connect(lambda _=False, k=kind: self.start_edit("arena", k))
            row.addWidget(bb, 0, i)
        for i, (kind, label) in enumerate((("rect", "＋矩形区域"), ("circle", "＋圆形区域"),
                                           ("polygon", "＋多边形区域"))):
            bb = QPushButton(label)
            bb.clicked.connect(lambda _=False, k=kind: self.start_edit("zone", k))
            row.addWidget(bb, 1, i)
        self.form.addRow(row)
        r2 = QHBoxLayout()
        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.finish_edit)
        self.del_zone_btn = QPushButton("删除所选区域")
        self.del_zone_btn.clicked.connect(self.delete_zone)
        r2.addWidget(self.ok_btn); r2.addWidget(self.del_zone_btn)
        self.form.addRow(r2)
        self.zone_list = QListWidget()
        self.zone_list.setMaximumHeight(90)
        self.form.addRow(self.zone_list)

        self.form.addRow(heading("比例尺"))
        r3 = QHBoxLayout()
        self.scale_btn = QPushButton("画比例尺")
        self.scale_btn.clicked.connect(lambda: self.start_edit("scale", "line"))
        self.scale_cm = self.dspin(0.1, 1000, 40, 1, 1, " cm")
        r3.addWidget(self.scale_btn); r3.addWidget(self.scale_cm)
        self.form.addRow(r3)
        self.scale_lbl = QLabel("未设置（结果以像素为单位）")
        self.form.addRow(self.scale_lbl)
        self.form.addRow(hint("沿着场地一条已知长度的边画一条线，填入实际长度，点「确定」。"))

        self.form.addRow(heading("追踪与僵直参数"))
        self.animal = self.add_param("动物颜色", self.combo(["自动", "深色（白色背景）", "浅色（深色背景）"]))
        self.freeze_thr = self.add_param("僵直阈值", self.dspin(0.001, 10, 0.1, 0.01, 3, " %"),
                                         "相邻两帧间变化的像素占场地的百分比低于这个值，算作不动。"
                                         "运行后在「速度与僵直曲线」里看动作指数，再调整。")
        self.freeze_min = self.add_param("最短僵直", self.dspin(0.1, 10, 1.0, 0.1, 1, " 秒"),
                                         "不动持续超过这个时间才算一次僵直（常用 1 秒）。")
        self.motion_thr = self.add_param("像素变化阈值", self.spin(3, 100, 15),
                                         "灰度变化超过这个值才算该像素在动，用来忽略相机噪声。")
        self.proc_scale = self.add_param("处理分辨率", self.combo(["1/2（推荐）", "原始", "1/4"]),
                                         "降低分辨率可以加快追踪，对位置精度影响很小。")
        self.max_interp = self.add_param(
            "最大插值时长", self.dspin(0, 3600, TrackingParams().max_interp_s, .1, 3, " 秒"),
            "只补两端都有观测的短缺口。设为 0 可关闭插值。修改后立即更新当前轨迹统计。")
        self.max_interp.setKeyboardTracking(False)
        self.max_interp.valueChanged.connect(self._set_gap_limit)

    def build_extra(self):
        self.quality_title = heading("追踪质量")
        self.lay.addWidget(self.quality_title)
        self.timeline = QualityTimeline()
        self.timeline.sampleSelected.connect(self._replay_quality)
        self.lay.addWidget(self.timeline)
        self.quality_stats = QLabel()
        self.quality_stats.setWordWrap(True)
        self.lay.addWidget(self.quality_stats)
        self._clear_quality()
        self.lay.addWidget(heading("导入其他软件的结果"))
        self.lay.addWidget(hint("不用本软件追踪时，可以导入 EthoVision XT 导出的原始数据"
                                "（txt / csv / xlsx，含 Trial time、X center、Y center，"
                                "有 Immobile / Freezing 列时一并读取），或 DeepLabCut 的 CSV。"))
        r = QHBoxLayout()
        ev = QPushButton("导入 EthoVision…")
        ev.clicked.connect(lambda: self.import_file("ethovision"))
        dl = QPushButton("导入 DeepLabCut…")
        dl.clicked.connect(lambda: self.import_file("dlc"))
        r.addWidget(ev); r.addWidget(dl)
        self.lay.addLayout(r)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("background:#f4f4f4; padding:6px; border-radius:4px;")
        self.lay.addWidget(self.summary)
        self.export_btn = QPushButton("导出行为结果…")
        self.export_btn.clicked.connect(self._export)
        self.lay.addWidget(self.export_btn)

    # ------------------------------------------------------------ context
    def on_context(self):
        cfg = BehaviorConfig.load(self.ws) if self.ws else None
        with QSignalBlocker(self.max_interp):
            self.max_interp.setValue((cfg.params if cfg else {}).get("max_interp_s", TrackingParams().max_interp_s))
        if cfg:
            self.cfg = cfg
            self.fps.setValue(cfg.fps)
            if cfg.video and Path(cfg.video).exists():
                self._open_video(cfg.video, cfg.fps, keep_cfg=True)
            p = TrackingParams(**cfg.params)
            self.animal.setCurrentIndex({"auto": 0, "dark": 1, "light": 2}.get(p.animal, 0))
            self.freeze_thr.setValue(p.freeze_threshold)
            self.freeze_min.setValue(p.freeze_min_s)
            self.motion_thr.setValue(p.motion_threshold)
        self._refresh_lists()
        self._update_summary()

    def update_status(self):
        super().update_status()
        ok = bool(self.ws and self.ws.step("behavior"))
        if hasattr(self, "export_btn"):
            self.export_btn.setEnabled(ok)

    def release_project(self):
        self.cancel_edit()
        self._clear_quality()
        self.summary.clear()
        if self.video is not None:
            self.video.close()
            self.video = None
        self.cfg = BehaviorConfig()
        self.video_lbl.setText("未导入")

    def pick_video(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择行为视频", "",
                                           "视频 (*.avi *.mp4 *.mov *.mkv *.wmv *.mpg *.mpeg);;所有文件 (*)")
        if f:
            self._open_video(f)

    def _open_video(self, path, fps=None, keep_cfg=False):
        try:
            video = BehaviorVideo(path, fps)
        except Exception as e:
            QMessageBox.warning(self, "无法打开视频", str(e))
            return
        if self.video is not None:
            self.video.close()
        self.video = video
        if not keep_cfg:
            self.cfg.video = str(path)
        self.fps.setValue(self.video.fps)
        self.video_lbl.setText(f"{Path(path).name}\n{self.video.width}×{self.video.height}，"
                               f"{self.video.n_frames} 帧，{self.video.fps:.2f} Hz，"
                               f"{self.video.duration_s / 60:.1f} 分钟")
        self.show_video()
        if self.ws:
            self.cfg.fps = self.fps.value()
            self.cfg.save(self.ws)

    def show_video(self):
        if self.video is None:
            return
        v = self.video
        self.main.trace.hide()
        self.main.view.set_movie(lambda i: v.get_frame(i), v.n_frames, v.fps, f"行为视频 · {v.path.name}")
        self.main.view.img.setLevels([0, 255])
        self._draw_shapes()

    # -------------------------------------------------------------- drawing
    def start_edit(self, what, kind):
        if self.video is None:
            QMessageBox.information(self, "提示", "请先导入行为视频")
            return
        self.cancel_edit()
        if self.main.view.title.text().startswith("行为视频") is False:
            self.show_video()
        w, h = self.video.width, self.video.height
        if kind == "rect":
            roi = pg.RectROI([w * 0.2, h * 0.2], [w * 0.6, h * 0.6], pen=ROI_PEN)
        elif kind == "circle":
            r = min(w, h) * 0.3
            roi = pg.CircleROI([w / 2 - r, h / 2 - r], radius=r, pen=ROI_PEN)
        elif kind == "polygon":
            roi = pg.PolyLineROI([[w * .3, h * .3], [w * .7, h * .3], [w * .7, h * .7], [w * .3, h * .7]],
                                 closed=True, pen=ROI_PEN)
        else:
            roi = pg.LineSegmentROI([[w * 0.2, h * 0.5], [w * 0.8, h * 0.5]], pen=ROI_PEN)
        self.main.view.vb.addItem(roi)
        self.edit_roi, self.edit_kind = roi, what
        self.main.statusBar().showMessage("拖动调整形状（多边形：拖动顶点，双击边可以加点），完成后点「确定」")

    def cancel_edit(self):
        if self.edit_roi is not None:
            self.main.view.vb.removeItem(self.edit_roi)
        self.edit_roi = self.edit_kind = None

    def finish_edit(self):
        if self.edit_roi is None:
            return
        roi, what = self.edit_roi, self.edit_kind
        if what == "scale":
            p1, p2 = [roi.mapToParent(h["item"].pos()) for h in roi.handles]
            px = float(np.hypot(p2.x() - p1.x(), p2.y() - p1.y()))
            self.cfg.px_per_cm = px / self.scale_cm.value()
        elif what == "arena":
            self.cfg.arena = roi_to_shape(roi, "arena").to_dict()
        else:
            name, ok = QInputDialog.getText(self, "区域名称", "给这个区域起个名字（如 center、corner1）：",
                                            QLineEdit.Normal, f"zone{len(self.cfg.zones) + 1}")
            if not ok or not name.strip():
                return
            self.cfg.zones.append(roi_to_shape(roi, name.strip()).to_dict())
        self.cancel_edit()
        if self.ws:
            self.cfg.save(self.ws)
        self._refresh_lists()
        self._draw_shapes()
        self._update_summary()

    def delete_zone(self):
        i = self.zone_list.currentRow()
        if i < 0:
            return
        if i == 0 and self.cfg.arena:
            self.cfg.arena = None
        else:
            del self.cfg.zones[i - (1 if self.cfg.arena else 0)]
        if self.ws:
            self.cfg.save(self.ws)
        self._refresh_lists()
        self._draw_shapes()

    def _refresh_lists(self):
        self.zone_list.clear()
        if self.cfg.arena:
            self.zone_list.addItem(f"场地（{ {'rect': '矩形', 'circle': '圆形', 'polygon': '多边形'}[self.cfg.arena['kind']] }）")
        for z in self.cfg.zones:
            self.zone_list.addItem(f"区域: {z['name']}")
        self.scale_lbl.setText(f"{self.cfg.px_per_cm:.2f} 像素/cm" if self.cfg.px_per_cm
                               else "未设置（结果以像素为单位）")

    def _draw_shapes(self):
        v = self.main.view
        v.clear_overlays()
        if self.cfg.arena:
            v.add_overlay(shape_to_item(Shape.from_dict(self.cfg.arena), (255, 200, 0)))
        for i, z in enumerate(self.cfg.zones):
            v.add_overlay(shape_to_item(Shape.from_dict(z), ZONE_COLORS[i % len(ZONE_COLORS)]))

    # ------------------------------------------------------------- running
    def _set_gap_limit(self, seconds):
        self.cfg.params["max_interp_s"] = seconds
        if self.ws:
            self.cfg.save(self.ws)
            self._update_summary()

    def set_running(self, running):
        super().set_running(running)
        self.max_interp.setEnabled(not running)

    def _clear_quality(self):
        self.timeline.clear()
        self.quality_stats.clear()
        for widget in (self.quality_title, self.timeline, self.quality_stats):
            widget.hide()

    def _replay_quality(self, index):
        time = self.timeline.times[index]
        state = QUALITY_LABELS[self.timeline.quality[index]]
        msg = f"行为时间 {time:.3f} 秒 · {state}"
        if self.video is None:
            self.main.statusBar().showMessage(msg + "（载入对应行为视频后可回放）")
            return
        # A matching frame count gives a direct correspondence even at variable
        # frame rates. External lower-rate tracks use the trial time in seconds.
        frame = index if len(self.timeline.times) == self.video.n_frames else round(time * self.video.fps)
        if not 0 <= frame < self.video.n_frames:
            self.main.statusBar().showMessage(msg + "（此时刻超出视频范围）")
            return
        self.show_video()
        self.main.view.slider.setValue(frame)
        self.main.view.toggle_play()
        self.main.statusBar().showMessage(msg)

    def params(self):
        self.cfg.fps = self.fps.value()
        scale = {0: 0.5, 1: 1.0, 2: 0.25}[self.proc_scale.currentIndex()]
        self.cfg.params = {**self.cfg.params, **dict(
            animal=["auto", "dark", "light"][self.animal.currentIndex()],
            freeze_threshold=self.freeze_thr.value(), freeze_min_s=self.freeze_min.value(),
            motion_threshold=self.motion_thr.value(), scale=scale)}
        return self.cfg

    def run_function(self):
        return run_tracking

    def _run(self):
        if self.video is None:
            QMessageBox.information(self, "提示", "请先导入行为视频")
            return
        self.cancel_edit()
        super()._run()

    def import_file(self, kind):
        if self.ws is None:
            return
        flt = "EthoVision 导出 (*.txt *.csv *.xlsx)" if kind == "ethovision" else "DeepLabCut (*.csv)"
        f, _ = QFileDialog.getOpenFileName(self, "选择文件", "", flt)
        if not f:
            return
        args = {"kind": kind, "path": f, "cfg": self.params()}
        if kind == "dlc":
            if self.video is None:
                fps, ok = QInputDialog.getDouble(self, "帧率", "DeepLabCut 视频的帧率 (Hz)：", 30, 1, 240, 2)
                if not ok:
                    return
                self.cfg.fps = fps
        self.runRequested.emit(run_import, args)

    def on_finished(self, result):
        self.cfg = BehaviorConfig.load(self.ws) or self.cfg
        self._update_summary()
        if "polarity" in result:
            self.status.setText(f"✓ 追踪完成：{result['detected_fraction'] * 100:.1f}% 的帧找到了动物"
                                f"（动物为{'深色' if result['polarity'] == 'dark' else '浅色'}）")
        else:
            self.status.setText(f"✓ 已导入 {result['n_frames']} 帧")

    def _update_summary(self):
        if not (self.ws and self.ws.step("behavior")):
            self._clear_quality()
            self.summary.setText("")
            return
        try:
            res = analyze(load_track(self.ws), self.params())
        except Exception as e:
            self._clear_quality()
            self.summary.setText(f"无法计算统计结果：{e}")
            return
        s, u = res["summary"], res["unit"]
        self.timeline.set_samples(res["t"], res["tracking_quality"])
        self.quality_stats.setText("　".join(
            f'<span style="color:{QUALITY_COLORS[code]}">■</span> {QUALITY_LABELS[code]} '
            f'{s["tracking_" + key + "_pct"]:.1f}%'
            for code, key in ((2, "observed"), (1, "interpolated"), (0, "invalid"))))
        for widget in (self.quality_title, self.timeline, self.quality_stats):
            widget.show()
        lines = [f"时长 {s['duration_s']:.1f} 秒，总路程 {s['distance']:.1f} {u}，平均速度 {s['mean_speed']:.2f} {u}/s"]
        if "freezing_percent" in s:
            lines.append(f"僵直 {s['freezing_percent']:.1f}%，共 {s['freezing_bouts']} 次，"
                         f"平均每次 {s['freezing_mean_bout_s']:.1f} 秒")
        for z in self.cfg.zones:
            n = z["name"]
            lines.append(f"区域 {n}：停留 {s[f'zone[{n}]_time_s']:.1f} 秒，进入 {s[f'zone[{n}]_entries']} 次")
        self.summary.setText("\n".join(lines))

    def _export(self):
        d = QFileDialog.getExistingDirectory(self, "选择导出文件夹", str(self.ws.root))
        if d:
            try:
                out = export_behavior(self.ws, self.params(), d)
                QMessageBox.information(self, "导出完成", f"已保存到:\n{out}")
            except Exception as e:
                QMessageBox.warning(self, "导出失败", str(e))

    # --------------------------------------------------------------- views
    def show_view(self, key):
        main = self.main
        if not (self.ws and self.ws.step("behavior")) and key != "beh_video":
            QMessageBox.information(self, "提示", "请先完成追踪或导入")
            return
        if key == "beh_video":
            self.show_video()
            return
        res = analyze(load_track(self.ws), self.params())
        if key == "beh_track":
            bgf = self.ws.path("behavior/background.npy")
            if self.video is not None:
                img = self.video.get_frame(0)
            elif bgf.exists():
                img = np.load(bgf)
            else:
                img = np.zeros((10, 10))
            main.view.set_image(img, "动物轨迹")
            if self.video is not None:
                main.view.img.setLevels([0, 255])
            self._draw_shapes()
            x, y = res["x"], res["y"]
            if res["unit"] == "cm" and self.cfg.source == "ethovision":
                main.statusBar().showMessage("EthoVision 坐标单位为 cm，与视频像素不对应，轨迹仅供参考")
            item = pg.PlotCurveItem(x, y, pen=pg.mkPen((255, 60, 60, 160), width=1))
            main.view.add_overlay(item)
            if "freezing" in res:
                fz = res["freezing"]
                sc = pg.ScatterPlotItem(x[fz][::5], y[fz][::5], size=4, brush=pg.mkBrush(60, 120, 255, 160),
                                        pen=None)
                main.view.add_overlay(sc)
            main.trace.hide()
        elif key == "beh_traces":
            tr = main.trace
            tr.clear()
            tr.addLegend()
            t = res["t"]
            sp = res["speed"]
            tr.plot(t, sp / (np.nanpercentile(sp, 99) or 1), pen=pg.mkPen((40, 110, 220)), name="速度（归一化）")
            if "motion" in res:
                m = res["motion"]
                tr.plot(t, m / (np.nanpercentile(m, 99) or 1), pen=pg.mkPen((120, 120, 120)), name="动作指数（归一化）")
                thr = self.freeze_thr.value() / (np.nanpercentile(m, 99) or 1)
                tr.addItem(pg.InfiniteLine(thr, angle=0, pen=pg.mkPen((220, 60, 60), style=Qt.DashLine)))
            if "freezing" in res:
                tr.plot(t, res["freezing"].astype(float) * 1.1, pen=pg.mkPen((220, 60, 60), width=2),
                        name="僵直", fillLevel=0, brush=(220, 60, 60, 40))
            tr.setLabel("bottom", "时间 (秒)")
            tr.setTitle("红色虚线 = 僵直阈值（动作指数低于它且持续够久 = 僵直）")
            tr.show()
        elif key == "beh_summary":
            self._update_summary()
            QMessageBox.information(self, "行为统计", self.summary.text() or "无")


class SyncPage(StepPage):
    step_key = "sync"
    title = "8. 同步分析"
    intro = ("把行为视频的时间换算到 miniscope 的时间轴上，然后分析神经活动和行为的关系。")
    run_label = "计算同步"
    results = (("sync_joint", "联合播放"), ("sync_qc", "同步检查"))

    def __init__(self, main, parent=None):
        self.main = main
        self.led_roi = None
        self.arduino_log = None
        super().__init__(parent)

    def build_params(self):
        self.form.addRow(heading("对齐方式"))
        self.m_led = QRadioButton("LED 闪光（推荐，配合 Arduino）")
        self.m_manual = QRadioButton("手动：指定同一时刻")
        self.m_led.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.m_led); grp.addButton(self.m_manual)
        self.form.addRow(self.m_led)
        self.form.addRow(self.m_manual)

        self.form.addRow(heading("LED 闪光"))
        self.form.addRow(hint("在行为视频上框住 LED（只框 LED，不要框到动物）。软件识别 1 秒的开始标记、"
                              "每隔一段时间的短闪和 2 秒的结束标记，分段校正时间。"))
        r = QHBoxLayout()
        b = QPushButton("框选 LED")
        b.clicked.connect(self.draw_led)
        self.log_btn = QPushButton("Arduino 记录（可选）…")
        self.log_btn.clicked.connect(self.pick_log)
        r.addWidget(b); r.addWidget(self.log_btn)
        self.form.addRow(r)
        self.interval = self.add_param("闪光间隔", self.dspin(5, 120, 30, 1, 1, " 秒"),
                                       "和 Arduino 程序里的设置一致。")
        self.mode = self.add_param("计时模式", self.combo(["自动", "TIME（未接 Sync 线）", "FRAME（接了 Sync 线）"]))

        self.form.addRow(heading("手动对齐"))
        self.form.addRow(hint("在行为视频里找到一个两边都能确定时间的时刻（例如 miniscope 开始录制那一刻），"
                              "把滑块拖到这一帧，点「使用当前帧」，再填入这一刻对应的 miniscope 时间。"))
        r2 = QHBoxLayout()
        self.b_frame = self.spin(0, 10_000_000, 0)
        use = QPushButton("使用当前帧")
        use.clicked.connect(self._use_current)
        r2.addWidget(QLabel("行为帧")); r2.addWidget(self.b_frame); r2.addWidget(use)
        self.form.addRow(r2)
        self.m_time = self.add_param("miniscope 时间", self.dspin(-3600, 36000, 0, 0.1, 3, " 秒"),
                                     "0 = miniscope 第一帧。若行为视频比 miniscope 早开始 2 秒，"
                                     "就把行为第 0 帧对应为 -2 秒。")

    def build_extra(self):
        self.lay.addWidget(heading("神经活动与行为"))
        self.lay.addWidget(hint("需要先完成第 6 步（信号）和第 7 步（行为）。"))
        g = QGridLayout()
        self.b_freeze = QPushButton("僵直 vs 运动")
        self.b_speed = QPushButton("与速度的相关")
        self.b_peth = QPushButton("事件前后平均")
        self.b_export = QPushButton("导出对齐数据…")
        for i, b in enumerate((self.b_freeze, self.b_speed, self.b_peth, self.b_export)):
            g.addWidget(b, i // 2, i % 2)
        self.lay.addLayout(g)
        self.event_src = self.combo(["僵直开始", "手动输入时间"])
        self.event_times = QLineEdit()
        self.event_times.setPlaceholderText("例如电击时间：120, 180, 240（秒，miniscope 时间）")
        r = QHBoxLayout()
        self.pre = self.dspin(0.5, 60, 5, 0.5, 1, " 秒前")
        self.post = self.dspin(0.5, 120, 10, 0.5, 1, " 秒后")
        r.addWidget(self.pre); r.addWidget(self.post)
        self.form2 = QGridLayout()
        self.lay.addWidget(QLabel("事件："))
        self.lay.addWidget(self.event_src)
        self.lay.addWidget(self.event_times)
        self.lay.addLayout(r)
        self.result_lbl = QLabel("")
        self.result_lbl.setWordWrap(True)
        self.result_lbl.setStyleSheet("background:#f4f4f4; padding:6px; border-radius:4px;")
        self.lay.addWidget(self.result_lbl)
        self.b_freeze.clicked.connect(lambda: self.main.joint_analysis("freezing"))
        self.b_speed.clicked.connect(lambda: self.main.joint_analysis("speed"))
        self.b_peth.clicked.connect(lambda: self.main.joint_analysis("peth"))
        self.b_export.clicked.connect(lambda: self.main.joint_analysis("export"))

    def on_context(self):
        bp = self.main.beh_page
        names = [f"进入区域 {z['name']}" for z in bp.cfg.zones]
        cur = self.event_src.currentText()
        self.event_src.clear()
        self.event_src.addItems(["僵直开始", *names, "手动输入时间"])
        if cur:
            self.event_src.setCurrentText(cur)
        st = self.ws.step("sync") if self.ws else None
        if st:   # restore the method used last time
            (self.m_manual if st["params"].get("method") == "manual" else self.m_led).setChecked(True)

    def draw_led(self):
        bp = self.main.beh_page
        if bp.video is None:
            QMessageBox.information(self, "提示", "请先在第 7 步导入行为视频")
            return
        bp.show_video()
        if self.led_roi is not None:
            self.main.view.vb.removeItem(self.led_roi)
        w, h = bp.video.width, bp.video.height
        self.led_roi = pg.RectROI([w * 0.05, h * 0.05], [max(8, w * 0.05), max(8, h * 0.05)],
                                  pen=pg.mkPen((0, 255, 0), width=2))
        self.main.view.vb.addItem(self.led_roi)

    def pick_log(self):
        f, _ = QFileDialog.getOpenFileName(self, "Arduino 串口记录", "", "文本 (*.txt *.log);;所有文件 (*)")
        self.arduino_log = f or None
        self.log_btn.setText("Arduino 记录 ✓" if f else "Arduino 记录（可选）…")

    def _use_current(self):
        self.b_frame.setValue(self.main.view.current_index())

    def params(self):
        bp = self.main.beh_page
        if self.m_led.isChecked():
            if self.led_roi is None:
                raise ValueError("请先点「框选 LED」并把方框放在 LED 上")
            x, y = self.led_roi.pos()
            w, h = self.led_roi.size()
            mode = {0: "auto", 1: "time", 2: "frame"}[self.mode.currentIndex()]
            return {"video": bp.cfg.video, "fps": bp.fps.value(), "roi": (x, y, w, h),
                    "params": LedParams(interval_s=self.interval.value(), mode=mode),
                    "arduino_log": self.arduino_log}
        return {"n_frames": bp.video.n_frames if bp.video else len(load_track(self.ws)["t"]),
                "fps": bp.fps.value(), "behavior_frame": self.b_frame.value(),
                "miniscope_time_s": self.m_time.value()}

    def run_function(self):
        return run_led_sync if self.m_led.isChecked() else run_manual_sync

    def _run(self):
        if self.main.beh_page.video is None and self.m_led.isChecked():
            QMessageBox.information(self, "提示", "请先在第 7 步导入行为视频")
            return
        try:
            prm = self.params()
        except ValueError as e:
            QMessageBox.information(self, "提示", str(e))
            return
        self.runRequested.emit(self.run_function(), prm)

    def on_finished(self, result):
        if result.get("mode") == "manual":
            self.status.setText("✓ 已按手动设置对齐")
            return
        msg = (f"✓ LED 同步完成（{result['mode'].upper()} 模式）：检测到 {result['n_flashes']} 次闪光，"
               f"{'有' if result['has_end_mark'] else '没有'}结束标记。"
               f"行为相机实际帧率 {result['behavior_fps_measured']:.3f} Hz。")
        if result.get("missing_flashes"):
            msg += f" 漏检第 {result['missing_flashes']} 次闪光，已用相邻闪光插值。"
        if result.get("arduino_clock_error_pct") is not None:
            msg += f" Arduino 时钟误差 {result['arduino_clock_error_pct']:+.3f}%，已校正。"
        self.status.setText(msg)
