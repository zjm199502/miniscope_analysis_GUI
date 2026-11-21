"""Device selection shared by image-processing steps."""

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QLabel, QPushButton

from .base import StepPage


@dataclass
class DeviceReport:
    inventory: dict


def probe_devices(session, ws, params, progress, should_stop):
    if should_stop():
        return DeviceReport({})
    from ...core.gpu import detect_gpus

    progress(0, "正在识别显卡并检查 CUDA 运算…")
    inventory = detect_gpus(refresh=params.get("refresh", True))
    return DeviceReport({} if should_stop() else inventory)


class ComputePage(StepPage):
    """Controls for processing steps that can execute on CPU or GPU."""

    def build_params(self):
        self.backend = QComboBox()
        for label, value in (("自动选择", "auto"), ("CPU", "cpu"), ("GPU", "gpu")):
            self.backend.addItem(label, value)
        self.add_param("计算设备", self.backend)
        self.device = QComboBox()
        self.device.addItem("自动选择显卡", None)
        self.add_param("CUDA 显卡", self.device)
        self.detect = QPushButton("检测 / 刷新显卡")
        self.form.addRow(self.detect)
        self.detect.clicked.connect(lambda: self.runRequested.emit(probe_devices, {}))
        self.device_info = QLabel()
        self.compute_result = QLabel()
        for label in (self.device_info, self.compute_result):
            label.setTextFormat(Qt.PlainText)
            label.setWordWrap(True)
            label.hide()
            self.form.addRow(label)
        self._inventory = {}
        self.backend.currentIndexChanged.connect(self._enable_devices)
        self._enable_devices()

    def _enable_devices(self):
        self.device.setEnabled(self.backend.isEnabled() and self.backend.currentData() != "cpu")

    def _set_cards(self, selected):
        self.device.clear()
        self.device.addItem("自动选择显卡", None)
        for d in self._inventory.get("cuda_devices", []):
            self.device.addItem(f"CUDA {d['id']} · {d['name']}", d["id"])
        if selected is not None:
            index = self.device.findData(selected)
            if index < 0:
                self.device.addItem(f"CUDA {selected}（已保存，待验证）", selected)
                index = self.device.count() - 1
            self.device.setCurrentIndex(index)

    def show_devices(self, inventory):
        self._inventory = inventory
        self._set_cards(self.device.currentData())
        adapters = inventory.get("adapters", [])
        lines = ["系统显卡：" + ("、".join(adapters) if adapters else "未读到型号")]
        for d in inventory.get("cuda_devices", []):
            lines.append(f"CUDA {d['id']} · {d['name']}，可用显存 "
                         f"{d['free_bytes'] / 2**30:.1f} / {d['total_bytes'] / 2**30:.1f} GiB")
        if inventory.get("detail"):
            lines.append(inventory["detail"])
        lines.extend(error for error in inventory.get("errors", []) if error not in lines)
        self.device_info.setText("\n".join(lines))
        self.device_info.show()

    def has_inventory(self):
        return bool(self._inventory)

    def compute_params(self):
        return {"backend": self.backend.currentData(),
                "device_id": self.device.currentData() if self.backend.currentData() != "cpu" else None}

    def set_running(self, running):
        super().set_running(running)
        self.backend.setEnabled(not running)
        self.detect.setEnabled(not running)
        self._enable_devices()

    def on_context(self):
        step = self.ws.step(self.step_key) if self.ws else None
        params = step["params"] if step else {}
        self.backend.setCurrentIndex(max(0, self.backend.findData(params.get("backend", "auto"))))
        self._set_cards(params.get("device_id"))

    def update_status(self):
        super().update_status()
        step = self.ws.step(self.step_key) if self.ws else None
        outputs = step["outputs"] if step else {}
        actual = outputs.get("compute_backend")
        self.compute_result.setVisible(bool(actual))
        if actual:
            device = outputs.get("compute_device")
            text = "上次实际使用：" + actual.upper()
            if device and device.upper() != actual.upper():
                text += " · " + device
            if actual == "cpu" and step["params"].get("backend", "auto") != "cpu" and outputs.get("fallback_reason"):
                text += "\n使用 CPU 的原因：" + outputs["fallback_reason"]
            self.compute_result.setText(text)
        else:
            self.compute_result.clear()
