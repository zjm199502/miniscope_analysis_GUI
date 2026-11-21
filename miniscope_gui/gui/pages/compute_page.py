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

    progress(0, "正在识别显卡并验证 GPU 运算…")
    inventory = detect_gpus(refresh=params.get("refresh", True))
    return DeviceReport({} if should_stop() else inventory)


class ComputePage(StepPage):
    """Controls for processing steps that can execute on CPU or GPU."""

    def build_params(self):
        self.backend = QComboBox()
        for label, value in (("自动选择", "auto"), ("CPU", "cpu"),
                             ("GPU（自动选择后端）", "gpu"),
                             ("NVIDIA CUDA", "cuda"), ("Apple Metal", "metal")):
            self.backend.addItem(label, value)
        self.add_param("计算设备", self.backend)
        self.device = QComboBox()
        self.device.addItem("自动选择显卡", None)
        self.add_param("显卡", self.device)
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
        self.backend.currentIndexChanged.connect(self._change_backend)
        self._enable_devices()

    def _enable_devices(self):
        self.device.setEnabled(self.backend.isEnabled() and self.backend.currentData() != "cpu")

    def _change_backend(self):
        self._set_cards(None)
        self._enable_devices()

    def _cards(self):
        if "devices" in self._inventory:
            return self._inventory["devices"]
        return [dict(d, kind=kind) for kind in ("cuda", "metal")
                for d in self._inventory.get(kind + "_devices", [])]

    def _set_cards(self, selected, selected_kind=None):
        backend = self.backend.currentData()
        self.device.clear()
        self.device.addItem("自动选择显卡", None)
        if backend == "cpu":
            return
        choice = 0
        for d in self._cards():
            kind = d["kind"]
            if backend in ("cuda", "metal") and kind != backend:
                continue
            kind_label = "Metal" if kind == "metal" else "CUDA"
            self.device.addItem(f"{kind_label} {d['id']} · {d['name']}", d["id"])
            index = self.device.count() - 1
            self.device.setItemData(index, kind, Qt.UserRole + 1)
            if d["id"] == selected and (kind == selected_kind or (selected_kind is None and not choice)):
                choice = index
        if selected is not None and not choice:
            label = {"cuda": "CUDA", "metal": "Metal"}.get(selected_kind, "设备")
            self.device.addItem(f"{label} {selected}（已保存，待验证）", selected)
            choice = self.device.count() - 1
            self.device.setItemData(choice, selected_kind, Qt.UserRole + 1)
        self.device.setCurrentIndex(choice)

    def show_devices(self, inventory):
        self._inventory = inventory
        self._set_cards(self.device.currentData(), self.device.currentData(Qt.UserRole + 1))
        adapters = inventory.get("adapters", [])
        lines = ["系统显卡：" + ("、".join(adapters) if adapters else "未读到型号")]
        for d in self._cards():
            if d["kind"] == "metal":
                line = f"Metal {d['id']} · {d['name']}"
                if d.get("total_bytes"):
                    line += f"，统一内存 {d['total_bytes'] / 2**30:.1f} GiB"
            else:
                line = f"CUDA {d['id']} · {d['name']}"
                if d.get("free_bytes") is not None and d.get("total_bytes"):
                    line += f"，可用显存 {d['free_bytes'] / 2**30:.1f} / {d['total_bytes'] / 2**30:.1f} GiB"
            lines.append(line)
        if inventory.get("detail"):
            lines.append(inventory["detail"])
        lines.extend(error for error in inventory.get("errors", []) if error not in lines)
        self.device_info.setText("\n".join(lines))
        self.device_info.show()

    def has_inventory(self):
        return bool(self._inventory)

    def compute_params(self):
        backend = self.backend.currentData()
        kind = self.device.currentData(Qt.UserRole + 1)
        # CUDA and Metal both number their devices from zero. A selected
        # card must retain its execution backend when its integer ID is sent.
        if backend in ("auto", "gpu") and kind:
            backend = kind
        return {"backend": backend, "device_id": self.device.currentData()}

    def set_running(self, running):
        super().set_running(running)
        self.backend.setEnabled(not running)
        self.detect.setEnabled(not running)
        self._enable_devices()

    def on_context(self):
        step = self.ws.step(self.step_key) if self.ws else None
        params = step["params"] if step else {}
        backend = params.get("backend", "auto")
        device_id = params.get("device_id")
        # In pre-Metal projects an explicitly selected card always referred to CUDA.
        if device_id is not None and backend in ("auto", "gpu"):
            backend = "cuda"
        self.backend.setCurrentIndex(max(0, self.backend.findData(backend)))
        self._set_cards(device_id, backend if device_id is not None else None)

    def update_status(self):
        super().update_status()
        step = self.ws.step(self.step_key) if self.ws else None
        outputs = step["outputs"] if step else {}
        actual = outputs.get("compute_backend")
        self.compute_result.setVisible(bool(actual))
        if actual:
            device = outputs.get("compute_device")
            text = "上次实际使用：" + ("Metal" if actual == "metal" else actual.upper())
            if device and device.upper() != actual.upper():
                text += " · " + device
            if actual == "cpu" and step["params"].get("backend", "auto") != "cpu" and outputs.get("fallback_reason"):
                text += "\n使用 CPU 的原因：" + outputs["fallback_reason"]
            self.compute_result.setText(text)
        else:
            self.compute_result.clear()
