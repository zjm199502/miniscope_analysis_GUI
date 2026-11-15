from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QScrollArea, QSizePolicy


def hint(text: str) -> QLabel:
    """Grey explanatory text under a parameter: the teaching layer of the GUI."""
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color: #777; font-size: 12px;")
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    # let the label shrink to the panel width and wrap instead of widening it
    lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    lbl.setMinimumWidth(50)
    return lbl


def heading(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 4px;")
    return lbl


class FitScrollArea(QScrollArea):
    """Vertical-only scroll area: content always has exactly the visible width."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.widget() is not None:
            self.widget().setFixedWidth(self.viewport().width())
