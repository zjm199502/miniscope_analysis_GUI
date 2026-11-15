from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel


def hint(text: str) -> QLabel:
    """Grey explanatory text under a parameter: the teaching layer of the GUI."""
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color: #777; font-size: 12px;")
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return lbl


def heading(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 4px;")
    return lbl
