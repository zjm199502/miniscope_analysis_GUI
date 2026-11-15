from PySide6.QtWidgets import QVBoxLayout, QWidget

from .common import heading, hint


class PlaceholderPage(QWidget):
    def __init__(self, title: str, text: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.addWidget(heading(title))
        lay.addWidget(hint(text + "\n\n（这一步正在开发中）"))
        lay.addStretch(1)
