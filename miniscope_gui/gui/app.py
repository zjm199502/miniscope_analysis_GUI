import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    # Fusion + explicit light palette: looks the same on every PC and avoids
    # unreadable text when Windows is in dark mode.
    app.setStyle("Fusion")
    app.setPalette(app.style().standardPalette())
    try:
        from PySide6.QtCore import Qt
        app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    except AttributeError:
        pass
    f = QFont()
    f.setFamilies(["Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", "sans-serif"])
    f.setPointSize(10)
    app.setFont(f)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
