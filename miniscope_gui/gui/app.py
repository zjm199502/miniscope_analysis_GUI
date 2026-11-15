import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    f = QFont()
    f.setFamilies(["Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", "sans-serif"])
    f.setPointSize(10)
    app.setFont(f)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
