"""图形界面。启动：python -m tidyaudio gui"""

import sys


def apply_theme(app) -> None:
    """Fusion 风格：在 macOS / Windows / Linux 上长得一样，样式表也最听话。"""
    from PySide6.QtGui import QColor, QPalette

    from .style import ACCENT, BG, TEXT

    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))   # 勾选框、选中项用主色
    palette.setColor(QPalette.ColorRole.Window, QColor(BG))
    palette.setColor(QPalette.ColorRole.Base, QColor("white"))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText,
                 QPalette.ColorRole.ButtonText):
        palette.setColor(role, QColor(TEXT))
    app.setPalette(palette)


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    from PySide6.QtGui import QIcon

    from .style import ICONS

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("TidyAudio")
    app.setWindowIcon(QIcon(f"{ICONS}/app.png"))   # 直接用 Python 运行时，Dock 上也显示 App 图标
    apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()
