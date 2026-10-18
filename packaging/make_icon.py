"""画 App 图标。

图标：白色圆角方块，里面是几根波形线，中间那根是向下的箭头（"把音频处理好、输出"）。
全部用同样粗细、圆头的线条，风格和界面的黑白灰一致。

生成两个文件：
  packaging/TidyAudio.icns          打包 App 用的图标（各种尺寸）
  tidyaudio/gui/icons/app.png       直接运行软件时，窗口和 Dock 上显示的图标

运行：.venv/bin/python packaging/make_icon.py
"""

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath, QPen)

HERE = Path(__file__).parent
ROOT = HERE.parent
SIZE = 1024
INK = "#1d1d1f"


def squircle(rect: QRectF, n: float = 5.0, steps: int = 360) -> QPainterPath:
    """苹果图标那种平滑的圆角（超椭圆），比普通圆角矩形过渡更自然。"""
    cx, cy = rect.center().x(), rect.center().y()
    a, b = rect.width() / 2, rect.height() / 2
    path = QPainterPath()
    for i in range(steps + 1):
        t = 2 * math.pi * i / steps
        c, s = math.cos(t), math.sin(t)
        x = cx + a * math.copysign(abs(c) ** (2 / n), c)
        y = cy + b * math.copysign(abs(s) ** (2 / n), s)
        path.moveTo(x, y) if i == 0 else path.lineTo(x, y)
    path.closeSubpath()
    return path


def draw(path: Path) -> None:
    img = QImage(SIZE, SIZE, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # macOS 图标网格：画布 1024，图标主体 824，四周留 100 的边
    body = QRectF(100, 100, 824, 824)
    p.setPen(Qt.PenStyle.NoPen)
    for i, alpha in enumerate((18, 12, 7)):            # 轻微的阴影
        p.setBrush(QColor(0, 0, 0, alpha))
        p.drawPath(squircle(body.adjusted(-2 - i * 3, 6 + i * 4, 2 + i * 3, 10 + i * 6)))
    grad = QLinearGradient(0, 100, 0, 924)
    grad.setColorAt(0, QColor("#ffffff"))
    grad.setColorAt(1, QColor("#ececf1"))
    p.setBrush(grad)
    p.drawPath(squircle(body))

    # 波形线 + 中间的向下箭头：全部是同样粗细、圆头的线
    p.setPen(QPen(QColor(INK), 42, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                  Qt.PenJoinStyle.RoundJoin))
    heights = [150, 190, None, 190, 150]             # None 的位置是箭头
    gap, head = 96, 58
    cy = SIZE / 2 - 18                               # 箭头往下伸得多，整体往上挪一点保持居中
    x = SIZE / 2 - (len(heights) - 1) * gap / 2
    for h in heights:
        if h is None:
            top, bottom = cy - 200, cy + 236
            p.drawLine(QPointF(x, top), QPointF(x, bottom))
            p.drawLine(QPointF(x - head, bottom - head), QPointF(x, bottom))
            p.drawLine(QPointF(x + head, bottom - head), QPointF(x, bottom))
        else:
            p.drawLine(QPointF(x, cy - h / 2), QPointF(x, cy + h / 2))
        x += gap
    p.end()
    img.save(str(path))


def main() -> None:
    QGuiApplication(sys.argv)
    png = HERE / "icon_1024.png"
    draw(png)

    # 窗口图标：直接运行软件时用
    window_icon = ROOT / "tidyaudio" / "gui" / "icons" / "app.png"
    subprocess.run(["sips", "-z", "256", "256", str(png), "--out", str(window_icon)],
                   check=True, capture_output=True)

    # 用 macOS 自带的 sips 和 iconutil 生成各种尺寸，打包成 .icns
    iconset = HERE / "TidyAudio.iconset"
    shutil.rmtree(iconset, ignore_errors=True)
    iconset.mkdir()
    for size in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            px = size * scale
            name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
            subprocess.run(["sips", "-z", str(px), str(px), str(png), "--out", str(iconset / name)],
                           check=True, capture_output=True)
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(HERE / "TidyAudio.icns")],
                   check=True)
    shutil.rmtree(iconset)
    png.unlink()
    print(f"Icon written: {HERE / 'TidyAudio.icns'} and {window_icon}")


if __name__ == "__main__":
    main()
