"""在软件里直接写 Python 代码用的编辑框。

CodeEditor：等宽字体、Tab 键插入 4 个空格、回车自动保持缩进
PythonHighlighter：简单的语法上色（关键字、字符串、注释、数字、函数名），颜色参考 Xcode 的浅色主题
"""

import keyword
import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import QPlainTextEdit


def _fmt(color: str, bold: bool = False, italic: bool = False) -> QTextCharFormat:
    f = QTextCharFormat()
    f.setForeground(QColor(color))
    if bold:
        f.setFontWeight(QFont.Weight.DemiBold)
    f.setFontItalic(italic)
    return f


class PythonHighlighter(QSyntaxHighlighter):
    """逐行给 Python 代码上色。按顺序套用规则，后面的会覆盖前面的（所以注释和字符串放最后）。"""

    BUILTINS = ["print", "len", "range", "int", "float", "str", "list", "dict", "tuple", "set",
                "min", "max", "abs", "sum", "enumerate", "zip", "isinstance", "None", "True", "False"]

    def __init__(self, document):
        super().__init__(document)
        # (规则, 颜色, 给第几组上色)：0 = 整个匹配；1 = 括号里的部分（比如 def 后面的函数名）
        self.rules = [
            (re.compile(r"\b(?:" + "|".join(keyword.kwlist) + r")\b"), _fmt("#ad3da4", bold=True), 0),
            (re.compile(r"\b(?:" + "|".join(self.BUILTINS) + r")\b"), _fmt("#6c36a9"), 0),
            (re.compile(r"\b\d+(?:\.\d+)?\b"), _fmt("#1c00cf"), 0),
            (re.compile(r"\bdef\s+(\w+)"), _fmt("#0f68a0", bold=True), 1),
            (re.compile(r"\"[^\"\n]*\"|'[^'\n]*'"), _fmt("#c41a16"), 0),
            (re.compile(r"#[^\n]*"), _fmt("#707f8c", italic=True), 0),
        ]
        self.docstring = _fmt("#707f8c", italic=True)

    def highlightBlock(self, text: str) -> None:
        for pattern, fmt, group in self.rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(group), m.end(group) - m.start(group), fmt)
        self._highlight_docstrings(text)

    def _highlight_docstrings(self, text: str) -> None:
        """三引号字符串可能跨好几行，用"块状态"记住上一行是不是还在字符串里。"""
        in_string = self.previousBlockState() == 1
        start = 0 if in_string else -1
        i = 0
        while True:
            pos = text.find('"""', i)
            if pos < 0:
                break
            if in_string:
                self.setFormat(start, pos + 3 - start, self.docstring)
                in_string = False
            else:
                start = pos
                in_string = True
            i = pos + 3
        if in_string:
            self.setFormat(start, len(text) - start, self.docstring)
        self.setCurrentBlockState(1 if in_string else 0)


class CodeEditor(QPlainTextEdit):
    """写 Python 用的编辑框。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(12)
        self.setFont(font)
        self.setObjectName("CodeEditor")
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * 4)
        self.highlighter = PythonHighlighter(self.document())

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Tab:                      # Tab = 4 个空格
            self.insertPlainText("    ")
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            # 回车后保持和上一行一样的缩进；上一行以冒号结尾就再多缩进一级
            line = self.textCursor().block().text()
            indent = line[: len(line) - len(line.lstrip(" "))]
            if line.rstrip().endswith(":"):
                indent += "    "
            super().keyPressEvent(event)
            self.insertPlainText(indent)
            return
        super().keyPressEvent(event)
