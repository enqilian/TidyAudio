"""界面上可以复用的小组件（macOS"系统设置"风格）。

ToggleSwitch：滑动开关（代替勾选框），用法和 QCheckBox 完全一样
Group：白色圆角分组，里面一行一行，行之间自动加细分隔线
ParamEditor：一个参数的输入控件，根据 Param 自动选择类型：
    有 choices → 下拉框；默认值是 True/False → 开关；整数 → 整数框；小数 → 小数框
StepCard：分组里的一行：功能名 + 一句说明 + 右边的开关；打开后下面展开参数
Collapsible：可以折叠/展开的区域
ScriptList：自定义脚本（添加、删除、选择执行位置、选择用哪个 Python 运行）
DropZone：可以把文件夹拖进来的区域
StatCard：统计数字小卡片
SidebarStep：侧栏里的一步（编号圆圈 + 标题 + 小字说明）
"""

import platform
from pathlib import Path

from PySide6.QtCore import (
    Property, QEasingCurve, QEvent, QPropertyAnimation, QRectF, QSize, Qt, Signal)
from PySide6.QtGui import QColor, QIcon, QPainter
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QInputDialog,
    QApplication, QLabel, QMenu, QMessageBox, QPushButton, QSizePolicy, QSpinBox, QToolButton,
    QVBoxLayout, QWidget)

from ..custom_script import (
    SCRIPTS_DIR, check_python, check_script_source, describe_env, new_script_path,
    resolve_python)
from ..steps import Param, Step
from .code_editor import CodeEditor
from .style import ICONS, INK, SWITCH_OFF

BIG = 1_000_000   # 参数没写 min/max 时用的范围

def without_flicker(widget: QWidget, change) -> None:
    """做一处会改变布局的修改（比如收起参数区），中间状态不画出来。

    1. 先暂停整个窗口的重画
    2. 做修改
    3. 马上把所有"待重新计算的布局"算完（Qt 默认会稍后才算）
    4. 再恢复重画：这时画出来的就已经是最终的样子
    不做第 3 步的话，会先画出"参数没了、下面的卡片还没挪上来"的空白状态，看起来就是闪一下。
    """
    window = widget.window()
    window.setUpdatesEnabled(False)
    try:
        change()
        # 布局是一层一层往上算的：卡片算完会再通知外面的分组、滚动区……这些通知又排进队列，
        # 所以多处理几轮，直到最外层也算完
        for _ in range(8):
            QApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)
    finally:
        window.setUpdatesEnabled(True)


def refresh_style(widget: QWidget) -> None:
    """改了 setProperty 之后，让样式表重新生效。"""
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class ToggleSwitch(QCheckBox):
    """iOS / macOS 风格的滑动开关。

    继承 QCheckBox，所以 isChecked()、setChecked()、toggled 信号都和勾选框一样，
    只是自己画成"圆角轨道 + 白色圆点"，切换时圆点有一个很短的滑动动画。
    """

    def __init__(self, parent=None, width: int = 38, height: int = 22):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._offset = 0.0                      # 0 = 关（圆点在左），1 = 开（圆点在右）
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)

    def _get_offset(self) -> float:
        return self._offset

    def _set_offset(self, value: float) -> None:
        self._offset = value
        self.update()

    offset = Property(float, _get_offset, _set_offset)

    def _animate(self, on: bool) -> None:
        end = 1.0 if on else 0.0
        if not self.isVisible():                # 还没显示出来，直接跳到终点，不用动画
            self._set_offset(end)
            return
        self._anim.stop()
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(end)
        self._anim.start()

    def sizeHint(self) -> QSize:
        return self.size()

    def hitButton(self, pos) -> bool:           # 点整个开关都算点击
        return self.rect().contains(pos)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        t = self._offset

        # 轨道颜色：关 = 浅灰，开 = 近黑，中间按动画进度混合
        off, on = QColor(SWITCH_OFF), QColor(INK)
        track = QColor(
            int(off.red() + (on.red() - off.red()) * t),
            int(off.green() + (on.green() - off.green()) * t),
            int(off.blue() + (on.blue() - off.blue()) * t))
        if not self.isEnabled():
            track.setAlpha(110)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)

        # 白色圆点 + 一点点阴影
        d = r.height() - 4
        x = r.left() + 2 + t * (r.width() - d - 4)
        p.setBrush(QColor(0, 0, 0, 28))
        p.drawEllipse(QRectF(x, r.top() + 2.6, d, d))
        p.setBrush(QColor("white"))
        p.drawEllipse(QRectF(x, r.top() + 2, d, d))


class Group(QFrame):
    """白色圆角分组。用 add() 加入的行之间会自动加一条细线。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Group")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._count = 0

    def add(self, widget: QWidget) -> None:
        if self._count:
            line = QFrame()
            line.setObjectName("Separator")
            wrap = QWidget()
            wrap_layout = QHBoxLayout(wrap)
            wrap_layout.setContentsMargins(16, 0, 0, 0)   # 分隔线左边缩进，和 macOS 一样
            wrap_layout.addWidget(line)
            self._layout.addWidget(wrap)
        self._layout.addWidget(widget)
        widget.show()                      # 之前拿出来过的行要重新显示
        self._count += 1

    def clear(self, keep=()) -> None:
        """清空分组。keep 里的行只是拿出来（之后可以再 add 回去），其他的（比如分隔线）删掉。"""
        keep = {id(w) for w in keep}
        while self._layout.count():
            widget = self._layout.takeAt(0).widget()
            if widget:
                # 先马上隐藏并拿出来，不然在真正删除之前还会画出来一帧（看起来像重影）
                widget.hide()
                widget.setParent(None)
                if id(widget) not in keep:
                    widget.deleteLater()
        self._count = 0


def setting_row(title: str, control: QWidget, subtitle: str = "") -> QFrame:
    """分组里的一行设置：左边是名字（和可选的小字说明），右边是控件。"""
    row = QFrame()
    row.setObjectName("Row")
    t = QLabel(title)
    t.setObjectName("RowTitle")
    text = QVBoxLayout()
    text.setSpacing(1)
    text.addWidget(t)
    if subtitle:
        s = QLabel(subtitle)
        s.setObjectName("Muted")
        s.setWordWrap(True)
        text.addWidget(s)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(16, 11, 16, 11)
    layout.addLayout(text, 1)
    layout.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter)
    return row


class ParamEditor(QWidget):
    """一个参数的输入控件。"""

    changed = Signal()

    def __init__(self, param: Param, parent=None):
        super().__init__(parent)
        self.param = param
        d = param.default

        if param.choices:
            self.editor = QComboBox()
            for c in param.choices:
                self.editor.addItem(str(c), c)       # 显示文字 + 真实的值
            self.editor.currentIndexChanged.connect(self.changed)
        elif isinstance(d, bool):                    # 注意 bool 要在 int 之前判断
            self.editor = ToggleSwitch(width=34, height=20)
            self.editor.toggled.connect(self.changed)
        elif isinstance(d, int):
            self.editor = QSpinBox()
            self.editor.setRange(int(_or(param.min, -BIG)), int(_or(param.max, BIG)))
            self.editor.valueChanged.connect(self.changed)
        else:
            self.editor = QDoubleSpinBox()
            lo, hi = _or(param.min, -BIG), _or(param.max, BIG)
            # 范围小的参数（比如 0.5~1 的阈值）需要更多小数位；dB 这种大范围的 1 位就够
            span = hi - lo
            self.editor.setDecimals(3 if span <= 1 else 2 if span <= 10 else 1)
            self.editor.setRange(lo, hi)
            self.editor.setSingleStep(0.001 if span <= 1 else 0.1 if span <= 10 else 1.0)
            self.editor.valueChanged.connect(self.changed)

        if isinstance(self.editor, QComboBox):
            # 下拉框按内容变宽：选项后面加了 "(current · 698 files)" 时也能完整显示
            self.editor.setMinimumWidth(170)
            self.editor.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        elif not isinstance(self.editor, QCheckBox):
            self.editor.setFixedWidth(170)
        if param.help:
            self.editor.setToolTip(param.help)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(self.editor)
        self.set_value(d)

    def annotate(self, notes: dict) -> None:
        """在下拉框选项后面加说明，比如 {16000: "current"} → "16000  (current)"。

        只改显示的文字，选项真正的值不变；传空字典就去掉所有说明。
        """
        if not isinstance(self.editor, QComboBox):
            return
        for i in range(self.editor.count()):
            value = self.editor.itemData(i)
            note = notes.get(value)
            self.editor.setItemText(i, f"{value}  ({note})" if note else str(value))

    def value(self):
        e = self.editor
        if isinstance(e, QComboBox):
            return e.currentData()
        if isinstance(e, QCheckBox):
            return e.isChecked()
        return e.value()

    def set_value(self, v) -> None:
        e = self.editor
        if isinstance(e, QComboBox):
            i = e.findData(v)
            if i < 0:   # 配置文件里的值可能是 16000.0 而选项是 16000，按文字再找一次
                i = e.findText(str(v))
            e.setCurrentIndex(max(i, 0))
        elif isinstance(e, QCheckBox):
            e.setChecked(bool(v))
        else:
            e.setValue(v)


class StepCard(QFrame):
    """分组里的一行：功能名 + 说明 + 右边的开关。打开后下面展开参数。"""

    toggled = Signal()

    def __init__(self, step_cls: type[Step], parent=None):
        super().__init__(parent)
        self.step_cls = step_cls
        self.setObjectName("Row")

        title = QLabel(step_cls.label)
        title.setObjectName("RowTitle")
        desc = QLabel(step_cls.description)
        desc.setObjectName("Muted")
        desc.setWordWrap(True)
        text = QVBoxLayout()
        text.setSpacing(1)
        text.addWidget(title)
        text.addWidget(desc)

        self.checkbox = ToggleSwitch()
        # 开关切换时：收起/展开参数 + 通知外面更新"几个打开"，放在同一次重画里完成
        self.checkbox.toggled.connect(
            lambda: without_flicker(self, lambda: (self._apply(), self.toggled.emit())))
        head = QHBoxLayout()
        head.setSpacing(16)
        head.addLayout(text, 1)
        head.addWidget(self.checkbox, 0, Qt.AlignmentFlag.AlignVCenter)

        # 参数区：左边是名字，右边是输入控件（靠右对齐，和 macOS 设置一样）
        self.params_box = QWidget()
        self.form = QFormLayout(self.params_box)
        self.form.setContentsMargins(0, 12, 0, 2)
        self.form.setHorizontalSpacing(16)
        self.form.setVerticalSpacing(10)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.editors: dict[str, ParamEditor] = {}
        for p in step_cls.params:
            ed = ParamEditor(p)
            ed.changed.connect(self._update)      # 参数变了，可能要显示/隐藏别的参数
            label = QLabel(p.label)
            label.setObjectName("ParamLabel")
            if p.help:
                label.setToolTip(p.help)
            self.form.addRow(label, ed)
            self.editors[p.name] = ed

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(0)
        layout.addLayout(head)
        layout.addWidget(self.params_box)
        self._update()

    def is_checked(self) -> bool:
        return self.checkbox.isChecked()

    def set_checked(self, on: bool) -> None:
        self.checkbox.setChecked(on)

    def values(self) -> dict:
        return {name: ed.value() for name, ed in self.editors.items()}

    def set_values(self, values: dict) -> None:
        for name, v in values.items():
            if name in self.editors:
                self.editors[name].set_value(v)
        self._update()

    def reset(self) -> None:
        self.set_values({p.name: p.default for p in self.step_cls.params})

    def _update(self) -> None:
        without_flicker(self, self._apply)

    def _apply(self) -> None:
        """打开后才显示参数；参数按 show_if 决定显示哪些。"""
        on = self.is_checked() and bool(self.editors)
        if on:                          # 关掉时不用管每一行，整个参数区直接收起
            values = self.values()
            for ed in self.editors.values():
                self.form.setRowVisible(ed, ed.param.is_active(values))
        self.params_box.setVisible(on)


class Collapsible(QWidget):
    """点标题就能折叠/展开的区域。"""

    def __init__(self, title: str, content: QWidget, expanded: bool = False, parent=None):
        super().__init__(parent)
        self.button = QToolButton()
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setChecked(expanded)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setIconSize(QSize(12, 12))
        self.button.setObjectName("SectionToggle")
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.toggled.connect(self._toggle)
        self.content = content

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.button)
        layout.addWidget(content)
        self._toggle(expanded)

    def set_title(self, title: str) -> None:
        self.button.setText(title)

    def _toggle(self, on: bool) -> None:
        icon = "chevron-down-section.svg" if on else "chevron-right.svg"
        self.button.setIcon(QIcon(f"{ICONS}/{icon}"))
        self.content.setVisible(on)


class ScriptRow(QFrame):
    """脚本列表里的一行：文件名 + 所在文件夹 + Edit + 删除。

    点 Edit 会在下面展开代码编辑框，可以直接改代码、保存。
    """

    removed = Signal(object)
    saved = Signal()

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.setObjectName("Row")
        self.path = str(Path(path).expanduser().resolve())

        name = QLabel(Path(self.path).name)
        name.setObjectName("RowTitle")
        folder = ElidedLabel(_short_path(Path(self.path).parent))
        folder.setObjectName("Muted")
        folder.setToolTip(self.path)
        text = QVBoxLayout()
        text.setSpacing(1)
        text.addWidget(name)
        text.addWidget(folder)

        self.edit_btn = QPushButton("Edit")
        self.edit_btn.setCheckable(True)
        self.edit_btn.toggled.connect(self._toggle_editor)
        remove = QToolButton()
        remove.setText("✕")
        remove.setObjectName("IconButton")
        remove.setToolTip("Remove this script from the list (the file is kept)")
        remove.clicked.connect(lambda: self.removed.emit(self))
        head = QHBoxLayout()
        head.setSpacing(6)
        head.addLayout(text, 1)
        head.addWidget(self.edit_btn)
        head.addWidget(remove)

        # 代码编辑区：默认收起
        self.editor = CodeEditor()
        self.editor.setMinimumHeight(300)
        self.editor.textChanged.connect(self._on_text_changed)
        self.status = QLabel("")
        self.status.setObjectName("Muted")
        self.status.setWordWrap(True)
        revert = QPushButton("Revert")
        revert.setToolTip("Discard unsaved changes")
        revert.clicked.connect(self.load)
        save = QPushButton("Save")
        save.setObjectName("Primary")
        save.clicked.connect(self.save)
        bar = QHBoxLayout()
        bar.addWidget(self.status, 1)
        bar.addWidget(revert)
        bar.addWidget(save)
        self.editor_box = QWidget()
        box = QVBoxLayout(self.editor_box)
        box.setContentsMargins(0, 10, 0, 0)
        box.setSpacing(8)
        box.addWidget(self.editor)
        box.addLayout(bar)
        self.editor_box.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 10, 10)
        layout.setSpacing(0)
        layout.addLayout(head)
        layout.addWidget(self.editor_box)
        self.dirty = False

    def to_dict(self) -> dict:
        return {"path": self.path, "position": "last"}

    def open_editor(self) -> None:
        self.edit_btn.setChecked(True)
        self.editor.setFocus()

    def _toggle_editor(self, on: bool) -> None:
        if on and not self.dirty:
            self.load()                        # 每次展开都读最新的文件内容
        self.editor_box.setVisible(on)
        self.edit_btn.setText("Done" if on else "Edit")

    def load(self) -> None:
        """从文件读代码（Revert 也是这个：丢掉没保存的修改）。"""
        try:
            code = Path(self.path).read_text(encoding="utf-8")
        except OSError as e:
            code = ""
            self._set_status(f"Could not read the file: {e}", error=True)
        self.editor.blockSignals(True)
        self.editor.setPlainText(code)
        self.editor.blockSignals(False)
        self.dirty = False
        self._set_status("")

    def save(self) -> bool:
        """检查代码、保存到文件。成功返回 True；有错误就在下面显示原因，返回 False。"""
        code = self.editor.toPlainText()
        try:
            check_script_source(code)
        except (SyntaxError, ValueError) as e:
            self._set_status(f"Not saved. {e}", error=True)
            return False
        Path(self.path).write_text(code, encoding="utf-8")
        self.dirty = False
        self._set_status("Saved")
        self.saved.emit()
        return True

    def _on_text_changed(self) -> None:
        self.dirty = True
        self._set_status("Unsaved changes")

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status.setText(text)
        self.status.setStyleSheet("color: #d70015;" if error else "")


class ScriptList(QWidget):
    """自定义脚本区域：说明 + 脚本列表 + 按钮 + "用哪个 Python 运行"。"""

    changed = Signal()

    def __init__(self, template_path: Path, parent=None):
        super().__init__(parent)
        self.template_path = template_path
        self.rows: list[ScriptRow] = []
        self._python: str | None = None      # None = 软件自己的 Python

        intro = QLabel(
            "Run your own Python code on every file, for anything the built-in steps don't "
            "cover, such as dropping certain classes or applying an effect. Click "
            "<b>New Script</b> to write one right here: it only needs a function "
            "<code>process(audio, sr)</code>. Set where it runs in the Workflow on the next page.")
        intro.setObjectName("Muted")
        intro.setWordWrap(True)

        # 脚本列表放在一个分组里；没有脚本时显示一行提示
        self.group = Group()
        self.empty = QLabel("No scripts yet")
        self.empty.setObjectName("Muted")
        self.empty.setContentsMargins(16, 12, 16, 12)
        self.group.add(self.empty)

        new_btn = QPushButton("New Script")
        new_btn.setToolTip(f"Create a script from the template in {_short_path(SCRIPTS_DIR)}")
        add_btn = QPushButton("Add Existing…")
        new_btn.clicked.connect(self._new_script)
        add_btn.clicked.connect(self._add)
        buttons = QHBoxLayout()
        buttons.addWidget(new_btn)
        buttons.addWidget(add_btn)
        buttons.addStretch()
        # 用哪个 Python 运行脚本
        self.env_value = ElidedLabel("")      # 路径很长时中间省略，不会把页面撑宽
        self.env_value.setObjectName("Muted")
        # "Choose…" 是一个菜单，三种指定环境的方式
        choose_py = QPushButton("Choose…")
        choose_py.setObjectName("MenuButton")
        menu = QMenu(choose_py)
        menu.addAction("From Project Folder…", self._choose_project_folder)
        menu.addAction("Python Interpreter…", self._choose_interpreter)
        menu.addAction("Enter Path…", self._enter_path)
        choose_py.setMenu(menu)
        builtin_py = QPushButton("Use Built-in")
        builtin_py.clicked.connect(lambda: self.set_python(None))
        env_controls = QWidget()
        env_buttons = QHBoxLayout(env_controls)
        env_buttons.setContentsMargins(0, 0, 0, 0)
        env_buttons.addWidget(choose_py)
        env_buttons.addWidget(builtin_py)
        self.env_title = QLabel("Python environment")
        self.env_title.setObjectName("RowTitle")
        env_row = QFrame()
        env_row.setObjectName("Row")
        env_text = QVBoxLayout()
        env_text.setSpacing(1)
        env_text.addWidget(self.env_title)
        env_text.addWidget(self.env_value)
        env_layout = QHBoxLayout(env_row)
        env_layout.setContentsMargins(16, 11, 16, 11)
        env_layout.addLayout(env_text, 1)
        env_layout.addWidget(env_controls)
        env_group = Group()
        env_group.add(env_row)
        env_hint = QLabel(
            "Scripts run in this app's own Python by default, which has numpy, scipy, librosa, "
            "soundfile and soxr. If your script needs other libraries, use another environment: "
            "pick a <b>project folder</b> that contains a .venv, pick the <b>Python interpreter</b> "
            "itself, or <b>paste a path</b> (e.g. the output of <code>which python</code>). "
            "The environment needs numpy installed.")
        env_hint.setObjectName("Muted")
        env_hint.setWordWrap(True)
        env_hint.setContentsMargins(4, 0, 4, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(intro)
        layout.addWidget(self.group)
        layout.addLayout(buttons)
        layout.addSpacing(8)
        layout.addWidget(env_group)
        layout.addWidget(env_hint)
        self.set_python(None)

    def scripts(self) -> list[dict]:
        return [r.to_dict() for r in self.rows]

    def python(self) -> str | None:
        return self._python

    def set_python(self, path: str | None, description: str = "") -> None:
        """设置用哪个 Python；None 表示软件自己的。"""
        self._python = path
        if path is None:
            self.env_value.setFullText(f"Built-in · Python {platform.python_version()}")
            self.env_value.setToolTip("")
        else:
            self.env_value.setFullText(describe_env(path)
                                       + (f"  ·  {description}" if description else ""))
            self.env_value.setToolTip(path)
        self.changed.emit()

    # ---- 三种指定环境的方式，最后都交给 use_environment ----

    def _choose_project_folder(self) -> None:
        """方式 1：选项目文件夹（或环境文件夹），自动用里面的 .venv。"""
        folder = QFileDialog.getExistingDirectory(
            self, "Choose a project folder that contains a .venv, or an environment folder")
        if folder:
            self.use_environment(folder)

    def _choose_interpreter(self) -> None:
        """方式 2：直接选 python 可执行文件。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a Python interpreter (in Finder, press ⌘⇧. to show hidden folders)")
        if path:
            self.use_environment(path)

    def _enter_path(self) -> None:
        """方式 3：直接粘贴路径，文件夹或 python 文件都可以。"""
        text, ok = QInputDialog.getText(
            self, "Enter Path",
            "Path to a Python interpreter or an environment folder\n"
            "(e.g. the output of: which python)")
        if ok and text.strip():
            self.use_environment(text)

    def use_environment(self, path: str) -> bool:
        """检查用户给的路径，能用就设置上；不能用就弹窗说明原因。"""
        try:
            python = resolve_python(path)
            description = check_python(python)    # 能运行、并且有 numpy 才接受
        except Exception as e:
            QMessageBox.warning(self, "This environment cannot be used", str(e))
            return False
        self.set_python(str(python), description)
        return True

    def set_scripts(self, scripts: list[dict]) -> None:
        self.rows = []
        self._rebuild()
        for s in scripts:
            self.add_script(s["path"])

    def add_script(self, path: str, open_editor: bool = False) -> None:
        resolved = str(Path(path).expanduser().resolve())
        if any(r.path == resolved for r in self.rows):
            return                               # 同一个脚本不重复添加
        row = ScriptRow(resolved)
        row.removed.connect(self._remove)
        self.rows.append(row)
        self._rebuild()
        self.changed.emit()
        if open_editor:
            row.open_editor()

    def _remove(self, row: ScriptRow) -> None:
        if row.dirty:
            answer = QMessageBox.question(
                self, "Unsaved changes",
                f"{Path(row.path).name} has unsaved changes. Remove it from the list anyway?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        self._rebuild()
        self.changed.emit()

    def unsaved(self) -> list[ScriptRow]:
        return [r for r in self.rows if r.dirty]

    def save_all(self) -> bool:
        """保存所有没保存的脚本。有一个保存失败（比如语法错误）就返回 False。"""
        ok = True
        for row in self.unsaved():
            if not row.save():
                row.open_editor()
                ok = False
        return ok

    def _rebuild(self) -> None:
        """重新排一遍分组里的行（行本身保留，里面没保存的代码不会丢）。"""
        self.group.clear(keep=self.rows + [self.empty])
        for row in self.rows:
            self.group.add(row)
        if not self.rows:
            self.group.add(self.empty)

    def _add(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add existing script", "", "Python (*.py)")
        for p in paths:
            self.add_script(p)

    def _new_script(self) -> None:
        """新建脚本：先用访达选位置、起名字，然后从模板生成，直接展开编辑框。

        如果选的文件已经存在，就直接打开它，不会用模板覆盖掉。
        """
        import tidyaudio.custom_script as cs          # 用模块里的值，测试时可以替换
        folder = cs.SCRIPTS_DIR
        folder.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "New Script", str(new_script_path("my_script", folder)), "Python (*.py)")
        if not path:
            return
        path = Path(path)
        if path.suffix != ".py":
            path = path.with_name(path.name + ".py")
        existed = path.exists()
        if not existed:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.template_path.read_text(encoding="utf-8"), encoding="utf-8")
        self.add_script(str(path), open_editor=True)
        if existed:
            row = next(r for r in self.rows if r.path == str(path.resolve()))
            row._set_status("This file already existed, so it was opened instead of replaced.")


class WorkflowList(QWidget):
    """Output 页的流程图：按执行顺序列出每一步，可以上移/下移，有问题的顺序会给出提醒。"""

    moved = Signal(int, int)           # (第几步, -1 上移 / +1 下移)
    reset_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        title = QLabel("Workflow")
        title.setObjectName("SectionLabel")
        self.state = QLabel("")
        self.state.setObjectName("Muted")
        self.reset_btn = QPushButton("Reset to Recommended")
        self.reset_btn.clicked.connect(self.reset_requested)
        head = QHBoxLayout()
        head.setContentsMargins(4, 0, 0, 0)
        head.addWidget(title)
        head.addStretch()
        head.addWidget(self.state)
        head.addWidget(self.reset_btn)

        self.group = Group()
        self.warnings = QLabel("")
        self.warnings.setWordWrap(True)
        self.warnings.setObjectName("Warning")
        self.warnings.setContentsMargins(4, 0, 4, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(head)
        layout.addWidget(self.group)
        layout.addWidget(self.warnings)

    def set_items(self, items: list[tuple[str, str]], custom: bool, warnings: list[str]) -> None:
        """items：[(名字, 类型说明), ...]，按执行顺序。"""
        self.group.clear()
        if not items:
            empty = QLabel("Nothing is turned on, so files are only copied / converted.")
            empty.setObjectName("Muted")
            empty.setContentsMargins(16, 12, 16, 12)
            self.group.add(empty)
        for i, (name, kind) in enumerate(items):
            self.group.add(self._row(i, name, kind, len(items)))
        self.state.setText("Custom order" if custom else "Recommended order")
        self.reset_btn.setVisible(custom)
        self.warnings.setText("<br>".join(f"⚠︎  {w}" for w in warnings))
        self.warnings.setVisible(bool(warnings))

    def _row(self, i: int, name: str, kind: str, total: int) -> QFrame:
        row = QFrame()
        row.setObjectName("Row")
        badge = QLabel(str(i + 1))
        badge.setObjectName("WorkflowBadge")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(22, 22)
        title = QLabel(name)
        title.setObjectName("RowTitle")
        tag = QLabel(kind)
        tag.setObjectName("Muted")
        up, down = QToolButton(), QToolButton()
        up.setIcon(QIcon(f"{ICONS}/chevron-up.svg"))
        down.setIcon(QIcon(f"{ICONS}/chevron-down.svg"))
        up.setToolTip("Run earlier")
        down.setToolTip("Run later")
        for b in (up, down):
            b.setObjectName("MoveButton")
            b.setIconSize(QSize(12, 12))
        up.setEnabled(i > 0)
        down.setEnabled(i < total - 1)
        up.clicked.connect(lambda: self.moved.emit(i, -1))
        down.clicked.connect(lambda: self.moved.emit(i, 1))
        layout = QHBoxLayout(row)
        layout.setContentsMargins(14, 8, 10, 8)
        layout.setSpacing(10)
        layout.addWidget(badge)
        layout.addWidget(title, 1)
        layout.addWidget(tag)
        layout.addSpacing(6)
        layout.addWidget(up)
        layout.addWidget(down)
        return row


class DropZone(QFrame):
    """可以把文件夹拖进来的区域，也可以点按钮选择。"""

    folder_chosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)

        icon = QLabel()
        icon.setPixmap(QIcon(f"{ICONS}/folder.svg").pixmap(44, 44))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("Drop a folder here")
        title.setObjectName("DropTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub = QLabel("WAV, FLAC, OGG, MP3 or AIFF")
        sub.setObjectName("Muted")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        browse = QPushButton("Choose Folder…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(browse)
        row.addStretch()

        # 高度固定：下面的内容变多时不会被挤扁（页面改成滚动）
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 32, 24, 28)
        layout.setSpacing(4)
        layout.addWidget(icon)
        layout.addSpacing(6)
        layout.addWidget(title)
        layout.addWidget(sub)
        layout.addSpacing(12)
        layout.addLayout(row)

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose input folder")
        if folder:
            self.folder_chosen.emit(folder)

    def _folder_from(self, event) -> str | None:
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile() and Path(urls[0].toLocalFile()).is_dir():
            return urls[0].toLocalFile()
        return None

    def dragEnterEvent(self, event) -> None:
        if self._folder_from(event):
            event.acceptProposedAction()
            self.setProperty("hover", True)
            refresh_style(self)

    def dragLeaveEvent(self, event) -> None:
        self.setProperty("hover", False)
        refresh_style(self)

    def dropEvent(self, event) -> None:
        self.setProperty("hover", False)
        refresh_style(self)
        folder = self._folder_from(event)
        if folder:
            self.folder_chosen.emit(folder)


class StatCard(QFrame):
    """一个统计数字的小卡片：大数字 + 小标签。"""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("StatCard")
        # 横向按比例分：同一排的卡片一样宽，不会因为里面字多就变宽
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.value = QLabel("—")
        self.value.setObjectName("StatValue")
        self.label = QLabel(label)
        self.label.setObjectName("Muted")
        # 明细：比如"不统一"时列出每种各有多少个文件；没有明细时隐藏
        self.details = QLabel("")
        self.details.setObjectName("StatDetails")
        self.details.setWordWrap(True)
        self.details.hide()
        # 建议和一键解决的按钮（比如"格式不统一"时）：没有时隐藏
        self.advice = QLabel("")
        self.advice.setObjectName("Muted")
        self.advice.setWordWrap(True)
        self.advice.hide()
        self.action = QPushButton("")
        self.action.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(2)
        layout.addWidget(self.value)
        layout.addWidget(self.label)
        layout.addWidget(self.details)
        layout.addWidget(self.advice)
        layout.addSpacing(4)
        layout.addWidget(self.action, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addStretch()

    def set(self, value: str, tooltip: str = "", details: list[str] | None = None,
            advice: str = "") -> None:
        self.value.setText(value)
        self.setToolTip(tooltip)
        self.details.setText("\n".join(details or []))
        self.details.setVisible(bool(details))
        self.advice.setText(advice)
        self.advice.setVisible(bool(advice))
        if not advice:
            self.action.hide()

    def set_action(self, text: str, enabled: bool) -> None:
        """显示一键按钮。enabled=False 表示已经处理好了（按钮变成灰色的"✓ 已打开"）。"""
        self.action.setText(text)
        self.action.setEnabled(enabled)
        self.action.show()


class SidebarStep(QFrame):
    """侧栏的一步：编号圆圈 + 标题 + 小字。状态：todo / active / done。"""

    clicked = Signal()

    def __init__(self, number: int, title: str, subtitle: str, parent=None):
        super().__init__(parent)
        self.setObjectName("SidebarStep")
        self.number = number
        self.badge = QLabel(str(number))
        self.badge.setObjectName("StepBadge")
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.setFixedSize(22, 22)
        t = QLabel(title)
        t.setObjectName("SidebarTitle")
        s = QLabel(subtitle)
        s.setObjectName("SidebarSub")
        text = QVBoxLayout()
        text.setSpacing(0)
        text.addWidget(t)
        text.addWidget(s)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)
        layout.addWidget(self.badge)
        layout.addLayout(text, 1)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_state(self, state: str) -> None:
        self.setProperty("state", state)
        self.badge.setText("✓" if state == "done" else str(self.number))
        for w in (self, self.badge):
            refresh_style(w)
        for child in self.findChildren(QLabel):
            refresh_style(child)

    def mousePressEvent(self, event) -> None:
        self.clicked.emit()


class ElidedLabel(QLabel):
    """太长时中间用 … 省略的标签（比如很长的文件夹路径），不会把窗口撑宽。"""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.full_text = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(40)

    def setFullText(self, text: str) -> None:
        self.full_text = text
        self.setText(self.fontMetrics().elidedText(text, Qt.TextElideMode.ElideMiddle, self.width()))

    def resizeEvent(self, event) -> None:
        metrics = self.fontMetrics()
        self.setText(metrics.elidedText(self.full_text, Qt.TextElideMode.ElideMiddle, self.width()))
        super().resizeEvent(event)


def _short_path(path: Path) -> str:
    """把用户主目录换成 ~，路径短一点。"""
    try:
        return "~/" + path.relative_to(Path.home()).as_posix()
    except ValueError:
        return str(path)


def _or(v, default):
    return default if v is None else v
