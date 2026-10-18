"""界面的样式表（类似网页的 CSS），参考 macOS"系统设置"的风格：

  - 浅灰背景 + 白色圆角分组，没有边框、没有阴影
  - 黑白灰为主，主按钮和开关用近黑色，不用彩色
  - 侧栏是浅灰色，选中项是更深一点的灰色圆角

改颜色只需要改这里；质量报告 report.html 用的也是这几个颜色。
"""

from pathlib import Path

# 样式表里的图标要写成文件路径；路径里不能有反斜杠，所以统一用 as_posix()
ICONS = (Path(__file__).parent / "icons").as_posix()

INK = "#1d1d1f"          # 主文字、主按钮、打开的开关
SECONDARY = "#6e6e73"    # 次要文字
TERTIARY = "#8e8e93"     # 更淡的文字、图标
HAIRLINE = "#e5e5ea"     # 分隔线
FIELD_BORDER = "#d2d2d7" # 输入框边框
BG = "#f5f5f7"           # 窗口背景
SIDEBAR = "#ececee"      # 侧栏背景
SIDEBAR_SELECTED = "#dcdce1"
CARD = "#ffffff"
SWITCH_OFF = "#d1d1d6"

# 兼容旧代码里用到的名字
ACCENT = INK
TEXT = INK

STYLE = f"""
QWidget {{ color: {INK}; font-size: 13px; }}
QMainWindow, QWidget#Content, QStackedWidget, QScrollArea, QWidget#ScrollBody {{ background: {BG}; }}
QToolTip {{ color: {INK}; background: white; border: 1px solid {HAIRLINE}; padding: 6px;
            border-radius: 6px; }}

/* ---------- 侧栏 ---------- */
QFrame#Sidebar {{ background: {SIDEBAR}; border-right: 1px solid #dedee3; }}
QFrame#Sidebar QLabel {{ background: transparent; }}
QLabel#AppName {{ font-size: 15px; font-weight: 700; }}
QLabel#AppTag {{ color: {SECONDARY}; font-size: 12px; }}
QLabel#SidebarFoot {{ color: {TERTIARY}; font-size: 11px; }}
QFrame#SidebarStep {{ border-radius: 8px; background: transparent; }}
QFrame#SidebarStep:hover {{ background: #e3e3e8; }}
QFrame#SidebarStep[state="active"] {{ background: {SIDEBAR_SELECTED}; }}
QLabel#SidebarTitle {{ color: {SECONDARY}; font-weight: 600; }}
QLabel#SidebarSub {{ color: {TERTIARY}; font-size: 11px; }}
QFrame#SidebarStep[state="active"] QLabel#SidebarTitle {{ color: {INK}; }}
QFrame#SidebarStep[state="done"] QLabel#SidebarTitle {{ color: {INK}; }}
QLabel#StepBadge {{ border-radius: 11px; border: 1px solid #c7c7cc; color: {TERTIARY};
                    font-weight: 600; font-size: 11px; background: transparent; }}
QFrame#SidebarStep[state="active"] QLabel#StepBadge {{ background: {INK}; border-color: {INK};
                                                      color: white; }}
QFrame#SidebarStep[state="done"] QLabel#StepBadge {{ background: #d1d1d6; border-color: #d1d1d6;
                                                    color: {INK}; }}

/* ---------- 文字 ---------- */
QLabel#PageTitle {{ font-size: 26px; font-weight: 700; }}
QLabel#PageSubtitle {{ color: {SECONDARY}; font-size: 13px; }}
QLabel#Muted {{ color: {SECONDARY}; }}
QLabel#SectionLabel {{ color: {SECONDARY}; font-size: 13px; font-weight: 600; }}
QLabel#RowTitle {{ font-size: 13px; font-weight: 600; }}
QLabel#ParamLabel {{ color: #3a3a3c; }}
QLabel#StatValue {{ font-size: 24px; font-weight: 600; }}
QLabel#StatDetails {{ color: #3a3a3c; font-size: 12px; padding-top: 6px; }}
QLabel#ResultTitle {{ font-size: 15px; font-weight: 600; }}
QLabel#DropTitle {{ font-size: 15px; font-weight: 600; }}

/* ---------- 分组（白色圆角块）和行 ---------- */
QFrame#Group, QFrame#Card, QFrame#StatCard {{ background: {CARD}; border: none; border-radius: 12px; }}
QFrame#Row {{ background: transparent; border: none; }}
QFrame#Separator {{ background: {HAIRLINE}; border: none; max-height: 1px; min-height: 1px; }}

QFrame#DropZone {{ background: {CARD}; border: 1.5px dashed #c7c7cc; border-radius: 14px; }}
QFrame#DropZone[hover="true"] {{ border-color: {INK}; background: #fafafa; }}

QFrame#Footer {{ background: {BG}; border-top: 1px solid {HAIRLINE}; }}

/* ---------- 按钮 ---------- */
QPushButton {{ background: #e8e8ed; border: none; border-radius: 8px; padding: 6px 14px;
               color: {INK}; }}
QPushButton:hover {{ background: #dedee3; }}
QPushButton:pressed {{ background: #d1d1d6; }}
QPushButton:disabled {{ color: #aeaeb2; background: #f0f0f2; }}
QPushButton#Primary {{ background: {INK}; color: white; font-weight: 600; padding: 7px 20px; }}
QPushButton#Primary:hover {{ background: #3a3a3c; }}
QPushButton#Primary:pressed {{ background: #000000; }}
QPushButton#Primary:disabled {{ background: #c7c7cc; color: white; }}
QPushButton::menu-indicator {{ image: url({ICONS}/chevron-down.svg); width: 10px; height: 10px;
                               subcontrol-position: right center; subcontrol-origin: padding;
                               right: 8px; }}
QPushButton#MenuButton {{ padding-right: 26px; }}
QMenu {{ background: white; border: 1px solid {HAIRLINE}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 16px; border-radius: 5px; }}
QMenu::item:selected {{ background: #f0f0f2; color: {INK}; }}
QToolButton#SectionToggle {{ border: none; background: transparent; color: {SECONDARY};
                             font-size: 13px; font-weight: 600; padding: 2px 0; }}
QToolButton#SectionToggle:hover {{ color: {INK}; }}
QToolButton#MoveButton {{ border: none; background: transparent; padding: 5px; border-radius: 6px; }}
QToolButton#MoveButton:hover {{ background: #f0f0f2; }}
QToolButton#MoveButton:disabled {{ background: transparent; }}
QToolButton#IconButton {{ border: none; background: transparent; color: {TERTIARY};
                          font-size: 13px; padding: 4px 8px; border-radius: 6px; }}
QToolButton#IconButton:hover {{ background: #f0f0f2; color: #ff3b30; }}

/* ---------- 输入框 ---------- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: white; border: 1px solid {FIELD_BORDER}; border-radius: 7px;
    padding: 4px 8px; min-height: 20px; selection-background-color: #d1d1d6;
    selection-color: {INK}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {SECONDARY}; }}
QLineEdit:read-only {{ background: {BG}; border-color: {BG}; color: #3a3a3c; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox::down-arrow {{ image: url({ICONS}/chevron-down.svg); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ background: white; border: 1px solid {HAIRLINE}; outline: none;
                               selection-background-color: #f0f0f2; selection-color: {INK}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border;
    subcontrol-position: top right; width: 22px; border: none; margin-top: 2px; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border;
    subcontrol-position: bottom right; width: 22px; border: none; margin-bottom: 2px; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({ICONS}/chevron-up.svg);
    width: 10px; height: 10px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({ICONS}/chevron-down.svg);
    width: 10px; height: 10px; }}

QPlainTextEdit#CodeEditor {{ background: #fbfbfd; border: 1px solid {HAIRLINE}; border-radius: 8px;
                             padding: 6px; selection-background-color: #d6e4ff;
                             selection-color: {INK}; }}
QPlainTextEdit#CodeEditor:focus {{ border-color: {SECONDARY}; }}
QPushButton:checked {{ background: #dedee3; }}
QLabel#WorkflowBadge {{ border-radius: 11px; background: {BG}; color: {SECONDARY};
                        font-weight: 600; font-size: 11px; }}
QLabel#Warning {{ color: #b25000; }}

/* ---------- 进度条和滚动条 ---------- */
QProgressBar {{ border: none; background: {HAIRLINE}; border-radius: 3px; max-height: 6px;
                color: transparent; }}
QProgressBar::chunk {{ background: {INK}; border-radius: 3px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #c7c7cc; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
"""
