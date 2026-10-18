"""主窗口：左边是步骤栏，右边是当前页面，底部是"上一步/下一步"。

  1 Input：选择（或拖入）输入文件夹 → 显示有多少条音频、有没有数据集 CSV
  2 Processing：打开要用的内置功能（常规 + 折叠的高级），添加自己的脚本，加载/保存预设
  3 Output：选择输出文件夹（不选就用默认的 "xxx_output"）→ 开始处理 → 打开结果和报告
"""

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget)

from .. import __version__
from ..custom_script import TEMPLATE, check_python
from ..dataset_csv import find_dataset_csv
from ..pipeline import (
    STEP_ORDER, PipelineConfig, RunSummary, default_order, default_output_dir, merge_order,
    order_warnings)
from ..scan import AudioProperties, read_properties, scan_folder
from ..steps import REGISTRY
from .style import STYLE
from .widgets import (
    Collapsible, DropZone, ElidedLabel, Group, ScriptList, SidebarStep, StatCard, StepCard,
    ToggleSwitch, WorkflowList, setting_row)
from .worker import PipelineWorker

# 打开软件时默认打开的功能：全部关闭，由用户自己决定
DEFAULT_CHECKED: set[str] = set()

FORMATS = [("WAV", "wav"), ("FLAC", "flac"), ("Same as source", "same")]

PAGES = [
    ("Input", "Choose your audio",
     "Input", "Choose the folder that contains your source audio."),
    ("Processing", "Pick what to apply",
     "Processing", "Turn on the steps you need. They always run top to bottom. "
                   "Hover over a setting to learn more."),
    ("Output", "Save and run",
     "Output", "Choose where to save the results, then start processing."),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TidyAudio")
        self.resize(1040, 760)
        self.setMinimumSize(880, 620)
        self.setStyleSheet(STYLE)

        self.input_dir: Path | None = None
        self.output_choice: str | None = None   # None = 用默认的输出文件夹
        # 用户在 Workflow 里调整过的顺序（内置功能用 key，脚本用 "script:<路径>"）；None = 推荐顺序
        self.custom_order: list[str] | None = None
        self.file_count = 0
        self.worker: PipelineWorker | None = None
        self.last_summary: RunSummary | None = None

        self.pages = QStackedWidget()
        self.pages.addWidget(self._page(0, self._build_input_page(), scroll=True))
        self.pages.addWidget(self._page(1, self._build_steps_page(), scroll=True))
        self.pages.addWidget(self._page(2, self._build_output_page(), scroll=True))

        right = QWidget()
        right.setObjectName("Content")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self.pages, 1)
        right_layout.addWidget(self._build_footer())

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_sidebar())
        layout.addWidget(right, 1)
        self.setCentralWidget(root)
        self._go(0)

    # ------------------------------------------------------------ 框架：侧栏、页面、底栏

    def _build_sidebar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("Sidebar")
        bar.setFixedWidth(220)
        name = QLabel("TidyAudio")
        name.setObjectName("AppName")
        tag = QLabel("Audio dataset preprocessing")
        tag.setObjectName("AppTag")
        foot = QLabel(f"Version {__version__}")
        foot.setObjectName("SidebarFoot")

        layout = QVBoxLayout(bar)
        layout.setContentsMargins(12, 28, 12, 16)
        layout.setSpacing(2)
        head = QVBoxLayout()
        head.setContentsMargins(10, 0, 0, 0)
        head.setSpacing(2)
        head.addWidget(name)
        head.addWidget(tag)
        layout.addLayout(head)
        layout.addSpacing(26)
        self.sidebar_steps = []
        for i, (title, sub, _, _) in enumerate(PAGES):
            item = SidebarStep(i + 1, title, sub)
            item.clicked.connect(lambda i=i: self._go(i))
            layout.addWidget(item)
            self.sidebar_steps.append(item)
        layout.addStretch()
        foot.setContentsMargins(10, 0, 0, 0)
        layout.addWidget(foot)
        return bar

    def _page(self, index: int, body: QWidget, scroll: bool = False) -> QWidget:
        """每一页：上面是大标题和说明，下面是内容（内容多的页面可以滚动）。"""
        _, _, title, subtitle = PAGES[index]
        t = QLabel(title)
        t.setObjectName("PageTitle")
        s = QLabel(subtitle)
        s.setObjectName("PageSubtitle")
        s.setWordWrap(True)

        inner = QWidget()
        inner.setObjectName("ScrollBody")
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(44, 36, 44, 28)
        inner_layout.setSpacing(4)
        inner_layout.addWidget(t)
        inner_layout.addWidget(s)
        inner_layout.addSpacing(22)
        inner_layout.addWidget(body)
        inner_layout.addStretch()
        if not scroll:
            return inner
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setWidget(inner)
        return area

    def _build_footer(self) -> QFrame:
        footer = QFrame()
        footer.setObjectName("Footer")
        self.back_btn = QPushButton("Back")
        self.next_btn = QPushButton("Continue")
        self.next_btn.setObjectName("Primary")
        self.back_btn.clicked.connect(lambda: self._go(self.pages.currentIndex() - 1))
        self.next_btn.clicked.connect(lambda: self._go(self.pages.currentIndex() + 1))
        self.footer_status = QLabel("")
        self.footer_status.setObjectName("Muted")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(44, 12, 44, 12)
        layout.addWidget(self.back_btn)
        layout.addStretch()
        layout.addWidget(self.footer_status)
        layout.addSpacing(14)
        layout.addWidget(self.next_btn)
        return footer

    # ------------------------------------------------------------ 1 Input

    def _build_input_page(self) -> QWidget:
        page = QWidget()
        drop = DropZone()
        drop.folder_chosen.connect(self.set_input_dir)

        # 选好文件夹后显示：路径、是否包括子文件夹、三个统计数字
        self.input_path = ElidedLabel("")
        self.input_path.setObjectName("Muted")
        self.recursive_box = ToggleSwitch()
        self.recursive_box.setChecked(True)
        self.recursive_box.toggled.connect(self._rescan)
        self.input_group = Group()
        folder_row = QFrame()
        folder_row.setObjectName("Row")
        folder_title = QLabel("Folder")
        folder_title.setObjectName("RowTitle")
        folder_layout = QVBoxLayout(folder_row)
        folder_layout.setContentsMargins(16, 11, 16, 11)
        folder_layout.setSpacing(1)
        folder_layout.addWidget(folder_title)
        folder_layout.addWidget(self.input_path)
        self.input_group.add(folder_row)
        self.input_group.add(setting_row("Include subfolders", self.recursive_box))
        self.input_group.hide()

        self.files_stat = StatCard("Audio files")
        self.formats_stat = StatCard("Formats")
        self.csv_stat = StatCard("Dataset CSV")
        self.stats_box = QWidget()
        stats = QHBoxLayout(self.stats_box)
        stats.setContentsMargins(0, 0, 0, 0)
        stats.setSpacing(12)
        for card in (self.files_stat, self.formats_stat, self.csv_stat):
            stats.addWidget(card, 1)
        self.stats_box.hide()

        # 第二行：原始音频的采样率、位深、声道
        self.sr_stat = StatCard("Sample rate")
        self.depth_stat = StatCard("Bit depth")
        self.ch_stat = StatCard("Channels")
        self.format_box_cards = QWidget()
        fmt = QHBoxLayout(self.format_box_cards)
        fmt.setContentsMargins(0, 0, 0, 0)
        fmt.setSpacing(12)
        for card in (self.sr_stat, self.depth_stat, self.ch_stat):
            fmt.addWidget(card, 1)
        # 卡片里的一键按钮：打开对应的统一格式的功能
        self.fix_steps = {self.sr_stat: "resample", self.depth_stat: "bit_depth",
                          self.ch_stat: "channels"}
        for card, key in self.fix_steps.items():
            card.action.clicked.connect(lambda _=False, key=key: self._turn_on(key))
        self.format_box_cards.hide()
        self.csv_hint = QLabel("")
        self.csv_hint.setObjectName("Muted")
        self.csv_hint.setWordWrap(True)
        self.csv_hint.setContentsMargins(4, 0, 4, 0)

        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(drop)
        layout.addSpacing(6)
        layout.addWidget(self.input_group)
        layout.addWidget(self.stats_box)
        layout.addWidget(self.format_box_cards)
        layout.addWidget(self.csv_hint)
        return page

    def set_input_dir(self, folder: str | Path) -> None:
        self.input_dir = Path(folder)
        self.input_path.setFullText(str(self.input_dir))
        self.input_path.setToolTip(str(self.input_dir))
        self.input_group.show()
        self.stats_box.show()
        self.format_box_cards.show()
        self._rescan()

    def _rescan(self) -> None:
        if not self.input_dir:
            return
        result = scan_folder(self.input_dir, recursive=self.recursive_box.isChecked())
        self.file_count = result.count
        self.files_stat.set(f"{result.count:,}")
        exts = sorted(result.by_extension.items(), key=lambda kv: -kv[1])
        self.formats_stat.set(", ".join(e.lstrip(".").upper() for e, _ in exts) or "—",
                              "\n".join(f"{e}: {n}" for e, n in exts))
        csv_path = find_dataset_csv(self.input_dir)
        self.csv_stat.set(csv_path.name if csv_path else "None")
        self._show_properties(read_properties(result.files))
        if result.count == 0:
            self.csv_hint.setText("No audio files found in this folder.")
        elif csv_path:
            self.csv_hint.setText(f"An updated copy of {csv_path.name} will be saved with the "
                                  "output, with durations, sample rates and segments reflecting "
                                  "the processing.")
        else:
            self.csv_hint.setText("No dataset CSV found. A manifest.csv describing every output "
                                  "file will still be saved.")
        self._update_output_hint()
        self._update_nav()

    def _show_properties(self, props: AudioProperties) -> None:
        """在 Input 页显示原始格式；全部一样时，在 Processing 页的下拉框里标出 (current)。"""
        for card, counter, kind in ((self.sr_stat, props.sample_rates, "sample_rate"),
                                    (self.depth_stat, props.subtypes, "bit_depth"),
                                    (self.ch_stat, props.channels, "channels")):
            card.set(*_describe(counter, kind, props))
        self._update_fix_buttons()

        # 不统一时不标：每个选项后面都挂个数字，反而看不清
        def current(counter, key=lambda v: v):
            return {key(next(iter(counter))): "current"} if len(counter) == 1 else {}
        self.cards["resample"].editors["target_sr"].annotate(current(props.sample_rates))
        self.cards["bit_depth"].editors["subtype"].annotate(current(props.subtypes))
        self.cards["channels"].editors["mode"].annotate(
            current(props.channels, key=lambda n: {1: "mono", 2: "stereo"}.get(n, n)))

    def _update_fix_buttons(self) -> None:
        """格式不统一的卡片显示按钮："Turn On xxx"，已经打开了就显示 "✓ xxx is on"。"""
        if not hasattr(self, "cards"):
            return
        for card, key in self.fix_steps.items():
            if not card.advice.isVisibleTo(card):
                continue
            label = REGISTRY[key].label
            if self.cards[key].is_checked():
                card.set_action("✓  Turned on", enabled=False)
                card.action.setToolTip(f"{label} is on. Adjust it on the Processing page.")
            else:
                card.set_action(_FIX_BUTTON[key], enabled=True)
                card.action.setToolTip(f"Turn on {label} on the Processing page")

    def _turn_on(self, key: str) -> None:
        """一键打开某个功能，并展开它所在的区域（去 Processing 页就能直接看到和调整）。"""
        self.cards[key].set_checked(True)
        section = self.advanced_section if REGISTRY[key].advanced else self.standard_section
        section.button.setChecked(True)
        self._update_fix_buttons()

    # ------------------------------------------------------------ 2 Processing

    def _build_steps_page(self) -> QWidget:
        page = QWidget()
        load_btn = QPushButton("Load Preset…")
        save_btn = QPushButton("Save Preset…")
        load_btn.setToolTip("Load a saved preset, or the config.yaml from any previous output")
        load_btn.clicked.connect(self._load_preset)
        save_btn.clicked.connect(self._save_preset)
        self.selected_label = QLabel("")
        self.selected_label.setObjectName("Muted")
        bar = QHBoxLayout()
        bar.addWidget(self.selected_label)
        bar.addStretch()
        bar.addWidget(load_btn)
        bar.addWidget(save_btn)

        # 按执行顺序排列，分成常规和高级两个分组
        self.cards: dict[str, StepCard] = {}
        normal, advanced = Group(), Group()
        order = STEP_ORDER + sorted(set(REGISTRY) - set(STEP_ORDER))
        for key in order:
            cls = REGISTRY[key]
            card = StepCard(cls)
            card.set_checked(key in DEFAULT_CHECKED)
            card.toggled.connect(self._update_selected_label)
            card.toggled.connect(self._update_fix_buttons)     # Input 页的一键按钮跟着变
            self.cards[key] = card
            (advanced if cls.advanced else normal).add(card)

        self.script_list = ScriptList(TEMPLATE)
        self.script_list.changed.connect(self._update_selected_label)

        # 三个区域都可以折叠；标题会显示打开了几个，收起来也能看到
        self.standard_section = Collapsible("", normal, expanded=False)
        self.advanced_section = Collapsible("", advanced, expanded=False)
        self.script_section = Collapsible("", self.script_list, expanded=True)

        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addLayout(bar)
        layout.addSpacing(4)
        layout.addWidget(self.standard_section)
        layout.addSpacing(8)
        layout.addWidget(self.advanced_section)
        layout.addSpacing(8)
        layout.addWidget(self.script_section)
        self._update_selected_label()
        return page

    def selected_steps(self) -> dict[str, dict]:
        return {key: card.values() for key, card in self.cards.items() if card.is_checked()}

    def apply_steps(self, steps: dict[str, dict]) -> None:
        """按预设设置开关和参数；预设里没有的功能关掉并恢复默认值。"""
        for key, card in self.cards.items():
            card.reset()
            card.set_checked(key in steps)
            if key in steps:
                card.set_values(steps[key] or {})

    def _update_selected_label(self) -> None:
        if not hasattr(self, "script_section"):   # 页面还在搭建中
            return
        cards = list(self.cards.values())
        std = [c for c in cards if not c.step_cls.advanced]
        adv = [c for c in cards if c.step_cls.advanced]
        n_std = sum(c.is_checked() for c in std)
        n_adv = sum(c.is_checked() for c in adv)
        n_scripts = len(self.script_list.scripts())
        self.standard_section.set_title(f"Standard  ·  {n_std} of {len(std)} on")
        self.advanced_section.set_title(f"Advanced  ·  {n_adv} of {len(adv)} on")
        self.script_section.set_title(f"Your Python Scripts  ·  {n_scripts} added")

        n = n_std + n_adv
        text = f"{n} step{'s' if n != 1 else ''} on"
        if n_scripts:
            text += f" · {n_scripts} Python script{'s' if n_scripts != 1 else ''}"
        self.selected_label.setText(text)

    def _load_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load preset", "", "Preset (*.yaml *.yml)")
        if not path:
            return
        try:
            config = PipelineConfig.load(path)
        except Exception as e:
            QMessageBox.warning(self, "Could not load preset", f"This file is not a valid preset:\n{e}")
            return
        unknown = set(config.steps) - set(self.cards)
        self.apply_steps({k: v for k, v in config.steps.items() if k in self.cards})
        self.script_list.set_scripts([s for s in config.scripts if Path(s["path"]).is_file()])
        missing = [s["path"] for s in config.scripts if not Path(s["path"]).is_file()]
        self._restore_order(config)
        self._set_format(config.output_format)
        notes = []
        # 预设里记的 Python 在这台电脑上不一定有（比如是别人电脑上的路径），检查一下
        self.script_list.set_python(None)
        if config.script_python:
            try:
                self.script_list.set_python(config.script_python, check_python(config.script_python))
            except Exception as e:
                notes.append(f"The preset's Python could not be used, so the built-in one is "
                             f"selected instead:\n{e}")
        if unknown:
            notes.append(f"Steps not available in this version: {', '.join(sorted(unknown))}")
        if missing:
            notes.append("Scripts not found:\n" + "\n".join(missing))
        if notes:
            QMessageBox.information(self, "Preset partly loaded", "\n\n".join(notes))

    def _restore_order(self, config: PipelineConfig) -> None:
        """把配置里的顺序（"script_0" 这种 key）换回界面用的写法（"script:<路径>"）。"""
        self.custom_order = None
        if not config.order:
            return
        ids = []
        for key in config.order:
            if key.startswith("script_") and key[7:].isdigit():
                i = int(key[7:])
                if i < len(config.scripts):
                    ids.append(_script_id(config.scripts[i]["path"]))
            else:
                ids.append(key)
        default, _ = self._workflow_keys(custom=None)
        merged = merge_order(default, ids)
        self.custom_order = None if merged == default else merged

    def _save_preset(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save preset", "my_preset.yaml", "Preset (*.yaml)")
        if path:
            self.build_config().save(path)

    # ------------------------------------------------------------ 3 Output

    def _build_output_page(self) -> QWidget:
        page = QWidget()

        # 输出设置：一个分组，两行
        self.output_path = ElidedLabel("")
        self.output_path.setObjectName("Muted")
        browse = QPushButton("Choose…")
        browse.clicked.connect(self._choose_output)
        reset = QPushButton("Use Default")
        reset.clicked.connect(self._reset_output)
        folder_buttons = QWidget()
        fb = QHBoxLayout(folder_buttons)
        fb.setContentsMargins(0, 0, 0, 0)
        fb.addWidget(browse)
        fb.addWidget(reset)
        folder_row = QFrame()
        folder_row.setObjectName("Row")
        folder_title = QLabel("Folder")
        folder_title.setObjectName("RowTitle")
        folder_text = QVBoxLayout()
        folder_text.setSpacing(1)
        folder_text.addWidget(folder_title)
        folder_text.addWidget(self.output_path)
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(16, 11, 16, 11)
        folder_layout.addLayout(folder_text, 1)
        folder_layout.addWidget(folder_buttons)

        self.format_box = QComboBox()
        self.format_box.setFixedWidth(170)
        for text, value in FORMATS:
            self.format_box.addItem(text, value)
        out_group = Group()
        out_group.add(folder_row)
        out_group.add(setting_row("Format", self.format_box))
        self.output_hint = QLabel("")
        self.output_hint.setObjectName("Muted")
        self.output_hint.setWordWrap(True)
        self.output_hint.setContentsMargins(4, 0, 4, 0)

        # Workflow：按执行顺序列出每一步，可以调整
        self.workflow = WorkflowList()
        self.workflow.moved.connect(self._move_step)
        self.workflow.reset_requested.connect(self._reset_order)

        # 运行：文件数 + 开始按钮 + 进度
        run_group = Group()
        run_body = QFrame()
        run_body.setObjectName("Row")
        self.plan_label = QLabel("")
        self.plan_label.setObjectName("RowTitle")
        self.run_btn = QPushButton("Start Processing")
        self.run_btn.setObjectName("Primary")
        self.run_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.hide()
        self.progress = QProgressBar()
        self.progress.hide()
        self.current_file = QLabel("")
        self.current_file.setObjectName("Muted")
        self.current_file.hide()                 # 只在处理时显示，不然会多出一块空白
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.cancel_btn)
        buttons.addStretch()
        run_layout = QVBoxLayout(run_body)
        run_layout.setContentsMargins(16, 14, 16, 16)
        run_layout.setSpacing(12)
        run_layout.addWidget(self.plan_label)
        run_layout.addLayout(buttons)
        run_layout.addWidget(self.progress)
        run_layout.addWidget(self.current_file)
        run_group.add(run_body)

        # 结果：处理完才显示
        self.result_box = QWidget()
        self.result_title = QLabel("")
        self.result_title.setObjectName("ResultTitle")
        self.result_path = QLabel("")
        self.result_path.setObjectName("Muted")
        self.result_path.setWordWrap(True)
        self.result_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.out_stat = StatCard("Output files")
        self.drop_stat = StatCard("Dropped")
        self.time_stat = StatCard("Processing time")
        stats = QHBoxLayout()
        stats.setSpacing(12)
        for c in (self.out_stat, self.drop_stat, self.time_stat):
            stats.addWidget(c, 1)
        open_report = QPushButton("Open Quality Report")
        open_report.setObjectName("Primary")
        open_folder = QPushButton("Show in Finder")
        open_report.clicked.connect(lambda: _open(self.last_summary.output_dir / "report.html"))
        open_folder.clicked.connect(lambda: _open(self.last_summary.output_dir))
        open_row = QHBoxLayout()
        open_row.addWidget(open_report)
        open_row.addWidget(open_folder)
        open_row.addStretch()
        res_layout = QVBoxLayout(self.result_box)
        res_layout.setContentsMargins(0, 0, 0, 0)
        res_layout.setSpacing(10)
        res_layout.addWidget(self.result_title)
        res_layout.addWidget(self.result_path)
        res_layout.addSpacing(4)
        res_layout.addLayout(stats)
        res_layout.addSpacing(4)
        res_layout.addLayout(open_row)
        self.result_box.hide()

        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(_section("Save to"))
        layout.addWidget(out_group)
        layout.addWidget(self.output_hint)
        layout.addSpacing(18)
        layout.addWidget(self.workflow)
        layout.addSpacing(18)
        layout.addWidget(_section("Run"))
        layout.addWidget(run_group)
        layout.addSpacing(22)
        layout.addWidget(self.result_box)
        return page

    def _choose_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if folder:
            self.output_choice = folder
            self._update_output_hint()

    def _reset_output(self) -> None:
        self.output_choice = None
        self._update_output_hint()

    def _update_output_hint(self) -> None:
        if self.output_choice:
            self.output_path.setFullText(self.output_choice)
            self.output_hint.setText("Results will be saved in this folder.")
        elif self.input_dir:
            self.output_path.setFullText(str(default_output_dir(self.input_dir)))
            self.output_hint.setText("By default the output goes next to the input folder, with "
                                     "_output added to the name. Earlier results are never "
                                     "overwritten; a number is added instead.")

    # ---- Workflow：执行顺序 ----

    def _workflow_keys(self, custom: list[str] | None) -> tuple[list[str], list[str]]:
        """返回 (推荐顺序, 实际顺序)。脚本在界面里用 "script:<路径>" 表示。"""
        builtin = [k for k, c in self.cards.items() if c.is_checked()]
        scripts = [(_script_id(s["path"]), "last") for s in self.script_list.scripts()]
        default = default_order(builtin, scripts)
        return default, merge_order(default, custom) if custom else default

    def _update_plan(self) -> None:
        """第三页：文件数，以及 Workflow 的执行顺序。"""
        default, keys = self._workflow_keys(self.custom_order)
        if keys == default:
            self.custom_order = None
        items = []
        for key in keys:
            if key.startswith("script:"):
                items.append((Path(key[7:]).name, "Python script"))
            else:
                items.append((REGISTRY[key].label, "Built-in"))
        warnings = order_warnings(keys, names={k: Path(k[7:]).name for k in keys
                                              if k.startswith("script:")})
        self.workflow.set_items(items, custom=self.custom_order is not None, warnings=warnings)
        self.plan_label.setText(f"{self.file_count:,} files · {len(keys)} "
                                f"step{'s' if len(keys) != 1 else ''}")

    def _move_step(self, index: int, delta: int) -> None:
        _, keys = self._workflow_keys(self.custom_order)
        target = index + delta
        if 0 <= target < len(keys):
            keys[index], keys[target] = keys[target], keys[index]
            self.custom_order = keys
            self._update_plan()

    def _reset_order(self) -> None:
        self.custom_order = None
        self._update_plan()

    def _set_format(self, value: str) -> None:
        i = self.format_box.findData(value)
        if i >= 0:
            self.format_box.setCurrentIndex(i)

    def build_config(self) -> PipelineConfig:
        return PipelineConfig(
            input_dir=str(self.input_dir) if self.input_dir else "",
            output_dir=self.output_choice,
            output_format=self.format_box.currentData(),
            recursive=self.recursive_box.isChecked(),
            steps=self.selected_steps(),
            scripts=self.script_list.scripts(),
            script_python=self.script_list.python(),
            order=self._config_order(),
        )

    def _config_order(self) -> list[str] | None:
        """把界面的顺序换成配置文件的写法：脚本 "script:<路径>" → "script_0"、"script_1"……"""
        if self.custom_order is None:
            return None
        _, keys = self._workflow_keys(self.custom_order)
        index = {_script_id(s["path"]): i for i, s in enumerate(self.script_list.scripts())}
        return [f"script_{index[k]}" if k in index else k for k in keys]

    def _start(self) -> None:
        # 编辑框里还没保存的脚本，开始前先保存；有语法错误就不开始，并展开那个脚本
        if not self.script_list.save_all():
            QMessageBox.warning(self, "Script has errors",
                                "A script could not be saved. Fix the error shown under it on the "
                                "Processing page, then start again.")
            return
        self.worker = PipelineWorker(self.build_config(), self)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)

        self.result_box.hide()
        self.progress.setRange(0, max(self.file_count, 1))
        self.progress.setValue(0)
        self.progress.show()
        self.current_file.setText("Starting…")
        self.current_file.show()
        self.run_btn.hide()
        self.cancel_btn.show()
        self.cancel_btn.setEnabled(True)
        self._set_busy(True)
        self.worker.start()

    def _cancel(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.current_file.setText("Stopping after the current file…")

    def _on_progress(self, done: int, total: int, name: str) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.current_file.setText(f"{done:,} of {total:,}   ·   {name}")
        self.footer_status.setText(f"Processing… {100 * done // max(total, 1)}%")

    def _on_finished(self, summary: RunSummary) -> None:
        self.last_summary = summary
        self.result_title.setText("Cancelled. Files processed so far were saved."
                                  if summary.cancelled else "Done")
        self.result_path.setText(str(summary.output_dir))
        self.out_stat.set(f"{summary.output_count:,}")
        self.drop_stat.set(f"{summary.dropped_count:,}")
        self.time_stat.set(f"{summary.seconds:.1f} s",
                           "Wall-clock time to read, process and save all audio files")
        self.result_box.show()
        self._finish_run()

    def _on_failed(self, message: str) -> None:
        self._finish_run()
        self.progress.hide()
        QMessageBox.critical(self, "Processing failed", message)

    def _finish_run(self) -> None:
        self.current_file.setText("")
        self.current_file.hide()
        self.progress.hide()
        self.footer_status.setText("")
        self.cancel_btn.hide()
        self.run_btn.show()
        self.run_btn.setText("Run Again")
        self._set_busy(False)
        self._update_output_hint()   # 默认输出文件夹已经被用掉了，更新成下一个编号

    # ------------------------------------------------------------ 翻页

    def _go(self, index: int) -> None:
        if self.worker and self.worker.isRunning():
            return                                  # 处理期间不能翻页
        index = max(0, min(index, self.pages.count() - 1))
        if index > 0 and self.file_count == 0:      # 没选好输入，不能往后走
            index = 0
        self.pages.setCurrentIndex(index)
        for i, item in enumerate(self.sidebar_steps):
            item.set_state("active" if i == index else "done" if i < index else "todo")
        if index == 2:
            self._update_plan()
        self._update_nav()

    def _update_nav(self) -> None:
        i = self.pages.currentIndex()
        self.back_btn.setVisible(i > 0)
        self.next_btn.setVisible(i < self.pages.count() - 1)
        self.next_btn.setEnabled(not (i == 0 and self.file_count == 0))
        self.footer_status.setText(f"Step {i + 1} of {self.pages.count()}")

    def _set_busy(self, busy: bool) -> None:
        """处理期间不能翻页、不能改输出设置。"""
        self.back_btn.setEnabled(not busy)
        self.format_box.setEnabled(not busy)

    def closeEvent(self, event) -> None:
        if self.worker and self.worker.isRunning():
            answer = QMessageBox.question(self, "Processing in progress",
                                          "Processing is still running. Quit anyway?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.worker.cancel()
            self.worker.wait()
        event.accept()


def _files(n: int) -> str:
    return f"{n:,} file" + ("" if n == 1 else "s")


def _format_rate(sr: int) -> str:
    return f"{sr / 1000:g} kHz"


def _format_subtype(subtype: str) -> str:
    """soundfile 的格式名 → 好懂的位深说法。"""
    return {"PCM_16": "16-bit", "PCM_24": "24-bit", "PCM_32": "32-bit", "PCM_S8": "8-bit",
            "PCM_U8": "8-bit", "FLOAT": "32-bit float", "DOUBLE": "64-bit float",
            "MPEG_LAYER_III": "MP3 (compressed)", "VORBIS": "Vorbis (compressed)",
            "OPUS": "Opus (compressed)"}.get(subtype, subtype)


def _format_channels(n: int) -> str:
    return {1: "Mono", 2: "Stereo"}.get(n, f"{n} channels")


# 每种格式："不统一"时怎么解释、建议打开哪个功能
_MIXED_ADVICE = {
    "sample_rate": ("{n} different sample rates.", "Most models expect one."),
    "bit_depth": ("{n} different formats.", "One format keeps the dataset consistent."),
    "channels": ("Mono and stereo are mixed.", "Most models expect one layout."),
}
# 一键按钮上的字（短一点，卡片放得下）
_FIX_BUTTON = {"resample": "Resample All", "bit_depth": "Convert Format",
               "channels": "Convert Channels"}
_FORMATTERS = {"sample_rate": lambda v: _format_rate(v), "bit_depth": lambda v: _format_subtype(v),
               "channels": lambda v: _format_channels(v)}


def _describe(counter, kind: str, props: AudioProperties) -> tuple[str, str, list[str], str]:
    """卡片的 (大字, 鼠标悬停说明, 卡片里的明细, 卡片里的建议)。"""
    fmt = _FORMATTERS[kind]
    sampled = (f"\nBased on the first {props.checked:,} of {props.total:,} files."
               if props.sampled else "")
    if not counter:
        return "—", "", [], ""
    if len(counter) == 1:
        value = fmt(next(iter(counter)))
        return value, f"All files: {value}.{sampled}", [], ""
    # 不统一：明细最多列 4 行，多的合并成"还有几种"
    rows = counter.most_common()
    details = [f"{fmt(v)} · {_files(n)}" for v, n in rows[:4]]
    if len(rows) > 4:
        details.append(f"+{len(rows) - 4} more")
    reason, advice = _MIXED_ADVICE[kind]
    text = f"{reason.format(n=len(rows))} {advice}"
    return "Mixed", text + sampled, details, text


def _script_id(path: str) -> str:
    return "script:" + str(Path(path).expanduser().resolve())


def _section(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("SectionLabel")
    label.setContentsMargins(4, 0, 0, 0)
    return label


def _open(path: Path) -> None:
    """用系统默认的程序打开（文件夹用访达，html 用浏览器）。"""
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
