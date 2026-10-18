"""测试 Output 页的 Workflow（调整执行顺序）和在软件里直接写脚本。"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path  # noqa: E402
from unittest import mock  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import soundfile as sf  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

import tidyaudio.custom_script as cs  # noqa: E402
from tidyaudio.custom_script import TEMPLATE  # noqa: E402
from tidyaudio.gui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dataset(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    sf.write(raw / "a.wav", 0.3 * np.sin(np.arange(16000) / 10), 16000)
    return raw


def workflow_names(w):
    return [label.text() for label in w.workflow.group.findChildren(QLabel, "RowTitle")]


def test_workflow_order_and_reorder(app, dataset, tmp_path):
    w = MainWindow()
    w.set_input_dir(dataset)
    for key in ("clipping_check", "resample", "normalize", "features"):
        w.cards[key].set_checked(True)
    w.script_list.add_script(str(TEMPLATE))
    w._go(2)
    # 推荐顺序：脚本在内置处理之后、特征提取之前
    assert workflow_names(w) == ["Clipping Detection", "Resampling", "Normalization",
                                 "custom_script_template.py", "Feature Extraction"]
    assert w.workflow.state.text() == "Recommended order"
    assert w.build_config().order is None

    w._move_step(0, 1)                                     # 削波检测往后挪
    w._move_step(1, 1)                                     # 再往后，挪到归一化后面
    assert workflow_names(w)[:3] == ["Resampling", "Normalization", "Clipping Detection"]
    assert w.workflow.state.text() == "Custom order"
    assert "clipping in the original audio" in w.workflow.warnings.text()
    assert w.build_config().order == ["resample", "normalize", "clipping_check", "script_0",
                                      "features"]

    # 保存成预设再加载，顺序还在
    preset = tmp_path / "p.yaml"
    w.build_config().save(preset)
    w2 = MainWindow()
    w2.set_input_dir(dataset)
    with mock.patch("tidyaudio.gui.main_window.QFileDialog.getOpenFileName",
                    return_value=(str(preset), "")):
        w2._load_preset()
    assert w2.build_config().order == w.build_config().order

    w._reset_order()
    assert w.build_config().order is None
    assert w.workflow.warnings.isHidden()


def test_new_steps_keep_custom_order(app, dataset):
    w = MainWindow()
    w.set_input_dir(dataset)
    w.cards["resample"].set_checked(True)
    w.cards["normalize"].set_checked(True)
    w._go(2)
    w._move_step(0, 1)                                     # 归一化放到重采样前面
    w.cards["clipping_check"].set_checked(True)            # 之后又打开一个功能
    w._go(2)
    # 自定义的部分保留，新功能按推荐位置插进去
    assert workflow_names(w) == ["Clipping Detection", "Normalization", "Resampling"]


def test_new_script_and_editor(app, tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SCRIPTS_DIR", tmp_path / "Scripts")
    w = MainWindow()
    sl = w.script_list
    chosen = tmp_path / "my folder" / "drop dogs"            # 用户在访达里选的位置和名字（没写 .py）
    chosen.parent.mkdir()
    with mock.patch("tidyaudio.gui.widgets.QFileDialog.getSaveFileName",
                    return_value=(str(chosen), "")) as dialog:
        sl._new_script()
    assert dialog.call_args[0][2].startswith(str(tmp_path / "Scripts"))   # 默认打开脚本文件夹
    row = sl.rows[0]
    path = Path(row.path)
    assert path == tmp_path / "my folder" / "drop dogs.py" and path.exists()
    assert row.editor_box.isVisibleTo(row)                 # 新建后直接展开编辑框
    assert "def process" in row.editor.toPlainText()

    row.editor.setPlainText("def process(audio, sr)\n    return audio, sr\n")   # 少了冒号
    assert row.dirty and sl.unsaved() == [row]
    assert row.save() is False                             # 语法错误：不保存
    assert "Line 1" in row.status.text()
    assert "def process(audio, sr)\n" not in path.read_text()

    row.editor.setPlainText("def helper(): pass\n")       # 没有 process 函数
    assert row.save() is False and "process" in row.status.text()

    good = "def process(audio, sr, info):\n    return audio, sr\n"
    row.editor.setPlainText(good)
    assert sl.save_all() is True                           # 开始处理前会自动保存
    assert path.read_text() == good and not row.dirty

    row.editor.setPlainText("x = 1\n")
    row.load()                                             # Revert：丢掉没保存的修改
    assert row.editor.toPlainText() == good


def test_same_script_not_added_twice(app):
    w = MainWindow()
    w.script_list.add_script(str(TEMPLATE))
    w.script_list.add_script(str(TEMPLATE))
    assert len(w.script_list.scripts()) == 1


def test_toggle_updates_layout_in_one_go(app):
    """关掉一个功能后，下面的卡片要立刻挪上来。

    如果要等下一轮事件处理才挪，中间会画出一帧"参数没了、下面还是空白"的画面，看起来就是闪一下。
    """
    w = MainWindow()
    w.resize(1040, 760)
    w.show()
    w.file_count = 1
    w._go(1)
    w.standard_section.button.setChecked(True)
    w.cards["clipping_check"].set_checked(True)
    for _ in range(5):
        app.processEvents()
    below = w.cards["resample"]
    y_open = below.mapTo(w, below.rect().topLeft()).y()

    w.cards["clipping_check"].set_checked(False)          # 不调用 processEvents
    y_closed = below.mapTo(w, below.rect().topLeft()).y()
    assert y_closed < y_open - 40                          # 已经挪上来了
    assert w.selected_label.text() == "0 steps on"
    w.close()


def test_new_script_does_not_overwrite_existing_file(app, tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SCRIPTS_DIR", tmp_path / "Scripts")
    existing = tmp_path / "mine.py"
    existing.write_text("def process(audio, sr):\n    return audio, sr  # my code\n")
    w = MainWindow()
    with mock.patch("tidyaudio.gui.widgets.QFileDialog.getSaveFileName",
                    return_value=(str(existing), "")):
        w.script_list._new_script()
    assert "# my code" in existing.read_text()             # 没被模板覆盖
    row = w.script_list.rows[0]
    assert "# my code" in row.editor.toPlainText()
    assert "already existed" in row.status.text()


def test_cancel_new_script(app, tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SCRIPTS_DIR", tmp_path / "Scripts")
    w = MainWindow()
    with mock.patch("tidyaudio.gui.widgets.QFileDialog.getSaveFileName", return_value=("", "")):
        w.script_list._new_script()
    assert w.script_list.scripts() == []


def test_source_format_shown(app, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    tone = 0.3 * np.sin(np.arange(1600) / 5)
    for i in range(3):
        sf.write(raw / f"s{i}.wav", tone, 16000, subtype="PCM_16")
    w = MainWindow()
    w.set_input_dir(raw)
    # Input 页的卡片
    assert w.sr_stat.value.text() == "16 kHz"
    assert w.depth_stat.value.text() == "16-bit"
    assert w.ch_stat.value.text() == "Mono"
    # 全部一样：只标 (current)
    combo = w.cards["bit_depth"].editors["subtype"].editor
    assert combo.itemText(combo.findData("PCM_16")) == "PCM_16  (current)"
    assert combo.itemText(combo.findData("PCM_24")) == "PCM_24"
    sr = w.cards["resample"].editors["target_sr"].editor
    assert sr.itemText(sr.findData(16000)) == "16000  (current)"
    ch = w.cards["channels"].editors["mode"].editor
    assert ch.itemText(ch.findData("mono")) == "mono  (current)"
    assert w.cards["bit_depth"].values()["subtype"] == "PCM_16"   # 选项的值没变

    # 不一样：下拉框不标；卡片显示 Mixed + 明细，悬停说明原因和建议
    sf.write(raw / "loud.wav", tone, 16000, subtype="PCM_24")
    w.set_input_dir(raw)
    assert combo.itemText(combo.findData("PCM_16")) == "PCM_16"
    assert combo.itemText(combo.findData("PCM_24")) == "PCM_24"
    assert w.depth_stat.value.text() == "Mixed"
    assert w.depth_stat.details.text() == "16-bit · 3 files\n24-bit · 1 file"
    # 建议直接写在卡片里，加一个一键按钮
    card = w.depth_stat
    assert "2 different formats" in card.advice.text()
    assert card.action.isVisibleTo(card) and card.action.text() == "Convert Format"
    assert not w.cards["bit_depth"].is_checked()
    card.action.click()                                         # 一键打开
    assert w.cards["bit_depth"].is_checked()
    assert card.action.text() == "✓  Turned on" and not card.action.isEnabled()
    assert w.standard_section.button.isChecked()                # 所在区域也展开了
    w.cards["bit_depth"].set_checked(False)                     # 在 Processing 页关掉
    assert card.action.text() == "Convert Format" and card.action.isEnabled()
    assert not w.sr_stat.action.isVisibleTo(w.sr_stat)          # 采样率统一：没有按钮
    assert w.sr_stat.details.isHidden()                         # 采样率统一：没有明细
    assert sr.itemText(sr.findData(16000)) == "16000  (current)"   # 统一的照样标
