"""测试图形界面。用 offscreen 模式运行，不会弹出窗口。"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import soundfile as sf  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tidyaudio.gui.main_window import MainWindow  # noqa: E402
from tidyaudio.gui.worker import PipelineWorker  # noqa: E402
from tidyaudio.pipeline import PipelineConfig  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dataset(tmp_path):
    raw = tmp_path / "raw"
    for name, sec in [("dog/a.wav", 3), ("dog/b.wav", 2), ("cat/c.flac", 2)]:
        t = np.arange(int(sec * 44100)) / 44100
        (raw / name).parent.mkdir(parents=True, exist_ok=True)
        sf.write(raw / name, 0.3 * np.sin(2 * np.pi * 440 * t), 44100)
    (raw / "file_list.csv").write_text("file,class\ndog/a.wav,dog\n", encoding="utf-8")
    return raw


def test_step1_counts_files(app, dataset):
    w = MainWindow()
    assert not w.next_btn.isEnabled()              # 还没选文件夹，不能下一步
    w.set_input_dir(dataset)
    assert w.files_stat.value.text() == "3"
    assert w.formats_stat.value.text() == "WAV, FLAC"
    assert w.csv_stat.value.text() == "file_list.csv"
    assert w.next_btn.isEnabled()


def test_step2_defaults_and_params(app):
    w = MainWindow()
    assert w.selected_steps() == {}                          # 默认全部关闭
    card = w.cards["normalize"]
    card.set_checked(True)
    card.set_values({"method": "rms", "target_rms_db": -18.0})
    assert w.selected_steps()["normalize"]["method"] == "rms"
    assert w.selected_steps()["normalize"]["target_rms_db"] == -18.0


def test_show_if_hides_params(app):
    w = MainWindow()
    card = w.cards["normalize"]
    card.set_checked(True)
    card.set_values({"method": "peak"})
    visible = lambda name: card.editors[name].isVisibleTo(card)   # noqa: E731
    assert visible("target_peak_db")
    assert not visible("target_lufs")                    # 选了 peak，不显示目标响度
    card.set_values({"method": "lufs"})
    assert visible("target_lufs")
    card.set_checked(False)
    assert not visible("target_lufs")                    # 取消勾选，参数全部收起


def test_preset_round_trip(app, dataset, tmp_path):
    w = MainWindow()
    w.set_input_dir(dataset)
    w.cards["resample"].set_checked(True)
    w.cards["resample"].set_values({"target_sr": 22050})
    preset = tmp_path / "preset.yaml"
    w.build_config().save(preset)

    w2 = MainWindow()
    w2.apply_steps(PipelineConfig.load(preset).steps)
    assert w2.selected_steps()["resample"]["target_sr"] == 22050


def test_default_output_hint(app, dataset):
    w = MainWindow()
    w.set_input_dir(dataset)
    assert w.output_path.full_text.endswith("raw_output")


def test_worker_runs_pipeline(app, dataset):
    w = MainWindow()
    w.set_input_dir(dataset)
    w.cards["resample"].set_checked(True)
    worker = PipelineWorker(w.build_config())
    results, progress = [], []
    worker.finished_ok.connect(results.append)
    worker.progress.connect(lambda d, t, n: progress.append(d))
    worker.run()                                  # 直接在当前线程跑，方便测试
    assert results[0].output_count == 3
    assert progress[-1] == 3
    assert (results[0].output_dir / "report.html").exists()


def test_worker_reports_errors(app, dataset):
    config = PipelineConfig(input_dir=str(dataset), output_dir=str(dataset / "inside"))
    worker = PipelineWorker(config)
    errors = []
    worker.failed.connect(errors.append)
    worker.run()
    assert "output folder cannot" in errors[0]


def test_cannot_skip_to_later_steps_without_input(app):
    w = MainWindow()
    w._go(2)
    assert w.pages.currentIndex() == 0


def test_section_titles_show_counts(app):
    w = MainWindow()
    assert w.standard_section.button.text() == "Standard  ·  0 of 8 on"
    assert w.advanced_section.button.text() == "Advanced  ·  0 of 5 on"
    w.cards["denoise"].set_checked(True)
    assert w.advanced_section.button.text() == "Advanced  ·  1 of 5 on"
    assert w.script_section.button.text() == "Your Python Scripts  ·  0 added"
    w.standard_section.button.setChecked(False)              # 收起来
    assert not w.standard_section.content.isVisibleTo(w.standard_section)


def test_python_setting_in_config(app, dataset, tmp_path):
    import sys
    w = MainWindow()
    w.set_input_dir(dataset)
    assert w.build_config().script_python is None
    assert "Built-in" in w.script_list.env_value.full_text
    w.script_list.set_python(sys.executable, "Python x · numpy y")
    assert w.build_config().script_python == sys.executable

    # 预设里的 Python 在这台电脑上不存在 → 自动改回内置的
    preset = tmp_path / "p.yaml"
    cfg = w.build_config()
    cfg.script_python = str(tmp_path / "missing_python")
    cfg.save(preset)
    from unittest import mock
    with mock.patch("tidyaudio.gui.main_window.QFileDialog.getOpenFileName",
                    return_value=(str(preset), "")), \
         mock.patch("tidyaudio.gui.main_window.QMessageBox.information") as info:
        w._load_preset()
    assert w.build_config().script_python is None
    assert "could not be used" in info.call_args[0][2]


def test_three_ways_to_choose_environment(app, tmp_path):
    import subprocess
    import sys
    from unittest import mock
    w = MainWindow()
    sl = w.script_list
    from PySide6.QtWidgets import QMenu
    menu = [a.text() for a in sl.findChildren(QMenu)[0].actions()]
    assert menu == ["From Project Folder…", "Python Interpreter…", "Enter Path…"]

    project = tmp_path / "proj"
    project.mkdir()
    made = subprocess.run([sys.executable, "-m", "venv", "--system-site-packages",
                           "--without-pip", str(project / ".venv")], capture_output=True)
    if made.returncode != 0:
        pytest.skip("could not create a venv")
    venv_python = project / ".venv" / "bin" / "python"

    W = "tidyaudio.gui.widgets"
    with mock.patch(f"{W}.QFileDialog.getExistingDirectory", return_value=str(project)):
        sl._choose_project_folder()                                  # 方式 1：选文件夹
    assert sl.python() == str(venv_python)

    sl.set_python(None)
    with mock.patch(f"{W}.QFileDialog.getOpenFileName", return_value=(str(venv_python), "")):
        sl._choose_interpreter()                                     # 方式 2：选 python 文件
    assert sl.python() == str(venv_python)

    sl.set_python(None)
    with mock.patch(f"{W}.QInputDialog.getText", return_value=(str(project / ".venv"), True)):
        sl._enter_path()                                             # 方式 3：粘贴路径
    assert sl.python() == str(venv_python)

    with mock.patch(f"{W}.QInputDialog.getText", return_value=(str(tmp_path / "nope"), True)), \
         mock.patch(f"{W}.QMessageBox.warning") as warn:
        sl._enter_path()                                             # 路径不对：弹窗，不改设置
    assert warn.called and sl.python() == str(venv_python)


def test_standard_and_advanced_start_collapsed(app):
    w = MainWindow()
    assert not w.standard_section.button.isChecked()
    assert not w.advanced_section.button.isChecked()
