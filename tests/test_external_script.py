"""测试"在另一个 Python 环境里运行脚本"。

大部分测试用软件自己的 Python（sys.executable）当作"另一个环境"，任何电脑上都能跑；
最后几个测试用电脑上真实的另一个环境（比如 Anaconda），找不到就跳过。
"""

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tidyaudio.custom_script import (
    CustomScript, ExternalScript, check_python, make_script)
from tidyaudio.pipeline import PipelineConfig, run_pipeline


def script(tmp_path, body, name="s.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def write_tone(path, seconds=1.0, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, 0.3 * np.sin(2 * np.pi * 440 * t), sr)


def other_python():
    """电脑上另一个装了 numpy 的 Python（不是软件自己的那个）。"""
    for candidate in ["/opt/anaconda3/bin/python", shutil.which("python3"), shutil.which("python")]:
        # 不用 resolve：虚拟环境的 python 是指向别处的快捷方式，真实文件可能相同
        if not candidate or Path(candidate).absolute() == Path(sys.executable).absolute():
            continue
        ok = subprocess.run([candidate, "-c", "import numpy"], capture_output=True).returncode == 0
        if ok:
            return candidate
    return None


OTHER = other_python()
needs_other = pytest.mark.skipif(OTHER is None, reason="no other Python with numpy found")


# ---------- 用 sys.executable 当外部环境 ----------

def test_external_matches_builtin(tmp_path, make_clip):
    body = """
        def process(audio, sr, info):
            return audio * 0.5, sr
    """
    clip_a, clip_b = make_clip(channels=2), make_clip(channels=2)
    [a] = CustomScript(script(tmp_path, body)).apply(clip_a)
    ext = ExternalScript(script(tmp_path, body, "e.py"), python=sys.executable)
    try:
        [b] = ext.apply(clip_b)
    finally:
        ext.close()
    assert np.array_equal(a.audio, b.audio)                 # 两种方式结果完全一样


def test_external_drop_split_and_info(tmp_path, make_clip):
    ext = ExternalScript(script(tmp_path, """
        def process(audio, sr, info):
            if info["name"] == "drop_me":
                return None
            half = len(audio) // 2
            return [(audio[:half], sr), (audio[half:], sr // 2)]
    """), python=sys.executable)
    try:
        clip = make_clip(seconds=2.0, sr=16000)
        clip.name = "drop_me"
        assert ext.apply(clip) == []
        parts = ext.apply(make_clip(seconds=2.0, sr=16000))
        assert [p.sr for p in parts] == [16000, 8000]
    finally:
        ext.close()


def test_print_in_script_does_not_break_messages(tmp_path, make_clip):
    ext = ExternalScript(script(tmp_path, """
        def process(audio, sr):
            print("hello from the script")       # print 会写到 stdout
            return audio, sr
    """), python=sys.executable)
    try:
        assert len(ext.apply(make_clip())) == 1
    finally:
        ext.close()


def test_error_inside_script_is_reported(tmp_path, make_clip):
    ext = ExternalScript(script(tmp_path, """
        def process(audio, sr):
            raise ValueError("bad file")
    """), python=sys.executable)
    try:
        with pytest.raises(RuntimeError, match="bad file"):
            ext.apply(make_clip())
    finally:
        ext.close()


def test_crash_restarts_for_next_file(tmp_path, make_clip):
    ext = ExternalScript(script(tmp_path, """
        import os
        def process(audio, sr, info):
            if info["name"] == "crash":
                os._exit(1)                          # 模拟脚本把整个 Python 弄崩
            return audio, sr
    """), python=sys.executable)
    try:
        bad = make_clip()
        bad.name = "crash"
        with pytest.raises(RuntimeError, match="stopped unexpectedly"):
            ext.apply(bad)
        assert len(ext.apply(make_clip())) == 1      # 自动重启，下一个文件照常处理
    finally:
        ext.close()


def test_load_error_is_reported(tmp_path):
    with pytest.raises(RuntimeError, match="Could not load"):
        ExternalScript(script(tmp_path, "import not_a_real_module\n"), python=sys.executable)


def test_bad_python_path(tmp_path):
    with pytest.raises(RuntimeError, match="Could not start"):
        ExternalScript(script(tmp_path, "def process(a, s): return a, s\n"),
                       python=str(tmp_path / "no_python"))
    with pytest.raises(RuntimeError, match="Could not run"):
        check_python(tmp_path / "no_python")


def test_check_python():
    assert check_python(sys.executable).startswith("Python ")


def test_make_script_uses_builtin_for_own_python(tmp_path):
    s = script(tmp_path, "def process(a, s): return a, s\n")
    assert type(make_script(s)) is CustomScript
    assert type(make_script(s, python=sys.executable)) is CustomScript


def test_symlinked_python_counts_as_different(tmp_path):
    """指向同一个真实文件的另一个 python 路径，也要当成"另一个环境"。"""
    s = script(tmp_path, "def process(a, s): return a, s\n")
    link = tmp_path / "python"
    link.symlink_to(Path(sys.executable).resolve())
    ext = make_script(s, python=str(link))
    try:
        assert type(ext) is ExternalScript
    finally:
        ext.close()


def test_pipeline_closes_process(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    write_tone(raw / "a.wav")
    s = script(tmp_path, "def process(a, s): return a, s\n")
    started = []
    original = ExternalScript._start

    def spy(self):
        original(self)
        started.append(self)
    monkeypatch.setattr(ExternalScript, "_start", spy)
    # 用一个指向软件自己 Python 的快捷方式，当作"另一个环境"
    link = tmp_path / "python"
    link.symlink_to(Path(sys.executable).resolve())

    run_pipeline(PipelineConfig(input_dir=str(raw), scripts=[{"path": str(s)}],
                                script_python=str(link)))
    assert started and all(e.proc is None for e in started)   # 处理完进程已经关掉


# ---------- 用电脑上真实的另一个环境 ----------

@needs_other
def test_library_only_in_other_environment(tmp_path):
    """pandas 只在另一个环境里有：内置 Python 加载失败，外部 Python 能用。"""
    has_pandas = subprocess.run([OTHER, "-c", "import pandas"], capture_output=True).returncode == 0
    in_venv = subprocess.run([sys.executable, "-c", "import pandas"], capture_output=True).returncode == 0
    if not has_pandas or in_venv:
        pytest.skip("needs pandas in the other environment only")

    raw = tmp_path / "raw"
    write_tone(raw / "dog" / "a.wav")
    s = script(tmp_path, """
        import pandas as pd
        def process(audio, sr, info):
            pd.DataFrame({"x": [1]})               # 真的用一下 pandas
            return audio, sr
    """)
    with pytest.raises(RuntimeError, match="pandas"):
        run_pipeline(PipelineConfig(input_dir=str(raw), scripts=[{"path": str(s)}]))

    summary = run_pipeline(PipelineConfig(input_dir=str(raw), scripts=[{"path": str(s)}],
                                          script_python=OTHER))
    assert summary.output_count == 1
    readme = (summary.output_dir / "dataset" / "README.txt").read_text(encoding="utf-8")
    assert f"Run with: {OTHER}" in readme


@needs_other
def test_example_script_in_other_environment(tmp_path):
    """examples 里的脚本用到 soxr；另一个环境没有的话，应该清楚地报错。"""
    example = Path(__file__).parent.parent / "examples" / "resample48k_drop_alarm_baby.py"
    has_soxr = subprocess.run([OTHER, "-c", "import soxr"], capture_output=True).returncode == 0
    raw = tmp_path / "raw"
    write_tone(raw / "dog" / "a.wav")
    write_tone(raw / "alarm" / "b.wav")
    config = PipelineConfig(input_dir=str(raw), scripts=[{"path": str(example)}],
                            script_python=OTHER)
    if has_soxr:
        summary = run_pipeline(config)
        assert summary.output_count == 1
    else:
        with pytest.raises(RuntimeError, match="soxr"):
            run_pipeline(config)


# ---------- 选文件夹，自动找里面的 Python ----------

def fake_env(folder, rel="bin/python"):
    path = folder / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_find_python_in_project_folder(tmp_path):
    from tidyaudio.custom_script import find_python_in
    project = tmp_path / "my_project"
    python = fake_env(project / ".venv")
    assert find_python_in(project) == python                  # 选项目文件夹
    assert find_python_in(project / ".venv") == python        # 选 .venv 本身


def test_find_python_other_layouts(tmp_path):
    from tidyaudio.custom_script import find_python_in
    venv_named = fake_env(tmp_path / "p1" / "venv")
    windows = fake_env(tmp_path / "p2" / ".venv", "Scripts/python.exe")
    conda = fake_env(tmp_path / "envs" / "torch")             # conda 环境：直接有 bin/python
    assert find_python_in(tmp_path / "p1") == venv_named
    assert find_python_in(tmp_path / "p2") == windows
    assert find_python_in(tmp_path / "envs" / "torch") == conda


def test_find_python_reports_missing(tmp_path):
    from tidyaudio.custom_script import find_python_in
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No Python environment found"):
        find_python_in(tmp_path / "empty")


def test_describe_env(tmp_path):
    from tidyaudio.custom_script import describe_env
    assert describe_env(Path.home() / "proj" / ".venv" / "bin" / "python") == "~/proj/.venv"


def test_real_venv_end_to_end(tmp_path):
    """真的建一个 .venv（借用系统里已经装好的库，不用联网），选项目文件夹就能用。"""
    from tidyaudio.custom_script import check_python, find_python_in
    project = tmp_path / "friend_project"
    project.mkdir()
    made = subprocess.run([sys.executable, "-m", "venv", "--system-site-packages",
                           "--without-pip", str(project / ".venv")], capture_output=True)
    if made.returncode != 0:
        pytest.skip("could not create a venv here")
    python = find_python_in(project)
    try:
        check_python(python)
    except RuntimeError:
        pytest.skip("the new venv has no numpy")

    raw = tmp_path / "raw"
    write_tone(raw / "dog" / "a.wav")
    s = script(tmp_path, """
        import sys
        def process(audio, sr, info):
            assert ".venv" in sys.prefix, sys.prefix        # 确实在这个 .venv 里运行
            return audio, sr
    """)
    summary = run_pipeline(PipelineConfig(input_dir=str(raw), scripts=[{"path": str(s)}],
                                          script_python=str(python)))
    assert summary.output_count == 1


def test_resolve_python_accepts_file_or_folder(tmp_path):
    from tidyaudio.custom_script import resolve_python
    python = fake_env(tmp_path / "proj" / ".venv")
    assert resolve_python(python) == python                        # 直接给 python 文件
    assert resolve_python(tmp_path / "proj") == python             # 给项目文件夹
    assert resolve_python(f"  '{python}'  ") == python             # 粘贴时带了引号和空格
    with pytest.raises(FileNotFoundError):
        resolve_python(tmp_path / "nope")


def test_restart_closes_previous_process_files(tmp_path, make_clip):
    """外部 Python 崩溃后重启时，上一个进程的日志文件和管道要马上关掉，不要等垃圾回收。"""
    ext = ExternalScript(script(tmp_path, """
        import os
        def process(audio, sr):
            os._exit(1)
    """), python=sys.executable)
    try:
        with pytest.raises(RuntimeError):
            ext.apply(make_clip())
        old_log, old_proc = ext.log, ext.proc
        with pytest.raises(RuntimeError):
            ext.apply(make_clip())                       # 这一次会先重启
        assert old_log.closed
        assert old_proc.stdin.closed and old_proc.stdout.closed
    finally:
        ext.close()
    assert ext.proc is None and ext.log is None
