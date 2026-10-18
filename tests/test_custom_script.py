"""测试自定义脚本。"""

import csv
import textwrap

import numpy as np
import pytest
import soundfile as sf

from tidyaudio.custom_script import TEMPLATE, CustomScript
from tidyaudio.pipeline import PipelineConfig, run_pipeline


def script(tmp_path, body, name="my_script.py"):
    """把一段代码写成脚本文件。"""
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def write_tone(path, seconds=2.0, sr=16000, amplitude=0.3):
    t = np.arange(int(seconds * sr)) / sr
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, amplitude * np.sin(2 * np.pi * 440 * t), sr)


def read_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ---------- 三种返回值 ----------

def test_one_in_one_out(tmp_path, make_clip):
    s = CustomScript(script(tmp_path, """
        def process(audio, sr):
            return audio * 0.5, sr
    """))
    clip = make_clip(amplitude=0.4)
    [out] = s.apply(clip)
    assert np.abs(out.audio).max() == pytest.approx(0.2, rel=1e-3)


def test_return_none_drops(tmp_path, make_clip):
    s = CustomScript(script(tmp_path, """
        def process(audio, sr):
            return None
    """))
    clip = make_clip()
    assert s.apply(clip) == []
    assert "my_script.py" in clip.info["dropped_reason"]


def test_split_into_several(tmp_path, make_clip):
    s = CustomScript(script(tmp_path, """
        def process(audio, sr):
            half = len(audio) // 2
            return [(audio[:half], sr), (audio[half:], sr)]
    """))
    out = s.apply(make_clip(seconds=2.0))
    assert [c.name for c in out] == ["test_part000", "test_part001"]
    assert all(c.duration == pytest.approx(1.0) for c in out)


def test_accepts_1d_and_librosa_layout(tmp_path, make_clip):
    s1 = CustomScript(script(tmp_path, """
        def process(audio, sr):
            return audio.mean(axis=1), sr          # 一维
    """, "a.py"))
    s2 = CustomScript(script(tmp_path, """
        def process(audio, sr):
            return audio.T, sr                     # (声道, 采样点)
    """, "b.py"))
    assert s1.apply(make_clip(channels=2))[0].audio.shape[1] == 1
    assert s2.apply(make_clip(channels=2))[0].audio.shape[1] == 2


def test_script_can_change_sample_rate(tmp_path, make_clip):
    s = CustomScript(script(tmp_path, """
        def process(audio, sr):
            return audio[::2], sr // 2
    """))
    [out] = s.apply(make_clip(sr=44100))
    assert out.sr == 22050


# ---------- 出错的情况 ----------

def test_missing_process_function(tmp_path):
    with pytest.raises(AttributeError, match="process"):
        CustomScript(script(tmp_path, "x = 1\n"))


def test_syntax_error_is_reported_on_load(tmp_path):
    with pytest.raises(RuntimeError, match="Could not load"):
        CustomScript(script(tmp_path, "def process(audio, sr)\n    return audio, sr\n"))


def test_bad_return_value(tmp_path, make_clip):
    s = CustomScript(script(tmp_path, """
        def process(audio, sr):
            return audio
    """))
    with pytest.raises(TypeError):
        s.apply(make_clip())


def test_template_works(make_clip):
    clip = make_clip()
    [out] = CustomScript(TEMPLATE).apply(clip)
    assert out.audio[0, 0] == 0                           # 模板做了淡入，第一个点是 0


# ---------- 放进 pipeline ----------

def test_pipeline_runs_scripts_in_position(tmp_path):
    raw = tmp_path / "raw"
    write_tone(raw / "a.wav", seconds=2.0)
    # first：记下它看到的采样率（原始的 16000）
    first = script(tmp_path, """
        def process(audio, sr):
            if sr != 16000:
                raise ValueError("first should see the original audio")
            return audio, sr
    """, "first.py")
    # last：在重采样之后，应该看到 8000
    last = script(tmp_path, """
        def process(audio, sr):
            if sr != 8000:
                raise ValueError("last should see the processed audio")
            return audio, sr
    """, "last.py")
    out = run_pipeline(PipelineConfig(
        input_dir=str(raw),
        steps={"resample": {"target_sr": 8000}, "features": {}},
        scripts=[{"path": str(last), "position": "last"},
                 {"path": str(first), "position": "first"}],
    )).output_dir

    rows = read_rows(out / "processing_steps.csv")
    assert [r["name"] for r in rows] == [
        "Read files", "Python script: first.py", "Resampling",
        "Python script: last.py", "Feature Extraction"]
    assert rows[-1]["dropped"] == "0"

    # 脚本复制了一份到输出文件夹；README 里有指纹；config.yaml 记下了脚本
    assert len(list((out / "scripts").glob("*.py"))) == 2
    assert "sha256" in (out / "dataset" / "README.txt").read_text(encoding="utf-8")
    saved = PipelineConfig.load(out / "config.yaml")
    assert {s["position"] for s in saved.scripts} == {"first", "last"}


def test_script_error_drops_file_not_run(tmp_path):
    raw = tmp_path / "raw"
    write_tone(raw / "a.wav")
    write_tone(raw / "b.wav")
    bad = script(tmp_path, """
        def process(audio, sr):
            raise RuntimeError("boom")
    """)
    summary = run_pipeline(PipelineConfig(input_dir=str(raw),
                                          scripts=[{"path": str(bad)}]))
    assert summary.dropped_count == 2
    reasons = [r["reason"] for r in read_rows(summary.output_dir / "dropped.csv")]
    assert all("boom" in r for r in reasons)


def test_missing_script_fails_before_processing(tmp_path):
    raw = tmp_path / "raw"
    write_tone(raw / "a.wav")
    with pytest.raises(FileNotFoundError):
        run_pipeline(PipelineConfig(input_dir=str(raw),
                                    scripts=[{"path": str(tmp_path / "nope.py")}]))
    assert not (tmp_path / "raw_output").exists()          # 一个文件都没开始处理


def test_info_argument(tmp_path):
    raw = tmp_path / "raw"
    write_tone(raw / "alarm" / "a.wav")
    write_tone(raw / "dog" / "d.wav")
    seen = tmp_path / "seen.txt"
    s = script(tmp_path, f"""
        def process(audio, sr, info):
            with open({str(seen)!r}, "a") as f:
                f.write(info["file"] + "|" + info["folder"] + "|" + info["name"] + "\\n")
            return None if info["folder"] == "alarm" else (audio, sr)
    """)
    summary = run_pipeline(PipelineConfig(input_dir=str(raw), scripts=[{"path": str(s)}]))
    assert summary.output_count == 1
    assert sorted(seen.read_text().splitlines()) == ["alarm/a.wav|alarm|a", "dog/d.wav|dog|d"]


def test_two_arg_scripts_still_work(make_clip):
    assert CustomScript(TEMPLATE).wants_info is False
