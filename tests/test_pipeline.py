"""测试第四步：整个 pipeline 从输入文件夹到输出文件夹。"""

import csv

import numpy as np
import pytest
import soundfile as sf

from tidyaudio.__main__ import main
from tidyaudio.pipeline import (
    STEP_ORDER, PipelineConfig, default_output_dir, run_pipeline)
from tidyaudio.steps import REGISTRY


def write_tone(path, seconds, sr=44100, channels=2, amplitude=0.3):
    t = np.arange(int(seconds * sr)) / sr
    wave = amplitude * np.sin(2 * np.pi * 440 * t)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.stack([wave] * channels, axis=1), sr)


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def dataset(tmp_path):
    """造一个小数据集：
    raw/
      dog/bark.wav    3 秒，正常
      cat/meow.wav    3 秒，正常
      cat/meow.flac   3 秒，和上面同名不同格式
      short.wav       0.2 秒，太短
      broken.wav      不是真的音频，打不开
    """
    raw = tmp_path / "raw"
    write_tone(raw / "dog" / "bark.wav", 3)
    write_tone(raw / "cat" / "meow.wav", 3)
    write_tone(raw / "cat" / "meow.flac", 3)
    write_tone(raw / "short.wav", 0.2)
    (raw / "broken.wav").write_bytes(b"this is not audio")
    return raw


def test_every_step_has_a_place_in_the_order():
    assert set(STEP_ORDER) == set(REGISTRY)


def test_full_run(dataset):
    config = PipelineConfig(
        input_dir=str(dataset),
        steps={
            "normalize": {"method": "peak"},
            "resample": {"target_sr": 16000},     # 故意写在后面，执行时会排到前面
            "channels": {"mode": "mono"},
            "duration_filter": {"min_sec": 1.0},
        },
    )
    summary = run_pipeline(config)
    out = summary.output_dir

    # 默认输出在输入的同级，名字加 _output
    assert out == dataset.parent / "raw_output"
    assert summary.input_count == 5
    assert summary.output_count == 3          # bark、meow.wav、meow.flac
    assert summary.dropped_count == 2         # short、broken

    # 子文件夹结构保留；同名文件自动改名，不互相覆盖
    files = sorted(p.relative_to(out / "dataset").as_posix() for p in (out / "dataset").rglob("*.wav"))
    assert files == ["cat/meow.wav", "cat/meow_2.wav", "dog/bark.wav"]

    info = sf.info(out / "dataset" / "dog" / "bark.wav")
    assert info.samplerate == 16000 and info.channels == 1

    manifest = read_csv(out / "manifest.csv")
    assert len(manifest) == 3
    assert {"path", "source", "duration_sec", "sample_rate", "normalize_gain_db"} <= set(manifest[0])

    dropped = {row["source"]: row["reason"] for row in read_csv(out / "dropped.csv")}
    assert "Could not read" in dropped["broken.wav"]
    assert "Too short" in dropped["short.wav"]


def test_config_yaml_can_rerun(dataset):
    config = PipelineConfig(input_dir=str(dataset), steps={"resample": {"target_sr": 8000}})
    first = run_pipeline(config)

    # 读回 config.yaml：设置一样，而且没写的默认值也被记录下来了
    loaded = PipelineConfig.load(first.output_dir / "config.yaml")
    assert loaded.steps["resample"]["target_sr"] == 8000

    config2 = PipelineConfig(input_dir=str(dataset), steps={"normalize": {}})
    saved = PipelineConfig.load(run_pipeline(config2).output_dir / "config.yaml")
    assert saved.steps["normalize"]["method"] == "lufs"
    assert saved.steps["normalize"]["prevent_clipping"] is True

    # 再跑一次：上次的输出不会被覆盖，自动用 raw_output_2
    second = run_pipeline(loaded)
    assert second.output_dir.name == "raw_output_3"
    assert (first.output_dir / "manifest.csv").exists()


def test_segment_and_features_output(dataset):
    config = PipelineConfig(
        input_dir=str(dataset),
        steps={"segment": {"segment_sec": 1.0, "last": "drop"}, "features": {"mfcc": True}},
    )
    out = run_pipeline(config).output_dir
    assert (out / "dataset" / "dog" / "bark_seg002.wav").exists()

    feats = np.load(out / "features" / "dog" / "bark_seg000.npz")
    assert {"rms", "mfcc"} <= set(feats.files)


def test_output_inside_input_is_rejected(dataset):
    config = PipelineConfig(input_dir=str(dataset), output_dir=str(dataset / "out"))
    with pytest.raises(ValueError):
        run_pipeline(config)


def test_unknown_step_is_rejected(dataset):
    with pytest.raises(ValueError):
        run_pipeline(PipelineConfig(input_dir=str(dataset), steps={"resampel": {}}))


def test_cancel(dataset):
    summary = run_pipeline(PipelineConfig(input_dir=str(dataset)), should_stop=lambda: True)
    assert summary.cancelled
    assert summary.output_count == 0


def test_progress_is_reported(dataset):
    calls = []
    run_pipeline(PipelineConfig(input_dir=str(dataset)), progress=lambda d, t, n: calls.append((d, t)))
    assert calls[-1] == (5, 5)


def test_default_output_dir_skips_empty_existing(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw_output").mkdir()          # 已存在但是空的，可以直接用
    assert default_output_dir(tmp_path / "raw").name == "raw_output"


def test_cli_rerun_with_new_input(dataset, tmp_path, capsys):
    out = run_pipeline(PipelineConfig(input_dir=str(dataset),
                                      steps={"resample": {"target_sr": 8000}})).output_dir
    new_raw = tmp_path / "new_batch"
    write_tone(new_raw / "x.wav", 2)

    # 换一批数据，用上次的设置再跑一遍
    assert main(["run", str(out / "config.yaml"), "-i", str(new_raw)]) == 0
    assert sf.info(tmp_path / "new_batch_output" / "dataset" / "x.wav").samplerate == 8000

    assert main(["scan", str(dataset)]) == 0
    assert "5 audio files" in capsys.readouterr().out


def test_processing_steps_csv(dataset):
    config = PipelineConfig(input_dir=str(dataset), steps={
        "normalize": {"method": "lufs"},
        "resample": {"target_sr": 16000},
        "duration_filter": {"min_sec": 1.0},
        "segment": {"segment_sec": 1.0, "last": "drop"},
    })
    out = run_pipeline(config).output_dir
    rows = read_csv(out / "processing_steps.csv")

    # 第 0 行是读取文件：5 个进来，broken.wav 打不开
    load, rows = rows[0], rows[1:]
    assert (load["name"], load["input"], load["output"], load["dropped"]) == (
        "Read files", "5", "4", "1")

    # 按实际执行顺序列出，不是按填写顺序
    assert [r["name"] for r in rows] == [
        "Resampling", "Duration Filtering", "Normalization", "Audio Segmentation"]
    assert rows[0]["parameters"] == "Target sample rate (Hz) = 16000"

    # 选了 lufs，就只列响度相关的参数，不列"目标峰值""目标 RMS"
    assert "Target loudness" in rows[2]["parameters"]
    assert "Target peak" not in rows[2]["parameters"]

    # 4 个能打开的文件进来，short.wav 被时长过滤丢掉；3 个 3 秒的文件各切成 3 段
    duration = rows[1]
    assert (duration["input"], duration["output"], duration["dropped"]) == ("4", "3", "1")
    assert rows[3]["output"] == "9"


# ---------- 自定义执行顺序 ----------

def test_custom_order_is_used(dataset):
    from tidyaudio.pipeline import order_warnings
    config = PipelineConfig(input_dir=str(dataset),
                            steps={"resample": {"target_sr": 8000}, "normalize": {"method": "peak"},
                                   "clipping_check": {}},
                            order=["normalize", "resample", "clipping_check"])
    assert [s.key for s in config.build_steps()] == ["normalize", "resample", "clipping_check"]
    out = run_pipeline(config).output_dir
    rows = read_csv(out / "processing_steps.csv")
    assert [r["name"] for r in rows[1:]] == ["Normalization", "Resampling", "Clipping Detection"]
    assert PipelineConfig.load(out / "config.yaml").order == ["normalize", "resample",
                                                               "clipping_check"]
    assert order_warnings(["normalize", "clipping_check"])


def test_merge_order():
    from tidyaudio.pipeline import merge_order
    default = ["validate", "resample", "normalize", "features"]
    assert merge_order(default, ["normalize", "resample"]) == \
        ["validate", "normalize", "resample", "features"]
    assert merge_order(["resample"], ["normalize", "resample"]) == ["resample"]   # 关掉的去掉


def test_unknown_key_in_order(dataset):
    with pytest.raises(ValueError, match="Unknown steps in order"):
        PipelineConfig(input_dir=str(dataset), steps={"resample": {}},
                       order=["resampel"]).build_steps()


def test_check_script_source():
    from tidyaudio.custom_script import check_script_source
    check_script_source("def process(audio, sr):\n    return audio, sr\n")
    with pytest.raises(SyntaxError, match="Line 1"):
        check_script_source("def process(audio, sr)\n")
    with pytest.raises(ValueError, match="process"):
        check_script_source("x = 1\n")
