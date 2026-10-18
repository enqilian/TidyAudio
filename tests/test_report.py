"""测试质量报告。"""

import numpy as np
import soundfile as sf

from tidyaudio.__main__ import main
from tidyaudio.pipeline import PipelineConfig, run_pipeline


def write_tone(path, seconds, sr=44100, channels=2, amplitude=0.3):
    t = np.arange(int(seconds * sr)) / sr
    wave = np.clip(amplitude * np.sin(2 * np.pi * 440 * t), -1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.stack([wave] * channels, axis=1), sr)


def make_dataset(tmp_path):
    raw = tmp_path / "raw"
    write_tone(raw / "dog" / "a.wav", 3, sr=44100)
    write_tone(raw / "dog" / "b.wav", 2, sr=48000, channels=1)
    write_tone(raw / "cat" / "loud.wav", 2, amplitude=3.0)      # 严重削波
    write_tone(raw / "cat" / "tiny.wav", 0.1)                    # 太短
    (raw / "file_list.csv").write_text(
        "file,class\ndog/a.wav,dog\ndog/b.wav,dog\ncat/loud.wav,cat\ncat/tiny.wav,cat\n",
        encoding="utf-8")
    return raw


def test_report_is_generated(tmp_path):
    raw = make_dataset(tmp_path)
    out = run_pipeline(PipelineConfig(input_dir=str(raw), steps={
        "clipping_check": {},
        "resample": {"target_sr": 16000},
        "duration_filter": {"min_sec": 0.5},
        "normalize": {"method": "peak"},
    })).output_dir

    page = (out / "report.html").read_text(encoding="utf-8")
    assert "<h2>Overview</h2>" in page
    assert "44100 Hz" in page and "48000 Hz" in page           # 原始采样率不统一
    assert "Mono" in page and "Stereo" in page
    assert "Clipping <span class='badge warn'>1 / 3 source files" in page
    assert "cat/loud.wav" in page
    assert "Classes <span class='muted'>by class" in page
    assert "Too short" in page                                 # 丢掉原因
    assert "data:image/png;base64," in page                    # 图嵌在网页里
    assert "Sample rates or channel counts differ" in page
    assert "Normalization target not reached <span class='badge ok'>0 / 3" in page


def test_report_finds_silent_segments(tmp_path):
    raw = tmp_path / "raw"
    sr = 16000
    sound = 0.3 * np.sin(2 * np.pi * 440 * np.arange(2 * sr) / sr)
    # 前 2 秒有声音，后 4 秒是静音 → 2 秒一段切成 3 段，后 2 段几乎没声音
    sf.write(raw.mkdir() or raw / "rec.wav", np.concatenate([sound, np.zeros(4 * sr)]), sr)
    out = run_pipeline(PipelineConfig(input_dir=str(raw), steps={
        "segment": {"segment_sec": 2.0}})).output_dir
    page = (out / "report.html").read_text(encoding="utf-8")
    assert "Nearly silent output (RMS below -60 dBFS) <span class='badge warn'>2 / 3" in page
    assert "All files share the same sample rate" in page


def test_manifest_has_original_and_level_columns(tmp_path):
    raw = make_dataset(tmp_path)
    out = run_pipeline(PipelineConfig(input_dir=str(raw),
                                      steps={"resample": {"target_sr": 16000}})).output_dir
    header = (out / "manifest.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    for col in ("orig_sample_rate", "orig_channels", "orig_duration_sec", "peak_dbfs", "rms_dbfs"):
        assert col in header


def test_report_without_checks(tmp_path):
    raw = make_dataset(tmp_path)
    out = run_pipeline(PipelineConfig(input_dir=str(raw))).output_dir
    page = (out / "report.html").read_text(encoding="utf-8")
    assert "Clipping Detection was not enabled" in page


def test_report_command_regenerates(tmp_path, capsys):
    raw = make_dataset(tmp_path)
    out = run_pipeline(PipelineConfig(input_dir=str(raw))).output_dir
    (out / "report.html").unlink()
    assert main(["report", str(out)]) == 0
    assert (out / "report.html").exists()
