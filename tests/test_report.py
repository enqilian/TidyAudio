"""测试质量报告。"""

import csv

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


def report(out):
    return (out / "report.html").read_text(encoding="utf-8")


def test_common_sections(tmp_path):
    raw = make_dataset(tmp_path)
    out = run_pipeline(PipelineConfig(input_dir=str(raw), steps={
        "resample": {"target_sr": 16000},
        "channels": {"mode": "mono"},
        "duration_filter": {"min_sec": 0.5},
    })).output_dir
    page = report(out)
    for title in ("Summary", "Overview", "Format", "Duration", "Classes", "Dropped files",
                  "Workflow"):
        assert f"<h2>{title}" in page
    # Format：处理前后对比，处理前包括后来被丢掉的 tiny.wav
    assert "44,100 Hz" in page and "48,000 Hz" in page and "16,000 Hz" in page
    assert "Mixed sample rates were unified to 16,000 Hz." in page
    assert "Mixed channel counts were unified to Mono." in page
    # 丢掉的文件写明是哪一步丢的
    assert "<td>Duration Filtering</td><td>Too short</td>" in page
    assert "data:image/png;base64," in page                    # 图嵌在网页里


def test_detail_sections_only_for_enabled_steps(tmp_path):
    raw = make_dataset(tmp_path)
    page = report(run_pipeline(PipelineConfig(input_dir=str(raw))).output_dir)
    assert "Processing details" not in page
    assert "<h2>Clipping Detection" not in page

    page = report(run_pipeline(PipelineConfig(input_dir=str(raw), steps={
        "clipping_check": {}, "resample": {"target_sr": 16000}})).output_dir)
    assert "Processing details" in page
    assert "<h2>Clipping Detection" in page
    assert "<h2>Resampling" not in page      # 重采样的结果在 Format 里，不单独成段


def test_clipping_levels(tmp_path):
    raw = make_dataset(tmp_path)
    page = report(run_pipeline(PipelineConfig(input_dir=str(raw), steps={
        "clipping_check": {}})).output_dir)
    assert "<td>cat/loud.wav</td><td>Severe</td>" in page
    assert "1 file is severely clipped (1 still in the dataset)." in page
    assert "thing worth checking" in page or "things worth checking" in page


def test_silent_and_padded_segments(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    sr = 16000
    sound = 0.3 * np.sin(2 * np.pi * 440 * np.arange(2 * sr) / sr)
    # 前 2 秒有声音，后 4 秒是静音 → 2 秒一段切成 3 段，后 2 段是数字静音
    sf.write(raw / "rec.wav", np.concatenate([sound, np.zeros(4 * sr)]), sr)
    # 2.5 秒 → 第二段只有 0.5 秒声音，补了 1.5 秒静音
    sf.write(raw / "short.wav", np.concatenate([sound, sound[: sr // 2]]), sr)
    out = run_pipeline(PipelineConfig(input_dir=str(raw), steps={
        "segment": {"segment_sec": 2.0}})).output_dir
    page = report(out)
    assert "2 / 5 output files" in page
    assert "<td>silent</td>" in page                           # 不再显示 -300 dBFS 这种数字
    assert "Mostly padding" in page and "short_seg001.wav" in page
    assert "Why the duration changed" in page and "of silence added" in page
    # 可以直接在报告里播放（相对路径，report.html 和 dataset/ 在同一个文件夹）
    assert "class='play' data-src='dataset/short_seg001.wav'" in page


def test_normalization_below_target(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    sr = 16000
    clicks = np.zeros(3 * sr)
    clicks[::sr // 2] = 0.9                                     # 很尖的峰、很小的平均响度
    sf.write(raw / "knock.wav", clicks, sr)
    sf.write(raw / "tone.wav", 0.05 * np.sin(2 * np.pi * 440 * np.arange(3 * sr) / sr), sr)
    out = run_pipeline(PipelineConfig(input_dir=str(raw), steps={
        "normalize": {"method": "lufs", "target_lufs": -14.0}})).output_dir
    page = report(out)
    assert "<h3>Below target</h3>" in page
    assert "<td>knock.wav</td>" in page
    assert "below the normalization target" in page
    rows = list(csv.DictReader((out / "manifest.csv").open(encoding="utf-8-sig")))
    tone = next(r for r in rows if r["source"] == "tone.wav")
    assert abs(float(tone["normalize_after"]) + 14.0) < 0.01


def test_script_path_not_in_report(tmp_path):
    raw = make_dataset(tmp_path)
    script = tmp_path / "private_folder" / "my_script.py"
    script.parent.mkdir()
    script.write_text("def process(audio, sr):\n    return audio, sr\n", encoding="utf-8")
    page = report(run_pipeline(PipelineConfig(input_dir=str(raw),
                                              scripts=[{"path": str(script)}])).output_dir)
    assert "<h2>Python scripts" in page
    assert "my_script.py" in page
    assert "private_folder" not in page


def test_manifest_has_original_and_level_columns(tmp_path):
    raw = make_dataset(tmp_path)
    out = run_pipeline(PipelineConfig(input_dir=str(raw),
                                      steps={"resample": {"target_sr": 16000}})).output_dir
    header = (out / "manifest.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    for col in ("orig_sample_rate", "orig_channels", "orig_duration_sec", "orig_subtype",
                "subtype", "peak_dbfs", "rms_dbfs"):
        assert col in header


def test_report_command_regenerates(tmp_path):
    raw = make_dataset(tmp_path)
    out = run_pipeline(PipelineConfig(input_dir=str(raw))).output_dir
    (out / "report.html").unlink()
    assert main(["report", str(out)]) == 0
    assert (out / "report.html").exists()


def test_report_for_older_output(tmp_path):
    """旧版本的输出文件夹（没有新加的列）也能重新生成报告。"""
    out = tmp_path / "old_output"
    out.mkdir()
    (out / "manifest.csv").write_text(
        "path,source,duration_sec,sample_rate,channels,rms_dbfs,orig_sample_rate,orig_channels,"
        "orig_duration_sec\ndataset/a.wav,a.wav,1.0,16000,1,-20,44100,2,1.0\n", encoding="utf-8")
    (out / "dropped.csv").write_text("source,segment,reason\nb.wav,,Too short (0.1s < 0.5s)\n",
                                     encoding="utf-8")
    (out / "processing_steps.csv").write_text(
        "step,name,parameters,input,output,dropped\n0,Read files,,2,2,0\n"
        "1,Duration Filtering,Minimum (s) = 0.5,2,1,1\n", encoding="utf-8")
    assert main(["report", str(out)]) == 0
    assert "<h2>Duration Filtering" in report(out)


def test_everything_dropped(tmp_path):
    raw = tmp_path / "raw"
    write_tone(raw / "tiny.wav", 0.1)
    page = report(run_pipeline(PipelineConfig(input_dir=str(raw), steps={
        "duration_filter": {"min_sec": 0.5}, "normalize": {}})).output_dir)
    assert "every file was dropped" in page
    assert "No problems found" not in page
    assert "No files reached this step" in page                # Normalization 什么都没处理
    assert "Every file was normalized" not in page
