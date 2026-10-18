"""测试数据集 CSV 的读取和更新。测试用的 CSV 仿照 NIGENS 的 file_list.csv 格式。"""

import csv

import numpy as np
import pytest
import soundfile as sf

from tidyaudio.dataset_csv import find_dataset_csv, read_dataset_csv
from tidyaudio.pipeline import PipelineConfig, run_pipeline

ORIGINAL_CSV = """file,class,duration_s
alarm/a1.wav,alarm,3.0
alarm/a2.wav,alarm,0.3
dog/d1.wav,dog,4.0
dog/missing.wav,dog,2.0

class,count,total_duration_s
alarm,2,3.3
dog,2,6.0
TOTAL,4,9.3

note: all files are 44100 Hz stereo (mono files upmixed L=R)
"""


def write_tone(path, seconds, sr=44100):
    t = np.arange(int(seconds * sr)) / sr
    wave = 0.3 * np.sin(2 * np.pi * 440 * t)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.stack([wave, wave], axis=1), sr)


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def dataset(tmp_path):
    raw = tmp_path / "NIGENS"
    write_tone(raw / "alarm" / "a1.wav", 3.0)
    write_tone(raw / "alarm" / "a2.wav", 0.3)
    write_tone(raw / "dog" / "d1.wav", 4.0)
    write_tone(raw / "dog" / "extra.wav", 1.0)       # CSV 里没有这个文件
    (raw / "file_list.csv").write_text(ORIGINAL_CSV, encoding="utf-8")
    return raw


# ---------- 读取 ----------

def test_reads_only_first_table(dataset):
    table = read_dataset_csv(dataset / "file_list.csv")
    assert table.fields == ["file", "class", "duration_s"]
    assert len(table.rows) == 4                          # 汇总表和 note 没混进来
    assert table.footer[0] == "class,count,total_duration_s"
    assert table.footer[-1].startswith("note:")


def test_find_dataset_csv(tmp_path):
    (tmp_path / "file_list.csv").touch()
    assert find_dataset_csv(tmp_path).name == "file_list.csv"
    (tmp_path / "other.csv").touch()
    assert find_dataset_csv(tmp_path).name == "file_list.csv"   # 名字里带 list 的优先
    (tmp_path / "labels.csv").touch()
    assert find_dataset_csv(tmp_path) is None                   # 两个都像，分不出来


# ---------- 更新 ----------

def test_csv_is_updated(dataset):
    config = PipelineConfig(input_dir=str(dataset), steps={
        "resample": {"target_sr": 16000},
        "channels": {"mode": "mono"},
        "duration_filter": {"min_sec": 0.5},
    })
    out = run_pipeline(config).output_dir / "dataset"
    rows = read_rows(out / "file_list.csv")
    by_file = {r["file"]: r for r in rows}

    # 原来的列在前，新加的采样率、声道在后
    assert list(rows[0]) == ["file", "class", "duration_s", "sample_rate", "channels"]
    # 标注保留，采样率和声道是处理后的值
    assert by_file["alarm/a1.wav"]["class"] == "alarm"
    assert by_file["alarm/a1.wav"]["sample_rate"] == "16000"
    assert by_file["alarm/a1.wav"]["channels"] == "1"
    # a2 太短被丢掉 → 这一行删除；missing.wav 找不到音频 → 也不在新 CSV 里
    assert "alarm/a2.wav" not in by_file
    assert "dog/missing.wav" not in by_file
    # extra.wav 原 CSV 里没有 → 加一行，类别留空
    assert by_file["dog/extra.wav"]["class"] == ""


def test_duration_column_is_changed(dataset):
    config = PipelineConfig(input_dir=str(dataset),
                            steps={"segment": {"segment_sec": 1.5, "last": "drop"}})
    out = run_pipeline(config).output_dir / "dataset"
    rows = read_rows(out / "file_list.csv")

    # 切片：1 行变多行，类别复制到每一行，时长改成片段的时长
    a1 = [r for r in rows if r["source_file"] == "alarm/a1.wav"]
    assert [r["file"] for r in a1] == ["alarm/a1_seg000.wav", "alarm/a1_seg001.wav"]
    assert all(r["class"] == "alarm" and r["duration_s"] == "1.5" for r in a1)
    assert a1[1]["segment_start_s"] == "1.5"
    assert list(rows[0])[-3:] == ["source_file", "segment_start_s", "segment_end_s"]


def test_summary_and_readme(dataset):
    out = run_pipeline(PipelineConfig(
        input_dir=str(dataset), steps={"duration_filter": {"min_sec": 0.5}})).output_dir / "dataset"

    summary = {r["class"]: r for r in read_rows(out / "summary.csv")}
    assert summary["alarm"]["count"] == "1"              # a2 被丢掉了，重新统计
    assert summary["dog"]["count"] == "1"
    assert summary["TOTAL"]["count"] == "3"              # 包括类别为空的 extra.wav

    readme = (out / "README.txt").read_text(encoding="utf-8")
    assert "Duration Filtering" in readme
    assert "1 rows in the original CSV had no matching audio" in readme   # missing.wav
    assert "1 audio files were not listed" in readme                  # extra.wav
    assert "note: all files are 44100 Hz stereo" in readme   # 原备注抄进来供参考


def test_filename_only_csv_keeps_style(tmp_path):
    # CSV 里只写文件名、不写子文件夹
    raw = tmp_path / "raw"
    write_tone(raw / "x" / "song.wav", 3.0)
    (raw / "meta.csv").write_text("filename,label\nsong.wav,music\n", encoding="utf-8")
    out = run_pipeline(PipelineConfig(
        input_dir=str(raw), steps={"segment": {"segment_sec": 1.0, "last": "drop"}})).output_dir
    rows = read_rows(out / "dataset" / "meta.csv")
    assert rows[0]["filename"] == "song_seg000.wav"     # 也只写文件名
    assert rows[0]["label"] == "music"


def test_unrelated_csv_is_left_alone(tmp_path):
    raw = tmp_path / "raw"
    write_tone(raw / "a.wav", 1.0)
    (raw / "scores.csv").write_text("name,score\nbob,3\n", encoding="utf-8")
    out = run_pipeline(PipelineConfig(input_dir=str(raw))).output_dir / "dataset"
    assert not (out / "scores.csv").exists()
    assert "matches the audio file names" in (out / "README.txt").read_text(encoding="utf-8")


def test_csv_can_be_turned_off(dataset):
    out = run_pipeline(PipelineConfig(input_dir=str(dataset), dataset_csv=None)).output_dir
    assert not (out / "dataset" / "file_list.csv").exists()
