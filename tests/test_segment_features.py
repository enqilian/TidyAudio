"""测试第 5 组：切片、特征提取。"""

import numpy as np
import pytest

from tidyaudio.audio_io import Clip
from tidyaudio.steps import REGISTRY

SR = 16000


def tone(seconds, freq=440.0, amplitude=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


def clip_of(*parts, name="song"):
    return Clip(audio=np.concatenate(parts)[:, None], sr=SR, name=name)


def durations(segments):
    return [round(s.duration, 2) for s in segments]


# ---------- 切片：固定长度 ----------

def test_fixed_pad_last():
    segs = REGISTRY["segment"](segment_sec=5, last="pad").apply(clip_of(tone(12)))
    assert durations(segs) == [5, 5, 5]                 # 最后 2 秒补成 5 秒
    assert segs[-1].info["segment_padded_sec"] == pytest.approx(3.0)


def test_fixed_keep_last():
    segs = REGISTRY["segment"](segment_sec=5, last="keep").apply(clip_of(tone(12)))
    assert durations(segs) == [5, 5, 2]


def test_fixed_drop_last():
    segs = REGISTRY["segment"](segment_sec=5, last="drop").apply(clip_of(tone(12)))
    assert durations(segs) == [5, 5]


@pytest.mark.parametrize("last,expected", [("pad", [5]), ("keep", [0.8]), ("drop", [])])
def test_clip_shorter_than_one_segment(last, expected):
    # 整条只有 0.8 秒，比一段还短：它本身就是"最后一段"，规则一样
    segs = REGISTRY["segment"](segment_sec=5, last=last).apply(clip_of(tone(0.8)))
    assert durations(segs) == expected


def test_fixed_with_overlap():
    segs = REGISTRY["segment"](segment_sec=5, overlap_sec=1, last="drop").apply(
        clip_of(tone(13)))
    starts = [s.info["segment_start_sec"] for s in segs]
    assert starts == [0, 4, 8]                          # 0-5, 4-9, 8-13


def test_overlap_does_not_add_redundant_tail():
    # 9 秒、5 秒一段、重叠 1 秒：0-5, 4-9 已经覆盖到结尾，不应该再多出一段
    segs = REGISTRY["segment"](segment_sec=5, overlap_sec=1, last="keep").apply(
        clip_of(tone(9)))
    assert len(segs) == 2


def test_overlap_must_be_shorter_than_segment():
    with pytest.raises(ValueError):
        REGISTRY["segment"](segment_sec=2, overlap_sec=2).apply(clip_of(tone(5)))


def test_segment_names_and_info():
    clip = clip_of(tone(10), name="song")
    clip.info["clipping_ratio_pct"] = 0.5               # 之前的检查结果
    segs = REGISTRY["segment"](segment_sec=5).apply(clip)
    assert [s.name for s in segs] == ["song_seg000", "song_seg001"]
    assert segs[1].info["segment_start_sec"] == 5.0
    assert segs[1].info["clipping_ratio_pct"] == 0.5    # 检查结果会继承下来
    segs[0].info["x"] = 1
    assert "x" not in segs[1].info                      # 每段的 info 互相独立


# ---------- 切片：按停顿 ----------

def test_silence_mode_splits_at_pauses():
    # 两句话，中间停 1 秒
    clip = clip_of(tone(2), silence(1), tone(3))
    segs = REGISTRY["segment"](mode="silence", min_silence_ms=500).apply(clip)
    assert len(segs) == 2
    assert durations(segs) == pytest.approx([2.1, 3.1], abs=0.1)   # 各含 0.1 秒边缘


def test_silence_mode_respects_max_sec():
    clip = clip_of(tone(7))                             # 一句 7 秒的长句
    segs = REGISTRY["segment"](mode="silence", max_sec=3).apply(clip)
    assert all(s.duration <= 3.0 + 1e-6 for s in segs)
    assert len(segs) == 3


def test_all_silent_is_dropped():
    clip = clip_of(silence(5))
    assert REGISTRY["segment"](mode="silence").apply(clip) == []


# ---------- 特征提取 ----------

def test_features_is_advanced():
    assert REGISTRY["features"].advanced is True


def test_default_features():
    clip = clip_of(tone(2, freq=1000))
    REGISTRY["features"]().apply(clip)
    assert {"rms", "spectral_centroid", "zcr"} <= set(clip.features)
    assert "mfcc" not in clip.features                  # 默认不开
    # 1000 Hz 正弦波的频谱质心应该在 1000 Hz 附近
    assert clip.info["spectral_centroid_mean"] == pytest.approx(1000, rel=0.05)


def test_feature_shapes():
    clip = clip_of(tone(2))                             # 2 秒，10ms 帧移 → 约 201 帧
    REGISTRY["features"](log_mel=True, n_mels=64, mfcc=True, n_mfcc=20).apply(clip)
    assert clip.features["log_mel"].shape == (64, 201)
    assert clip.features["mfcc"].shape == (20, 201)
    assert clip.features["log_mel"].dtype == np.float32


def test_features_do_not_change_audio(make_clip):
    clip = make_clip(channels=2, sr=SR)
    before = clip.audio.copy()
    REGISTRY["features"](log_mel=True, mfcc=True).apply(clip)
    assert np.array_equal(clip.audio, before)
