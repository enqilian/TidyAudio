"""测试第 2 组：文件校验、时长过滤、削波检测。"""

import numpy as np

from tidyaudio.steps import REGISTRY


# ---------- 文件校验 ----------

def test_validation_keeps_normal_audio(make_clip):
    assert len(REGISTRY["validate"]().apply(make_clip())) == 1


def test_validation_drops_nan(make_clip):
    clip = make_clip()
    clip.audio[100, 0] = np.nan
    assert REGISTRY["validate"]().apply(clip) == []
    assert "NaN" in clip.info["dropped_reason"]


def test_validation_drops_silent(make_clip):
    clip = make_clip(amplitude=0.0001)       # 约 -80 dBFS
    assert REGISTRY["validate"]().apply(clip) == []
    assert "Near-silent" in clip.info["dropped_reason"]


def test_validation_can_keep_silent(make_clip):
    clip = make_clip(amplitude=0.0001)
    assert len(REGISTRY["validate"](drop_silent=False).apply(clip)) == 1


# ---------- 时长过滤 ----------

def test_duration_filter(make_clip):
    step = REGISTRY["duration_filter"](min_sec=1.0, max_sec=5.0)
    assert step.apply(make_clip(seconds=0.5)) == []      # 太短
    assert len(step.apply(make_clip(seconds=2.0))) == 1  # 正好
    assert step.apply(make_clip(seconds=6.0)) == []      # 太长


def test_duration_filter_no_max(make_clip):
    step = REGISTRY["duration_filter"](min_sec=0.0, max_sec=0.0)
    assert len(step.apply(make_clip(seconds=60.0, sr=8000))) == 1


# ---------- 削波检测 ----------

def test_clean_audio_has_no_clipping(make_clip):
    clip = make_clip(amplitude=0.5)
    REGISTRY["clipping_check"]().apply(clip)
    assert clip.info["clipping_ratio_pct"] == 0
    assert clip.info["longest_clip_run"] == 0


def test_clipping_is_detected_and_marked(make_clip):
    clip = make_clip(amplitude=2.0)
    clip.audio = np.clip(clip.audio, -1, 1)   # 模拟被"削平"的爆音
    out = REGISTRY["clipping_check"](action="mark").apply(clip)
    assert len(out) == 1                       # mark 模式不丢
    assert clip.info["clipping_ratio_pct"] > 10
    assert clip.info["longest_clip_run"] > 10


def test_clipping_drop_mode(make_clip):
    clip = make_clip(amplitude=2.0)
    clip.audio = np.clip(clip.audio, -1, 1)
    assert REGISTRY["clipping_check"](action="drop", max_ratio=1.0).apply(clip) == []
    assert "Heavy clipping" in clip.info["dropped_reason"]
