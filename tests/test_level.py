"""测试第 3 组：归一化（peak / rms / lufs）、直流偏移。"""

import numpy as np
import pyloudnorm as pyln
import pytest

from tidyaudio.steps import REGISTRY
from tidyaudio.steps.level import rms


def lufs(clip):
    return pyln.Meter(clip.sr).integrated_loudness(clip.audio)


def db(x):
    return 20 * np.log10(x)


# ---------- 归一化：三种方式 ----------

def test_peak_method(make_clip):
    clip = make_clip(amplitude=0.1)
    REGISTRY["normalize"](method="peak", target_peak_db=-1.0).apply(clip)
    assert db(np.abs(clip.audio).max()) == pytest.approx(-1.0, abs=0.01)


def test_rms_method(make_clip):
    clip = make_clip(amplitude=0.05)
    REGISTRY["normalize"](method="rms", target_rms_db=-20.0).apply(clip)
    assert db(rms(clip.audio)) == pytest.approx(-20.0, abs=0.01)


def test_lufs_method(make_clip):
    clip = make_clip(seconds=3.0, amplitude=0.05, sr=16000, channels=1)
    REGISTRY["normalize"](method="lufs", target_lufs=-23.0).apply(clip)
    assert lufs(clip) == pytest.approx(-23.0, abs=0.1)
    assert "loudness_before_lufs" in clip.info


def test_lufs_quiet_and_loud_end_up_equal(make_clip):
    quiet = make_clip(seconds=3.0, amplitude=0.02, sr=16000, channels=1)
    loud = make_clip(seconds=3.0, amplitude=0.3, sr=16000, channels=1)
    step = REGISTRY["normalize"](method="lufs", target_lufs=-26.0)
    step.apply(quiet)
    step.apply(loud)
    assert lufs(quiet) == pytest.approx(lufs(loud), abs=0.1)


@pytest.mark.parametrize("method,target", [("rms", {"target_rms_db": -3.0}),
                                           ("lufs", {"target_lufs": -3.0})])
def test_peak_ceiling_prevents_clipping(make_clip, method, target):
    # 目标值很响，正弦波调上去会超过 -1 dBFS，应该被限制住
    clip = make_clip(seconds=3.0, amplitude=0.1, sr=16000, channels=1)
    REGISTRY["normalize"](method=method, peak_ceiling_db=-1.0, **target).apply(clip)
    assert np.abs(clip.audio).max() <= 10 ** (-1 / 20) + 1e-5
    assert clip.info["normalize_limited"] is True


def test_lufs_skips_short_clip(make_clip):
    clip = make_clip(seconds=0.2, sr=16000)
    before = clip.audio.copy()
    REGISTRY["normalize"](method="lufs").apply(clip)
    assert "normalize_skipped" in clip.info
    assert np.array_equal(clip.audio, before)


def test_normalize_leaves_silence_alone(make_clip):
    clip = make_clip(amplitude=0.0)
    REGISTRY["normalize"](method="peak").apply(clip)
    assert np.abs(clip.audio).max() == 0


# ---------- 直流偏移（高级选项） ----------

def test_dc_offset_is_advanced():
    assert REGISTRY["dc_offset"].advanced is True


def test_dc_check_mode_only_records(make_clip):
    clip = make_clip(amplitude=0.3)
    clip.audio += 0.2
    before = clip.audio.copy()
    REGISTRY["dc_offset"]().apply(clip)                 # 默认是 check
    assert clip.info["dc_offset"] == pytest.approx(0.2, abs=1e-3)
    assert np.array_equal(clip.audio, before)           # 音频没被改


def test_dc_remove_mode(make_clip):
    clip = make_clip(amplitude=0.3)
    clip.audio += 0.2
    REGISTRY["dc_offset"](mode="remove").apply(clip)
    assert np.abs(clip.audio.mean(axis=0)).max() < 1e-6


def test_dc_threshold_mode(make_clip):
    small = make_clip(amplitude=0.3)
    small.audio += 0.005                                # 低于阈值 0.01，不动
    big = make_clip(amplitude=0.3)
    big.audio += 0.2                                    # 高于阈值，去除
    step = REGISTRY["dc_offset"](mode="threshold", threshold=0.01)
    step.apply(small)
    step.apply(big)
    assert "dc_offset_removed" not in small.info
    assert big.info["dc_offset_removed"] is True
