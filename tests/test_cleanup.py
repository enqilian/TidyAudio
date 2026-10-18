"""测试第 4 组：去静音、高通滤波、降噪。"""

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


def clip_of(*parts):
    """把几段声音/静音拼成一个单声道 Clip。"""
    return Clip(audio=np.concatenate(parts)[:, None], sr=SR, name="test")


def band_energy(audio, low, high):
    """某个频段的能量，用来检查滤波效果。"""
    spectrum = np.abs(np.fft.rfft(audio[:, 0])) ** 2
    freqs = np.fft.rfftfreq(audio.shape[0], 1 / SR)
    return spectrum[(freqs >= low) & (freqs <= high)].sum()


# ---------- 去静音 ----------

def test_trim_removes_edges_only():
    # 1 秒静音 + 1 秒声音 + 1 秒静音 + 1 秒声音 + 1 秒静音 = 5 秒
    clip = clip_of(silence(1), tone(1), silence(1), tone(1), silence(1))
    REGISTRY["silence_removal"](mode="trim", keep_ms=0).apply(clip)
    # 去掉头尾各 1 秒，中间的停顿保留 → 约 3 秒
    assert clip.duration == pytest.approx(3.0, abs=0.1)


def test_all_mode_removes_long_pauses():
    clip = clip_of(silence(1), tone(1), silence(1), tone(1), silence(1))
    REGISTRY["silence_removal"](mode="all", keep_ms=0, min_silence_ms=300).apply(clip)
    # 中间 1 秒停顿也去掉 → 约 2 秒
    assert clip.duration == pytest.approx(2.0, abs=0.1)
    assert clip.info["silence_removed_sec"] == pytest.approx(3.0, abs=0.1)


def test_all_mode_keeps_short_pauses():
    # 中间只停 0.1 秒，短于 min_silence_ms=300，不应该被去掉
    clip = clip_of(tone(1), silence(0.1), tone(1))
    REGISTRY["silence_removal"](mode="all", keep_ms=0, min_silence_ms=300).apply(clip)
    assert clip.duration == pytest.approx(2.1, abs=0.05)


def test_keep_ms_leaves_margin():
    clip = clip_of(silence(1), tone(1), silence(1))
    REGISTRY["silence_removal"](mode="trim", keep_ms=200).apply(clip)
    assert clip.duration == pytest.approx(1.4, abs=0.1)   # 前后各留 0.2 秒


def test_all_silent_clip_is_dropped():
    clip = clip_of(silence(2))
    assert REGISTRY["silence_removal"]().apply(clip) == []
    assert "silent" in clip.info["dropped_reason"]


# ---------- 高通滤波 ----------

def test_highpass_removes_low_keeps_high():
    # 30 Hz 低频哼声 + 1000 Hz 有用声音
    clip = clip_of(tone(2, freq=30) + tone(2, freq=1000))
    low_before = band_energy(clip.audio, 20, 40)
    high_before = band_energy(clip.audio, 990, 1010)

    REGISTRY["highpass"](cutoff_hz=80).apply(clip)

    assert band_energy(clip.audio, 20, 40) < low_before * 0.01      # 低频被削掉 99% 以上
    assert band_energy(clip.audio, 990, 1010) > high_before * 0.95  # 高频基本不变


def test_highpass_rejects_cutoff_above_nyquist():
    clip = clip_of(tone(1))
    with pytest.raises(ValueError):
        REGISTRY["highpass"](cutoff_hz=9000).apply(clip)   # 16k 采样率最高只到 8k


# ---------- 降噪 ----------

@pytest.mark.parametrize("key", ["silence_removal", "highpass", "denoise"])
def test_cleanup_steps_are_advanced(key):
    assert REGISTRY[key].advanced is True


def test_denoise_reduces_noise_and_keeps_shape():
    rng = np.random.default_rng(0)
    noise = (0.05 * rng.standard_normal(3 * SR)).astype(np.float32)
    signal = np.concatenate([silence(1), tone(1), silence(1)])
    clip = clip_of(signal + noise)

    REGISTRY["denoise"](strength=1.0).apply(clip)

    assert clip.audio.shape == (3 * SR, 1)
    # 开头那 1 秒本来只有噪声，降噪后应该明显更安静
    noise_before = np.sqrt(np.mean(noise[:SR] ** 2))
    noise_after = np.sqrt(np.mean(clip.audio[: SR // 2, 0] ** 2))
    assert noise_after < noise_before * 0.5


def test_denoise_works_on_stereo(make_clip):
    clip = make_clip(seconds=1.0, sr=SR, channels=2)
    REGISTRY["denoise"]().apply(clip)
    assert clip.channels == 2
