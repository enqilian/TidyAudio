"""测试第 1 组：读写音频 + 重采样、声道转换、位深转换。"""

import numpy as np
import pytest
import soundfile as sf

from tidyaudio.audio_io import load_audio, save_audio
from tidyaudio.steps import REGISTRY


def test_all_steps_registered():
    assert {"resample", "channels", "bit_depth"} <= set(REGISTRY)


def test_resample_changes_rate_and_length(make_clip):
    clip = make_clip(seconds=1.0, sr=44100)
    [out] = REGISTRY["resample"](target_sr=16000).apply(clip)
    assert out.sr == 16000
    assert out.audio.shape[0] == 16000      # 1 秒 = 16000 个采样点
    assert out.duration == pytest.approx(1.0)


def test_stereo_to_mono(make_clip):
    [out] = REGISTRY["channels"](mode="mono").apply(make_clip(channels=2))
    assert out.channels == 1


def test_mono_to_stereo(make_clip):
    [out] = REGISTRY["channels"](mode="stereo").apply(make_clip(channels=1))
    assert out.channels == 2


def test_unknown_param_is_rejected():
    with pytest.raises(ValueError):
        REGISTRY["resample"](target_rate=16000)   # 拼错了参数名


def test_save_and_load_with_bit_depth(tmp_path, make_clip):
    clip = make_clip(channels=1, sr=16000)
    [clip] = REGISTRY["bit_depth"](subtype="PCM_24").apply(clip)
    path = tmp_path / "out.wav"
    save_audio(clip, path)

    assert sf.info(path).subtype == "PCM_24"
    loaded = load_audio(path)
    assert loaded.sr == 16000
    assert loaded.channels == 1
    assert np.allclose(loaded.audio, clip.audio, atol=1e-4)
