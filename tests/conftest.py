"""所有测试共用的工具。pytest 会自动加载这个文件。"""

import numpy as np
import pytest

from tidyaudio.audio_io import Clip


@pytest.fixture
def make_clip():
    """返回一个函数，用来生成测试用的正弦波 Clip。

    在测试函数的参数里写 make_clip，pytest 就会自动把它传进来。
    """

    def _make(seconds=1.0, sr=44100, channels=2, amplitude=0.5, freq=440.0):
        t = np.arange(int(seconds * sr)) / sr
        wave = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        audio = np.stack([wave] * channels, axis=1)
        return Clip(audio=audio, sr=sr, name="test")

    return _make
