"""第 1 组：格式类功能 —— 重采样、声道转换、位深转换。"""

import numpy as np
import soxr

from ..audio_io import Clip
from .base import Param, Step, register


@register
class Resample(Step):
    key = "resample"
    label = "Resampling"
    description = "Convert every file to the same sample rate."
    params = [
        Param("target_sr", "Target sample rate (Hz)", 16000,
              choices=[8000, 16000, 22050, 24000, 32000, 44100, 48000]),
    ]

    def apply(self, clip: Clip) -> list[Clip]:
        target = self.values["target_sr"]
        if clip.sr != target:
            # soxr 是高质量的重采样库，内部会先做抗混叠滤波
            clip.audio = soxr.resample(clip.audio, clip.sr, target).astype(np.float32)
            clip.sr = target
        return [clip]


@register
class ChannelConversion(Step):
    key = "channels"
    label = "Channel Conversion"
    description = "Convert every file to mono or stereo."
    params = [
        Param("mode", "Convert to", "mono", choices=["mono", "stereo"],
              help="mono: average all channels; stereo: duplicate mono into two channels"),
    ]

    def apply(self, clip: Clip) -> list[Clip]:
        if self.values["mode"] == "mono":
            if clip.channels > 1:
                clip.audio = clip.audio.mean(axis=1, keepdims=True)
        else:
            if clip.channels == 1:
                clip.audio = np.repeat(clip.audio, 2, axis=1)
            elif clip.channels > 2:
                clip.audio = clip.audio[:, :2]
        return [clip]


@register
class BitDepthConversion(Step):
    key = "bit_depth"
    label = "Bit Depth Conversion"
    description = "Set the bit depth / sample format used when saving."
    params = [
        Param("subtype", "Sample format", "PCM_16",
              choices=["PCM_16", "PCM_24", "PCM_32", "FLOAT"],
              help="PCM_16 is the most common and smallest; FLOAT is the most precise and largest"),
    ]

    def apply(self, clip: Clip) -> list[Clip]:
        # 处理过程中音频一直是 float32，不会损失精度；
        # 这里只是记下"保存时用什么格式"，真正的转换在 save_audio 里发生。
        clip.subtype = self.values["subtype"]
        return [clip]
