"""第 4 组：清理类功能 —— 去静音、高通滤波、降噪。

这组功能都会"删掉"一部分声音，可能破坏你要研究的特征，
所以整组都放在高级选项里（默认不勾选），默认参数也都比较保守：
  - 去静音：默认只切掉开头和结尾，不动中间
  - 高通滤波：默认 80 Hz，只去掉人声以下的低频隆隆声
  - 降噪：默认只降一半强度
"""

import warnings

import librosa
import noisereduce as nr
import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..audio_io import Clip
from .base import Param, Step, register
from .checks import drop, to_db

# 整条音频峰值低于这个值，直接当作全静音
# （top_db 是"相对最响处"判断，整条都很安静时就失效了，需要一个绝对的底线）
SILENT_FLOOR_DB = -60.0


def sound_regions(clip: Clip, top_db: float, min_silence_ms: float,
                  keep_ms: float) -> list[tuple[int, int]]:
    """找出有声音的区域，返回 [(起点, 终点), ...]（单位：采样点）。

    - 间隔短于 min_silence_ms 的两段声音会合并成一段（字与字之间的短停顿不算）
    - 每段前后各多留 keep_ms 毫秒
    - 整条都是静音时返回空列表
    去静音和按停顿切片都用这个函数。
    """
    peak = float(np.abs(clip.audio).max()) if clip.audio.size else 0.0
    if to_db(peak) < SILENT_FLOOR_DB:
        return []

    # librosa 要求的形状是 (声道, 采样点)，和我们的 (采样点, 声道) 相反，所以转置
    # 用 25ms 窗口、10ms 步长判断（语音处理的常用设置），切点更精确
    intervals = librosa.effects.split(
        clip.audio.T,
        top_db=top_db,
        frame_length=int(0.025 * clip.sr),
        hop_length=int(0.010 * clip.sr),
    )
    if len(intervals) == 0:
        return []

    min_gap = int(min_silence_ms / 1000 * clip.sr)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - merged[-1][1] < min_gap:
            merged[-1][1] = e           # 停顿太短，算作同一段
        else:
            merged.append([s, e])

    keep = int(keep_ms / 1000 * clip.sr)
    n = clip.audio.shape[0]
    return [(max(0, s - keep), min(n, e + keep)) for s, e in merged]


@register
class SilenceRemoval(Step):
    key = "silence_removal"
    label = "Silence Removal"
    description = "Trim leading/trailing silence, or also remove long pauses in between."
    advanced = True
    params = [
        Param("mode", "Remove", "trim", choices=["trim", "all"],
              help="trim: only the start and end (safe); "
                   "all: also long pauses in between (changes timing and rhythm)"),
        Param("top_db", "Silence threshold (dB below peak)", 40.0, min=10, max=80,
              help="Anything this far below the loudest part counts as silence. Smaller = more aggressive"),
        Param("keep_ms", "Padding kept (ms)", 100, min=0, max=1000,
              help="Keep a little audio around each sound so onsets are not cut off"),
        Param("min_silence_ms", "Only remove pauses longer than (ms)", 300, min=50, max=5000,
              show_if={"mode": "all"}),
    ]

    def apply(self, clip: Clip) -> list[Clip]:
        v = self.values
        regions = sound_regions(clip, v["top_db"], v["min_silence_ms"], v["keep_ms"])
        if not regions:
            return drop(clip, "Nothing left after silence removal (all silent)")

        if v["mode"] == "trim":
            # 只保留"第一段声音的开头"到"最后一段声音的结尾"
            regions = [(regions[0][0], regions[-1][1])]

        original = clip.duration
        clip.audio = np.concatenate([clip.audio[s:e] for s, e in regions], axis=0)
        clip.info["silence_removed_sec"] = round(original - clip.duration, 3)
        return [clip]


@register
class HighPassFilter(Step):
    key = "highpass"
    label = "High-Pass Filtering"
    description = "Remove low-frequency rumble (air conditioning, wind, desk bumps, mains hum)."
    advanced = True
    params = [
        Param("cutoff_hz", "Cutoff frequency (Hz)", 80.0, min=10, max=2000,
              help="Sound below this frequency is attenuated. 60-100 Hz is typical for speech; "
                   "be careful with low-frequency sources (drums, engines, growls)"),
        Param("order", "Filter order", 4, choices=[2, 4, 6, 8],
              help="Higher order = steeper cutoff"),
    ]

    def apply(self, clip: Clip) -> list[Clip]:
        cutoff = self.values["cutoff_hz"]
        if cutoff >= clip.sr / 2:
            raise ValueError(f"Cutoff {cutoff} Hz must be below half the sample rate ({clip.sr / 2} Hz)")
        # Butterworth 滤波器：通带最平坦，不会给保留下来的频率加起伏
        sos = butter(self.values["order"], cutoff, btype="highpass", fs=clip.sr, output="sos")
        # sosfiltfilt 正着滤一遍、倒着滤一遍，相位不变（波形不会被"推后"）
        clip.audio = sosfiltfilt(sos, clip.audio, axis=0).astype(np.float32)
        return [clip]


@register
class Denoising(Step):
    key = "denoise"
    label = "Denoising"
    description = "Reduce background noise. May add artifacts and remove details you care about."
    advanced = True
    params = [
        Param("strength", "Strength (0-1)", 0.5, min=0.0, max=1.0,
              help="1 removes as much noise as possible; 0.5 is conservative with fewer artifacts"),
        Param("stationary", "Noise type", "stationary",
              choices=["stationary", "non-stationary"],
              help="stationary: constant noise (fans, hiss); "
                   "non-stationary: changing noise (crowds, street), slower"),
    ]

    def apply(self, clip: Clip) -> list[Clip]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = nr.reduce_noise(
                y=clip.audio.T,     # noisereduce 也要 (声道, 采样点)
                sr=clip.sr,
                stationary=self.values["stationary"] == "stationary",
                prop_decrease=self.values["strength"],
            )
        clip.audio = np.asarray(out, dtype=np.float32).reshape(clip.audio.T.shape).T
        return [clip]
