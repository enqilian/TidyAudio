"""第 3 组：电平类功能 —— 归一化、去直流偏移（高级选项）。

推荐顺序：去直流偏移 → 归一化
  直流偏移会让峰值"虚高"，所以要先去掉，否则归一化算出来的增益不准。
"""

import warnings

import numpy as np
import pyloudnorm as pyln

from ..audio_io import Clip
from .base import Param, Step, register
from .checks import to_db


def db_to_gain(db: float) -> float:
    """dB 转线性倍数。比如 -6 dB → 约 0.5。"""
    return float(10 ** (db / 20))


def peak(audio: np.ndarray) -> float:
    return float(np.abs(audio).max()) if audio.size else 0.0


def rms(audio: np.ndarray) -> float:
    """均方根：整段音频的平均能量。"""
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0


@register
class Normalization(Step):
    key = "normalize"
    label = "Normalization"
    description = "Bring every file to the same level."
    params = [
        Param("method", "Method", "lufs", choices=["peak", "rms", "lufs"],
              help="peak: same maximum peak; rms: same average energy; "
                   "lufs: same perceived loudness (closest to how we hear)"),
        Param("target_peak_db", "Target peak (dBFS)", -1.0, min=-60, max=0,
              show_if={"method": "peak"}),
        Param("target_rms_db", "Target RMS (dBFS)", -20.0, min=-60, max=0,
              show_if={"method": "rms"}),
        Param("target_lufs", "Target loudness (LUFS)", -23.0, min=-70, max=0,
              help="-23 is the EBU R128 broadcast standard; speech datasets often use -20 to -26",
              show_if={"method": "lufs"}),
        Param("prevent_clipping", "Prevent clipping", True,
              help="If reaching the target would exceed the peak ceiling, apply less gain",
              show_if={"method": ["rms", "lufs"]}),
        Param("peak_ceiling_db", "Peak ceiling (dBFS)", -1.0, min=-20, max=0,
              show_if={"method": ["rms", "lufs"]}),
    ]

    def apply(self, clip: Clip) -> list[Clip]:
        method = self.values["method"]
        p = peak(clip.audio)
        if p == 0:
            clip.info["normalize_skipped"] = "silent"
            return [clip]

        # 第 1 步：按选定的方式测出现在的电平，算出要调多少 dB
        if method == "peak":
            before, target = to_db(p), self.values["target_peak_db"]
        elif method == "rms":
            before, target = to_db(rms(clip.audio)), self.values["target_rms_db"]
        else:
            before, target = self._measure_lufs(clip), self.values["target_lufs"]
            if before is None:
                return [clip]
        gain_db = target - before

        # 第 2 步：rms / lufs 可能把峰值推过上限，需要时少调一点
        if method != "peak" and self.values["prevent_clipping"]:
            headroom_db = self.values["peak_ceiling_db"] - to_db(p)
            if gain_db > headroom_db:
                clip.info["normalize_limited"] = True   # 没能达到目标值
                clip.info["normalize_shortfall_db"] = round(float(gain_db - headroom_db), 2)
                gain_db = headroom_db

        clip.audio = (clip.audio * db_to_gain(gain_db)).astype(np.float32)
        clip.info["normalize_gain_db"] = round(float(gain_db), 2)
        # 处理前后的电平（单位跟着方法走：peak / rms 是 dBFS，lufs 是 LUFS），报告里画对比图
        clip.info["normalize_before"] = round(float(before), 2)
        clip.info["normalize_after"] = round(float(before + gain_db), 2)
        return [clip]

    @staticmethod
    def _measure_lufs(clip: Clip) -> float | None:
        """按 ITU-R BS.1770 标准测响度。测不了时返回 None 并记录原因。"""
        if clip.duration < 0.4:   # 标准规定至少要 0.4 秒
            clip.info["normalize_skipped"] = "too short to measure loudness"
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # 安静片段 pyloudnorm 会发警告
            loudness = pyln.Meter(clip.sr).integrated_loudness(clip.audio)
        if not np.isfinite(loudness):
            clip.info["normalize_skipped"] = "too quiet to measure loudness"
            return None
        return float(loudness)


@register
class DCOffset(Step):
    key = "dc_offset"
    label = "DC Offset"
    description = "Check whether the waveform is shifted away from zero, and optionally remove it. Rarely needed for modern recordings."
    advanced = True
    params = [
        Param("mode", "Mode", "check", choices=["check", "threshold", "remove"],
              help="check: only record the offset, audio unchanged; "
                   "threshold: remove only above the threshold; remove: always remove"),
        Param("threshold", "Removal threshold (linear)", 0.01, min=0, max=0.5,
              show_if={"mode": "threshold"}),
    ]

    def apply(self, clip: Clip) -> list[Clip]:
        # 直流偏移 = 每个声道的平均值。正常音频平均值应该非常接近 0。
        # 用 float64 计算：float32 累加几十万个采样点会有明显的舍入误差
        offset = clip.audio.mean(axis=0, dtype=np.float64)
        size = float(np.abs(offset).max())
        clip.info["dc_offset"] = round(size, 6)

        mode = self.values["mode"]
        if mode == "remove" or (mode == "threshold" and size > self.values["threshold"]):
            clip.audio = (clip.audio - offset).astype(np.float32)
            clip.info["dc_offset_removed"] = True
        return [clip]
