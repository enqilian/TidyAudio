"""第 2 组：检查类功能 —— 文件校验、时长过滤、削波检测。

这组功能大多不改音频，只做两件事之一：
  - 把检查结果写进 clip.info（最后会汇总到报告里）
  - 不合格就返回 []，把这条丢掉，并在 clip.info["dropped_reason"] 里写明原因
"""

import numpy as np

from ..audio_io import Clip
from .base import Param, Step, register


def drop(clip: Clip, reason: str) -> list[Clip]:
    """丢掉这条音频，并记下原因。"""
    clip.info["dropped_reason"] = reason
    return []


def to_db(x: float) -> float:
    """线性幅度转 dBFS。0 → -inf。"""
    return float(20 * np.log10(x)) if x > 0 else float("-inf")


@register
class FileValidation(Step):
    key = "validate"
    label = "File Validation"
    description = "Drop empty files, corrupted data (NaN/Inf), and near-silent files."
    params = [
        Param("drop_silent", "Drop near-silent files", True),
        Param("silent_db", "Silence threshold (dBFS)", -60.0, min=-120, max=0,
              help="A file whose peak is below this level counts as silent"),
    ]

    def apply(self, clip: Clip) -> list[Clip]:
        # 注意：完全打不开的文件在"读取"那一步就会失败，
        # 那种情况由后面的 pipeline 统一记录，不会走到这里。
        if clip.audio.shape[0] == 0:
            return drop(clip, "Empty file (0 samples)")
        if not np.isfinite(clip.audio).all():
            return drop(clip, "Corrupted data (contains NaN or Inf)")

        peak_db = to_db(float(np.abs(clip.audio).max()))
        if self.values["drop_silent"] and peak_db < self.values["silent_db"]:
            return drop(clip, f"Near-silent (peak {peak_db:.1f} dBFS)")
        return [clip]


@register
class DurationFilter(Step):
    key = "duration_filter"
    label = "Duration Filtering"
    description = "Drop files that are too short or too long."
    params = [
        Param("min_sec", "Minimum (s)", 0.5, min=0),
        Param("max_sec", "Maximum (s), 0 = no limit", 0.0, min=0),
    ]

    def apply(self, clip: Clip) -> list[Clip]:
        d = clip.duration
        clip.info["duration_checked_sec"] = round(d, 3)   # 报告里画"过滤时看到的时长"
        lo, hi = self.values["min_sec"], self.values["max_sec"]
        if d < lo:
            return drop(clip, f"Too short ({_seconds(d, lo)} < {lo}s)")
        if hi > 0 and d > hi:
            return drop(clip, f"Too long ({_seconds(d, hi)} > {hi}s)")
        return [clip]


@register
class ClippingDetection(Step):
    key = "clipping_check"
    label = "Clipping Detection"
    description = "Measure the share of clipped samples; optionally drop heavily clipped files."
    params = [
        Param("threshold", "Clipping threshold (linear)", 0.999, min=0.5, max=1.0,
              help="Samples whose absolute value reaches this level count as clipped"),
        Param("action", "When clipping is found", "mark", choices=["mark", "drop"],
              help="mark: only record it in the report; drop: drop the file above the ratio below"),
        Param("max_ratio", "Max allowed clipped ratio (%)", 0.1, min=0, max=100,
              show_if={"action": "drop"}),
    ]

    def apply(self, clip: Clip) -> list[Clip]:
        hits = np.abs(clip.audio) >= self.values["threshold"]
        # 任一声道削波，这个时间点就算削波
        clipped = hits.any(axis=1)
        ratio = 100.0 * clipped.mean() if clipped.size else 0.0

        clip.info["clipping_ratio_pct"] = round(float(ratio), 4)
        clip.info["clipped_samples"] = int(clipped.sum())
        clip.info["longest_clip_run"] = _longest_run(clipped)

        if self.values["action"] == "drop" and ratio > self.values["max_ratio"]:
            return drop(clip, f"Heavy clipping ({ratio:.3f}% > {self.values['max_ratio']}%)")
        return [clip]


def _seconds(d: float, limit: float) -> str:
    """时长保留两位小数；如果四舍五入后看起来和界限一样（0.4996 → 0.50），多显示几位。"""
    text = f"{d:.2f}"
    return f"{d:.3f}s" if float(text) == limit else f"{text}s"


def _longest_run(mask: np.ndarray) -> int:
    """最长连续 True 的长度。真实削波通常是连续一段被"削平"，
    孤立的单个满幅采样点不一定是削波。"""
    if not mask.any():
        return 0
    # 在首尾补 0，找出每段连续 True 的起点和终点
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return int((ends - starts).max())
