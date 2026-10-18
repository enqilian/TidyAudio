"""第 5 组（上）：切片 —— 把长音频切成短片段。

这是唯一一个"1 段进、N 段出"的功能。
切出来的每一段都是一个新的 Clip，名字是 "原名_seg000"、"原名_seg001"……
并在 info 里记下它来自原文件的第几秒到第几秒，方便以后回查。
"""

import numpy as np

from ..audio_io import Clip
from .base import Param, Step, register
from .checks import drop
from .cleanup import sound_regions


@register
class Segmentation(Step):
    key = "segment"
    label = "Audio Segmentation"
    description = "Cut long files into shorter segments."
    params = [
        Param("mode", "Split by", "fixed", choices=["fixed", "silence"],
              help="fixed: fixed-length segments; silence: split at pauses so sounds are not cut"),
        # ---- fixed 模式 ----
        Param("segment_sec", "Segment length (s)", 5.0, min=0.1,
              show_if={"mode": "fixed"}),
        Param("overlap_sec", "Overlap (s)", 0.0, min=0,
              help="e.g. 5 s segments with 1 s overlap: 0-5, 4-9, 8-13 ...",
              show_if={"mode": "fixed"}),
        Param("last", "Last short segment", "pad", choices=["pad", "keep", "drop"],
              help="pad: pad with silence to full length; keep: keep as is; drop: discard. "
                   "A file shorter than one segment is treated as its own last segment",
              show_if={"mode": "fixed"}),
        # ---- silence 模式 ----
        Param("top_db", "Silence threshold (dB below peak)", 40.0, min=10, max=80,
              show_if={"mode": "silence"}),
        Param("min_silence_ms", "Split at pauses longer than (ms)", 500, min=50, max=5000,
              show_if={"mode": "silence"}),
        Param("max_sec", "Max segment length (s), 0 = no limit", 0.0, min=0,
              help="Segments still longer than this after splitting at pauses are cut further",
              show_if={"mode": "silence"}),
    ]

    def apply(self, clip: Clip) -> list[Clip]:
        if self.values["mode"] == "fixed":
            spans = self._fixed_spans(clip)
        else:
            spans = self._silence_spans(clip)

        if not spans:
            if self.values["mode"] == "fixed":
                return drop(clip, f"Shorter than one segment ({clip.duration:.2f}s < "
                                  f"{self.values['segment_sec']}s) and last segment is set to drop")
            return drop(clip, "No segments after splitting (all silent)")
        return [self._make_segment(clip, i, s, e) for i, (s, e) in enumerate(spans)]

    def _fixed_spans(self, clip: Clip) -> list[tuple[int, int]]:
        v = self.values
        length = int(v["segment_sec"] * clip.sr)
        step = length - int(v["overlap_sec"] * clip.sr)
        if step <= 0:
            raise ValueError("Overlap must be shorter than the segment length")
        n = clip.audio.shape[0]

        spans = []
        for start in range(0, n, step):
            # 有重叠时，上一段可能已经覆盖到结尾了，剩下的部分不用再切
            if spans and spans[-1][1] >= n:
                break
            end = start + length
            if end <= n:
                spans.append((start, end))
                continue
            # 最后不够长的一段：pad 补静音（end 超出原长度，_make_segment 里补）、keep 原样、drop 不要
            if v["last"] == "pad":
                spans.append((start, end))
            elif v["last"] == "keep":
                spans.append((start, n))
            break
        return spans

    def _silence_spans(self, clip: Clip) -> list[tuple[int, int]]:
        v = self.values
        regions = sound_regions(clip, v["top_db"], v["min_silence_ms"], keep_ms=100)
        if v["max_sec"] <= 0:
            return regions
        # 太长的段再硬切
        limit = int(v["max_sec"] * clip.sr)
        spans = []
        for s, e in regions:
            spans.extend((i, min(i + limit, e)) for i in range(s, e, limit))
        return spans

    @staticmethod
    def _make_segment(clip: Clip, index: int, start: int, end: int) -> Clip:
        audio = clip.audio[start:end]
        missing = end - start - audio.shape[0]
        if missing > 0:   # 补静音
            audio = np.pad(audio, ((0, missing), (0, 0)))
        info = dict(clip.info)   # 复制一份，每段各自独立
        info.update(
            segment_index=index,
            segment_start_sec=round(start / clip.sr, 3),
            segment_end_sec=round(min(end, clip.audio.shape[0]) / clip.sr, 3),
        )
        if missing > 0:
            info["segment_padded_sec"] = round(missing / clip.sr, 3)
        return Clip(
            audio=audio.copy(),
            sr=clip.sr,
            name=f"{clip.name}_seg{index:03d}",
            source=clip.source,
            info=info,
            subtype=clip.subtype,
            rel_path=clip.rel_path,
        )
