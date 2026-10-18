"""读写音频文件。

程序内部统一用 Clip 表示"一段音频"，不管原文件是什么格式。
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class Clip:
    """一段音频。

    audio: 形状是 (采样点数, 声道数) 的 float32 数组，数值范围大约在 -1 到 1。
           即使是单声道，也是二维的 (N, 1)，这样所有功能处理起来写法统一。
    sr:    采样率，比如 16000、44100。
    name:  输出时用的文件名（不含后缀），切片后会变成 "xxx_seg000" 这样。
    source: 原始文件路径，用于记录这段音频是从哪来的。
    info:  各个功能往这里写检查结果，比如 {"clipping_ratio": 0.001}，
           最后会汇总进 manifest。
    subtype: 保存时用的位深格式，比如 "PCM_16"。None 表示保持原样。
    features: 特征提取的结果，比如 {"mfcc": 数组}，最后会存成 .npy 文件。
    rel_path: 相对于输入文件夹的路径，比如 "alarm/xxx.wav"（pipeline 读文件时填）。
    """

    audio: np.ndarray
    sr: int
    name: str
    source: Path | None = None
    info: dict = field(default_factory=dict)
    subtype: str | None = None
    features: dict = field(default_factory=dict)
    rel_path: str | None = None

    @property
    def channels(self) -> int:
        return self.audio.shape[1]

    @property
    def duration(self) -> float:
        """时长（秒）。"""
        return self.audio.shape[0] / self.sr


def load_audio(path: str | Path) -> Clip:
    """读取一个音频文件，返回 Clip。"""
    path = Path(path)
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    subtype = sf.info(path).subtype
    return Clip(audio=audio, sr=sr, name=path.stem, source=path, subtype=subtype)


def save_audio(clip: Clip, path: str | Path) -> None:
    """把 Clip 保存成音频文件。格式由后缀决定（.wav / .flac ...）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, clip.audio, clip.sr, subtype=_compatible_subtype(clip.subtype, path))


def _compatible_subtype(subtype: str | None, path: Path) -> str | None:
    """原文件的位深格式不一定能存成目标格式（比如 mp3 的格式不能直接写进 wav），
    不兼容时交给 soundfile 用默认值。"""
    if subtype is None:
        return None
    fmt = path.suffix.lstrip(".").upper()
    if fmt == "AIF":
        fmt = "AIFF"
    return subtype if sf.check_format(fmt, subtype) else None
