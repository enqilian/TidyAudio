"""第二步：扫描输入文件夹，找出里面所有的音频文件。"""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# 支持的音频格式（soundfile 能直接读的）
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}


@dataclass
class ScanResult:
    """扫描结果。"""

    folder: Path
    files: list[Path] = field(default_factory=list)

    @property
    def count(self) -> int:
        """音频文件总数。"""
        return len(self.files)

    @property
    def by_extension(self) -> dict[str, int]:
        """每种格式各有多少个，比如 {".wav": 120, ".flac": 3}。"""
        return dict(Counter(f.suffix.lower() for f in self.files))


def scan_folder(folder: str | Path, recursive: bool = True) -> ScanResult:
    """扫描文件夹，返回所有音频文件。

    recursive=True 时会连子文件夹一起扫描。
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    pattern = "**/*" if recursive else "*"
    files = sorted(
        p
        for p in folder.glob(pattern)
        if p.is_file()
        and p.suffix.lower() in AUDIO_EXTENSIONS
        # 跳过隐藏文件，比如 macOS 在 U 盘上生成的 "._xxx.wav"
        and not p.name.startswith(".")
    )
    return ScanResult(folder=folder, files=files)


# 读文件头很快（大约 0.15 毫秒一个），但文件特别多时只看前这么多个，避免界面卡住
PROPERTY_SAMPLE_LIMIT = 5000


@dataclass
class AudioProperties:
    """输入音频的格式统计：每种采样率 / 位深 / 声道数各有多少个文件。"""

    sample_rates: Counter = field(default_factory=Counter)   # {44100: 700, 48000: 4}
    subtypes: Counter = field(default_factory=Counter)       # {"PCM_16": 698, "FLOAT": 4}
    channels: Counter = field(default_factory=Counter)       # {2: 704}
    checked: int = 0          # 实际读了多少个文件
    total: int = 0            # 一共有多少个文件
    unreadable: int = 0       # 读不出文件头的（坏文件）

    @property
    def sampled(self) -> bool:
        """是不是只看了一部分文件。"""
        return self.checked < self.total


def read_properties(files: list[Path], limit: int = PROPERTY_SAMPLE_LIMIT) -> AudioProperties:
    """只读文件头（不读音频数据），统计采样率、位深、声道数。"""
    import soundfile as sf

    props = AudioProperties(total=len(files))
    for path in files[:limit]:
        props.checked += 1
        try:
            info = sf.info(path)
        except Exception:
            props.unreadable += 1
            continue
        props.sample_rates[info.samplerate] += 1
        props.subtypes[info.subtype] += 1
        props.channels[info.channels] += 1
    return props
