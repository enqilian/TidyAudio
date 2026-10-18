"""测试 scan.py：确认能正确数出音频文件。"""

import pytest

from tidyaudio.scan import scan_folder


def test_scan_counts_audio_files(tmp_path):
    # tmp_path 是 pytest 自动提供的临时文件夹，测试完会自动删掉
    (tmp_path / "a.wav").touch()
    (tmp_path / "b.FLAC").touch()          # 大写后缀也要认
    (tmp_path / "notes.txt").touch()       # 不是音频，不算
    (tmp_path / "._a.wav").touch()         # macOS 隐藏文件，不算
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.wav").touch()   # 子文件夹里的

    result = scan_folder(tmp_path)

    assert result.count == 3
    assert result.by_extension == {".wav": 2, ".flac": 1}


def test_scan_without_subfolders(tmp_path):
    (tmp_path / "a.wav").touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.wav").touch()

    assert scan_folder(tmp_path, recursive=False).count == 1


def test_scan_rejects_missing_folder(tmp_path):
    with pytest.raises(NotADirectoryError):
        scan_folder(tmp_path / "does_not_exist")


def test_read_properties(tmp_path):
    import numpy as np
    import soundfile as sf
    from tidyaudio.scan import read_properties
    tone = 0.3 * np.sin(np.arange(1600) / 5)
    sf.write(tmp_path / "a.wav", tone, 16000, subtype="PCM_16")
    sf.write(tmp_path / "b.wav", tone, 16000, subtype="PCM_24")
    sf.write(tmp_path / "c.wav", np.stack([tone, tone], 1), 44100, subtype="PCM_16")
    (tmp_path / "broken.wav").write_bytes(b"xx")
    props = read_properties(scan_folder(tmp_path).files)
    assert props.sample_rates == {16000: 2, 44100: 1}
    assert props.subtypes == {"PCM_16": 2, "PCM_24": 1}
    assert props.channels == {1: 2, 2: 1}
    assert props.unreadable == 1 and not props.sampled

    sampled = read_properties(scan_folder(tmp_path).files, limit=2)
    assert sampled.checked == 2 and sampled.sampled
