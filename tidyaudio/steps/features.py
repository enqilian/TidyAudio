"""第 5 组（下）：特征提取。

和其他功能不同，它不改音频，而是额外算出"特征"，有两种输出：
  1. 逐帧特征（每 10ms 一组数）→ 存进 clip.features，最后保存成 .npy 文件
     形状统一是 (特征维度, 帧数)，和 librosa 的习惯一致
  2. 整条音频的平均值（每条一个数）→ 存进 clip.info，会出现在 manifest 和报告里，
     方便直接比较、画分布图

多声道音频会先混成单声道再提取特征。
建议放在所有处理的最后，这样提取的是处理完的音频的特征。
"""

import librosa
import numpy as np

from ..audio_io import Clip
from .base import Param, Step, register


@register
class FeatureExtraction(Step):
    key = "features"
    label = "Feature Extraction"
    description = "Extract RMS, spectral features, log-mel and MFCC, saved as .npz files."
    advanced = True
    params = [
        Param("rms", "RMS energy", True),
        Param("spectral", "Spectral features (centroid, bandwidth, rolloff, flatness, ZCR)", True),
        Param("log_mel", "Log-mel spectrogram", False,
              help="The most common input for deep learning models"),
        Param("n_mels", "Mel bands", 80, choices=[40, 64, 80, 128],
              show_if={"log_mel": True}),
        Param("mfcc", "MFCC", False),
        Param("n_mfcc", "MFCC coefficients", 13, choices=[13, 20, 40],
              show_if={"mfcc": True}),
        Param("win_ms", "Window (ms)", 25.0, min=5, max=200),
        Param("hop_ms", "Hop (ms)", 10.0, min=1, max=100),
    ]

    def apply(self, clip: Clip) -> list[Clip]:
        v = self.values
        y = clip.audio.mean(axis=1)   # 混成单声道
        win = int(v["win_ms"] / 1000 * clip.sr)
        hop = int(v["hop_ms"] / 1000 * clip.sr)
        n_fft = 1 << (win - 1).bit_length()   # 不小于窗长的 2 的幂，FFT 更快
        common = dict(n_fft=n_fft, hop_length=hop, win_length=win)

        feats = {}
        if v["rms"]:
            feats["rms"] = librosa.feature.rms(
                y=y, frame_length=n_fft, hop_length=hop)
        if v["spectral"]:
            S = np.abs(librosa.stft(y, **common))   # 频谱只算一次，几个特征共用
            feats["spectral_centroid"] = librosa.feature.spectral_centroid(S=S, sr=clip.sr)
            feats["spectral_bandwidth"] = librosa.feature.spectral_bandwidth(S=S, sr=clip.sr)
            feats["spectral_rolloff"] = librosa.feature.spectral_rolloff(S=S, sr=clip.sr)
            feats["spectral_flatness"] = librosa.feature.spectral_flatness(S=S)
            feats["zcr"] = librosa.feature.zero_crossing_rate(
                y, frame_length=n_fft, hop_length=hop)
        if v["log_mel"]:
            mel = librosa.feature.melspectrogram(
                y=y, sr=clip.sr, n_mels=v["n_mels"], **common)
            feats["log_mel"] = librosa.power_to_db(mel, ref=np.max)
        if v["mfcc"]:
            feats["mfcc"] = librosa.feature.mfcc(
                y=y, sr=clip.sr, n_mfcc=v["n_mfcc"], **common)

        for name, arr in feats.items():
            arr = arr.astype(np.float32)
            clip.features[name] = arr
            # 一维特征（rms、频谱质心等）顺便记一个整条的平均值
            if arr.shape[0] == 1:
                clip.info[f"{name}_mean"] = round(float(arr.mean()), 6)
        return [clip]
