"""Custom processing script for TidyAudio.

TidyAudio calls `process(audio, sr)` once for every audio file
(or every segment, if your script runs after Audio Segmentation).

Arguments
    audio : numpy.ndarray, shape (samples, channels), float32, roughly -1.0 to 1.0
            (mono audio has shape (samples, 1))
    sr    : int, sample rate in Hz

Return one of
    (audio, sr)                      keep the file (most common)
    None                             drop the file
    [(audio1, sr1), (audio2, sr2)]   turn one file into several

The returned audio may be (samples, channels), a 1-D mono array,
or (channels, samples) as librosa uses; all three are accepted.

Need to know which file you are processing? Add a third argument:

    def process(audio, sr, info):
        info["file"]    path relative to the input folder, e.g. "alarm/door_01.wav"
        info["folder"]  the folder the file is in, e.g. "alarm" (often the class)
        info["name"]    file name without extension ("door_01_seg000" after segmentation)

You can import anything installed in the app's Python environment
(numpy, scipy, librosa, soundfile, ...).
"""

import numpy as np


def process(audio, sr):
    # Example: apply a 10 ms fade-in and fade-out to avoid clicks.
    n = min(int(0.010 * sr), audio.shape[0] // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return audio, sr
