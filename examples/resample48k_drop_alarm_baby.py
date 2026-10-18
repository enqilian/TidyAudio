"""Example custom script for TidyAudio.

1. Drops every file whose folder (class) is "alarm" or "baby".
2. Resamples everything else to 48 kHz.

Add it in the app under Processing -> Your Python Scripts -> Add Existing…
"""

import soxr

DROP_CLASSES = {"alarm", "baby"}
TARGET_SR = 48000


def process(audio, sr, info):
    # info["folder"] is the folder the file sits in; in NIGENS that is the class
    if info["folder"] in DROP_CLASSES:
        return None

    if sr != TARGET_SR:
        audio = soxr.resample(audio, sr, TARGET_SR)
    return audio, TARGET_SR
