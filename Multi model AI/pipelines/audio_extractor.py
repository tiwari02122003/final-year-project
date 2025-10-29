# pipelines/audio_extractor.py
import numpy as np
import subprocess
import os
import pandas as pd

def parse_opensmile_csv(path):
    # reads opensmile output CSV (feature per row)
    df = pd.read_csv(path, sep=';|,', engine='python')
    # take last row stats or mean across frames
    # simplest: take numeric columns, compute mean
    numeric = df.select_dtypes(include=["number"])
    return numeric.mean().values

def fallback_audio_features(wav_path):
    import librosa
    y, sr = librosa.load(wav_path, sr=16000)
    f0 = np.mean(librosa.feature.rms(y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = np.mean(librosa.feature.mfcc(y, sr=sr), axis=1)
    return np.concatenate([[f0, zcr], mfcc])

if __name__ == "__main__":
    # demo fallback
    arr = fallback_audio_features("example.wav") if os.path.exists("example.wav") else np.random.randn(40)
    print("audio vec len:", len(arr))
