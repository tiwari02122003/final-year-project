# pipelines/visual_extractor.py
import numpy as np
import pandas as pd
import os

def parse_openface_csv(path):
    # expects OpenFace CSV; returns summary vector of AUs and blink rate etc.
    df = pd.read_csv(path)
    # example features: AU12_r (smile), AU1_r, AU4_r, gaze, headpose
    au_cols = [c for c in df.columns if c.startswith("AU")]
    numeric = df[au_cols].fillna(0)
    stats = numeric.mean().values
    blink_rate = np.mean(df.get("blink", pd.Series(0)))
    head_pose_cols = [c for c in df.columns if "pose" in c.lower()]
    head_stats = df[head_pose_cols].mean().fillna(0).values if head_pose_cols else np.zeros(3)
    return np.concatenate([stats, [blink_rate], head_stats])

def fallback_visual_features():
    return np.random.randn(128)

if __name__ == "__main__":
    print("visual vec len:", len(fallback_visual_features()))
