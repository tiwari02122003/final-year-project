# data/synthetic_dataset.py
import os
import json
import random
import numpy as np

OUT = "data/synthetic_samples.jsonl"
NUM = 120

def generate_sample(i):
    # Each sample: id, text, audio_vec (placeholder), visual_vec (placeholder), stats, phq9, label
    phq9 = float(np.clip(np.random.normal(8, 6), 0, 27))
    label = 1 if phq9 >= 10 else 0  # threshold
    text = "I feel " + random.choice(["fine", "sad", "lonely", "tired", "happy"])
    # placeholder vectors
    audio = np.random.randn(256).tolist()
    visual = np.random.randn(128).tolist()
    stats = {
        "neg_prop": round(random.random(), 3),
        "late_night_ratio": round(random.random(), 3),
        "posts_per_week": random.randint(0, 30),
        "std_post_time": round(random.random()*10, 3),
        "image_freq": round(random.random(), 3)
    }
    return {
        "id": f"user_{i}",
        "text": text,
        "audio_vec": audio,
        "visual_vec": visual,
        "stats": stats,
        "phq9": phq9,
        "label": label
    }

def main():
    os.makedirs("data", exist_ok=True)
    with open(OUT, "w") as f:
        for i in range(NUM):
            s = generate_sample(i)
            f.write(json.dumps(s) + "\n")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
