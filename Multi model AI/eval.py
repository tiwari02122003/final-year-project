# eval.py
# Basic evaluation wrapper — you can extend for subgroup analysis & calibration plots.
from train import MultimodalDataset, collate_fn
from torch.utils.data import DataLoader
import torch
from models.mffnc import MFFNC
from pipelines.text_extractor import TextEncoder
from utils.metrics import compute_class_metrics, compute_reg_metrics, expected_calibration_error
import json, numpy as np

def load_model(path=None):
    m = MFFNC()
    if path:
        m.load_state_dict(torch.load(path))
    return m

def subgroup_analysis(samples, preds, labels, subgroup_fn):
    groups = {}
    for s, p, l in zip(samples, preds, labels):
        g = subgroup_fn(s)
        groups.setdefault(g, {"pred": [], "label": []})
        groups[g]["pred"].append(p); groups[g]["label"].append(l)
    out = {}
    for g,v in groups.items():
        out[g] = compute_class_metrics(np.array(v["label"]), np.array(v["pred"]), np.array(v["pred"]))
    return out

if __name__ == "__main__":
    print("Run eval.py with your trained checkpoint.")
