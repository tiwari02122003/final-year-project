# train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import random
import numpy as np
from models.mffnc import MFFNC
from utils.metrics import compute_class_metrics, compute_reg_metrics, expected_calibration_error
from tqdm import tqdm

class MultimodalDataset(Dataset):
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return s

def collate_fn(batch):
    import numpy as np
    texts = [b["text"] for b in batch]
    audio = np.stack([np.array(b["audio_vec"]) for b in batch])
    visual = np.stack([np.array(b["visual_vec"]) for b in batch])
    stats = np.stack([[b["stats"]["neg_prop"], b["stats"]["late_night_ratio"], b["stats"]["posts_per_week"], b["stats"]["std_post_time"], b["stats"]["image_freq"]] for b in batch])
    phq = np.array([b["phq9"] for b in batch], dtype=np.float32)
    label = np.array([b["label"] for b in batch], dtype=np.int64)
    return texts, torch.from_numpy(audio).float(), torch.from_numpy(visual).float(), torch.from_numpy(stats).float(), torch.from_numpy(phq).float(), torch.from_numpy(label)

def train_one_epoch(model, dataloader, optimizer, device, text_encoder=None, modality_dropout=0.15):
    model.train()
    total_loss = 0.0
    for texts, audio, visual, stats, phq, label in tqdm(dataloader):
        batch_size = audio.shape[0]
        # optionally encode text (if using text encoder)
        if text_encoder is not None:
            text_emb = text_encoder.encode(texts)  # numpy
            text_emb = torch.tensor(text_emb).float()
        else:
            text_emb = torch.randn(batch_size, 384)
        text_emb = text_emb.to(device)
        audio = audio.to(device)
        visual = visual.to(device)
        stats = stats.to(device)
        phq = phq.to(device)
        label = label.to(device)

        # modality dropout simulation
        if random.random() < modality_dropout:
            # randomly drop one modality
            drop = random.choice(["text", "audio", "visual"])
            if drop == "text": text_emb = None
            if drop == "audio": audio = None
            if drop == "visual": visual = None

        optimizer.zero_grad()
        logits, reg = model(text_emb=text_emb, audio_emb=audio, visual_emb=visual, stats_vec=stats)
        ce = F.cross_entropy(logits, label)
        mae = torch.abs(reg - phq).mean()
        loss = ce + 0.5 * mae
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dl, device, text_encoder=None):
    model.eval()
    ys, preds, scores, regs = [], [], [], []
    with torch.no_grad():
        for texts, audio, visual, stats, phh, label in dl:
            B = audio.shape[0]
            if text_encoder is not None:
                text_emb = text_encoder.encode(texts)
                text_emb = torch.tensor(text_emb).float().to(device)
            else:
                text_emb = torch.randn(B, 384).to(device)
            audio, visual, stats = audio.to(device), visual.to(device), stats.to(device)
            logits, reg = model(text_emb=text_emb, audio_emb=audio, visual_emb=visual, stats_vec=stats)
            prob = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            pred = (prob >= 0.5).astype(int)
            ys.extend(label.numpy().tolist())
            preds.extend(pred.tolist())
            scores.extend(prob.tolist())
            regs.extend(reg.cpu().numpy().tolist())
    cmetrics = compute_class_metrics(np.array(ys), np.array(preds), np.array(scores))
    rmetrics = compute_reg_metrics(np.array([s["phq9"] for s in open_dataset_samples("data/synthetic_samples.jsonl")]), np.array(regs))
    ece = expected_calibration_error(np.array(scores), np.array(ys))
    return cmetrics, rmetrics, ece

def open_dataset_samples(path):
    import json
    res = []
    with open(path, "r") as f:
        for l in f: res.append(json.loads(l))
    return res

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_samples.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bs", type=int, default=16)
    args = parser.parse_args()

    ds = MultimodalDataset(args.data)
    # train/test split
    train_idx = list(range(int(0.8*len(ds))))
    test_idx = list(range(int(0.8*len(ds)), len(ds)))
    from torch.utils.data import Subset
    train_ds = Subset(ds, train_idx)
    test_ds = Subset(ds, test_idx)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False, collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MFFNC()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    # optionally instantiate a text encoder (slow on first run)
    from pipelines.text_extractor import TextEncoder
    text_enc = TextEncoder()  # you can set device with model.to()

    for e in range(args.epochs):
        loss = train_one_epoch(model, train_dl, opt, device, text_encoder=text_enc)
        print(f"epoch {e} loss {loss:.4f}")
    # quick eval
    cmetrics, rmetrics, ece = evaluate(model, test_dl, device, text_encoder=text_enc)
    print("classification:", cmetrics)
    print("regression:", rmetrics)
    print("ECE:", ece)
