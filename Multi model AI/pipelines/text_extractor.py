# pipelines/text_extractor.py
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class TextEncoder:
    def __init__(self, model_name="hfl/chinese-macbert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts, device="cpu"):
        # texts: list[str]
        with torch.no_grad():
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            outputs = self.model(**inputs)
            # Use pooled CLS or mean
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            else:
                emb = outputs.last_hidden_state.mean(dim=1)
            return emb.cpu().numpy()

if __name__ == "__main__":
    te = TextEncoder()
    v = te.encode(["I feel tired", "Today was good"])
    print("text emb shape:", v.shape)
