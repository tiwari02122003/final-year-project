# models/mffnc.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model*2), nn.ReLU(), nn.Linear(d_model*2, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, q, kv, q_mask=None, kv_mask=None):
        # q: (B, Lq, D), kv: (B, Lkv, D)
        attn_out, _ = self.mha(query=q, key=kv, value=kv, key_padding_mask=None)
        q = self.ln1(q + attn_out)
        ff = self.ff(q)
        q = self.ln2(q + ff)
        return q

class FusionCrossAttention(nn.Module):
    def __init__(self, dim_text=256, dim_audio=256, dim_visual=256, fusion_dim=256):
        super().__init__()
        # projectors
        self.pt = nn.Linear(dim_text, fusion_dim)
        self.pa = nn.Linear(dim_audio, fusion_dim)
        self.pv = nn.Linear(dim_visual, fusion_dim)
        # cross-attention blocks: text<-audio/visual, audio<-visual/text, visual<-audio/text (pairwise)
        self.ta = CrossAttentionBlock(fusion_dim)
        self.tv = CrossAttentionBlock(fusion_dim)
        self.at = CrossAttentionBlock(fusion_dim)
        self.av = CrossAttentionBlock(fusion_dim)
        self.vt = CrossAttentionBlock(fusion_dim)
        self.va = CrossAttentionBlock(fusion_dim)
        # gating
        self.gate = nn.Parameter(torch.ones(3))  # text/audio/visual gates

    def forward(self, text_vec, audio_vec, visual_vec, mask=(True,True,True)):
        # All inputs are (B, D) or None. Convert to (B,1,D) sequence
        B = text_vec.shape[0] if text_vec is not None else (audio_vec.shape[0] if audio_vec is not None else visual_vec.shape[0])
        # Prepare sequences
        def proj(x, p):
            return p(x).unsqueeze(1)  # (B,1,D)
        t = proj(text_vec, self.pt) if text_vec is not None else None
        a = proj(audio_vec, self.pa) if audio_vec is not None else None
        v = proj(visual_vec, self.pv) if visual_vec is not None else None

        # pairwise cross-attention (if missing, skip)
        # For each modality, let it query others, then pool results
        out = []
        # Text as query
        if t is not None:
            klist = []
            if a is not None:
                klist.append(self.ta(t, a))
            if v is not None:
                klist.append(self.tv(t, v))
            if klist:
                t_out = torch.mean(torch.stack(klist, dim=0), dim=0).squeeze(1)
            else:
                t_out = t.squeeze(1)
            out.append(t_out * self.gate[0])
        # Audio as query
        if a is not None:
            klist = []
            if t is not None:
                klist.append(self.at(a, t))
            if v is not None:
                klist.append(self.av(a, v))
            if klist:
                a_out = torch.mean(torch.stack(klist, dim=0), dim=0).squeeze(1)
            else:
                a_out = a.squeeze(1)
            out.append(a_out * self.gate[1])
        # Visual as query
        if v is not None:
            klist = []
            if t is not None:
                klist.append(self.vt(v, t))
            if a is not None:
                klist.append(self.va(v, a))
            if klist:
                v_out = torch.mean(torch.stack(klist, dim=0), dim=0).squeeze(1)
            else:
                v_out = v.squeeze(1)
            out.append(v_out * self.gate[2])

        if not out:
            raise ValueError("No modalities provided")
        fused = torch.cat(out, dim=1)  # (B, D*num_present)
        return fused

class MFFNC(nn.Module):
    def __init__(self, text_dim=384, audio_dim=256, visual_dim=128, stats_dim=16, fusion_dim=256, hidden=256):
        super().__init__()
        # per-modality projectors (if already embeddings, you can fine-tune sizes)
        self.text_proj = nn.Linear(text_dim, 256)
        self.audio_proj = nn.Linear(audio_dim, 256)
        self.visual_proj = nn.Linear(visual_dim, 256)
        self.stats_proj = MLP(stats_dim, 64, hidden=128)
        self.fuser = FusionCrossAttention(dim_text=256, dim_audio=256, dim_visual=256, fusion_dim=fusion_dim)
        # combine fused + stats
        fusion_out_dim = fusion_dim * 3  # because we concatenate present ones; safe upper-bound
        self.comb = nn.Sequential(
            nn.Linear(fusion_out_dim + 64, hidden),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # heads
        self.class_head = nn.Linear(hidden, 2)
        self.reg_head = nn.Linear(hidden, 1)

    def forward(self, text_emb=None, audio_emb=None, visual_emb=None, stats_vec=None):
        t = self.text_proj(text_emb) if text_emb is not None else None
        a = self.audio_proj(audio_emb) if audio_emb is not None else None
        v = self.visual_proj(visual_emb) if visual_emb is not None else None
        s = self.stats_proj(stats_vec) if stats_vec is not None else torch.zeros((t.shape[0] if t is not None else (a.shape[0] if a is not None else v.shape[0]),64), device=next(self.parameters()).device)
        fused = self.fuser(t, a, v)
        x = torch.cat([fused, s], dim=1)
        x = self.comb(x)
        logits = self.class_head(x)
        reg = self.reg_head(x).squeeze(1)
        return logits, reg
