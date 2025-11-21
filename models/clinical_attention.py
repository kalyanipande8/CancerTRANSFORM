import torch
import torch.nn as nn


class ClinicalAttentionModel(nn.Module):
    """Self-contained Clinical Attention model.

    The model treats the 7 clinical features as a sequence and applies
    a single-head self-attention over them. The output is pooled and
    passed to an MLP for binary classification.
    """

    def __init__(self, n_features: int = 7, embed_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.n_features = n_features
        self.embed = nn.Linear(1, embed_dim)
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Sequential(nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor):
        # x: [batch, n_features]
        b = x.shape[0]
        x = x.unsqueeze(-1)  # [b, n_features, 1]
        emb = self.embed(x)  # [b, n_features, embed_dim]
        q = self.q(emb)
        k = self.k(emb)
        v = self.v(emb)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (emb.shape[-1] ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_weights, v)
        feat_vec = attended.sum(dim=1)
        logits = self.out(feat_vec)
        # Return average attention per feature for explainability
        return logits, attn_weights.mean(dim=1)
