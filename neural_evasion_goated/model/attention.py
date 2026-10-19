"""
Cross-threat temporal attention head.

A GRUCell carries one hidden state per tracked missile across frames, an
attention scorer pools those states into a single situation vector, and a
sigmoid head turns that vector into a scalar danger score.

Copied verbatim from main3.py section 1.
"""

import torch
import torch.nn as nn


class TemporalThreatAttention(nn.Module):
    def __init__(self, embed_dim=32, context_dim=5, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(embed_dim + context_dim, hidden_dim)
        self.attn_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.LayerNorm(32), nn.Linear(32, 1)
        )
        self.danger_head = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.ReLU(), nn.LayerNorm(16),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, threat_embeds, contexts, hidden=None):
        if threat_embeds.size(0) == 0:
            dev = threat_embeds.device
            return torch.tensor([0.0], device=dev), torch.empty(0, device=dev), None
        K = threat_embeds.size(0)
        if hidden is None or hidden.size(0) != K:
            hidden = torch.zeros(K, self.hidden_dim, device=threat_embeds.device)
        combined    = torch.cat([threat_embeds, contexts], dim=-1)
        new_hidden  = self.gru(combined, hidden)
        scores      = self.attn_scorer(new_hidden)
        attn_weights = torch.softmax(scores / 0.5, dim=0)
        z      = torch.sum(attn_weights * new_hidden, dim=0, keepdim=True)
        danger = self.danger_head(z).squeeze(-1)
        return danger, attn_weights.squeeze(-1), new_hidden
