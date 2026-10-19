"""
NeuralEvasionBrain — the full inference stack.

    sub-graphs -> ThreatGNN -> TemporalThreatAttention -> danger score
                                                        -> maneuver logits

Copied verbatim from main3.py section 1.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from ..config import MANEUVER_NAMES
from .attention import TemporalThreatAttention
from .layers import ThreatGNN


class NeuralEvasionBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.threat_gnn = ThreatGNN()
        self.attention  = TemporalThreatAttention()
        # Maneuver head: 64 (GRU hidden pooled) + 1 (danger) → 6 logits
        self.maneuver_head = nn.Sequential(
            nn.Linear(65, 32), nn.ReLU(), nn.LayerNorm(32),
            nn.Linear(32, len(MANEUVER_NAMES))
        )

    def forward(self, sub_graphs_list, contexts, hidden=None):
        if not sub_graphs_list:
            dev = contexts.device if contexts is not None else torch.device('cpu')
            return (torch.tensor([0.0], device=dev),
                    torch.empty(0, device=dev),
                    None,
                    torch.zeros(len(MANEUVER_NAMES), device=dev))
        batched = Batch.from_data_list(sub_graphs_list)
        embeds  = self.threat_gnn(batched)
        danger, attn_weights, new_hidden = self.attention(embeds, contexts, hidden)
        # Pool GRU hidden states for maneuver head
        pooled      = new_hidden.mean(dim=0, keepdim=True)       # [1, 64]
        m_input     = torch.cat([pooled, danger.unsqueeze(-1).unsqueeze(0)
                                 if danger.dim()==0 else danger.unsqueeze(-1).reshape(1,1)], dim=-1)  # [1, 65]
        maneuver_logits = self.maneuver_head(m_input).squeeze(0) # [6]
        return danger, attn_weights, new_hidden, maneuver_logits
