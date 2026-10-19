"""
param_count.py
Run: python param_count.py
Saves result to paper_data/v2_gru/model_params_v2.txt
"""
import os
import torch
import torch.nn as nn
from torch_geometric.nn import MetaLayer
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

# ── Tier 2 GRU architecture only (no pygame) ──────────────────
class EdgeModel(nn.Module):
    def __init__(self, node_dim=64, edge_dim=32, msg_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, msg_dim), nn.ReLU(), nn.LayerNorm(msg_dim)
        )

    def forward(self, src, dest, edge_attr, u, batch):
        return self.mlp(torch.cat([src, dest, edge_attr], dim=-1))


class NodeModel(nn.Module):
    def __init__(self, node_dim=64, msg_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim + msg_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, node_dim), nn.ReLU(), nn.LayerNorm(node_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        agg = scatter(edge_attr, col, dim=0, dim_size=x.size(0), reduce='sum')
        return self.mlp(torch.cat([x, agg], dim=-1))


class ThreatGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_enc = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.LayerNorm(64))
        self.edge_enc = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.LayerNorm(32))
        self.layers = nn.ModuleList([
            MetaLayer(EdgeModel(64, 32, 32), NodeModel(64, 32)),
            MetaLayer(EdgeModel(64, 32, 32), NodeModel(64, 32)),
            MetaLayer(EdgeModel(64, 32, 32), NodeModel(64, 32))
        ])
        self.pool_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU())

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_enc(x)
        edge_attr = self.edge_enc(edge_attr)
        for layer in self.layers:
            x, edge_attr, _ = layer(x, edge_index, edge_attr, None, batch)
        pooled = scatter(x, batch, dim=0, reduce='mean')
        return self.pool_head(pooled)


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
        combined = torch.cat([threat_embeds, contexts], dim=-1)
        new_hidden = self.gru(combined, hidden)
        scores = self.attn_scorer(new_hidden)
        attn_weights = torch.softmax(scores / 0.5, dim=0)
        z = torch.sum(attn_weights * new_hidden, dim=0, keepdim=True)
        danger = self.danger_head(z).squeeze(-1)
        return danger, attn_weights.squeeze(-1), new_hidden


MANEUVER_NAMES = ["NORMAL", "BARREL_ROLL", "JINKING", "FALLING_LEAF", "COBRA", "IMMELMANN"]

class NeuralEvasionBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.threat_gnn    = ThreatGNN()
        self.attention     = TemporalThreatAttention()
        self.maneuver_head = nn.Sequential(
            nn.Linear(65, 32), nn.ReLU(), nn.LayerNorm(32),
            nn.Linear(32, len(MANEUVER_NAMES))
        )

    def forward(self, sub_graphs_list, contexts, hidden=None):
        if not sub_graphs_list:
            device = contexts.device if contexts is not None else torch.device('cpu')
            return (torch.tensor([0.0], device=device),
                    torch.empty(0, device=device),
                    None,
                    torch.zeros(len(MANEUVER_NAMES), device=device))
        batched = Batch.from_data_list(sub_graphs_list)
        embeds  = self.threat_gnn(batched)
        danger, attn_weights, new_hidden = self.attention(embeds, contexts, hidden)
        pooled  = new_hidden.mean(dim=0, keepdim=True)
        d_s     = danger.unsqueeze(0) if danger.dim() == 0 else danger.reshape(1, 1)
        m_in    = torch.cat([pooled, d_s], dim=-1)
        m_logits = self.maneuver_head(m_in).squeeze(0)
        return danger, attn_weights, new_hidden, m_logits


model = NeuralEvasionBrain()
try:
    model.load_state_dict(torch.load('jet_brain_v3.pt', map_location='cpu'), strict=False)
    print('Weights loaded from jet_brain_v3.pt')
except FileNotFoundError:
    print('jet_brain_v3.pt not found — counting architecture params only (same number regardless)')
except RuntimeError as e:
    print('jet_brain_v3.pt is not compatible with this GRU architecture.\nCounting parameters from the model definition only.')
    print('Load error:', e)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

lines = [
    'Architecture: "ThreatGNN + TemporalThreatAttention (GRU hidden_dim=64) + ManeuverHead, curriculum 5-stage"',
    f'Total parameters:     {total:,}',
    f'Trainable parameters: {trainable:,}',
    '',
    'Layer breakdown:',
]
for name, p in model.named_parameters():
    lines.append(f'  {name:<55} {p.numel():>8,}')

output = '\n'.join(lines)
print('\n' + output)

os.makedirs('paper_data/v2_gru', exist_ok=True)
with open('paper_data/v2_gru/model_params_v2.txt', 'w') as f:
    f.write(output + '\n')
print('\nSaved to paper_data/v2_gru/model_params_v2.txt')
