"""
Graph-network building blocks: per-edge message MLP, per-node update MLP and the
ThreatGNN encoder that turns one (jet + missile + flares) sub-graph into a
32-dim threat embedding.

Copied verbatim from main3.py section 1 — weight shapes must keep matching
jet_brain_v3.pt.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MetaLayer
from torch_geometric.utils import scatter


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
