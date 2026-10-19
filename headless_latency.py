"""
headless_latency.py  —  v2_gru inference latency benchmark
Run: python headless_latency.py
Saves to paper_data/v2_gru/inference_latency_v2.txt

Architecture: ThreatGNN + TemporalThreatAttention (GRU hidden=64) + ManeuverHead
Weights:      jet_brain_v3.pt
Mirrors main3.py inference exactly, including 5-dim context and 4-return forward.
"""

import math
import random
import time
import os
import statistics
import torch
import torch.nn as nn
from torch_geometric.nn import MetaLayer
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter

# ── CONFIG ────────────────────────────────────────────────────
WEIGHTS_FILE    = "jet_brain_v3.pt"
DURATION_SECS   = 60          # how long to run the benchmark
MANEUVER_NAMES  = ["NORMAL", "BARREL_ROLL", "JINKING", "FALLING_LEAF", "COBRA", "IMMELMANN"]

# ── ARCHITECTURE  (must match main3.py exactly) ───────────────
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
        self.node_enc  = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.LayerNorm(64))
        self.edge_enc  = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.LayerNorm(32))
        self.layers    = nn.ModuleList([
            MetaLayer(EdgeModel(64, 32, 32), NodeModel(64, 32)),
            MetaLayer(EdgeModel(64, 32, 32), NodeModel(64, 32)),
            MetaLayer(EdgeModel(64, 32, 32), NodeModel(64, 32)),
        ])
        self.pool_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU())

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch)
        x         = self.node_enc(x)
        edge_attr = self.edge_enc(edge_attr)
        for layer in self.layers:
            x, edge_attr, _ = layer(x, edge_index, edge_attr, None, batch)
        pooled = scatter(x, batch, dim=0, reduce='mean')
        return self.pool_head(pooled)   # [K, 32]


class TemporalThreatAttention(nn.Module):
    """GRU-based attention with per-missile temporal memory."""
    def __init__(self, embed_dim=32, context_dim=5, hidden_dim=64):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.gru         = nn.GRUCell(embed_dim + context_dim, hidden_dim)
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
        combined     = torch.cat([threat_embeds, contexts], dim=-1)   # [K, 37]
        new_hidden   = self.gru(combined, hidden)                      # [K, 64]
        scores       = self.attn_scorer(new_hidden)                    # [K,  1]
        attn_weights = torch.softmax(scores / 0.5, dim=0)             # [K,  1]
        z            = torch.sum(attn_weights * new_hidden, dim=0, keepdim=True)  # [1, 64]
        danger       = self.danger_head(z).squeeze(-1)
        return danger, attn_weights.squeeze(-1), new_hidden


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
        """
        Returns: (danger, attn_weights, new_hidden, maneuver_logits)
        Matches main3.py API exactly.
        """
        if not sub_graphs_list:
            dev = contexts.device if contexts is not None else torch.device('cpu')
            return (torch.tensor([0.0], device=dev),
                    torch.empty(0, device=dev),
                    None,
                    torch.zeros(len(MANEUVER_NAMES), device=dev))
        batched = Batch.from_data_list(sub_graphs_list)
        embeds  = self.threat_gnn(batched)
        danger, attn_weights, new_hidden = self.attention(embeds, contexts, hidden)
        pooled        = new_hidden.mean(dim=0, keepdim=True)           # [1, 64]
        d_scalar      = danger.unsqueeze(0) if danger.dim() == 0 else danger.reshape(1, 1)
        m_input       = torch.cat([pooled, d_scalar], dim=-1)          # [1, 65]
        maneuver_logits = self.maneuver_head(m_input).squeeze(0)       # [6]
        return danger, attn_weights, new_hidden, maneuver_logits


# ── GRAPH BUILDER  (mirrors main3.py / notebook Cell 4 exactly) ──
def build_candidate_sub_graphs(jet_sim, missiles_sim, flares_sim):
    """
    Context vector per missile (5-dim):
        [norm_dist, tti_norm, closing_norm, aoa, mtype]
    """
    sub_graphs = []
    for m in missiles_sim:
        nearby_flares = [
            f for f in flares_sim
            if math.hypot(f['pos'][0]-m['pos'][0], f['pos'][1]-m['pos'][1]) < 200
        ][:6]
        nodes = [jet_sim, m] + nearby_flares
        x_rows, pos, vel = [], [], []
        for node in nodes:
            p = [node['pos'][0], node['pos'][1]]
            v = [node['vel'][0], node['vel'][1]]
            x_rows.append(p + v + [float(node['type'])])
            pos.append(torch.tensor(p, dtype=torch.float))
            vel.append(torch.tensor(v, dtype=torch.float))
        x  = torch.tensor(x_rows, dtype=torch.float)
        n  = len(nodes)
        ei, ea = [], []
        for i in range(n):
            for j in range(n):
                if i == j: continue
                ei.append([i, j])
                ea.append(torch.cat([pos[i]-pos[j], vel[i]-vel[j]], dim=0))
        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
        edge_attr  = torch.stack(ea, dim=0).float()

        dx   = jet_sim['pos'][0] - m['pos'][0]
        dy   = jet_sim['pos'][1] - m['pos'][1]
        dist = math.hypot(dx, dy) + 1e-5
        nd   = min(dist / 800.0, 1.0)
        rvx  = m['vel'][0] - jet_sim['vel'][0]
        rvy  = m['vel'][1] - jet_sim['vel'][1]
        closing      = (dx * rvx + dy * rvy) / dist
        closing_norm = max(-1.0, min(1.0, closing / 10.0))
        tti          = min((dist / closing) / 60.0, 1.0) if closing > 0.05 else 1.0
        mh_len       = math.hypot(rvx, rvy) + 1e-5
        aoa          = max(0.0, (dx * rvx + dy * rvy) / (dist * mh_len))
        # Node 'type' is the role (jet 0 / missile 1 / flare 2). The context's
        # mtype is the guidance type (0 homing, 1 ballistic) -- training kept
        # these separate, see architecture.ipynb cell 4.
        mtype        = float(m.get('mtype', 0))
        ctx  = torch.tensor([nd, tti, closing_norm, aoa, mtype], dtype=torch.float)
        sub_graphs.append(Data(x=x, edge_index=edge_index,
                               edge_attr=edge_attr, context=ctx))
    return sub_graphs


def sample_state():
    """Generate a random game state for benchmarking."""
    jet = {
        'pos':  [400.0, 300.0],
        'vel':  [4.0, 0.0],
        'heading': 0.0,
        'type': 0,
    }
    missiles = []
    for _ in range(random.randint(1, 4)):
        angle = random.uniform(0, 2 * math.pi)
        dist  = random.uniform(80, 250)
        speed = random.uniform(3.5, 7.5)
        missiles.append({
            'pos':  [jet['pos'][0] + math.cos(angle)*dist,
                     jet['pos'][1] + math.sin(angle)*dist],
            'vel':  [math.cos(angle)*speed, math.sin(angle)*speed],
            'type':  1,                          # node role: missile
            'mtype': random.choice([0, 1]),      # guidance: 0=homing, 1=ballistic
        })
    flares = []
    for _ in range(random.randint(0, 3)):
        angle = random.uniform(0, 2 * math.pi)
        dist  = random.uniform(30, 150)
        speed = random.uniform(2.0, 4.0)
        flares.append({
            'pos':  [jet['pos'][0] + math.cos(angle)*dist,
                     jet['pos'][1] + math.sin(angle)*dist],
            'vel':  [math.cos(angle)*speed, math.sin(angle)*speed],
            'type': 2,
        })
    return jet, missiles, flares


def main(duration_seconds=DURATION_SECS):
    device = torch.device('cpu')
    model  = NeuralEvasionBrain().to(device)

    # ── Robust weight loading (shape-filtered) ─────────────────
    try:
        checkpoint   = torch.load(WEIGHTS_FILE, map_location=device)
        model_dict   = model.state_dict()
        filtered     = {k: v for k, v in checkpoint.items()
                        if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        loaded       = len(filtered)
        skipped      = len(checkpoint) - loaded
        print(f"Weights loaded: {WEIGHTS_FILE}  ({loaded} tensors loaded, {skipped} skipped)")
    except FileNotFoundError:
        print(f"WARNING: {WEIGHTS_FILE} not found — benchmarking with random weights.")

    model.eval()

    # ── Latency by missile count ────────────────────────────────
    by_k      = {k: [] for k in range(1, 6)}   # K = number of missiles
    all_lat   = []
    manu_dist = {m: 0 for m in MANEUVER_NAMES}
    gru_hidden = None

    print(f"\nRunning latency benchmark for {duration_seconds}s...")
    t_start    = time.perf_counter()
    iterations = 0

    while time.perf_counter() - t_start < duration_seconds:
        jet, missiles, flares = sample_state()
        sgs  = build_candidate_sub_graphs(jet, missiles, flares)
        if not sgs:
            continue
        ctxs = torch.stack([g.context for g in sgs]).to(device)
        K    = len(sgs)

        # Reset GRU if missile count changed (mirrors main3.py behavior)
        if gru_hidden is not None and gru_hidden.size(0) != K:
            gru_hidden = None

        t0 = time.perf_counter()
        with torch.no_grad():
            danger, attn_w, gru_hidden, m_logits = model(sgs, ctxs, gru_hidden)
        t1 = time.perf_counter()

        lat_ms = (t1 - t0) * 1000.0
        all_lat.append(lat_ms)
        if K in by_k:
            by_k[K].append(lat_ms)

        manu = MANEUVER_NAMES[m_logits.argmax().item()]
        manu_dist[manu] = manu_dist.get(manu, 0) + 1
        iterations += 1

    if iterations == 0:
        raise RuntimeError("No inference iterations completed.")

    avg    = statistics.mean(all_lat)
    med    = statistics.median(all_lat)
    p95    = sorted(all_lat)[int(0.95 * len(all_lat))]
    stdev  = statistics.stdev(all_lat) if len(all_lat) > 1 else 0.0
    total  = sum(manu_dist.values()) or 1

    lines = [
        "=" * 60,
        "v2_gru — ThreatGNN + TemporalThreatAttention (GRU) + ManeuverHead",
        f"Weights:      {WEIGHTS_FILE}",
        f"Device:       CPU",
        f"Context dim:  5  [norm_dist, tti_norm, closing_norm, aoa, mtype]",
        "=" * 60,
        f"Total iterations:    {iterations}",
        f"Avg latency (ms):    {avg:.3f}",
        f"Median latency (ms): {med:.3f}",
        f"Std dev (ms):        {stdev:.3f}",
        f"P95 latency (ms):    {p95:.3f}",
        "",
        "Latency by missile count K:",
    ]
    for k in range(1, 6):
        if by_k[k]:
            lines.append(
                f"  K={k}  n={len(by_k[k]):>5}  "
                f"avg={statistics.mean(by_k[k]):.3f}ms  "
                f"p95={sorted(by_k[k])[int(0.95*len(by_k[k]))]:.3f}ms"
            )
    lines += [
        "",
        "GNN Maneuver Recommendation Distribution:",
    ]
    for m in MANEUVER_NAMES:
        pct = 100.0 * manu_dist.get(m, 0) / total
        lines.append(f"  {m:<16} {manu_dist.get(m,0):>6}  ({pct:.1f}%)")

    output = "\n".join(lines)
    print("\n" + output)

    out_dir = os.path.join("paper_data", "v2_gru")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "inference_latency_v2.txt")
    with open(out_path, "w") as f:
        f.write(output + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
