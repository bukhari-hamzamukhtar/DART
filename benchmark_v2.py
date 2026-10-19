"""
benchmark.py  —  v2_gru survival benchmark
Run: python benchmark.py
Saves to paper_data/v2_gru/survival_benchmark_v2.txt + .json

Architecture: ThreatGNN + TemporalThreatAttention (GRU) + ManeuverHead
Weights:      jet_brain_v3.pt
Mirrors main3.py inference exactly, including GRU hidden state management.
"""

import os, math, random, time, json, statistics
import torch
import torch.nn as nn
from torch_geometric.nn import MetaLayer
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter

# ── CONFIG ────────────────────────────────────────────────────
NUM_TRIALS     = 50
FRAMES_PER_SIM = 300
GNN_INTERVAL   = 6
WIDTH, HEIGHT  = 800, 600
JET_RADIUS     = 11
MISSILE_RADIUS = 3
FLARE_RADIUS   = 6
HIT_DISTANCE   = JET_RADIUS + MISSILE_RADIUS
WEIGHTS_FILE   = "jet_brain_v3.pt"
MANEUVER_NAMES = ["NORMAL", "BARREL_ROLL", "JINKING", "FALLING_LEAF", "COBRA", "IMMELMANN"]
# ─────────────────────────────────────────────────────────────

# ── ARCHITECTURE — must match main3.py exactly ────────────────
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
        self.layers   = nn.ModuleList([
            MetaLayer(EdgeModel(64, 32, 32), NodeModel(64, 32)),
            MetaLayer(EdgeModel(64, 32, 32), NodeModel(64, 32)),
            MetaLayer(EdgeModel(64, 32, 32), NodeModel(64, 32))
        ])
        self.pool_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x         = self.node_enc(x)
        edge_attr = self.edge_enc(edge_attr)
        for layer in self.layers:
            x, edge_attr, _ = layer(x, edge_index, edge_attr, None, batch)
        pooled = scatter(x, batch, dim=0, reduce='mean')
        return self.pool_head(pooled)

class TemporalThreatAttention(nn.Module):
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
        scores       = self.attn_scorer(new_hidden)                    # [K, 1]
        attn_weights = torch.softmax(scores / 0.5, dim=0)             # [K, 1]
        z            = torch.sum(attn_weights * new_hidden, dim=0, keepdim=True)  # [1, 64]
        danger       = self.danger_head(z).squeeze(-1)
        return danger, attn_weights.squeeze(-1), new_hidden

class NeuralEvasionBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.threat_gnn    = ThreatGNN()
        self.attention     = TemporalThreatAttention()
        # ManeuverHead: pool(GRU hidden)[64] + danger[1] → 6 logits
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

        # Maneuver head
        pooled          = new_hidden.mean(dim=0, keepdim=True)             # [1, 64]
        danger_scalar   = danger.unsqueeze(0) if danger.dim() == 0 else danger.reshape(1, 1)
        m_input         = torch.cat([pooled, danger_scalar], dim=-1)       # [1, 65]
        maneuver_logits = self.maneuver_head(m_input).squeeze(0)           # [6]
        return danger, attn_weights, new_hidden, maneuver_logits
# ─────────────────────────────────────────────────────────────

# ── GRAPH BUILDER ─────────────────────────────────────────────
def build_candidate_sub_graphs(jet_sim, missiles_sim, flares_sim):
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
            pos.append(torch.tensor(p))
            vel.append(torch.tensor(v))
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
        dist = math.hypot(jet_sim['pos'][0]-m['pos'][0], jet_sim['pos'][1]-m['pos'][1])
        nd   = min(dist/800.0, 1.0)
        dx   = jet_sim['pos'][0] - m['pos'][0]
        dy   = jet_sim['pos'][1] - m['pos'][1]
        rvx  = m['vel'][0] - jet_sim['vel'][0]
        rvy  = m['vel'][1] - jet_sim['vel'][1]
        closing      = (dx*rvx + dy*rvy) / (dist + 1e-5)
        closing_norm = max(-1.0, min(1.0, closing / 10.0))
        tti          = min((dist / closing) / 60.0, 1.0) if closing > 0.05 else 1.0
        mh_len       = math.hypot(rvx, rvy) + 1e-5
        aoa          = max(0.0, (dx*rvx + dy*rvy) / ((dist+1e-5) * mh_len))
        # Node 'type' is the role (jet 0 / missile 1 / flare 2). The context's
        # mtype is the guidance type (0 homing, 1 ballistic) -- training kept
        # these separate, see architecture.ipynb cell 4.
        mtype        = float(m.get('mtype', 0))
        ctx          = torch.tensor([nd, tti, closing_norm, aoa, mtype], dtype=torch.float)
        sub_graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, context=ctx))
    return sub_graphs

def normalize_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ── GNN HEADING PICKER ────────────────────────────────────────
def gnn_pick_heading(model, jet, missiles, flares, gru_hidden, frame):
    jx, jy     = jet['pos']
    candidates = [0, 0.20, -0.20, 0.40, -0.40, 0.65, -0.65, 1.0, -1.0, 1.4, -1.4, math.pi]
    best_heading  = jet['heading']
    lowest_total  = 999.0
    t_start       = time.perf_counter()

    for delta in candidates:
        th  = jet['heading'] + delta
        tvx = math.cos(th) * 4.0
        tvy = math.sin(th) * 4.0
        px  = jx + tvx * 30
        py  = jy + tvy * 30
        if px < 100 or px > 700 or py < 100 or py > 450:
            total_danger = 10.0
        else:
            js = {'pos': [jx, jy], 'vel': [tvx, tvy], 'type': 0}
            ms = [{'pos': m['pos'], 'vel': m['vel'], 'type': 1,
                   'mtype': m.get('type', 0)} for m in missiles]
            fs = [{'pos': f['pos'], 'vel': f['vel'], 'type': 2} for f in flares]
            if ms:
                sgs  = build_candidate_sub_graphs(js, ms, fs)
                ctxs = torch.stack([g.context for g in sgs])
                with torch.no_grad():
                    # 4-value unpack — candidate evals read hidden state but DON'T update it
                    dnn_danger, _, _, _ = model(sgs, ctxs, gru_hidden)
                gnn_d = dnn_danger.item()
            else:
                gnn_d = 0.0
            esc = 0.0
            for m in ms:
                cd = math.hypot(jx-m['pos'][0], jy-m['pos'][1])
                pd = math.hypot(px-m['pos'][0], py-m['pos'][1])
                esc += -0.04 if pd > cd else 0.04
            cp           = (math.hypot(px-400, py-275)/800.0) * 0.10
            total_danger = gnn_d + esc + cp

        if total_danger < lowest_total:
            lowest_total = total_danger
            best_heading = th

    # ── Real forward pass: updates GRU hidden state + gets maneuver ──
    new_hidden       = gru_hidden
    current_danger   = 0.0
    recommended_manu = "NORMAL"
    if missiles:
        js_real = {'pos': [jx, jy], 'vel': jet['vel'], 'type': 0}
        ms_real = [{'pos': m['pos'], 'vel': m['vel'], 'type': 1,
                   'mtype': m.get('type', 0)} for m in missiles]
        fs_real = [{'pos': f['pos'], 'vel': f['vel'], 'type': 2} for f in flares]
        sgs_real = build_candidate_sub_graphs(js_real, ms_real, fs_real)
        if sgs_real:
            ctxs_real = torch.stack([g.context for g in sgs_real])
            with torch.no_grad():
                d_real, _, new_hidden, m_logits = model(sgs_real, ctxs_real, gru_hidden)
            current_danger   = d_real.item()
            if frame % 30 == 0:
                    print(f"[DBG] frame={frame} danger={current_danger:.4f} missiles={len(missiles)}")
            recommended_manu = MANEUVER_NAMES[m_logits.argmax().item()]

    inference_ms = (time.perf_counter() - t_start) * 1000.0
    drop_flare   = (current_danger > 0.25) and len(missiles) > 0
    return best_heading, drop_flare, inference_ms, new_hidden, current_danger, recommended_manu

# ── MAIN BENCHMARK LOOP ───────────────────────────────────────
def run_benchmark(model):
    print(f"\nRunning {NUM_TRIALS} benchmark trials...")
    results = []

    for trial in range(NUM_TRIALS):
        jet = {
            'pos':     [random.uniform(200, WIDTH-200), random.uniform(150, HEIGHT-150)],
            'vel':     [0.0, 0.0],
            'heading': random.uniform(0, 2*math.pi),
            'speed':   5.0,
            'drag':    0.92
        }
        num_missiles = random.randint(1, 5)
        is_hot       = random.random() < 0.30
        missiles = []
        for _ in range(num_missiles):
            m_type = 0 if random.random() < 0.75 else 1
            if is_hot:
                angle = random.uniform(0, 2*math.pi)
                dist  = random.uniform(80, 180)
                mx    = clamp(jet['pos'][0] + math.cos(angle)*dist, 20, WIDTH-20)
                my    = clamp(jet['pos'][1] + math.sin(angle)*dist, 20, HEIGHT-20)
            else:
                mx = random.choice([random.uniform(0, 50), random.uniform(WIDTH-50, WIDTH)])
                my = random.choice([random.uniform(0, 50), random.uniform(HEIGHT-50, HEIGHT)])
            missiles.append({
                'pos':     [mx, my],
                'vel':     [0.0, 0.0],
                'heading': math.atan2(jet['pos'][1]-my, jet['pos'][0]-mx),
                'speed':   random.uniform(3.5, 8.0) if is_hot else random.uniform(3.5, 7.0),
                'type':    m_type,
                'dead':    False
            })

        flares            = []
        next_flare_frame  = random.randint(120, 240)
        flares_deployed   = 0
        latencies         = []
        maneuver_counts   = {m: 0 for m in MANEUVER_NAMES}
        hit_frame         = -1
        gru_hidden        = None
        prev_missile_count = num_missiles

        for frame in range(FRAMES_PER_SIM):
            live_m = [m for m in missiles if not m['dead']]

            # Reset GRU if missile count changed
            if len(live_m) != prev_missile_count:
                gru_hidden = None
            prev_missile_count = len(live_m)

            # GNN decision every GNN_INTERVAL frames
            if frame % GNN_INTERVAL == 0:
                best_h, drop_f, lat, gru_hidden, cur_danger, rec_manu = gnn_pick_heading(
                    model, jet, live_m, flares, gru_hidden, frame
                )
                latencies.append(lat)
                jet['heading'] = best_h
                maneuver_counts[rec_manu] = maneuver_counts.get(rec_manu, 0) + 1

                if drop_f:
                    start_angle = jet['heading'] + math.radians(45)
                    for i in range(8):
                        angle = start_angle + i * math.radians(270) / 7
                        spd   = jet['speed'] + 2.0
                        flares.append({
                            'pos':  [jet['pos'][0], jet['pos'][1]],
                            'vel':  [math.cos(angle)*spd, math.sin(angle)*spd],
                            'drag': 0.85, 'life': 120
                        })
                    flares_deployed  += 1
                    next_flare_frame += random.randint(120, 240)

            # Jet physics
            jet['vel'][0] = jet['vel'][0]*jet['drag'] + math.cos(jet['heading'])*jet['speed']*0.1
            jet['vel'][1] = jet['vel'][1]*jet['drag'] + math.sin(jet['heading'])*jet['speed']*0.1
            jet['pos'][0] = clamp(jet['pos'][0] + jet['vel'][0], 30, WIDTH-30)
            jet['pos'][1] = clamp(jet['pos'][1] + jet['vel'][1], 30, HEIGHT-30)

            # Flare update
            live_flares = []
            for f in flares:
                f['vel'][0] *= f['drag']; f['vel'][1] *= f['drag']
                f['pos'][0] += f['vel'][0]; f['pos'][1] += f['vel'][1]
                f['life'] -= 1
                if f['life'] > 0:
                    live_flares.append(f)
            flares = live_flares

            # Missile update
            hit = False
            for m in missiles:
                if m['dead']:
                    continue
                if m['type'] == 0:
                    dx = jet['pos'][0] - m['pos'][0]
                    dy = jet['pos'][1] - m['pos'][1]
                    desired = math.atan2(dy, dx)
                    diff    = normalize_angle(desired - m['heading'])
                    m['heading'] += max(-0.15, min(0.15, diff))
                m['vel'][0] = math.cos(m['heading']) * m['speed']
                m['vel'][1] = math.sin(m['heading']) * m['speed']
                m['pos'][0] += m['vel'][0]
                m['pos'][1] += m['vel'][1]
                if not (0 <= m['pos'][0] <= WIDTH and 0 <= m['pos'][1] <= HEIGHT):
                    m['dead'] = True; continue
                if math.hypot(jet['pos'][0]-m['pos'][0], jet['pos'][1]-m['pos'][1]) < HIT_DISTANCE:
                    hit = True; m['dead'] = True; continue
                for f in flares:
                    if math.hypot(f['pos'][0]-m['pos'][0], f['pos'][1]-m['pos'][1]) < (MISSILE_RADIUS+FLARE_RADIUS):
                        m['dead'] = True; f['life'] = 0; break

            if hit:
                hit_frame = frame
                break

        survived_frames = hit_frame if hit_frame >= 0 else FRAMES_PER_SIM
        top_maneuver    = max(maneuver_counts, key=maneuver_counts.get)
        results.append({
            'trial':            trial + 1,
            'hit':              hit_frame >= 0,
            'survived_frames':  survived_frames,
            'flares_deployed':  flares_deployed,
            'avg_latency_ms':   statistics.mean(latencies) if latencies else 0.0,
            'num_missiles':     num_missiles,
            'maneuver_counts':  maneuver_counts,
            'top_maneuver':     top_maneuver,
        })

        status = f"HIT  @ frame {hit_frame:>3}" if hit_frame >= 0 else f"SURVIVED ({FRAMES_PER_SIM} frames)"
        print(f"  Trial {trial+1:>2}: {status} | missiles={num_missiles} | "
              f"flares={flares_deployed} | lat={results[-1]['avg_latency_ms']:.1f}ms | "
              f"top_manu={top_maneuver}")

    return results

def save_results(results):
    hits          = [r for r in results if r['hit']]
    hit_rate      = 100.0 * len(hits) / len(results)
    survival_rate = 100.0 - hit_rate
    avg_frames    = statistics.mean(r['survived_frames'] for r in results)
    avg_lat       = statistics.mean(r['avg_latency_ms'] for r in results)
    med_lat       = statistics.median(r['avg_latency_ms'] for r in results)
    avg_flares    = statistics.mean(r['flares_deployed'] for r in results)

    # Aggregate maneuver distribution across all trials
    total_manu = {m: 0 for m in MANEUVER_NAMES}
    for r in results:
        for m, cnt in r['maneuver_counts'].items():
            total_manu[m] = total_manu.get(m, 0) + cnt
    total_decisions = sum(total_manu.values()) or 1

    lines = [
        "=" * 65,
        "V2_GRU — ThreatGNN + TemporalThreatAttention (GRU) + ManeuverHead",
        "Curriculum: 5-stage (slow_singles → fast_homing → multi_threat → cluster_swarm → adversarial)",
        "Weights: jet_brain_v3.pt",
        "=" * 65,
        f"Trials:              {len(results)}",
        f"Hit rate:            {hit_rate:.1f}%",
        f"Survival rate:       {survival_rate:.1f}%",
        f"Avg frames survived: {avg_frames:.1f} / {FRAMES_PER_SIM}",
        f"Avg flares/trial:    {avg_flares:.2f}",
        f"Avg inference (ms):  {avg_lat:.2f}",
        f"Median latency (ms): {med_lat:.2f}",
        "",
        "GNN Maneuver Recommendation Distribution:",
    ]
    for m in MANEUVER_NAMES:
        pct = 100.0 * total_manu.get(m, 0) / total_decisions
        lines.append(f"  {m:<14} {total_manu.get(m,0):>6} decisions  ({pct:.1f}%)")

    lines += [
        "",
        "Per-trial log:",
        f"{'Trial':>6}  {'Hit':>5}  {'Frames':>7}  {'Flares':>7}  {'Lat(ms)':>8}  {'Missiles':>9}  {'TopManeuver':<14}",
    ]
    for r in results:
        lines.append(
            f"{r['trial']:>6}  {'YES' if r['hit'] else 'NO':>5}  "
            f"{r['survived_frames']:>7}  {r['flares_deployed']:>7}  "
            f"{r['avg_latency_ms']:>8.2f}  {r['num_missiles']:>9}  {r['top_maneuver']:<14}"
        )

    output = "\n".join(lines)
    print("\n" + output)

    os.makedirs("paper_data/v2_gru", exist_ok=True)
    with open("paper_data/v2_gru/survival_benchmark_v2.txt", "w", encoding="utf-8") as f:
        f.write(output + "\n")
    with open("paper_data/v2_gru/survival_benchmark_v2.json", "w") as f:
        json.dump({
            "version":              "v2_gru",
            "architecture":         "ThreatGNN + TemporalThreatAttention (GRU, hidden=64) + ManeuverHead",
            "training":             "5-stage curriculum, jet_brain_v3.pt",
            "trials":               len(results),
            "hit_rate_pct":         round(hit_rate, 2),
            "survival_rate_pct":    round(survival_rate, 2),
            "avg_survived_frames":  round(avg_frames, 2),
            "avg_flares_per_trial": round(avg_flares, 2),
            "avg_inference_ms":     round(avg_lat, 2),
            "median_inference_ms":  round(med_lat, 2),
            "maneuver_distribution": {m: total_manu.get(m, 0) for m in MANEUVER_NAMES},
            "per_trial":            results
        }, f, indent=2)

    print("\nSaved:")
    print("  paper_data/v2_gru/survival_benchmark_v2.txt")
    print("  paper_data/v2_gru/survival_benchmark_v2.json")

# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device('cpu')
    model  = NeuralEvasionBrain().to(device)
    try:
        # 1. Load the raw dictionary from the file
        checkpoint = torch.load(WEIGHTS_FILE, map_location=device)
        model_dict = model.state_dict()

        # 2. Filter out anything that doesn't have the exact same shape
        filtered_dict = {k: v for k, v in checkpoint.items() 
                        if k in model_dict and v.shape == model_dict[k].shape}

        # 3. Update the model's dictionary and load it safely
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f"Weights loaded: {WEIGHTS_FILE} (Mismatched shapes were skipped!)")
    
    except FileNotFoundError:
        print(f"WARNING: {WEIGHTS_FILE} not found — running with random weights.")
    model.eval()

    results = run_benchmark(model)
    save_results(results)
