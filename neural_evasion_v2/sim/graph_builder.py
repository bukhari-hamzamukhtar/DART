"""
Scene -> graph conversion (main3.py section 3).

One sub-graph per missile: nodes are [jet, missile, <=6 nearby flares], edges are
fully connected with relative position/velocity attributes, and a 5-dim context
vector (normalised distance, time-to-impact, closing speed, angle of attack,
missile type) rides alongside for the attention head.
"""

import math

import torch
from torch_geometric.data import Data

from ..config import GUIDANCE_HOMING

FLARE_RADIUS_OF_INTEREST = 200   # only flares this close to a missile join its graph
MAX_FLARE_NODES          = 6


def build_candidate_sub_graphs(jet_sim, missiles_sim, flares_sim):
    sub_graphs = []
    for m in missiles_sim:
        nearby_flares = [
            f for f in flares_sim
            if math.hypot(f['pos'][0]-m['pos'][0], f['pos'][1]-m['pos'][1]) < FLARE_RADIUS_OF_INTEREST
        ][:MAX_FLARE_NODES]
        nodes = [jet_sim, m] + nearby_flares
        x_rows, pos, vel = [], [], []
        for node in nodes:
            p = [node['pos'][0], node['pos'][1]]
            v = [node['vel'][0], node['vel'][1]]
            x_rows.append(p + v + [float(node['type'])])
            pos.append(torch.tensor(p))
            vel.append(torch.tensor(v))
        x = torch.tensor(x_rows, dtype=torch.float)
        n = len(nodes)
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
        # Signed closing speed
        dx   = jet_sim['pos'][0] - m['pos'][0]
        dy   = jet_sim['pos'][1] - m['pos'][1]
        rvx  = m['vel'][0] - jet_sim['vel'][0]
        rvy  = m['vel'][1] - jet_sim['vel'][1]
        closing = (dx*rvx + dy*rvy) / (dist + 1e-5)
        closing_norm = max(-1.0, min(1.0, closing / 10.0))
        if closing > 0.05:
            tti  = min((dist / closing) / 60.0, 1.0)
        else:
            tti  = 1.0
        # Angle of attack
        mh_len = math.hypot(rvx, rvy) + 1e-5
        aoa    = max(0.0, (dx*rvx + dy*rvy) / ((dist+1e-5) * mh_len))
        # Node 'type' is the role (jet 0 / missile 1 / flare 2); the context's
        # mtype is the guidance type (0 homing, 1 ballistic). Training kept these
        # separate -- see architecture.ipynb cell 4 -- so we do too.
        mtype  = float(m.get('mtype', GUIDANCE_HOMING))
        ctx  = torch.tensor([nd, tti, closing_norm, aoa, mtype], dtype=torch.float)
        sub_graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, context=ctx))
    return sub_graphs
