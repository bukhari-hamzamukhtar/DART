"""
Temporal graph network over the dynamic scene graph (the v3 brain).

Built on the Temporal Graph Networks recipe (Rossi et al., 2020):

  memory     one vector per node key, carried from tick to tick and updated by a
             GRU. A node seen for the first time starts from zero; a node that
             leaves the graph has its memory deleted.
  embedding  typed message passing over the current snapshot, reading both each
             node's features and its memory, so a node whose own memory is stale
             is refreshed from its neighbours.
  heads      per track -- is it guided at all, how hard it turns, how much it
             leads, which flank it aims for and how far off, how much it weaves
             or darts, does it hold a lock now, will it threaten the jet; per
             LINK edge -- is this datalink live.

None of those quantities is visible on the radar. Turn rate only shows while the
missile is turning; a flight style only shows in how the track moves over many
ticks; lock state and datalink membership only show in how a missile's behaviour
correlates with terrain, flares and other missiles over time. That is what needs
both the graph and the memory, and the ablations (`use_edges=False`,
`use_memory=False`) exist to measure how much each one buys.
"""

import json
import os

import torch
import torch.nn as nn

from ..config import PROJECT_ROOT
from ..sim.scene_graph import EDGE_DIM, EDGE_TYPES, NODE_DIM

HIDDEN      = 64
EDGE_EMB    = 24
N_TRACK_OUT = 11         # pursuit, turn rate, lead, which flank side, guided, threat,
                         # turn rate (90% quantile), lateral bias,
                         # flank angle, weave/dart swing, has a flank side
MAX_TURN    = 0.2        # rad/frame; the turn-rate heads are scaled into [0, MAX_TURN]
TURN_QUANTILE = 0.9      # the planner flies against this quantile of turn rate
MAX_BIAS    = 0.03       # rad/frame; persistent curve of a cluster missile
MAX_FLANK   = 1.0        # tan of the flank deviation; head range
MAX_WOBBLE  = 1.2        # tan of weave + dart swing; head range
CHECKPOINT  = "jet_brain_tgn.pt"


class MessageLayer(nn.Module):
    """One round of typed, gated message passing with a residual update."""

    def __init__(self, hidden, edge_dim):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden))
        self.gate = nn.Linear(2 * hidden + edge_dim, 1)
        self.upd = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, src, dst, e):
        z = torch.cat([h[src], h[dst], e], dim=-1)
        gate = torch.sigmoid(self.gate(z))
        m = self.msg(z) * gate
        agg = torch.zeros_like(h).index_add_(0, dst, m)
        deg = torch.zeros(h.size(0), 1, dtype=h.dtype).index_add_(
            0, dst, torch.ones(dst.size(0), 1, dtype=h.dtype))
        agg = agg / deg.clamp(min=1.0).sqrt()
        return self.norm(h + self.upd(torch.cat([h, agg], dim=-1))), gate.squeeze(-1)


class TemporalGraphNet(nn.Module):
    def __init__(self, hidden=HIDDEN, layers=2, use_edges=True, use_memory=True):
        super().__init__()
        self.hidden = hidden
        self.use_edges = use_edges
        self.use_memory = use_memory
        self.node_in = nn.Sequential(nn.Linear(NODE_DIM, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden))
        self.mem_in = nn.Linear(hidden, hidden, bias=False)
        self.in_norm = nn.LayerNorm(hidden)
        self.etype_emb = nn.Embedding(len(EDGE_TYPES), EDGE_EMB)
        self.eattr_in = nn.Linear(EDGE_DIM, EDGE_EMB)
        self.layers = nn.ModuleList([MessageLayer(hidden, EDGE_EMB) for _ in range(layers)])
        self.memory = nn.GRUCell(hidden, hidden)
        self.track_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                        nn.Linear(hidden, N_TRACK_OUT))
        self.link_head = nn.Sequential(nn.Linear(4 * hidden + 2 * EDGE_EMB, hidden), nn.ReLU(),
                                       nn.Linear(hidden, 1))

    def config(self):
        return {'hidden': self.hidden, 'layers': len(self.layers),
                'use_edges': self.use_edges, 'use_memory': self.use_memory}

    def forward(self, x, src, dst, etype, eattr, mem):
        """Returns (z, new_memory, edge_embeddings, last-layer edge gates)."""
        h = self.node_in(x)
        if self.use_memory:
            h = h + self.mem_in(mem)
        h = self.in_norm(h)
        e = torch.relu(self.etype_emb(etype) + self.eattr_in(eattr))
        gates = None
        if self.use_edges and src.numel():
            for layer in self.layers:
                h, gates = layer(h, src, dst, e)
        new_mem = self.memory(h, mem) if self.use_memory else torch.zeros_like(mem)
        return torch.cat([h, new_mem], dim=-1), new_mem, e, gates

    def link_logits(self, z, e, src, dst, rows):
        """Symmetric logit per LINK pair; `rows` index the forward edge, row+1 is its reverse."""
        a, b = src[rows], dst[rows]
        fwd = torch.cat([z[a], z[b], e[rows], e[rows + 1]], dim=-1)
        rev = torch.cat([z[b], z[a], e[rows + 1], e[rows]], dim=-1)
        return 0.5 * (self.link_head(fwd) + self.link_head(rev)).squeeze(-1)


def decode_tracks(raw):
    """Raw head outputs -> named estimates (tensors, one entry per track)."""
    return {
        'pursuit':   torch.sigmoid(raw[:, 0]),
        'turn_rate': torch.sigmoid(raw[:, 1]) * MAX_TURN,
        'lead':      torch.sigmoid(raw[:, 2]),
        'side':      torch.sign(raw[:, 3]) * (raw[:, 10] > 0.0).float(),   # -1 / 0 / +1
        'guided':    torch.sigmoid(raw[:, 4]),
        'threat':    torch.sigmoid(raw[:, 5]),
        'turn_hi':   torch.sigmoid(raw[:, 6]) * MAX_TURN,
        'bias':      torch.tanh(raw[:, 7]) * MAX_BIAS,
        'flank':     torch.sigmoid(raw[:, 8]) * MAX_FLANK,
        'wobble':    torch.sigmoid(raw[:, 9]) * MAX_WOBBLE,
    }


def graph_tensors(g):
    return (torch.from_numpy(g.x), torch.from_numpy(g.src), torch.from_numpy(g.dst),
            torch.from_numpy(g.etype), torch.from_numpy(g.eattr))


class GraphBrain(object):
    """
    Runtime wrapper: holds the per-node memory for one match and turns a scene
    graph into per-track estimates.
    """

    def __init__(self, model):
        self.model = model.eval()
        self.memory = {}
        self.zero = torch.zeros(model.hidden)

    def reset(self):
        self.memory = {}

    @torch.no_grad()
    def step(self, g):
        n = len(g.keys)
        if n == 0:
            return [], [], None
        mem = torch.stack([self.memory.get(k, self.zero) for k in g.keys])
        x, src, dst, et, ea = graph_tensors(g)
        z, new_mem, e, gates = self.model(x, src, dst, et, ea, mem)
        # Nodes absent from this snapshot are deleted from memory (TGN node deletion).
        self.memory = {k: new_mem[i] for i, k in enumerate(g.keys)}

        rows = []
        if g.track_rows:
            est = decode_tracks(self.model.track_head(z[g.track_rows]))
            cols = {k: v.tolist() for k, v in est.items()}
            for i in range(len(g.track_rows)):
                rows.append({k: cols[k][i] for k in cols})

        links = []
        if g.link_rows:
            idx = torch.tensor(g.link_rows, dtype=torch.long)
            p = torch.sigmoid(self.model.link_logits(z, e, src, dst, idx)).tolist()
            tid_of = dict(zip(g.track_rows, g.track_tids))
            for r, pr in zip(g.link_rows, p):
                links.append((tid_of[int(g.src[r])], tid_of[int(g.dst[r])], pr))
        return rows, links, (gates.numpy() if gates is not None else None)


def checkpoint_path(name=CHECKPOINT):
    for cand in (name, os.path.join(PROJECT_ROOT, os.path.basename(name))):
        if os.path.isfile(cand):
            return cand
    return None


def load_graph_brain(name=CHECKPOINT, verbose=True):
    """Build the network and load trained weights; None if no usable checkpoint exists."""
    path = checkpoint_path(name)
    if path is None:
        if verbose:
            print("WARNING: %s not found -- v3 runs on prior estimates only." % name)
        return None
    blob = torch.load(path, map_location="cpu")
    model = TemporalGraphNet(**blob['config'])
    try:
        model.load_state_dict(blob['state_dict'])
    except RuntimeError:
        # A checkpoint from before the current heads: retrain (training/train.py).
        if verbose:
            print("WARNING: %s has a different head layout -- v3 runs on prior estimates only."
                  % os.path.basename(path))
        return None
    model.eval()
    # Per-level conformal correction for the turn-rate quantile, if this
    # checkpoint has been calibrated (training/calibrate.py).
    model.turn_margin = {}
    calib = os.path.splitext(path)[0] + ".calib.json"
    if os.path.isfile(calib):
        with open(calib) as fh:
            model.turn_margin = {int(k): float(v['margin'])
                                 for k, v in json.load(fh)['levels'].items()}
    if verbose:
        print("Temporal graph brain online (%s, %d parameters%s)."
              % (os.path.basename(path), sum(p.numel() for p in model.parameters()),
                 ", conformal turn calibration available" if model.turn_margin else ""))
    return model
