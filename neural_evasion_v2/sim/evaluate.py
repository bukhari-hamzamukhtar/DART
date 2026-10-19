"""
Batched scene evaluation.

The brain's `forward` handles one scene at a time, but the expensive part — the
three MetaLayers of message passing — does not care which scene a sub-graph
belongs to. So for a tick that needs the network's opinion on several scenes
(the current one plus the shortlisted rollout futures) we can concatenate all
their sub-graphs into a single batch, encode once, and then run the cheap
per-scene attention/danger/maneuver heads on the slices.

Results are identical to calling `model(sgs, ctxs, hidden)` per scene — the
sub-graphs are disjoint and pooling is per-sub-graph — but one encoder pass
replaces N of them.
"""

import torch
from torch_geometric.data import Batch

from ..config import MANEUVER_NAMES


class SceneResult(object):
    __slots__ = ("danger", "attention", "hidden", "maneuver_logits")

    def __init__(self, danger, attention, hidden, maneuver_logits):
        self.danger = danger
        self.attention = attention
        self.hidden = hidden
        self.maneuver_logits = maneuver_logits

    @property
    def maneuver(self):
        return MANEUVER_NAMES[int(self.maneuver_logits.argmax().item())]


def evaluate_scenes(model, scenes, hidden=None):
    """
    Score several scenes in one encoder pass.

    `scenes` is a list of sub-graph lists (each the output of
    `build_candidate_sub_graphs`).  Empty scenes yield a zero result.
    Returns a list of SceneResult, one per scene, in the same order.
    """
    flat, sizes = [], []
    for sgs in scenes:
        sizes.append(len(sgs))
        flat.extend(sgs)

    if not flat:
        return [_empty_result() for _ in scenes]

    with torch.no_grad():
        embeds = model.threat_gnn(Batch.from_data_list(flat))

        out, off = [], 0
        for sgs, n in zip(scenes, sizes):
            if n == 0:
                out.append(_empty_result())
                continue
            e = embeds[off:off + n]
            off += n
            ctxs = torch.stack([g.context for g in sgs]).to(e.device)
            danger, attn, new_hidden = model.attention(e, ctxs, hidden)
            pooled = new_hidden.mean(dim=0, keepdim=True)
            m_in = torch.cat([pooled, danger.unsqueeze(-1).reshape(1, 1)], dim=-1)
            logits = model.maneuver_head(m_in).squeeze(0)
            out.append(SceneResult(float(danger.item()),
                                   attn.cpu().numpy().tolist(),
                                   new_hidden, logits))
    return out


def _empty_result():
    return SceneResult(0.0, [], None, torch.zeros(len(MANEUVER_NAMES)))
