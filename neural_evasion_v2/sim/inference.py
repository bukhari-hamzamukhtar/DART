"""
Background inference thread.

The render loop pushes a snapshot of the world onto `state_queue` every few
frames; this worker plans an action with the world-model planner, runs one
"real" forward pass that advances the GRU memory, and pushes an `AIDecision`
back on `decision_queue`.

Both queues have maxsize=1, so a slow pass drops frames rather than queueing
them up — the game thread never blocks on the brain.
"""

import queue

import torch
from torch_geometric.data import Batch

from ..model import get_brain
from .decision import AIDecision
from .planner import plan

# ── Thread plumbing ───────────────────────────────────────────
state_queue    = queue.Queue(maxsize=1)
decision_queue = queue.Queue(maxsize=1)

FLARE_DANGER_THRESHOLD = 0.55

# How far ahead to aim, in frames. Seeded at a sane value and then driven by the
# staleness the game actually reports back, so it adapts to the machine it is
# running on instead of assuming a fixed budget.
DEFAULT_LEAD = 8
MAX_LEAD     = 30
MAX_OVERLAY_FLARES     = 6      # flare nodes shown in the [G] graph overlay


def _edge_overlay(model, scene):
    """
    Per-edge message magnitudes for the [G] debug overlay.

    Re-runs the encoder stack so the intermediate edge features can be read out
    (the brain's forward pass only returns the pooled embedding).  Only called
    when the overlay is actually on — it used to run on every improving
    candidate whether or not anyone was looking at it.
    """
    sgs, js, ms, fs = scene
    with torch.no_grad():
        batched = Batch.from_data_list(sgs)
        x  = model.threat_gnn.node_enc(batched.x)
        ea = model.threat_gnn.edge_enc(batched.edge_attr)
        for layer in model.threat_gnn.layers:
            x, ea, _ = layer(x, batched.edge_index, ea, None, batched.batch)
        magnitudes = ea.norm(dim=1).cpu().numpy()

    edge_data = []
    for k, sg in enumerate(sgs):
        nodes_pos = [
            [js['pos'][0], js['pos'][1]],
            [ms[k]['pos'][0], ms[k]['pos'][1]]
        ] + [[f['pos'][0], f['pos'][1]] for f in fs[:MAX_OVERLAY_FLARES]]
        for idx, (i, j) in enumerate(sg.edge_index.t().tolist()):
            if i < len(nodes_pos) and j < len(nodes_pos):
                edge_data.append({
                    'p1': nodes_pos[i],
                    'p2': nodes_pos[j],
                    'mag': float(magnitudes[idx]) if idx < len(magnitudes) else 0.0
                })
    return edge_data


def gnn_inference_worker(stop_event=None):
    """Consume world snapshots and emit AIDecisions until told to stop."""
    model = get_brain()
    gru_hidden         = None   # persists missile trajectory memory across frames
    prev_missile_count = 0      # used to detect when to reset hidden state
    lead               = DEFAULT_LEAD

    while True:
        state = state_queue.get()
        if state is None or (stop_event is not None and stop_event.is_set()):
            break

        # The per-missile GRU memory is keyed by position in the threat list,
        # so it goes stale the moment the threat set changes.
        ms_count = len(state['missiles'])
        if ms_count != prev_missile_count:
            gru_hidden = None
        prev_missile_count = ms_count

        # ── Plan: roll the world forward under every candidate action, then
        #    let the network re-rank the shortlist. Also advances GRU memory.
        # The game reports how old the last plan was when it got used; aim that
        # far ahead, plus half a push interval for the hold time.
        reported = state.get('plan_age')
        if reported is not None:
            lead = int(min(MAX_LEAD, max(2, 0.5 * lead + 0.5 * (reported + 3))))

        result = plan(model, state['jet'], state['missiles'],
                      state['flares'], gru_hidden, lead=lead)
        gru_hidden = result.hidden

        edges = []
        if state.get('want_overlay') and result.scene is not None:
            edges = _edge_overlay(model, result.scene)

        # Countermeasures when the world model says no heading escapes, or when
        # the learned danger score is high.
        drop_flare = ms_count > 0 and (result.cornered
                                       or result.danger > FLARE_DANGER_THRESHOLD)

        if not decision_queue.full():
            decision_queue.put(AIDecision(
                heading=result.heading,
                throttle=result.throttle,
                drop_flare=drop_flare,
                attention=result.attention,
                danger=result.danger,
                edges=edges,
                candidate_scores=result.candidate_scores,
                maneuver=result.maneuver,
                fresh=True,
                stamp=state.get('frame', 0),
                cornered=result.cornered,
            ))


def drain_decisions():
    """Drop any decision the previous match left behind."""
    while not decision_queue.empty():
        decision_queue.get()


def drain_queues():
    """Empty both queues so a new match starts with no stale advice."""
    while not state_queue.empty():
        state_queue.get()
    drain_decisions()


def stop_worker():
    """Ask the worker thread to exit, without ever blocking the game thread.

    main3.py called `state_queue.put(None)` directly; on a maxsize=1 queue that
    blocks forever if the sentinel is sent more than once after the worker has
    already exited.  Same effect, no deadlock: if the slot is taken, drop the
    stale snapshot sitting in it and put the sentinel there instead.
    """
    try:
        state_queue.put_nowait(None)
        return
    except queue.Full:
        pass
    try:
        state_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        state_queue.put_nowait(None)
    except queue.Full:
        pass
