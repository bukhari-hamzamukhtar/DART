"""
Checkpoint loading (main3.py section 2).

Same shape-filtered load as the original — any tensor whose shape does not match
the current architecture is skipped rather than raising — plus a lazily cached
singleton so the brain is only built once per process.
"""

import os

import torch

from ..config import CHECKPOINT_NAME, PROJECT_ROOT
from .brain import NeuralEvasionBrain

device = torch.device('cpu')

_BRAIN = None


def resolve_checkpoint(path=None):
    """
    Return the first existing checkpoint path, or None.

    Tries the caller's path / the plain filename relative to the current working
    directory first (main3.py's original behaviour when launched from the repo
    root), then the same filename next to the repo root so the game also starts
    when launched from another directory.
    """
    name = path or CHECKPOINT_NAME
    for candidate in (name, os.path.join(PROJECT_ROOT, os.path.basename(name))):
        if os.path.isfile(candidate):
            return candidate
    return None


def load_brain(path=None, verbose=True):
    """Build a NeuralEvasionBrain and load jet_brain_v3.pt into it."""
    if verbose:
        print("Loading Neural Engine...")
    model = NeuralEvasionBrain().to(device)

    checkpoint_path = resolve_checkpoint(path)
    if checkpoint_path is None:
        if verbose:
            print("WARNING: %s not found — running with random weights." % (path or CHECKPOINT_NAME))
    else:
        # 1. Load the raw dictionary from the file
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_dict = model.state_dict()

        # 2. Filter out anything that doesn't have the exact same shape
        filtered_dict = {k: v for k, v in checkpoint.items()
                         if k in model_dict and v.shape == model_dict[k].shape}

        # 3. Update the model's dictionary and load it safely
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        if verbose:
            print("Engine Online. Weights loaded (Mismatched shapes skipped!).")

    model.eval()
    return model


def get_brain():
    """Process-wide singleton — loads on first use, reused afterwards."""
    global _BRAIN
    if _BRAIN is None:
        _BRAIN = load_brain()
    return _BRAIN
