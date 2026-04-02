"""
Minimal DlbtAgent training example.

Trains on a single task (right) with synthetic behavioral data
generated from a simple ground-truth observer: objects on the right
side of the image (latent_state bit DIM_LEFT_RIGHT=1) are chosen
with probability 0.8; objects on the left with probability 0.2.

Run from repo root:
    python examples/03_train_dlbt.py
"""

import numpy as np
import torch

from dlbt.constants import DIM_LEFT_RIGHT
from dlbt.data.dataset import BehavioralDataset, Observation
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import TASKS
from dlbt.agents.dlbt import DlbtAgent
from dlbt.training.train_dlbt import train_dlbt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
METADATA  = "stimuli/imgs/metadata.jsonl"
TASK_NAME = "right"
N_TRIALS  = 50
SEED      = 42

# ---------------------------------------------------------------------------
# Load images
# ---------------------------------------------------------------------------
refs_dict = load_image_refs(METADATA)
refs      = image_refs_as_list(refs_dict)
print(f"Loaded {len(refs)} images.")

# ---------------------------------------------------------------------------
# Simulate observer choices
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)

records = []
for ref in refs:
    p_right = 0.8 if (ref.latent_state >> DIM_LEFT_RIGHT) & 1 else 0.2
    c1 = int(rng.binomial(N_TRIALS, p_right))
    records.append(Observation(
        uid=ref.uid, task_name=TASK_NAME,
        count_0=N_TRIALS - c1, count_1=c1,
    ))

ds = BehavioralDataset.from_records(records)
print(f"Dataset: {ds}")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent  = DlbtAgent(freeze_encoder=True, device=device)
agent.precompute_features(refs)

result = train_dlbt(agent, ds, ds, refs_dict, n_epochs=200, patience=30)
print(f"\nBest val MSE: {result.best_val_mse:.4f}  (epoch {result.best_epoch})")
