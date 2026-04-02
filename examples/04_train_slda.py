"""
Minimal SldaAgent training example.

Fits an SLDA model on a single task (right) using the same simple
ground-truth observer as examples/03_train_dlbt.py: objects on the right
(latent_state bit DIM_LEFT_RIGHT=1) are chosen with probability 0.8;
objects on the left with probability 0.2.

Run from repo root:
    python examples/04_train_slda.py
"""

import numpy as np  # for rng.binomial
import torch

from dlbt.constants import DIM_LEFT_RIGHT
from dlbt.data.dataset import BehavioralDataset, Observation
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.agents.slda import SldaAgent
from dlbt.training.train_slda import fit_slda

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
# Fit SLDA
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent  = SldaAgent(device=device)

result = fit_slda(agent, train_dataset=ds, val_dataset=ds, image_refs=refs_dict,
                  n_epochs=200, patience=30)

print(f"\nBest val MSE:  {result.best_val_mse:.4f}  (epoch {result.best_epoch})")
tau = agent.log_temperature.exp().item()
print(f"Temperature τ: {tau:.3f}")
