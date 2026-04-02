"""
Smoke-test all three agents on a handful of real images.

Checks:
  - choice_probs() returns [B, 2] tensors summing to 1.
  - SLDA fits and queries without errors.
  - DlbtAgent alpha values are strictly positive.
  - DummyAgent always returns 0.5 / 0.5.

Run from repo root:
    python examples/02_check_agents.py
"""

import torch
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import TASKS
from dlbt.agents.dlbt import DlbtAgent
from dlbt.agents.slda import SldaAgent
from dlbt.agents.dummy import DummyAgent

METADATA  = "stimuli/imgs/metadata.jsonl"
N_IMAGES  = 10   # use only the first N images to keep this fast
TASKS_TO_CHECK = ["right", "transparent", "right_and_transparent"]

# ---------------------------------------------------------------------------
# Load images
# ---------------------------------------------------------------------------
refs_dict = load_image_refs(METADATA)
refs      = image_refs_as_list(refs_dict)[:N_IMAGES]

print(f"Loaded {len(refs_dict)} image refs. Using first {N_IMAGES}.")
print(f"Latent states present: {sorted(set(r.latent_state for r in refs))}\n")

# ---------------------------------------------------------------------------
# DummyAgent
# ---------------------------------------------------------------------------
dummy = DummyAgent()
for task_name in TASKS_TO_CHECK:
    task  = TASKS[task_name]
    probs = dummy.choice_probs(refs, task)
    assert probs.shape == (N_IMAGES, 2), f"Bad shape: {probs.shape}"
    assert torch.allclose(probs.sum(dim=1), torch.ones(N_IMAGES)), "Probs don't sum to 1"
    assert torch.allclose(probs, torch.full_like(probs, 0.5)), "Dummy should be 50/50"
print("DummyAgent:  OK")

# ---------------------------------------------------------------------------
# DlbtAgent (frozen encoder)
# ---------------------------------------------------------------------------
agent = DlbtAgent(freeze_encoder=True, n_mc_samples=100)
agent.precompute_features(refs)

for task_name in TASKS_TO_CHECK:
    task  = TASKS[task_name]
    alpha = agent.get_alpha(refs)
    assert alpha.shape == (N_IMAGES, 16), f"Bad alpha shape: {alpha.shape}"
    assert (alpha > 0).all(), "Alpha must be strictly positive"

    agent.eval()
    with torch.no_grad():
        probs = agent.choice_probs(refs, task)
    assert probs.shape == (N_IMAGES, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(N_IMAGES), atol=1e-5)

    p_right = probs[:, 1].tolist()
    print(f"  DlbtAgent [{task_name}]: P(right) = {[f'{p:.2f}' for p in p_right]}")

print("DlbtAgent:   OK\n")

# ---------------------------------------------------------------------------
# SldaAgent  (untrained — smoke-test forward pass only)
# ---------------------------------------------------------------------------
slda = SldaAgent()
slda.precompute_features(refs)
slda.eval()

for task_name in TASKS_TO_CHECK:
    task = TASKS[task_name]
    with torch.no_grad():
        probs = slda.choice_probs(refs, task)
    assert probs.shape == (N_IMAGES, 2), f"Bad shape: {probs.shape}"
    assert torch.allclose(probs.sum(dim=1), torch.ones(N_IMAGES), atol=1e-5)

    p_right = probs[:, 1].tolist()
    print(f"  SldaAgent  [{task_name}]: P(right) = {[f'{p:.2f}' for p in p_right]}")

print("SldaAgent:   OK  (untrained; use examples/04_train_slda.py for full training)")
