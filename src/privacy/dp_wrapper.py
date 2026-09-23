"""
dp_wrapper.py

wraps a model/optimizer/dataloader with Opacus's PrivacyEngine to
enable differentially private training. gradients are clipped per
sample and calibrated noise is added before the optimizer step,
giving a formal (epsilon, delta) privacy guarantee.
"""

import json
import os

from opacus import PrivacyEngine

MAX_GRAD_NORM = 1.0   # per-sample gradient clipping threshold
DELTA = 1e-5          # privacy failure probability, standard choice
STATE_DIR = "data/privacy_state"


def save_accountant_history(client_id: int, privacy_engine) -> None:
    """persists this client's privacy accountant history so epsilon
    correctly accumulates across FL rounds instead of resetting each round."""
    os.makedirs(STATE_DIR, exist_ok=True)
    history = [list(entry) for entry in privacy_engine.accountant.history]
    path = f"{STATE_DIR}/client_{client_id}_history.json"
    with open(path, "w") as f:
        json.dump(history, f)


def load_accountant_history(client_id: int) -> list:
    path = f"{STATE_DIR}/client_{client_id}_history.json"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return [tuple(entry) for entry in data]


def make_private(model, optimizer, data_loader, noise_multiplier: float):
    """
    wraps model/optimizer/dataloader for DP training.
    higher noise_multiplier -> stronger privacy, lower utility.
    returns the wrapped (private_model, private_optimizer, private_loader, privacy_engine)
    so epsilon can be queried later via privacy_engine.get_epsilon(delta=DELTA)
    """
    privacy_engine = PrivacyEngine(accountant="rdp")

    private_model, private_optimizer, private_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=MAX_GRAD_NORM,
    )

    return private_model, private_optimizer, private_loader, privacy_engine


def get_epsilon(privacy_engine, delta: float = DELTA) -> float:
    return privacy_engine.get_epsilon(delta=delta)


def scaled_noise_multiplier(base_noise: float, client_size: int, reference_size: int = 20000) -> float:
    """
    scales noise inversely with sqrt(dataset size), so small clients
    (like a 628-row client) get calibrated noise relative to a large
    client (e.g. 159,296 rows), rather than identical fixed noise for both.
    reference_size is the dataset size at which base_noise applies unscaled.
    """
    import math
    scale_factor = math.sqrt(reference_size / client_size)
    return base_noise * scale_factor
