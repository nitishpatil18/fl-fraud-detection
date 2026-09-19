"""
dp_wrapper.py

wraps a model/optimizer/dataloader with Opacus's PrivacyEngine to
enable differentially private training. gradients are clipped per
sample and calibrated noise is added before the optimizer step,
giving a formal (epsilon, delta) privacy guarantee.
"""

from opacus import PrivacyEngine

MAX_GRAD_NORM = 1.0   # per-sample gradient clipping threshold
DELTA = 1e-5          # privacy failure probability, standard choice


def make_private(model, optimizer, data_loader, noise_multiplier: float):
    """
    wraps model/optimizer/dataloader for DP training.
    higher noise_multiplier -> stronger privacy, lower utility.
    returns the wrapped (private_model, private_optimizer, private_loader, privacy_engine)
    so epsilon can be queried later via privacy_engine.get_epsilon(delta=DELTA)
    """
    privacy_engine = PrivacyEngine()

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
