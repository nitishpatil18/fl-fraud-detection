"""
strategy_fedprox.py

FedProx strategy: same as FedAvg but adds a proximal term (mu) to
client-side training, penalizing local models for drifting too far
from the global model. helps convergence under non-IID data.
"""

import flwr as fl
from server import weighted_average

NUM_CLIENTS = 5
NUM_ROUNDS = 15
PROXIMAL_MU = 0.01  # strength of the proximal term


def get_fedprox_strategy():
    return fl.server.strategy.FedProx(
        proximal_mu=PROXIMAL_MU,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
