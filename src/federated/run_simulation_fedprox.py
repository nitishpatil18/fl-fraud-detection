"""
run_simulation_fedprox.py

launches the federated learning simulation using FedProx instead of
FedAvg, to compare non-IID robustness against the plain FedAvg run.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import flwr as fl

from client import client_fn
from strategy_fedprox import get_fedprox_strategy, NUM_CLIENTS, NUM_ROUNDS

if __name__ == "__main__":
    strategy = get_fedprox_strategy()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print("\nfedprox simulation finished")
    print(history)
