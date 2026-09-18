"""
run_simulation.py

launches the federated learning simulation: NUM_CLIENTS simulated
institutions training FraudNet together via FedAvg, coordinated by
a central Flower server, for NUM_ROUNDS communication rounds.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import flwr as fl

from client import client_fn
from server import get_fedavg_strategy, NUM_CLIENTS, NUM_ROUNDS

if __name__ == "__main__":
    strategy = get_fedavg_strategy()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print("\nsimulation finished")
    print(history)
