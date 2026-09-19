"""
run_simulation_dp.py

launches FedAvg with differential privacy enabled on clients.
set DP_NOISE_MULTIPLIER env var to control the privacy strength
(higher = more privacy, lower utility).

usage:
    DP_NOISE_MULTIPLIER=0.5 python3 src/federated/run_simulation_dp.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import flwr as fl

from client import client_fn, DP_NOISE_MULTIPLIER
from server import get_fedavg_strategy, NUM_CLIENTS, NUM_ROUNDS

if __name__ == "__main__":
    print(f"running FedAvg + DP with noise_multiplier={DP_NOISE_MULTIPLIER}")

    strategy = get_fedavg_strategy()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print("\nfedavg+dp simulation finished")
    print(history)
