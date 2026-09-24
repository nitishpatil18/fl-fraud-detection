"""
plot_convergence.py

runs FedAvg and FedProx, captures per-round F1 scores, and plots
convergence curves for the report.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import flwr as fl

PLOT_PATH = "experiments/results/plots/convergence_f1.png"


def run_and_extract_f1(strategy_fn):
    from client import client_fn
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=15),
        strategy=strategy_fn(),
    )
    f1_list = history.metrics_distributed.get("f1", [])
    rounds = [r for r, _ in f1_list]
    f1_scores = [f1 for _, f1 in f1_list]
    return rounds, f1_scores


if __name__ == "__main__":
    from server import get_fedavg_strategy
    from strategy_fedprox import get_fedprox_strategy

    print("running FedAvg...")
    rounds_avg, f1_avg = run_and_extract_f1(get_fedavg_strategy)

    print("running FedProx...")
    rounds_prox, f1_prox = run_and_extract_f1(get_fedprox_strategy)

    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(rounds_avg, f1_avg, marker="o", label="FedAvg", linewidth=2)
    plt.plot(rounds_prox, f1_prox, marker="s", label="FedProx (mu=0.01)", linewidth=2)
    plt.xlabel("Communication Round")
    plt.ylabel("F1 Score")
    plt.title("FedAvg vs FedProx: F1 Convergence Over Rounds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"\nplot saved to {PLOT_PATH}")
