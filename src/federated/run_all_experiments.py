"""
run_all_experiments.py

runs centralized, FedAvg, FedProx, and FedAvg+DP experiments in
sequence, extracts final-round metrics, and writes a comparison
table to experiments/results/comparison.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import csv
import flwr as fl

RESULTS_PATH = "experiments/results/comparison.csv"


def extract_final_metrics(history):
    f1_list = history.metrics_distributed.get("f1", [])
    precision_list = history.metrics_distributed.get("precision", [])
    recall_list = history.metrics_distributed.get("recall", [])
    loss_list = history.losses_distributed

    final_f1 = f1_list[-1][1] if f1_list else None
    final_precision = precision_list[-1][1] if precision_list else None
    final_recall = recall_list[-1][1] if recall_list else None
    final_loss = loss_list[-1][1] if loss_list else None

    return {
        "final_loss": final_loss,
        "final_precision": final_precision,
        "final_recall": final_recall,
        "final_f1": final_f1,
    }


def run_fedavg():
    from client import client_fn
    from server import get_fedavg_strategy, NUM_CLIENTS, NUM_ROUNDS

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=get_fedavg_strategy(),
    )
    return extract_final_metrics(history)


def run_fedprox():
    from client import client_fn
    from strategy_fedprox import get_fedprox_strategy, NUM_CLIENTS, NUM_ROUNDS

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=get_fedprox_strategy(),
    )
    return extract_final_metrics(history)


def run_fedavg_dp(noise_multiplier: float):
    os.environ["DP_NOISE_MULTIPLIER"] = str(noise_multiplier)
    from client import client_fn
    from server import get_fedavg_strategy, NUM_CLIENTS, NUM_ROUNDS

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=get_fedavg_strategy(),
    )
    return extract_final_metrics(history)


if __name__ == "__main__":
    results = []

    print("running FedAvg...")
    results.append({"variant": "FedAvg", **run_fedavg()})

    print("running FedProx (mu=0.01)...")
    results.append({"variant": "FedProx (mu=0.01)", **run_fedprox()})

    print("running FedAvg + DP (noise=0.1)...")
    results.append({"variant": "FedAvg+DP (noise=0.1)", **run_fedavg_dp(0.1)})

    print("running FedAvg + DP (noise=0.2)...")
    results.append({"variant": "FedAvg+DP (noise=0.2)", **run_fedavg_dp(0.2)})

    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "final_loss", "final_precision", "final_recall", "final_f1"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nresults written to {RESULTS_PATH}")
    for r in results:
        print(r)
