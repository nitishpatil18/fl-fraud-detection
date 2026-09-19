import os
"""
server.py

FedAvg strategy configuration for the federated simulation,
with weighted aggregation of precision/recall/F1 across clients.
"""

import flwr as fl

NUM_CLIENTS = 5
NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", 15))


def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    precision = sum(num_examples * m["precision"] for num_examples, m in metrics) / total_examples
    recall = sum(num_examples * m["recall"] for num_examples, m in metrics) / total_examples
    f1 = sum(num_examples * m["f1"] for num_examples, m in metrics) / total_examples
    return {"precision": precision, "recall": recall, "f1": f1}


def fit_metrics_aggregation(metrics):
    epsilons = [m["epsilon"] for _, m in metrics if "epsilon" in m]
    if not epsilons:
        return {}
    return {"avg_epsilon": sum(epsilons) / len(epsilons), "max_epsilon": max(epsilons)}


def get_fedavg_strategy():
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
    )
