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


class SaveModelFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy that saves the aggregated global model after
    the final round, so it can be attacked/evaluated later."""

    def __init__(self, save_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = save_path
        self.latest_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            import numpy as np
            np.savez(self.save_path, *ndarrays)
        return aggregated_parameters, metrics


def get_fedavg_strategy(save_path: str = "experiments/results/fedavg_global_model.npz"):
    return SaveModelFedAvg(
        save_path=save_path,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
    )
