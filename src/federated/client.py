"""
client.py

Flower client wrapper around FraudNet. each simulated institution
runs one of these: trains locally on its own client_N.csv, never
exposes raw data, only sends model weights to the server.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flwr as fl
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from net import FraudNet
from privacy.dp_wrapper import make_private, get_epsilon, save_accountant_history, load_accountant_history, scaled_noise_multiplier

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 256))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", 2))
LR = 0.001
DP_NOISE_MULTIPLIER = float(os.environ.get("DP_NOISE_MULTIPLIER", 0.0))


def load_client_data(client_id: int):
    path = f"data/client_splits/client_{client_id}.csv"
    df = pd.read_csv(path)
    X = torch.tensor(df.drop(columns=["Class"]).values, dtype=torch.float32)
    y = torch.tensor(df["Class"].values, dtype=torch.float32).unsqueeze(1)
    return X, y


class FraudClient(fl.client.NumPyClient):
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.model = FraudNet()
        X, y = load_client_data(client_id)
        self.X, self.y = X, y

        num_neg = (y == 0).sum().item()
        num_pos = max((y == 1).sum().item(), 1)  # avoid divide-by-zero if a client has 0 fraud
        pos_weight = torch.tensor([(num_neg / num_pos) ** 0.5])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        global_params = [p.clone().detach() for p in self.model.parameters()]
        proximal_mu = config.get("proximal_mu", 0.0)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)

        ds = TensorDataset(self.X, self.y)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

        epsilon = None
        if DP_NOISE_MULTIPLIER > 0:
            client_noise = scaled_noise_multiplier(DP_NOISE_MULTIPLIER, len(self.X), BATCH_SIZE, LOCAL_EPOCHS)
            self.model, optimizer, loader, privacy_engine = make_private(
                self.model, optimizer, loader, client_noise
            )
            privacy_engine.accountant.history = load_accountant_history(self.client_id)

        self.model.train()
        for _ in range(LOCAL_EPOCHS):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = self.criterion(self.model(X_batch), y_batch)

                if proximal_mu > 0:
                    proximal_term = 0.0
                    for local_p, global_p in zip(self.model.parameters(), global_params):
                        proximal_term += (local_p - global_p).norm(2) ** 2
                    loss += (proximal_mu / 2) * proximal_term

                loss.backward()
                optimizer.step()

        if DP_NOISE_MULTIPLIER > 0:
            save_accountant_history(self.client_id, privacy_engine)
            epsilon = get_epsilon(privacy_engine)
            print(f"[client {self.client_id}] finished DP training (noise={client_noise:.3f}), cumulative epsilon={epsilon:.3f}")
        else:
            print(f"[client {self.client_id}] finished local training on {len(self.X)} samples")

        metrics = {"epsilon": epsilon} if epsilon is not None else {}
        return self.get_parameters(config={}), len(self.X), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X)
            loss = self.criterion(logits, self.y).item()
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

        from sklearn.metrics import precision_score, recall_score, f1_score

        precision = precision_score(self.y, preds, zero_division=0)
        recall = recall_score(self.y, preds, zero_division=0)
        f1 = f1_score(self.y, preds, zero_division=0)

        return loss, len(self.X), {'precision': precision, 'recall': recall, 'f1': f1}


def client_fn(cid: str):
    return FraudClient(int(cid)).to_client()
