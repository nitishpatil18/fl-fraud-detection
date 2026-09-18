"""
client.py

Flower client wrapper around FraudNet. each simulated institution
runs one of these: trains locally on its own client_N.csv, never
exposes raw data, only sends model weights to the server.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))

import flwr as fl
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from net import FraudNet

BATCH_SIZE = 256
LOCAL_EPOCHS = 2
LR = 0.001


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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)

        ds = TensorDataset(self.X, self.y)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

        self.model.train()
        for _ in range(LOCAL_EPOCHS):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = self.criterion(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        print(f"[client {self.client_id}] finished local training on {len(self.X)} samples")
        return self.get_parameters(config={}), len(self.X), {}

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
