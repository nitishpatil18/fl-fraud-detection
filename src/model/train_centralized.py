"""
train_centralized.py

trains FraudNet on the full pooled (centralized) training data.
this is the baseline that FedAvg, FedProx, and FedAvg+DP get compared against.
"""

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from net import FraudNet

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_SAVE_PATH = "experiments/results/centralized_model.pt"
BATCH_SIZE = 256
EPOCHS = 10
LR = 0.001
RANDOM_STATE = 42

torch.manual_seed(RANDOM_STATE)


def load_tensors(path: str):
    df = pd.read_csv(path)
    X = torch.tensor(df.drop(columns=["Class"]).values, dtype=torch.float32)
    y = torch.tensor(df["Class"].values, dtype=torch.float32).unsqueeze(1)
    return X, y


def train():
    X_train, y_train = load_tensors(TRAIN_PATH)
    X_test, y_test = load_tensors(TEST_PATH)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = FraudNet()

    # class imbalance handling: weight positive (fraud) class heavily
    num_neg = (y_train == 0).sum().item()
    num_pos = (y_train == 1).sum().item()
    pos_weight = torch.tensor([(num_neg / num_pos) ** 0.5])

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"epoch {epoch+1}/{EPOCHS} - loss: {total_loss / len(train_loader):.4f}")

    evaluate(model, X_test, y_test)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"model saved to {MODEL_SAVE_PATH}")


def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print("\ntest set evaluation:")
    print(f"  precision: {precision:.4f}")
    print(f"  recall:    {recall:.4f}")
    print(f"  f1:        {f1:.4f}")
    print(f"  auc:       {auc:.4f}")


if __name__ == "__main__":
    train()
