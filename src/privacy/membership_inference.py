"""
membership_inference.py

runs a loss-threshold membership inference attack (MIA) against a
trained model: attempts to distinguish training-set samples ("members")
from held-out test samples ("non-members") using per-sample loss.
higher attack accuracy = more privacy leakage from the model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score

from net import FraudNet

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
SAMPLE_SIZE = 5000  # samples per group (members/non-members), kept equal for a fair 50/50 baseline


def load_model_from_npz(npz_path: str) -> FraudNet:
    model = FraudNet()
    data = np.load(npz_path)
    state_dict = model.state_dict()
    for key, arr_key in zip(state_dict.keys(), data.files):
        state_dict[key] = torch.tensor(data[arr_key])
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_model_from_pt(pt_path: str) -> FraudNet:
    model = FraudNet()
    model.load_state_dict(torch.load(pt_path, weights_only=True))
    model.eval()
    return model


def compute_per_sample_loss(model, X, y):
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    with torch.no_grad():
        logits = model(X)
        losses = criterion(logits, y).squeeze().numpy()
    return losses


def run_mia(model_path: str):
    if model_path.endswith(".pt"):
        model = load_model_from_pt(model_path)
    else:
        model = load_model_from_npz(model_path)

    train_df = pd.read_csv(TRAIN_PATH).sample(n=SAMPLE_SIZE, random_state=42)
    test_df = pd.read_csv(TEST_PATH).sample(n=min(SAMPLE_SIZE, len(pd.read_csv(TEST_PATH))), random_state=42)

    X_train = torch.tensor(train_df.drop(columns=["Class"]).values, dtype=torch.float32)
    y_train = torch.tensor(train_df["Class"].values, dtype=torch.float32).unsqueeze(1)

    X_test = torch.tensor(test_df.drop(columns=["Class"]).values, dtype=torch.float32)
    y_test = torch.tensor(test_df["Class"].values, dtype=torch.float32).unsqueeze(1)

    member_losses = compute_per_sample_loss(model, X_train, y_train)
    nonmember_losses = compute_per_sample_loss(model, X_test, y_test)

    # attack: lower loss -> predicted member. use each sample's loss as the score.
    all_losses = np.concatenate([member_losses, nonmember_losses])
    true_labels = np.concatenate([np.ones(len(member_losses)), np.zeros(len(nonmember_losses))])

    # threshold at median loss of the combined set; predict "member" for anything below it
    threshold = np.median(all_losses)
    predictions = (all_losses < threshold).astype(int)

    attack_accuracy = accuracy_score(true_labels, predictions)
    # AUC uses -loss as the score since lower loss should indicate "member"
    attack_auc = roc_auc_score(true_labels, -all_losses)

    print(f"member avg loss:     {member_losses.mean():.4f}")
    print(f"non-member avg loss: {nonmember_losses.mean():.4f}")
    print(f"attack accuracy:     {attack_accuracy:.4f}  (0.50 = no leakage, higher = more leakage)")
    print(f"attack AUC:          {attack_auc:.4f}")

    return {
        "member_avg_loss": member_losses.mean(),
        "nonmember_avg_loss": nonmember_losses.mean(),
        "attack_accuracy": attack_accuracy,
        "attack_auc": attack_auc,
    }


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "experiments/results/fedavg_global_model.npz"
    print(f"running MIA against: {model_path}\n")
    run_mia(model_path)
