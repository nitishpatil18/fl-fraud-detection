"""
split_clients.py

partitions the training data across N simulated clients using a
Dirichlet distribution over class labels, producing non-IID splits
(each client sees a different fraud ratio, mimicking real institutions
with different fraud exposure).
"""

import numpy as np
import pandas as pd

TRAIN_DATA_PATH = "data/processed/train.csv"
OUTPUT_DIR = "data/client_splits"
NUM_CLIENTS = 5
ALPHA = 0.5          # lower = more non-IID (more skewed), higher = more uniform
RANDOM_STATE = 42


def dirichlet_partition(df: pd.DataFrame, num_clients: int, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]

    for class_label in df["Class"].unique():
        class_indices = df[df["Class"] == class_label].index.values.copy()
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        splits = np.split(class_indices, split_points)

        for client_id, idx in enumerate(splits):
            client_indices[client_id].extend(idx.tolist())

    return client_indices


def save_client_splits(df: pd.DataFrame, client_indices: list) -> None:
    for client_id, indices in enumerate(client_indices):
        client_df = df.loc[indices]
        path = f"{OUTPUT_DIR}/client_{client_id}.csv"
        client_df.to_csv(path, index=False)

        total = len(client_df)
        fraud = client_df["Class"].sum()
        fraud_pct = (fraud / total * 100) if total > 0 else 0
        print(f"client_{client_id}: {total} rows, {fraud} fraud ({fraud_pct:.3f}%)")


if __name__ == "__main__":
    df = pd.read_csv(TRAIN_DATA_PATH)
    client_indices = dirichlet_partition(df, NUM_CLIENTS, ALPHA, RANDOM_STATE)
    save_client_splits(df, client_indices)
