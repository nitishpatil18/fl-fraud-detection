"""
preprocess.py

scales Amount and Time (V1-V28 are already PCA-transformed and scaled),
then splits into train/test sets. saves processed data to data/processed/.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from load_data import load_raw_data, RAW_DATA_PATH

PROCESSED_DIR = "data/processed"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"] = scaler.fit_transform(df[["Time"]])
    return df


def split_and_save(df: pd.DataFrame) -> None:
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df["Class"] = y_train
    test_df = X_test.copy()
    test_df["Class"] = y_test

    train_df.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
    test_df.to_csv(f"{PROCESSED_DIR}/test.csv", index=False)

    print(f"train shape: {train_df.shape}")
    print(f"test shape: {test_df.shape}")
    print(f"train fraud count: {train_df['Class'].sum()}")
    print(f"test fraud count: {test_df['Class'].sum()}")


if __name__ == "__main__":
    df = load_raw_data(RAW_DATA_PATH)
    df = preprocess(df)
    split_and_save(df)
