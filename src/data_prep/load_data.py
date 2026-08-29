"""
load_data.py

loads the raw credit card fraud dataset, prints basic stats
(shape, class balance, missing values), and confirms the data
is ready for preprocessing.
"""

import pandas as pd

RAW_DATA_PATH = "data/raw/creditcard.csv"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def inspect_data(df: pd.DataFrame) -> None:
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print(f"missing values total: {df.isnull().sum().sum()}")

    class_counts = df["Class"].value_counts()
    fraud_ratio = class_counts[1] / len(df) * 100

    print("\nclass balance:")
    print(f"  normal (0): {class_counts[0]}")
    print(f"  fraud  (1): {class_counts[1]}")
    print(f"  fraud percentage: {fraud_ratio:.4f}%")

    print("\nAmount stats:")
    print(df["Amount"].describe())


if __name__ == "__main__":
    df = load_raw_data()
    inspect_data(df)
