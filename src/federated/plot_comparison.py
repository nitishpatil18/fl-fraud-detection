"""
plot_comparison.py

reads the final comparison CSV and plots a grouped bar chart of
precision/recall/F1 across all tested variants.
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "experiments/results/comparison.csv"
PLOT_PATH = "experiments/results/plots/comparison_bars.png"


def load_results():
    variants, precisions, recalls, f1s = [], [], [], []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            variants.append(row["variant"])
            precisions.append(float(row["final_precision"]))
            recalls.append(float(row["final_recall"]))
            f1s.append(float(row["final_f1"]))
    return variants, precisions, recalls, f1s


if __name__ == "__main__":
    variants, precisions, recalls, f1s = load_results()

    x = np.arange(len(variants))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precisions, width, label="Precision")
    plt.bar(x, recalls, width, label="Recall")
    plt.bar(x + width, f1s, width, label="F1")

    plt.xlabel("Variant")
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1 Across FL Variants")
    plt.xticks(x, variants, rotation=20, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"plot saved to {PLOT_PATH}")
