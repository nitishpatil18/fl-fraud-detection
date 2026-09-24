"""
plot_privacy_tradeoff.py

visualizes the core DP finding: no single noise-scaling approach
protects both the smallest and largest client simultaneously.
uses the actual epsilon values measured in earlier experiments.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOT_PATH = "experiments/results/plots/privacy_tradeoff.png"

# measured epsilon values from experiments (client_1: 628 rows, client_4: 159296 rows)
configs = ["Fixed noise\n(no scaling)", "Size-only\nscaling", "Size+step\nscaling"]
client1_epsilon = [98, 4.99, 157.2]      # smallest client
client4_epsilon = [485320, 485320, 506.5]  # largest client

if __name__ == "__main__":
    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width/2, client1_epsilon, width, label="Client 1 (628 rows)")
    ax.bar(x + width/2, client4_epsilon, width, label="Client 4 (159,296 rows)")

    ax.set_yscale("log")
    ax.set_ylabel("Epsilon (log scale, lower = more private)")
    ax.set_title("Privacy-Utility Tradeoff: No Single Scaling Protects Both Clients")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=10, color="green", linestyle="--", alpha=0.5, label="Meaningful privacy threshold (~10)")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"plot saved to {PLOT_PATH}")
