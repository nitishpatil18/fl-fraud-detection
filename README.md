# Privacy-Preserving Fraud Detection using Federated Learning with Differential Privacy

A project implementing and evaluating federated learning (FedAvg, FedProx) with
differential privacy for collaborative fraud detection across simulated financial
institutions, without sharing raw transaction data.

## Overview

Five simulated institutions ("clients") hold non-identically distributed (non-IID)
slices of a credit card fraud dataset. A shared fraud-detection model is trained
collaboratively via Federated Averaging (FedAvg) and FedProx, with differential
privacy applied at the client level using Opacus. Privacy claims are empirically
validated via a membership inference attack, and adaptive per-client noise scaling
is explored to address client heterogeneity.

## Key Findings

- FedAvg matches/exceeds centralized training: F1 = 0.798 vs 0.697 centralized baseline, despite non-IID data.
- FedProx is mu-sensitive: mu = 0.01 stabilizes training; mu = 0.1 degrades it.
- Naive fixed-noise DP has a fundamental privacy-utility conflict: no noise level and round count tested achieves both a meaningful epsilon (<10) and non-zero utility.
- No measurable membership leakage detected in either the centralized or FedAvg model (attack accuracy ~0.49-0.50, i.e. random guessing).
- Adaptive per-client noise scaling (by dataset size) achieves a strong privacy guarantee for the smallest client (epsilon = 4.99) while maintaining F1 = 0.74, but cannot simultaneously protect the largest client without also accounting for training step count -- no single global scaling formula optimizes privacy for all clients under heterogeneous non-IID data.

Full details, methodology, and results: report/final_report.pdf

## Project Structure

fl-fraud-detection/
    data/                     raw, processed, and per-client data splits (gitignored)
    src/
        data_prep/            loading, preprocessing, non-IID client partitioning
        model/                FraudNet model + centralized baseline training
        federated/            Flower client/server, FedAvg, FedProx, experiment scripts, plots
        privacy/              differential privacy wrapper, membership inference attack
    experiments/results/      comparison CSVs, saved models, plots
    report/                   final project report

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Download the dataset (Kaggle: mlg-ulb/creditcardfraud) into data/raw/creditcard.csv.

## Running

    # data pipeline
    python3 src/data_prep/load_data.py
    python3 src/data_prep/preprocess.py
    python3 src/data_prep/split_clients.py

    # centralized baseline
    python3 src/model/train_centralized.py

    # federated learning
    python3 src/federated/run_simulation.py                # FedAvg
    python3 src/federated/run_simulation_fedprox.py         # FedProx
    DP_NOISE_MULTIPLIER=0.2 python3 src/federated/run_simulation_dp.py   # FedAvg + DP

    # full comparison + plots
    python3 src/federated/run_all_experiments.py
    python3 src/federated/plot_convergence.py
    python3 src/federated/plot_comparison.py
    python3 src/federated/plot_privacy_tradeoff.py

    # membership inference attack
    python3 src/privacy/membership_inference.py experiments/results/fedavg_global_model.npz

## Tech Stack

Flower (federated learning), PyTorch (model training), Opacus (differential privacy), scikit-learn and matplotlib (evaluation)
