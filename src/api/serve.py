"""
serve.py

FastAPI backend for testing the trained FedAvg model. Loads the saved
global model and serves predictions on real test-set transactions, so
the index.html frontend can be used to sanity-check the whole pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from net import FraudNet

MODEL_PATH = "experiments/results/fedavg_global_model.npz"
TEST_DATA_PATH = "data/processed/test.csv"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model():
    model = FraudNet()
    data = np.load(MODEL_PATH)
    state_dict = model.state_dict()
    for key, arr_key in zip(state_dict.keys(), data.files):
        state_dict[key] = torch.tensor(data[arr_key])
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_model()
test_df = pd.read_csv(TEST_DATA_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/sample")
def get_sample(label: int = 0):
    row = test_df[test_df["Class"] == label].sample(n=1).iloc[0]
    features = row.drop("Class").tolist()
    return {"features": features, "true_label": int(row["Class"])}


@app.post("/predict")
def predict(req: PredictRequest):
    x = torch.tensor([req.features], dtype=torch.float32)
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()
    return {"fraud_probability": prob, "predicted_label": int(prob >= 0.5)}
