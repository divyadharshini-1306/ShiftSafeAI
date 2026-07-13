
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# Load all assets once when this module is imported.
with open(os.path.join(MODELS_DIR, "xgboost_aqi_model.pkl"), "rb") as f:
    xgb_model = pickle.load(f)

with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODELS_DIR, "feature_cols.json"), "r") as f:
    feature_cols = json.load(f)

with open(os.path.join(MODELS_DIR, "ensemble_weights.json"), "r") as f:
    weights = json.load(f)

XGB_WEIGHT = weights["xgb_weight"]
GRU_WEIGHT = weights["gru_weight"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(BiGRUModel, self).__init__()

        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        last_out = self.dropout(last_out)
        return self.fc(last_out)


# These four values must match training exactly.
bigru_model = BiGRUModel(
    input_size=17,
    hidden_size=64,
    num_layers=2,
    dropout=0.3
)

bigru_model.load_state_dict(
    torch.load(
        os.path.join(MODELS_DIR, "bigru_final.pt"),
        map_location=device
    )
)
bigru_model = bigru_model.to(device)
bigru_model.eval()


def predict_aqi(feature_dict: dict) -> float:
    """
    Predict next-hour AQI from exactly 17 raw, unscaled features.
    """
    missing = [column for column in feature_cols if column not in feature_dict]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # XGBoost receives raw values.
    input_df = pd.DataFrame([feature_dict])[feature_cols]
    xgb_pred = float(xgb_model.predict(input_df)[0])

    # Bi-GRU receives internally scaled data as a 24-step sequence.
    input_array = np.array(
        [[feature_dict[column] for column in feature_cols]],
        dtype=float
    )
    input_scaled = scaler.transform(input_array)
    sequence = np.tile(input_scaled, (24, 1))

    sequence_t = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        gru_pred = float(bigru_model(sequence_t).cpu().item())

    return round((XGB_WEIGHT * xgb_pred) + (GRU_WEIGHT * gru_pred), 1)
