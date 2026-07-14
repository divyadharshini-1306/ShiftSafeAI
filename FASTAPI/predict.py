"""
predict.py
Loads the XGBoost + Bi-GRU ensemble once at import time and exposes
predict_aqi(). Logic is identical to README Section 3 — the only change
is that file paths are built from BASE_DIR instead of hardcoded relative
strings, which is README's own "Mistake 4" (paths break on Render
because the working directory isn't guaranteed to be the project root).
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Path setup (works locally and on Render) ───────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Load all model assets once at startup ──────────────────────────────

with open(os.path.join(MODELS_DIR, "xgboost_aqi_model.pkl"), "rb") as f:
    xgb_model = pickle.load(f)

with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODELS_DIR, "feature_cols.json"), "r") as f:
    feature_cols = json.load(f)

with open(os.path.join(MODELS_DIR, "ensemble_weights.json"), "r") as f:
    weights = json.load(f)

XGB_WEIGHT = weights["xgb_weight"]   # 0.9
GRU_WEIGHT = weights["gru_weight"]   # 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Bi-GRU architecture (must match training exactly) ──────────────────

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


bigru_model = BiGRUModel(
    input_size=17,
    hidden_size=64,
    num_layers=2,
    dropout=0.3
)
bigru_model.load_state_dict(
    torch.load(os.path.join(MODELS_DIR, "bigru_final.pt"), map_location=device)
)
bigru_model = bigru_model.to(device)
bigru_model.eval()


# ── Main prediction function ─────────────────────────────────────────

def predict_aqi(feature_dict: dict) -> float:
    """
    Accepts a dict of 17 raw, unscaled features.
    Returns predicted next-hour AQI as a float.
    """

    # --- XGBoost prediction ---
    input_df = pd.DataFrame([feature_dict])[feature_cols]
    xgb_pred = float(xgb_model.predict(input_df)[0])

    # --- Bi-GRU prediction ---
    input_array = np.array([[feature_dict[col] for col in feature_cols]])
    input_scaled = scaler.transform(input_array)

    # Repeat the row 24 times to create a mock 24-hour sequence.
    # In production, swap this for a real 24-hour window from the DB.
    sequence = np.tile(input_scaled, (24, 1))          # shape: (24, 17)
    sequence_t = torch.tensor(
        sequence, dtype=torch.float32
    ).unsqueeze(0).to(device)                          # shape: (1, 24, 17)

    with torch.no_grad():
        gru_pred = float(bigru_model(sequence_t).cpu().item())

    # --- Weighted ensemble ---
    final_pred = (XGB_WEIGHT * xgb_pred) + (GRU_WEIGHT * gru_pred)
    return round(final_pred, 1)
