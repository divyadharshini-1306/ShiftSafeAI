# ShiftSafeAI
# ShiftSafe AI — Model & Data Pipeline Handoff
### 
### 
### 

---

## What has been built and handed off to you

Two complete systems are ready for you to plug into FastAPI:

1. **The ML model** — a trained ensemble (XGBoost + Bi-GRU) that predicts next-hour AQI from 17 input features. Built by Divyadarshini.
2. **The data pipeline** — a live sensor simulator with a `get_latest_features()` function that returns those exact 17 features in the exact format the model expects. Built by Atharvi.

Your job is to wrap both of these inside a FastAPI backend with three endpoints.

---

## Section 1 — Files you need from Google Drive

All files are in: `ShiftSafe_AI/models/`

| File | What it is | Used by |
|---|---|---|
| `xgboost_aqi_model.pkl` | Trained XGBoost model | predict endpoint |
| `bigru_final.pt` | Trained Bi-GRU weights | predict endpoint |
| `scaler.pkl` | StandardScaler fitted on 48,189 rows | Bi-GRU preprocessing |
| `feature_cols.json` | Ordered list of 17 feature names | Both models |
| `ensemble_weights.json` | `{"xgb_weight": 0.9, "gru_weight": 0.1}` | Ensemble fusion |

Download all 5 files and put them in a folder called `models/` inside your FastAPI project directory.

Also get from data pipeline:
- `aqi_sensor.db` — the SQLite database with 48,189 rows of Bengaluru AQI data
- `data_pipeline.ipynb` — Atharvi's pipeline notebook for reference

---

## Section 2 — The 17 features your API must handle

This is the exact input the model expects. Every key name, spelling, and data type must match precisely. One wrong key name will silently break predictions.

```python
{
    # Raw pollutant readings — all float
    'PM2.5':        float,   # e.g. 45.2   (range roughly 0–200)
    'PM10':         float,   # e.g. 78.1
    'NO':           float,   # e.g. 3.1
    'NO2':          float,   # e.g. 18.4
    'NH3':          float,   # e.g. 12.1
    'CO':           float,   # e.g. 0.8
    'SO2':          float,   # e.g. 5.2
    'O3':           float,   # e.g. 22.1

    # Time features — all int
    'hour':         int,     # 0–23
    'month':        int,     # 1–12
    'day_of_week':  int,     # 0=Monday, 6=Sunday
    'is_weekend':   int,     # 0 or 1
    'is_shift_hour':int,     # 1 if hour is between 6 and 18, else 0

    # Engineered history features — all float
    'AQI_lag1':     float,   # AQI from 1 hour ago
    'AQI_lag3':     float,   # AQI from 3 hours ago
    'PM25_rolling6':float,   # 6-hour rolling average of PM2.5
    'AQI_rolling6': float,   # 6-hour rolling average of AQI
}
```

**These values must be RAW and UNSCALED.** The model handles scaling internally for the Bi-GRU. Do not normalise anything before passing to the prediction function.

---

## Section 3 — How to load and call the prediction model

Install dependencies first:
```
pip install xgboost torch scikit-learn pandas numpy fastapi uvicorn
```

Copy this exactly into a file called `predict.py` in your project root:

```python
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Load all model assets once at startup ──────────────────────────────

with open('models/xgboost_aqi_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/feature_cols.json', 'r') as f:
    feature_cols = json.load(f)

with open('models/ensemble_weights.json', 'r') as f:
    weights = json.load(f)

XGB_WEIGHT = weights['xgb_weight']   # 0.9
GRU_WEIGHT = weights['gru_weight']   # 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Define Bi-GRU architecture (must match training exactly) ──────────

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

# Load Bi-GRU weights
bigru_model = BiGRUModel(
    input_size=17,
    hidden_size=64,
    num_layers=2,
    dropout=0.3
)
bigru_model.load_state_dict(
    torch.load('models/bigru_final.pt', map_location=device)
)
bigru_model = bigru_model.to(device)
bigru_model.eval()

# ── Main prediction function ───────────────────────────────────────────

def predict_aqi(feature_dict: dict) -> float:
    """
    Accepts a dict of 17 raw features.
    Returns predicted next-hour AQI as a float.
    This is the function to call from your FastAPI endpoints.
    """

    # --- XGBoost prediction ---
    # XGBoost takes raw unscaled features as a single-row DataFrame
    input_df = pd.DataFrame([feature_dict])[feature_cols]
    xgb_pred = float(xgb_model.predict(input_df)[0])

    # --- Bi-GRU prediction ---
    # Bi-GRU needs scaled input — use the saved scaler
    input_array = np.array([[feature_dict[col] for col in feature_cols]])
    input_scaled = scaler.transform(input_array)

    # Repeat the row 24 times to create a mock 24-hour sequence
    # In production replace with real 24-hour window from Atharvi's DB
    sequence = np.tile(input_scaled, (24, 1))          # shape: (24, 17)
    sequence_t = torch.tensor(
        sequence, dtype=torch.float32
    ).unsqueeze(0).to(device)                          # shape: (1, 24, 17)

    with torch.no_grad():
        gru_pred = float(bigru_model(sequence_t).cpu().item())

    # --- Weighted ensemble ---
    final_pred = (XGB_WEIGHT * xgb_pred) + (GRU_WEIGHT * gru_pred)
    return round(final_pred, 1)
```

---

## Section 4 — How to use  data pipeline

Atharvi's `get_latest_features()` function connects to the SQLite database and returns the latest sensor reading as the exact 17-key dict the model expects.

Here is how to integrate it in your FastAPI app:

```python
import sqlite3
import pandas as pd

# Connect to the database — do this once at startup
conn = sqlite3.connect('aqi_sensor.db')

FEATURE_COLS = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NH3', 'CO', 'SO2', 'O3',
    'hour', 'month', 'day_of_week', 'is_weekend', 'is_shift_hour',
    'AQI_lag1', 'AQI_lag3', 'PM25_rolling6', 'AQI_rolling6'
]

def get_latest_features() -> dict:
    row = pd.read_sql_query(
        'SELECT * FROM sensor_data ORDER BY rowid DESC LIMIT 1',
        conn
    )
    if row.empty:
        raise RuntimeError('No data in sensor_data table yet.')

    features = {}
    for col in FEATURE_COLS:
        val = row[col].iloc[0]
        if col in ('hour', 'month', 'day_of_week', 'is_weekend', 'is_shift_hour'):
            features[col] = int(val)
        else:
            features[col] = float(val)
    return features
```

---

## Section 5 — Your three FastAPI endpoints

### Endpoint 1 — POST /predict

**What it does:** Takes a worker role and returns the predicted AQI for the next hour using live data from the pipeline.

**Request body:**
```json
{
  "worker_role": "construction"
}
```

**Response:**
```json
{
  "predicted_aqi": 94.5,
  "hour": 14,
  "timestamp": "2026-06-16T14:00:00"
}
```

**Logic inside the endpoint:**
1. Call `get_latest_features()` to get the current feature dict from SQLite
2. Pass that dict to `predict_aqi(feature_dict)` from predict.py
3. Return the result as JSON

---

### Endpoint 2 — POST /risk-score

**What it does:** Computes the worker's cumulative exposure score for their shift and returns a risk tier with a safety directive.

**Request body:**
```json
{
  "worker_role": "construction",
  "shift_duration_hours": 8
}
```

**Response:**
```json
{
  "exposure_score": 423.6,
  "risk_tier": "High",
  "directive": "Wear N95 mask. Limit outdoor tasks to 30-minute intervals.",
  "predicted_aqi": 94.5
}
```

**Logic inside the endpoint:**
1. Call `get_latest_features()` → call `predict_aqi()` → get current AQI
2. Apply the exposure formula: `exposure_score = predicted_aqi × shift_duration_hours × MET_value`
3. Look up MET value from your MET dictionary using worker_role
4. Map exposure score to risk tier using your risk engine
5. Return all fields as JSON

**MET values to use:**
```python
MET_VALUES = {
    'construction':   5.0,   # heavy outdoor physical labour
    'traffic_police': 3.0,   # standing and walking outdoors
    'factory':        4.0,   # moderate indoor/outdoor assembly
    'delivery':       6.0,   # cycling with load
}
```

**Risk tier thresholds:**
```python
# exposure_score = AQI × hours × MET
# Safe:     below 200   → "Normal operations"
# Moderate: 200 – 400   → "Wear dust mask. Hydrate regularly."
# High:     400 – 600   → "Wear N95. Limit outdoor tasks to 30-min intervals."
# Critical: above 600   → "Halt all outdoor operations immediately."
```

---

### Endpoint 3 — GET /shift-plan

**What it does:** Returns an 8-slot hourly schedule labelling each hour as Safe, Moderate, or High based on the current predicted AQI.

**Request parameters:**
```
worker_role=construction&shift_start_hour=6
```

**Response:**
```json
{
  "worker_role": "construction",
  "shift_start_hour": 6,
  "schedule": [
    {"hour": 6,  "predicted_aqi": 87.2, "risk_level": "Safe",     "recommended_intensity": "Heavy"},
    {"hour": 7,  "predicted_aqi": 89.1, "risk_level": "Safe",     "recommended_intensity": "Heavy"},
    {"hour": 8,  "predicted_aqi": 94.5, "risk_level": "Moderate", "recommended_intensity": "Moderate"},
    {"hour": 9,  "predicted_aqi": 98.3, "risk_level": "Moderate", "recommended_intensity": "Moderate"},
    {"hour": 10, "predicted_aqi": 92.1, "risk_level": "Safe",     "recommended_intensity": "Heavy"},
    {"hour": 11, "predicted_aqi": 88.7, "risk_level": "Safe",     "recommended_intensity": "Heavy"},
    {"hour": 12, "predicted_aqi": 105.2,"risk_level": "High",     "recommended_intensity": "Light"},
    {"hour": 13, "predicted_aqi": 110.4,"risk_level": "High",     "recommended_intensity": "Rest"}
  ]
}
```

**Logic inside the endpoint:**
1. Call `predict_aqi()` once to get the current AQI
2. For each of the 8 shift hours, apply a simple offset to simulate how AQI changes across the shift (use your EDA finding — evening hours are worst)
3. Label each hour: Safe (AQI < 100), Moderate (100–150), High (> 150)
4. Map risk level to recommended task intensity

---

## Section 6 — FastAPI app structure

Your project folder should look like this:

```
ShiftSafe_backend/
├── main.py                  ← FastAPI app with all 3 endpoints
├── predict.py               ← Model loading and predict_aqi() function
├── exposure.py              ← MET values, exposure formula, risk tiers
├── pipeline.py              ← get_latest_features() from Atharvi's code
├── aqi_sensor.db            ← SQLite database from Atharvi
├── models/
│   ├── xgboost_aqi_model.pkl
│   ├── bigru_final.pt
│   ├── scaler.pkl
│   ├── feature_cols.json
│   └── ensemble_weights.json
└── requirements.txt
```

Your `requirements.txt`:
```
fastapi
uvicorn
xgboost
torch
scikit-learn
pandas
numpy
pydantic
```

To run locally:
```
uvicorn main.py:app --reload --port 8000
```

Then open `http://localhost:8000/docs` in your browser — FastAPI generates a Swagger UI automatically where you can test all endpoints without writing any frontend code.

---

## Section 7 — CORS setup (critical for frontend)

You must add CORS middleware to your FastAPI app or Likhita's React frontend will get blocked by the browser when calling your API. Add this to `main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Section 8 — Common mistakes and how to avoid them

**Mistake 1 — Wrong feature key names**
The model is extremely sensitive to key names. `PM2.5` is not the same as `PM25` or `pm2_5`. Copy the exact list from Section 2 of this README. Do not rename anything.

**Mistake 2 — Scaling the input before passing to predict_aqi()**
The `predict_aqi()` function in predict.py handles scaling internally for the Bi-GRU. If you scale the inputs before calling the function, the XGBoost half of the ensemble will receive wrong values and predictions will be garbage. Always pass raw unscaled values.

**Mistake 3 — Loading models inside the endpoint function**
Never load `.pkl` or `.pt` files inside an endpoint function. Loading happens at startup once. If you load inside the endpoint, it reloads the model on every API call — this will make each request take 3–5 seconds instead of milliseconds.

**Mistake 4 — Wrong file paths to models**
When you deploy to Render.com, file paths change. Use `os.path.join` with a base directory variable rather than hardcoded paths like `'models/xgboost_aqi_model.pkl'`. This ensures the app finds the files regardless of where it runs.

**Mistake 5 — Forgetting to start the SQLite connection before calling get_latest_features()**
The `conn = sqlite3.connect('aqi_sensor.db')` line must run before `get_latest_features()` is ever called. Put the connection in your app startup event, not inside the function itself.

**Mistake 6 — The Bi-GRU architecture in predict.py must match training exactly**
The `BiGRUModel` class in predict.py must be identical to the one used during training — same hidden_size (64), num_layers (2), dropout (0.3). If you change any of these values, `load_state_dict()` will throw a shape mismatch error.

**Mistake 7 — Not testing with the Swagger UI before connecting to Likhita**
Always test all three endpoints at `localhost:8000/docs` and confirm correct JSON responses before telling Likhita the API is ready. One broken endpoint will break the entire dashboard.

---

## Section 9 — Model performance reference

Use these numbers in any documentation or presentation:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost alone | 3.55 | 5.05 | 0.9445 |
| Bi-GRU alone | 4.80 | 6.62 | 0.9046 |
| Ensemble (XGB:0.9, GRU:0.1) | 3.53 | — | 0.9454 |
| Transfer learning on 6 months | 5.13 | 7.14 | 0.5714 |
| Scratch model on 6 months | 6.88 | 8.14 | 0.4432 |

The ensemble is the production model. R² of 0.9454 means the model explains 94.5% of all AQI variation on unseen test data.

---

## Section 10 — Deployment to Render.com

1. Push your entire `ShiftSafe_backend/` folder to a GitHub repository
2. Go to render.com → New → Web Service → connect your GitHub repo
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Click Deploy — Render gives you a public URL in about 2 minutes
6. Share that URL with Likhita — she replaces all mock data with calls to this URL

---

## Quick checklist before you start coding

- [ ] Downloaded all 5 model files from Google Drive into `models/` folder
- [ ] Got `aqi_sensor.db` from Atharvi
- [ ] Installed all dependencies from requirements.txt
- [ ] Copied `predict.py` exactly as written in Section 3
- [ ] Confirmed `python predict.py` runs without errors locally
- [ ] Called `predict_aqi()` with a test dict and got a float back
- [ ] FastAPI app runs on localhost:8000
- [ ] All 3 endpoints return correct JSON in Swagger UI
- [ ] CORS middleware added
- [ ] Deployed to Render.com and shared URL with Likhita

---

*Built by Divyadarshini M.B (ML Lead) and Atharvi Desurkar (Data Pipeline Lead)*
*ShiftSafe AI — Internship Project 2026*
