# ShiftSafe AI — Complete Project Handoff
### ML Lead: Divyadarshini M.B
### Data Pipeline Lead: Atharvi Desurkar
### Backend Lead: Aadish Sarin (your task)
### Frontend Lead: Marella Likhita Sri Durga
### Project ID: 105 | IEEE CS Bangalore Chapter Internship 2026

---

## What has been built and is ready for you

Three complete workstreams have been finished and verified:

1. **ML Model (Divyadarshini)** — A trained ensemble of XGBoost + Bi-GRU that predicts next-hour AQI from 17 input features. Verified end-to-end. Ensemble R²: 0.9454, MAE: 3.53. The model is packaged as a single callable function `predict_aqi()`.

2. **Data Pipeline (Atharvi)** — A fully verified software pipeline that loads 48,189 rows of real Bengaluru AQI data into SQLite, simulates live sensor arrival, applies KNN imputation for sensor failures, and exposes `get_latest_features()` returning the exact 17 keys the model needs. End-to-end test confirmed: `get_latest_features()` → `predict_aqi()` → predicted AQI: 46.4 

3. **Frontend (Likhita)** — React dashboard built with mock data. Exposure score panel, shift schedule chart, hourly risk slots, and alert system are complete. Waiting for your API URL to wire real data.

**Your job: wrap everything above into a FastAPI backend with 3 endpoints and deploy to Render.com.**

---

## What was built — full summary

### Divyadarshini's ML workstream

| Notebook | What it did | Key result |
|---|---|---|
| 01_data_exploration | Loaded city_hour.csv, audited 48,192 Bengaluru rows | Missing value map confirmed |
| 02_data_cleaning + EDA | KNN imputation, IQR clipping, 8 engineered features, 5 EDA plots | blr_clean.csv — 48,189 rows, 20 cols, 0 missing |
| 03_xgboost_model | XGBoost trained with time-ordered 80/20 split | R²: 0.9445, MAE: 3.55, RMSE: 5.05 |
| 04_bigru_model | Bi-GRU (PyTorch), 24-hr lookback, ensemble fusion | Bi-GRU R²: 0.9046 / Ensemble R²: 0.9454 |
| 05_transfer_learning | Froze GRU layers, fine-tuned on 6-month subset | +0.1282 R² over scratch (28.9% improvement) |
| 06_handoff | Packaged predict_aqi_ensemble() + all model files | 6 files saved, function tested |

**Key EDA finding:** Evening hours 18:00–22:00 have the highest AQI in Bengaluru (avg ~97–99). Morning hours 7:00–10:00 are cleanest (avg ~86–87). This directly justifies the shift-planning feature.

**Top 3 model features by XGBoost importance:** AQI_lag1 > AQI_rolling6 > AQI_lag3. Engineered features outranked all raw pollutants.

### Atharvi's data pipeline workstream

| Component | What it does | Status |
|---|---|---|
| aqi_sensor.db | SQLite DB with 48,189 rows of blr_clean.csv loaded safely |  Complete |
| Sensor simulator | Replays historical data in time order, mimics live arrival |  Complete |
| get_latest_features() | Returns 17 correct keys, raw unscaled, correct types |  Verified |
| IoT simulation (live_iot.db) | ESP8266/DHT-11/MQ-135 software simulation with dropout |  Complete |
| KNN imputation service | Fills missing sensor readings automatically |  Tested (6 tests pass) |
| APScheduler | Polls every 5 minutes (10 seconds in demo mode) |  Complete |
| background.py | asyncio version of ingestion loop for FastAPI |  Written and saved |
| End-to-end test | get_latest_features() → XGBoost → 46.4 AQI predicted |  Verified |

---

## Section 1 — Files you need from Google Drive

### From `ShiftSafe_AI/models/`

| File | What it is | Used by |
|---|---|---|
| `xgboost_aqi_model.pkl` | Trained XGBoost model (primary model, 90% weight) | predict endpoint |
| `bigru_final.pt` | Trained Bi-GRU weights (10% weight in ensemble) | predict endpoint |
| `scaler.pkl` | StandardScaler fitted on 48,189 rows — required for Bi-GRU input | predict.py internally |
| `feature_cols.json` | Exact ordered list of 17 feature names | Both models |
| `ensemble_weights.json` | `{"xgb_weight": 0.9, "gru_weight": 0.1}` | Ensemble fusion |
| `background.py` | asyncio ingestion loop for FastAPI startup | main.py startup event |

### From `ShiftSafe_AI/data/`

| File | What it is |
|---|---|
| `aqi_sensor.db` | SQLite database — 48,189 rows of Bengaluru AQI data. This is what `get_latest_features()` reads from. |
| `live_iot.db` | SQLite database — live IoT simulation stream. Monitoring only, not used by ML model directly. |

Download all files. Put model files in `models/` and database files in your project root.

---

## Section 2 — The 17 features your API must handle

Every key name, spelling, and data type must match exactly. One typo silently breaks predictions with no error message.

```python
{
    # Raw pollutant readings — all float, raw unscaled values
    'PM2.5':         float,   # e.g. 45.2   (range roughly 0–200)
    'PM10':          float,   # e.g. 78.1
    'NO':            float,   # e.g. 3.1
    'NO2':           float,   # e.g. 18.4
    'NH3':           float,   # e.g. 12.1
    'CO':            float,   # e.g. 0.8
    'SO2':           float,   # e.g. 5.2
    'O3':            float,   # e.g. 22.1

    # Time features — all int
    'hour':          int,     # 0–23
    'month':         int,     # 1–12
    'day_of_week':   int,     # 0=Monday, 6=Sunday
    'is_weekend':    int,     # 0 or 1
    'is_shift_hour': int,     # 1 if hour is between 6 and 18, else 0

    # Engineered history features — all float
    'AQI_lag1':      float,   # AQI value from 1 hour ago
    'AQI_lag3':      float,   # AQI value from 3 hours ago
    'PM25_rolling6': float,   # 6-hour rolling average of PM2.5
    'AQI_rolling6':  float,   # 6-hour rolling average of AQI
}
```

**These values must be RAW and UNSCALED.** The model handles scaling internally for the Bi-GRU. Never normalise anything before passing to predict_aqi().

---

## Section 3 — How to load and call the prediction model

Install dependencies:
```
pip install xgboost torch scikit-learn pandas numpy fastapi uvicorn pydantic apscheduler
```

Copy this exactly into `predict.py` in your project root:

```python
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Load all model assets ONCE at startup — never inside endpoint functions ──

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

# ── Bi-GRU architecture — must match training EXACTLY, do not change values ──

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
    input_size=17,    # must be 17
    hidden_size=64,   # must be 64
    num_layers=2,     # must be 2
    dropout=0.3       # must be 0.3
)
bigru_model.load_state_dict(
    torch.load('models/bigru_final.pt', map_location=device)
)
bigru_model = bigru_model.to(device)
bigru_model.eval()

# ── Main prediction function — this is what your endpoints call ──

def predict_aqi(feature_dict: dict) -> float:
    """
    Input:  dict with exactly 17 keys (see Section 2), raw unscaled values.
    Output: predicted next-hour AQI as a float.

    Verified end-to-end: get_latest_features() → predict_aqi() → 46.4 AQI 
    """
    # XGBoost — raw unscaled features
    input_df = pd.DataFrame([feature_dict])[feature_cols]
    xgb_pred = float(xgb_model.predict(input_df)[0])

    # Bi-GRU — scale internally, then create 24-step sequence
    input_array = np.array([[feature_dict[col] for col in feature_cols]])
    input_scaled = scaler.transform(input_array)
    sequence = np.tile(input_scaled, (24, 1))
    sequence_t = torch.tensor(
        sequence, dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        gru_pred = float(bigru_model(sequence_t).cpu().item())

    # Weighted ensemble
    return round((XGB_WEIGHT * xgb_pred) + (GRU_WEIGHT * gru_pred), 1)
```

---

## Section 4 — How to use Atharvi's data pipeline

Copy this exactly into `pipeline.py` in your project root:

```python
import sqlite3
import pandas as pd

# Connect once at startup — never inside endpoint functions
conn = sqlite3.connect('aqi_sensor.db', check_same_thread=False)

FEATURE_COLS = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NH3', 'CO', 'SO2', 'O3',
    'hour', 'month', 'day_of_week', 'is_weekend', 'is_shift_hour',
    'AQI_lag1', 'AQI_lag3', 'PM25_rolling6', 'AQI_rolling6'
]

INT_COLS = {'hour', 'month', 'day_of_week', 'is_weekend', 'is_shift_hour'}

def get_latest_features() -> dict:
    """
    Returns the most recent row from aqi_sensor.db as a
    17-key model-ready feature dict. Raw unscaled values.
    Verified: returns correct types and keys 
    """
    row = pd.read_sql_query(
        'SELECT * FROM sensor_data ORDER BY rowid DESC LIMIT 1',
        conn
    )
    if row.empty:
        raise RuntimeError('sensor_data table is empty.')

    features = {}
    for col in FEATURE_COLS:
        val = row[col].iloc[0]
        features[col] = int(val) if col in INT_COLS else float(val)
    return features


def get_live_data() -> dict:
    """
    Alias for get_latest_features().
    Use whichever name you prefer in your endpoints.
    """
    return get_latest_features()
```

### background.py — already saved in Drive, copy into project root

Add this to your `main.py` startup event to run the live IoT ingestion loop automatically on Render.com:

```python
import asyncio
from background import run_ingestion_loop

@app.on_event("startup")
async def start_background():
    asyncio.create_task(run_ingestion_loop())
```

---

## Section 5 — Your three FastAPI endpoints

### Endpoint 1 — POST /predict

**Request body:**
```json
{ "worker_role": "construction" }
```

**Response:**
```json
{
  "predicted_aqi": 94.5,
  "hour": 14,
  "timestamp": "2026-06-26T14:00:00"
}
```

**Logic:**
1. Call `get_latest_features()` from pipeline.py
2. Pass dict to `predict_aqi()` from predict.py
3. Return result as JSON

---

### Endpoint 2 — POST /risk-score

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

**Exposure formula:** `exposure_score = predicted_aqi × shift_duration_hours × MET_value`

**MET values:**
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
# Below 200   → Safe     → "Normal operations. Standard PPE applies."
# 200 – 400   → Moderate → "Wear dust mask. Hydrate every 30 minutes."
# 400 – 600   → High     → "Wear N95. Limit outdoor tasks to 30-min intervals."
# Above 600   → Critical → "Halt all outdoor operations immediately."
```

---

### Endpoint 3 — GET /shift-plan

**Request parameters:** `?worker_role=construction&shift_start_hour=6`

**Response:**
```json
{
  "worker_role": "construction",
  "shift_start_hour": 6,
  "schedule": [
    {"hour": 6,  "predicted_aqi": 87.2,  "risk_level": "Safe",     "recommended_intensity": "Heavy"},
    {"hour": 7,  "predicted_aqi": 89.1,  "risk_level": "Safe",     "recommended_intensity": "Heavy"},
    {"hour": 8,  "predicted_aqi": 94.5,  "risk_level": "Moderate", "recommended_intensity": "Moderate"},
    {"hour": 9,  "predicted_aqi": 98.3,  "risk_level": "Moderate", "recommended_intensity": "Moderate"},
    {"hour": 10, "predicted_aqi": 92.1,  "risk_level": "Safe",     "recommended_intensity": "Heavy"},
    {"hour": 11, "predicted_aqi": 88.7,  "risk_level": "Safe",     "recommended_intensity": "Heavy"},
    {"hour": 12, "predicted_aqi": 105.2, "risk_level": "High",     "recommended_intensity": "Light"},
    {"hour": 13, "predicted_aqi": 110.4, "risk_level": "High",     "recommended_intensity": "Rest"}
  ]
}
```

**Logic:**
1. Call `predict_aqi()` for base AQI
2. Apply hourly offsets — EDA finding: evening hours add +10–15 AQI over morning
3. Label: Safe = AQI below 100, Moderate = 100–150, High = above 150
4. Map to intensity: Safe → Heavy, Moderate → Moderate, High → Light or Rest

---

## Section 6 — FastAPI project structure

```
ShiftSafe_backend/
├── main.py              ← FastAPI app — all 3 endpoints + startup event
├── predict.py           ← Model loading + predict_aqi() — copy from Section 3
├── exposure.py          ← MET values, exposure formula, risk tier logic
├── pipeline.py          ← get_latest_features() — copy from Section 4
├── background.py        ← asyncio ingestion loop — get from Drive
├── aqi_sensor.db        ← 48,189 rows — get from Drive
├── live_iot.db          ← IoT stream — get from Drive
├── models/
│   ├── xgboost_aqi_model.pkl
│   ├── bigru_final.pt
│   ├── scaler.pkl
│   ├── feature_cols.json
│   └── ensemble_weights.json
└── requirements.txt
```

**requirements.txt:**
```
fastapi
uvicorn
xgboost
torch
scikit-learn
pandas
numpy
pydantic
apscheduler
```

**Run locally:**
```bash
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/docs` to test all endpoints in Swagger UI before connecting to Likhita.

---

## Section 7 — CORS setup (required for Likhita's frontend)

Add this to `main.py` or her React app will be blocked by the browser:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Section 8 — Common mistakes and how to avoid them

**Mistake 1 — Wrong feature key names**
`PM2.5` is not the same as `PM25` or `pm2_5`. Copy the exact list from Section 2. One wrong key produces a silent KeyError that breaks every prediction.

**Mistake 2 — Scaling inputs before calling predict_aqi()**
predict_aqi() handles scaling internally. If you scale before calling it, XGBoost gets wrong values. Always pass raw unscaled values.

**Mistake 3 — Loading model files inside endpoint functions**
Load `.pkl` and `.pt` files once at startup only. Loading inside an endpoint adds 3–5 seconds per request.

**Mistake 4 — Hardcoded file paths**
Use `os.path.join(BASE_DIR, 'models', 'xgboost_aqi_model.pkl')` where `BASE_DIR = os.path.dirname(__file__)`. Hardcoded paths break on Render.com.

**Mistake 5 — Forgetting to open SQLite connection before calling get_latest_features()**
`conn = sqlite3.connect('aqi_sensor.db')` must run at startup, not inside the function itself.

**Mistake 6 — Changing BiGRUModel parameters**
`input_size=17, hidden_size=64, num_layers=2, dropout=0.3` must be identical to training. Any change causes a shape mismatch error when loading weights.

**Mistake 7 — Not adding background.py startup task**
Without `asyncio.create_task(run_ingestion_loop())` in your startup event, the live IoT pipeline does not run on Render.com. The model will still predict using the last historical row from aqi_sensor.db but live data will not stream.

**Mistake 8 — Not testing in Swagger before telling Likhita**
Test all 3 endpoints at `localhost:8000/docs`. Confirm correct JSON shapes. One broken endpoint breaks Likhita's entire dashboard.

---

## Section 9 — Model performance reference

| Model | MAE | RMSE | R² | Data |
|---|---|---|---|---|
| XGBoost alone | 3.55 | 5.05 | 0.9445 | 48,189 rows |
| Bi-GRU alone | 4.80 | 6.62 | 0.9046 | 48,189 rows |
| Ensemble (XGB 0.9 / GRU 0.1) | 3.53 | — | 0.9454 | 48,189 rows |
| Transfer learning (6 months) | 5.13 | 7.14 | 0.5714 | ~4,380 rows |
| Scratch model (6 months) | 6.88 | 8.14 | 0.4432 | ~4,380 rows |

Production model is the ensemble. R²: 0.9454 means the model explains 94.5% of all AQI variation on unseen test data.

End-to-end pipeline verification: `get_latest_features()` → `predict_aqi()` → **predicted AQI: 46.4** 

---

## Section 10 — Deployment to Render.com

1. Push entire `ShiftSafe_backend/` folder to a GitHub repository
2. Go to render.com → New → Web Service → connect your GitHub repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Click Deploy — public URL ready in about 2 minutes
6. Share URL with Likhita — she replaces all mock data with calls to this URL

---

## Quick checklist before you start coding

- [ ] All 6 model files downloaded from Drive into `models/`
- [ ] `aqi_sensor.db` and `live_iot.db` in project root
- [ ] `background.py` in project root
- [ ] `predict.py` copied exactly from Section 3
- [ ] `pipeline.py` copied exactly from Section 4
- [ ] `python predict.py` runs with no errors
- [ ] `predict_aqi(test_dict)` returns a float
- [ ] FastAPI app runs on localhost:8000
- [ ] All 3 endpoints return correct JSON at /docs
- [ ] CORS middleware added to main.py
- [ ] background.py startup task added to main.py
- [ ] Deployed to Render.com
- [ ] Public URL shared with Likhita

---

*ML Lead: Divyadarshini M.B — XGBoost R²: 0.9445 | Bi-GRU R²: 0.9046 | Ensemble R²: 0.9454 | Transfer learning: +0.1282 R²*  
*Data Pipeline Lead: Atharvi Desurkar — 48,189 rows | 6 tests passing | End-to-end verified *  
*ShiftSafe AI — IEEE CS Bangalore Chapter Internship 2026 | Project ID: 105*
