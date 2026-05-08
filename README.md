# Exoplanet Classifier — Backend API

A machine learning backend for classifying **Kepler Objects of Interest (KOI)** — candidate exoplanets detected by NASA's Kepler Space Telescope. Given a set of physical and orbital measurements, the model predicts whether the object is a **CONFIRMED** planet, a **CANDIDATE**, or a **FALSE POSITIVE**.

This API powers the [Cosmic Zoom Verse](https://github.com/QaisRjoob/Nasa-exoplanet-explorer-cosmic-zoom-verse) frontend — an interactive 3D space explorer built with React and Three.js.

---

## Live Demo

> **Backend API:** https://nasa-kepler-sight-backend.onrender.com
> **Frontend App:** https://nasa-kepler-sight.onrender.com

---

## Team

This project was built collaboratively by:

| Name | GitHub |
|------|--------|
| Qais Rjoob | [@QaisRjoob](https://github.com/QaisRjoob) |
| Mohammed Nnimer | [@mohammednnimer](https://github.com/mohammednnimer) |
| Laith | [@laithw2](https://github.com/laithw2) |
| Baraa | [@Baraa-Rj](https://github.com/Baraa-Rj) |

---

## About the Project

The NASA Kepler Space Telescope observed over 150,000 stars and produced thousands of planet candidates — Kepler Objects of Interest (KOIs). Each KOI is described by a set of transit photometry measurements: how much the star dims, for how long, how often, and what the star itself looks like. Most candidates turn out to be false positives caused by eclipsing binary stars or instrument noise. Only a fraction are true confirmed planets.

This project trains a machine learning model on the official KOI dataset from the NASA Exoplanet Archive to automatically classify new candidates. The result is a REST API that takes KOI measurements as input and returns a classification with a confidence score.

---

## The Machine Learning Model

### Problem

Multi-class classification across three labels:

| Label | Meaning |
|-------|---------|
| `CONFIRMED` | The object has been verified as a real planet |
| `CANDIDATE` | Transit signal looks planetary but needs follow-up |
| `FALSE POSITIVE` | Signal is not caused by a planet |

### Architecture

The model is a **stacking ensemble** — a technique where multiple independent models each make a prediction, and a final meta-learner combines their outputs into one decision:

```
Input features (40+ KOI parameters)
        │
   ┌────┴────┐
   │         │
LightGBM   XGBoost        ← base learners (GPU-accelerated)
   │         │
   └────┬────┘
        │
 Logistic Regression      ← meta-learner
        │
   Classification
 CONFIRMED / CANDIDATE / FALSE POSITIVE
```

**Base learners:**
- **LightGBM** — gradient boosted trees optimized for speed and memory efficiency
- **XGBoost** — gradient boosted trees with regularization, strong on tabular data

**Meta-learner:**
- **Logistic Regression** — learns how to combine the base learner outputs

**Training details:**
- Dataset: NASA Kepler KOI cumulative table (~9,000 objects)
- Features: 40+ parameters covering orbital, stellar, and transit signal properties
- Validation: 5-fold cross-validation
- Test accuracy: 95%+
- Class imbalance handled through stratified splitting

### Input Features

The model uses the following categories of KOI parameters:

| Category | Features |
|----------|---------|
| Orbital | `koi_period`, `koi_time0bk`, `koi_impact`, `koi_duration` |
| Transit signal | `koi_depth`, `koi_model_snr` |
| Planet properties | `koi_prad`, `koi_teq`, `koi_insol` |
| Stellar properties | `koi_steff`, `koi_slogg`, `koi_srad`, `koi_smass`, `koi_kepmag` |

---

## Backend Architecture

The backend is a **FastAPI** application organized into the following layers:

```
backend/
├── app.py                  # Application entry point, all route definitions
├── ml_model.py             # Model loading, caching, and prediction logic
├── models.py               # Pydantic request/response schemas
├── config.py               # Settings (host, port, model paths, CORS)
├── requirements.txt
├── .env.example
├── data/
│   ├── training_data.csv   # Uploaded training data
│   ├── planets/            # Saved planet prediction records (JSON)
│   └── flexible_planets/   # Flexible-format planet records
└── saved_models/
    ├── stacking_model.pkl  # Trained ensemble model
    └── model_metadata.pkl  # Accuracy stats, feature names, hyperparameters
```

### Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Framework | FastAPI 0.104 |
| Server | Uvicorn |
| ML models | LightGBM 4.6, XGBoost 3.0, scikit-learn 1.7 |
| Data processing | Pandas, NumPy |
| Validation | Pydantic v2 |
| Serialization | joblib |

### Key Design Decisions

- **Model loaded once at startup** — not per request, for fast response times
- **Background training** — retraining runs as an async job, API stays responsive
- **Multi-encoding CSV support** — handles UTF-8, Latin-1, Windows-1252 automatically
- **Graceful degradation** — API starts even if model file is missing; endpoints return 503 with a clear message

---

## API Reference

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info and route map |
| `GET` | `/health` | Model load status and version |

### Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Classify a single KOI |
| `POST` | `/predict/batch` | Classify a list of KOIs |
| `POST` | `/predict/csv` | Upload a CSV file for bulk prediction (up to 10,000 rows) |

**Example request:**

```json
POST /predict
{
  "features": {
    "koi_period": 9.48,
    "koi_prad": 2.26,
    "koi_teq": 793.0,
    "koi_model_snr": 49.2,
    "koi_steff": 5455.0,
    "koi_srad": 0.95
  }
}
```

**Example response:**

```json
{
  "prediction": "0",
  "prediction_label": "CONFIRMED",
  "confidence": 0.94,
  "probabilities": {
    "0": 0.94,
    "1": 0.04,
    "2": 0.02
  },
  "model_version": "1.0.0"
}
```

### Model Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/model/info` | Model type, version, feature count |
| `GET` | `/model/statistics` | Accuracy, F1, confusion matrix, class distribution |
| `GET` | `/model/hyperparameters` | Hyperparameters used during training |

### Training

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/training/start` | Start a background retraining job |
| `GET` | `/training/status/{job_id}` | Poll training progress |
| `GET` | `/training/history` | List past training runs |

### Data Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/data/ingest` | Upload new CSV training data |
| `GET` | `/data/info` | Dataset size and class distribution |

### Planet Storage

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/planets/predict-and-save` | Predict and save in one request |
| `GET` | `/planets/list` | List all saved planets |
| `GET` | `/planets/{planet_id}` | Retrieve a planet by ID |
| `DELETE` | `/planets/{planet_id}` | Delete a planet record |

---

## Running Locally

### Prerequisites

- Python 3.10 or higher

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/QaisRjoob/exoplanet-classifier-BackEnd.git
cd exoplanet-classifier-BackEnd/backend

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment file
cp .env.example .env

# 5. Start the server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

---

## Training the Model

A pre-trained model is included at `backend/saved_models/stacking_model.pkl`. To retrain with new data:

```bash
# Full training (~8–15 min, 500 trees, recommended)
python train_3class_gpu.py

# Fast training (~3–8 min, 150 trees)
python train_3class_gpu_fast.py
```

Or trigger retraining at runtime via `POST /training/start`.

---

## Sample Data

The `sample_data/` folder contains 23 ready-to-use CSV files for testing the API:

| File | Rows | Use |
|------|------|-----|
| `sample_small_test.csv` | 5 | Quick smoke test |
| `sample_mixed_balanced.csv` | 30 | Balanced class test |
| `sample_large_test.csv` | 1000+ | Performance test |

```bash
curl -X POST http://localhost:8000/predict/csv \
  -F "file=@sample_data/sample_small_test.csv"
```

---

## Deploying to Render

This repository includes a `render.yaml` file. To deploy:

1. Push the repository to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect this repository
4. Render will detect `render.yaml` and configure the service automatically
5. The start command is: `uvicorn app:app --host 0.0.0.0 --port $PORT`

---

## Related Repository

**Frontend — Cosmic Zoom Verse**  
[github.com/QaisRjoob/Nasa-exoplanet-explorer-cosmic-zoom-verse](https://github.com/QaisRjoob/Nasa-exoplanet-explorer-cosmic-zoom-verse)

---

## Data Source

NASA Exoplanet Archive — Kepler Objects of Interest cumulative table  
[exoplanetarchive.ipac.caltech.edu](https://exoplanetarchive.ipac.caltech.edu/)
