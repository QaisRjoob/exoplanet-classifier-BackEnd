# KOI Exoplanet Classification API

A GPU-accelerated machine learning backend for classifying **Kepler Objects of Interest (KOI)** — candidate exoplanets detected by NASA's Kepler Space Telescope. Given a set of KOI parameters, the model predicts whether the object is a **CONFIRMED** planet, a **CANDIDATE**, or a **FALSE POSITIVE**.

> This API powers the [Cosmic Zoom Verse](https://github.com/QaisRjoob/Nasa-exoplanet-explorer-cosmic-zoom-verse) frontend — an interactive 3D space explorer built with React and Three.js.

---

## Features

- **3-class classification** — CONFIRMED / CANDIDATE / FALSE POSITIVE
- **Stacking ensemble model** — LightGBM + XGBoost (95%+ accuracy on test set)
- **Single, batch, and CSV prediction** endpoints
- **In-browser model retraining** via background job API
- **Planet data storage** — save, retrieve, and delete prediction records
- **Live model metrics** — accuracy, precision, recall, F1, confusion matrix

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Framework | FastAPI 0.104 |
| Server | Uvicorn |
| ML | LightGBM 4.6, XGBoost 3.0, scikit-learn 1.7 |
| Data | Pandas, NumPy |
| Validation | Pydantic v2 |
| Serialization | joblib |

---

## Project Structure

```
trying-to-train-main/
├── backend/
│   ├── app.py                  # FastAPI application & all route definitions
│   ├── ml_model.py             # Model loading & prediction logic
│   ├── models.py               # Pydantic request/response schemas
│   ├── config.py               # Settings (host, port, paths, CORS)
│   ├── requirements.txt
│   ├── .env.example
│   ├── data/
│   │   ├── training_data.csv   # Ingested training data
│   │   ├── planets/            # Saved planet JSON records
│   │   └── flexible_planets/   # Flexible-format planet records
│   └── saved_models/
│       ├── stacking_model.pkl  # Trained ensemble model
│       └── model_metadata.pkl  # Accuracy stats & hyperparameters
├── train_3class_gpu.py         # Full training run (~8-15 min, 300 trees)
├── train_3class_gpu_fast.py    # Fast training run (~3-8 min, 150 trees)
├── generate_various_samples.py # Sample data generator
├── sample_data/                # 23 pre-built sample datasets
├── test_*.py                   # API and model test scripts
└── start_server.bat            # Windows quick-launch script
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- `pip`
- (Optional) CUDA-compatible GPU for faster training

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/trying-to-train.git
cd trying-to-train/backend

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the environment file
cp .env.example .env
```

### Environment Variables

Edit `backend/.env` as needed:

```env
API_TITLE="KOI Classification API"
API_VERSION="1.0.0"
API_HOST="0.0.0.0"
API_PORT=8000
MODEL_PATH="saved_models/stacking_model.pkl"
METADATA_PATH="saved_models/model_metadata.pkl"
CORS_ORIGINS="http://localhost:3000,http://localhost:5173,http://localhost:8000"
LOG_LEVEL="INFO"
```

### Running the Server

```bash
# From the backend/ directory
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Or on Windows, double-click **`start_server.bat`** in the project root.

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## Training the Model

A pre-trained model is included at `backend/saved_models/stacking_model.pkl`. To retrain:

```bash
# Full accuracy training (~8-15 min, recommended)
python train_3class_gpu.py

# Fast training (~3-8 min)
python train_3class_gpu_fast.py
```

You can also trigger retraining at runtime via the `/training/start` endpoint.

---

## API Reference

### Health & Info

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info and available routes |
| `GET` | `/health` | Health check |

### Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Classify a single KOI |
| `POST` | `/predict/batch` | Classify multiple KOIs |
| `POST` | `/predict/csv` | Upload a CSV for bulk prediction (up to 10,000 rows) |

**Single prediction request body:**

```json
{
  "koi_period": 9.48,
  "koi_prad": 2.26,
  "koi_teq": 793.0,
  "koi_insol": 93.4,
  "koi_model_snr": 49.2,
  "koi_steff": 5455.0,
  "koi_slogg": 4.46,
  "koi_srad": 0.95,
  "... (40+ KOI parameters)": "..."
}
```

**Response:**

```json
{
  "prediction": "CONFIRMED",
  "confidence": 0.94,
  "probabilities": {
    "CONFIRMED": 0.94,
    "CANDIDATE": 0.04,
    "FALSE POSITIVE": 0.02
  }
}
```

### Model Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/model/info` | Model type, version, feature count, GPU status |
| `GET` | `/model/statistics` | Accuracy, F1, confusion matrix, class distribution |
| `GET` | `/model/hyperparameters` | Hyperparameter configuration |

### Training

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/training/start` | Start retraining (background job) |
| `GET` | `/training/status/{job_id}` | Poll training progress |
| `GET` | `/training/history` | Past training runs |

### Data Ingestion

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/data/ingest` | Upload CSV training data |
| `GET` | `/data/info` | Dataset statistics and column info |

### Planet Storage

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/planets/predict-and-save` | Predict and persist in one call |
| `POST` | `/planets/save` | Save a planet record |
| `GET` | `/planets/{planet_id}` | Retrieve a planet |
| `GET` | `/planets/list` | List all saved planets |
| `DELETE` | `/planets/{planet_id}` | Delete a planet record |

---

## Sample Data for Testing

The `sample_data/` folder contains 23 ready-to-use CSV files. Two good starting points:

### 1. `sample_small_test.csv` — Quick smoke test (5 rows)

A tiny file with 5 KOI entries — ideal for a first test to confirm the API is working.

```bash
curl -X POST http://localhost:8000/predict/csv \
  -F "file=@sample_data/sample_small_test.csv"
```

| Rows | Classes present | Use case |
|------|----------------|----------|
| 5 | Mixed | Fast API check, zero wait |

---

### 2. `sample_mixed_balanced.csv` — Balanced batch test (30 rows)

30 real-world KOI entries with an even spread of CONFIRMED, CANDIDATE, and FALSE POSITIVE labels — useful for verifying the model gives all three output classes.

```bash
curl -X POST http://localhost:8000/predict/csv \
  -F "file=@sample_data/sample_mixed_balanced.csv"
```

| Rows | Classes present | Use case |
|------|----------------|----------|
| 30 | CONFIRMED · CANDIDATE · FALSE POSITIVE | Balanced end-to-end batch test |

> **In the UI:** go to **Add Exoplanet Data → Upload CSV File from many planets**, drag either file in, and hit Upload.

---

## Model Details

The classifier is a **stacking ensemble**:

- **Base learners:** LightGBM + XGBoost (GPU-accelerated when available)
- **Meta-learner:** Logistic Regression
- **Features:** 40+ KOI parameters covering orbital, stellar, and planetary properties
- **Classes:** CONFIRMED · CANDIDATE · FALSE POSITIVE
- **Test accuracy:** 95%+

Feature categories include:
- Orbital parameters (period, impact parameter, transit duration)
- Planetary properties (radius, equilibrium temperature, insolation flux)
- Stellar properties (effective temperature, surface gravity, stellar radius)
- Transit signal quality (SNR, depth, model fit statistics)

---

## Frontend

The companion frontend is **Cosmic Zoom Verse** — an immersive 3D space explorer that lets users:

- Navigate from Earth to the Milky Way in a live 3D scene (Three.js)
- Submit KOI parameters and receive real-time classifications
- Browse and manage saved planet predictions
- Monitor model training metrics and performance charts

**Repository:** [cosmic-zoom-verse](https://github.com/QaisRjoob/Nasa-exoplanet-explorer-cosmic-zoom-verse)  
**Stack:** React 18 · TypeScript · Three.js · Tailwind CSS · shadcn/ui

To connect the frontend to this backend, set `VITE_API_URL=http://localhost:8000` in the frontend `.env`.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [NASA Kepler Mission](https://www.nasa.gov/mission_pages/kepler/main/index.html) for the KOI dataset
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) for public data access
- FastAPI, LightGBM, and XGBoost open-source communities
