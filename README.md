# Laptop Price Predictor

A full-stack machine learning web app that predicts Indian laptop prices based on hardware specifications — all in a **single file**.

| Layer | Technology |
|---|---|
| ML Model | XGBoost Regressor (log-price transform) |
| Backend API | FastAPI + Uvicorn (background thread) |
| Frontend UI | Streamlit + Plotly |
| Data | 893 real Indian e-commerce laptop listings |

---

## Project Structure

```
laptop_price_predictor/
├── data/
│   └── data.csv                   # Raw dataset (893 listings)
├── models/                        # Auto-created on first run
│   ├── laptop_price_model.pkl     # Trained XGBoost pipeline
│   └── feature_meta.pkl           # Dropdown options + metrics
├── main.py                        # Entire app — train + API + UI
├── requirements.txt
└── README.md
```

Everything lives in **`main.py`** — no separate backend or frontend files needed.

---

## Quick Start

### 1. Install dependencies

```bash
cd laptop_price_predictor
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run main.py
```

That's it. On first launch `main.py` will:
1. **Train** the XGBoost model automatically (~30 seconds, one-time only)
2. **Start** the FastAPI backend on port `8000` in a background thread
3. **Open** the Streamlit UI at **http://localhost:8501**

On all subsequent runs the saved model is loaded instantly — no retraining.

---

## How main.py is structured

| Section | Contents |
|---|---|
| Section 0 | Imports |
| Section 1 | Paths & constants |
| Section 2 | Feature engineering helpers |
| Section 3 | Model training (`train_model()`) |
| Section 4 | FastAPI app + `/predict`, `/meta`, `/health` endpoints |
| Section 5 | Streamlit frontend (`run_frontend()`) |
| Section 6 | Entrypoint — wires sections 3–5 together |

---

## API Reference

The FastAPI backend runs internally on `http://localhost:8000`. You can also hit it directly from any HTTP client.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Model status + R² score |
| GET | `/meta` | Dropdown options + model metrics |
| POST | `/predict` | Predict laptop price |

Interactive API docs: **http://localhost:8000/docs**

### POST `/predict` — Request Body

```json
{
  "brand":            "HP",
  "ram_gb":           16,
  "rom_gb":           512,
  "gpu_vram_gb":      4,
  "dedicated_gpu":    1,
  "cpu_gen":          12,
  "cpu_tier":         "I5",
  "display_size":     15.6,
  "spec_rating":      72.0,
  "is_ssd":           1,
  "warranty":         1,
  "ram_type":         "DDR4",
  "resolution_class": "FHD",
  "os_clean":         "Windows"
}
```

### POST `/predict` — Response

```json
{
  "predicted_price":         68999.00,
  "predicted_price_rounded": 69000,
  "price_range_low":         62098,
  "price_range_high":        75898,
  "currency":                "INR"
}
```

The `price_range_low` / `price_range_high` represent a ±10% confidence band around the prediction.

---

## Feature Engineering

| Raw Column | Engineered Feature | Description |
|---|---|---|
| `Ram` | `ram_gb` | Numeric GB extracted from string |
| `ROM` | `rom_gb` | GB extracted — TB values converted (×1024) |
| `GPU` | `gpu_vram_gb` | Dedicated VRAM in GB (0 if integrated) |
| `GPU` | `dedicated_gpu` | Binary flag — 1 if NVIDIA/AMD discrete GPU |
| `processor` | `cpu_gen` | Generation number (e.g. 12 for "12th Gen") |
| `processor` | `cpu_tier` | I3 / I5 / I7 / I9 / RYZEN3–9 / M1 / M2 |
| `resolution_width/height` | `resolution_class` | HD / FHD / QHD / 4K+ |
| `OS` | `os_clean` | Normalised OS name (Windows / Mac / Android) |
| `ROM_type` | `is_ssd` | Binary flag — 1 if SSD |

---

## Dataset Columns

| Column | Description |
|---|---|
| `brand` | Laptop brand |
| `name` | Full product name |
| `price` | Price in INR — prediction target |
| `spec_rating` | Aggregate spec score (0–100) |
| `processor` | CPU description string |
| `CPU` | Core / thread configuration |
| `Ram` | RAM size string (e.g. "16GB") |
| `Ram_type` | DDR4 / DDR5 / LPDDR5 / LPDDR5X etc. |
| `ROM` | Storage size string (e.g. "512GB", "1TB") |
| `ROM_type` | SSD / HDD / Hard-Disk |
| `GPU` | GPU description string |
| `display_size` | Diagonal size in inches |
| `resolution_width` / `resolution_height` | Display resolution in pixels |
| `OS` | Operating system |
| `warranty` | Warranty period in years |

---

## Architecture

```
streamlit run main.py
        |
        |-- Section 3: train_model()          (one-time, saves .pkl files)
        |
        |-- Section 4: FastAPI on :8000       (background daemon thread)
        |       GET  /meta    --> dropdown options + R2/MAE/RMSE
        |       POST /predict --> XGBoost inference --> price + range
        |
        `-- Section 5: Streamlit on :8501     (main thread)
                Sidebar inputs --> POST /predict --> price card + gauge chart
```

---

## Model Performance

| Metric | Value |
|---|---|
| R² Score | 0.90 |
| MAE | Rs.10,728 |
| RMSE | Rs.18,331 |
| 5-fold CV R² | 0.85 ± 0.06 |
| Training samples | ~714 |
| Test samples | ~179 |
