"""
main.py – Laptop Price Predictor (All-in-one)
=============================================
Single file that contains:
  1. Data preprocessing + XGBoost model training
  2. FastAPI backend  (runs in a background thread on port 8000)
  3. Streamlit frontend (main thread)

Run:
    streamlit run main.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 – Imports
# ─────────────────────────────────────────────────────────────────────────────
import re
import os
import sys
import time
import threading
import numpy as np
import pandas as pd
import joblib
import requests
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import uvicorn

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – Paths & Constants
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "laptop_price_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "feature_meta.pkl")
API_URL    = "http://localhost:8000"

NUM_FEATURES = [
    "ram_gb", "rom_gb", "gpu_vram_gb", "dedicated_gpu",
    "cpu_gen", "display_size", "spec_rating", "is_ssd", "warranty",
]
CAT_FEATURES = ["brand", "cpu_tier", "Ram_type", "resolution_class", "os_clean"]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – Feature Engineering Helpers
# ─────────────────────────────────────────────────────────────────────────────
def extract_ram_gb(val: str) -> float:
    m = re.search(r"(\d+)", str(val))
    return float(m.group(1)) if m else 0.0


def extract_rom_gb(val: str) -> float:
    s = str(val).upper()
    m = re.search(r"(\d+)", s)
    if not m:
        return 0.0
    num = float(m.group(1))
    return num * 1024 if "TB" in s else num


def extract_gpu_vram(val: str) -> float:
    m = re.search(r"(\d+)GB", str(val), re.IGNORECASE)
    return float(m.group(1)) if m else 0.0


def is_dedicated_gpu(val: str) -> int:
    keywords = ["nvidia", "amd radeon rx", "rtx", "gtx", "rx 6", "rx 7"]
    return int(any(k in str(val).lower() for k in keywords))


def extract_cpu_gen(val: str) -> float:
    m = re.search(r"(\d+)(st|nd|rd|th)\s+gen", str(val), re.IGNORECASE)
    return float(m.group(1)) if m else 0.0


def extract_cpu_tier(val: str) -> str:
    v = str(val).lower()
    for tier in ["i9", "i7", "i5", "i3",
                 "ryzen 9", "ryzen 7", "ryzen 5", "ryzen 3",
                 "m2", "m1", "celeron", "pentium"]:
        if tier in v:
            return tier.replace(" ", "").upper()
    return "OTHER"


def extract_resolution_class(w: float, h: float) -> str:
    w, h   = (w or 0), (h or 0)
    # Normalise so width is always the longer side
    longer = max(w, h)
    shorter = min(w, h)
    if longer >= 3840 and shorter >= 2160:   # true 4K
        return "4K+"
    if longer >= 2240:                        # QHD / 2.5K / MacBook Retina
        return "QHD"
    if longer >= 1920:                        # FHD 1080p
        return "FHD"
    return "HD"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ram_gb"]           = df["Ram"].apply(extract_ram_gb)
    df["rom_gb"]           = df["ROM"].apply(extract_rom_gb)
    df["gpu_vram_gb"]      = df["GPU"].apply(extract_gpu_vram)
    df["dedicated_gpu"]    = df["GPU"].apply(is_dedicated_gpu)
    df["cpu_gen"]          = df["processor"].apply(extract_cpu_gen)
    df["cpu_tier"]         = df["processor"].apply(extract_cpu_tier)
    df["resolution_class"] = df.apply(
        lambda r: extract_resolution_class(
            r.get("resolution_width", 0), r.get("resolution_height", 0)
        ), axis=1
    )
    df["os_clean"] = (
        df["OS"].str.replace(r"\s+OS$", "", regex=True).str.strip()
        .replace({"Windows 11": "Windows", "Windows 10": "Windows"})
    )
    df["is_ssd"]      = df["ROM_type"].str.upper().str.contains("SSD").astype(int)
    df["spec_rating"] = df["spec_rating"].fillna(df["spec_rating"].median())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – Model Training
# ─────────────────────────────────────────────────────────────────────────────
def train_model() -> dict:
    """Train XGBoost pipeline and save to disk. Returns metrics dict."""
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    df = build_features(df)

    all_features = NUM_FEATURES + CAT_FEATURES
    df = df.dropna(subset=all_features)

    X = df[all_features]
    y = np.log1p(df["price"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
        )),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = np.expm1(pipeline.predict(X_test))
    y_true = np.expm1(y_test)
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2     = r2_score(y_true, y_pred)

    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    meta = {
        "num_features":  NUM_FEATURES,
        "cat_features":  CAT_FEATURES,
        "all_features":  all_features,
        "brands":        sorted(df["brand"].dropna().unique().tolist()),
        "cpu_tiers":     sorted(df["cpu_tier"].dropna().unique().tolist()),
        "ram_types":     sorted(df["Ram_type"].dropna().unique().tolist()),
        "res_classes":   sorted(df["resolution_class"].dropna().unique().tolist()),
        "os_options":    sorted(df["os_clean"].dropna().unique().tolist()),
        "metrics":       {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)},
    }
    joblib.dump(meta, META_PATH)
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – FastAPI Backend
# ─────────────────────────────────────────────────────────────────────────────
_model = None
_meta  = None

def _load_artefacts():
    global _model, _meta
    _model = joblib.load(MODEL_PATH)
    _meta  = joblib.load(META_PATH)


api = FastAPI(
    title="Laptop Price Predictor API",
    description="XGBoost regression model for laptop price prediction.",
    version="1.0.0",
)
api.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class LaptopFeatures(BaseModel):
    brand:            str
    ram_gb:           float = Field(..., ge=1)
    rom_gb:           float = Field(..., ge=1)
    gpu_vram_gb:      float = Field(0.0, ge=0)
    dedicated_gpu:    int   = Field(0, ge=0, le=1)
    cpu_gen:          float = Field(0.0, ge=0)
    cpu_tier:         str
    display_size:     float = Field(..., gt=0)
    spec_rating:      float = Field(69.0, ge=0, le=100)
    is_ssd:           int   = Field(1, ge=0, le=1)
    warranty:         int   = Field(1, ge=0)
    ram_type:         str
    resolution_class: str
    os_clean:         str


class PredictionResponse(BaseModel):
    predicted_price:         float
    predicted_price_rounded: int
    price_range_low:         int
    price_range_high:        int
    currency:                str = "INR"


@api.get("/", tags=["Health"])
def root():
    return {"status": "ok"}


@api.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "r2_score": round(_meta["metrics"]["r2"], 4)}


@api.get("/meta", tags=["Metadata"])
def get_meta():
    return {
        "brands":        _meta["brands"],
        "cpu_tiers":     _meta["cpu_tiers"],
        "ram_types":     _meta["ram_types"],
        "res_classes":   _meta["res_classes"],
        "os_options":    _meta["os_options"],
        "model_metrics": _meta["metrics"],
    }


@api.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: LaptopFeatures):
    try:
        row = pd.DataFrame([{
            "ram_gb":           features.ram_gb,
            "rom_gb":           features.rom_gb,
            "gpu_vram_gb":      features.gpu_vram_gb,
            "dedicated_gpu":    features.dedicated_gpu,
            "cpu_gen":          features.cpu_gen,
            "display_size":     features.display_size,
            "spec_rating":      features.spec_rating,
            "is_ssd":           features.is_ssd,
            "warranty":         features.warranty,
            "brand":            features.brand,
            "cpu_tier":         features.cpu_tier,
            "Ram_type":         features.ram_type,
            "resolution_class": features.resolution_class,
            "os_clean":         features.os_clean,
        }])
        price = float(np.expm1(_model.predict(row)[0]))
        return PredictionResponse(
            predicted_price=round(price, 2),
            predicted_price_rounded=int(round(price, -2)),
            price_range_low=int(price * 0.90),
            price_range_high=int(price * 1.10),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _start_api():
    """Load artefacts then serve FastAPI in this thread."""
    _load_artefacts()
    uvicorn.run(api, host="127.0.0.1", port=8000, log_level="error")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – Streamlit Frontend
# ─────────────────────────────────────────────────────────────────────────────
def format_inr(amount: float) -> str:
    amount = int(amount)
    s      = str(amount)
    if len(s) <= 3:
        return f"Rs.{s}"
    last3 = s[-3:]
    rest  = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.append(rest)
    return "Rs." + ",".join(reversed(parts)) + "," + last3


def gauge_chart(price: float):
    max_val = 300000
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=price,
        number={"prefix": "Rs.", "valueformat": ",.0f",
                "font": {"size": 22, "color": "#1e3a5f"}},
        gauge={
            "axis": {"range": [0, max_val], "tickformat": ",.0f",
                     "tickfont": {"size": 10}},
            "bar": {"color": "#2563eb"},
            "steps": [
                {"range": [0,       50000], "color": "#dbeafe"},
                {"range": [50000,  100000], "color": "#bfdbfe"},
                {"range": [100000, 200000], "color": "#93c5fd"},
                {"range": [200000, max_val],"color": "#60a5fa"},
            ],
            "threshold": {"line": {"color": "#ef4444", "width": 3},
                          "thickness": 0.75, "value": price},
        },
        title={"text": "Predicted Price Position", "font": {"size": 14}},
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
    return fig


def brand_bar_chart():
    data = {
        "Apple":   135000, "MSI":     89000, "Samsung": 85000,
        "Dell":    72000,  "Asus":    68000, "HP":      65000,
        "Lenovo":  58000,  "Acer":    54000, "Infinix": 45000,
    }
    df = (pd.DataFrame(data.items(), columns=["Brand", "Avg Price"])
            .sort_values("Avg Price", ascending=True))
    fig = px.bar(
        df, x="Avg Price", y="Brand", orientation="h",
        color="Avg Price", color_continuous_scale=["#bfdbfe", "#2563eb"],
        title="Average Price by Brand",
    )
    fig.update_layout(
        height=360, showlegend=False, coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=20),
        plot_bgcolor="#f9fafb", paper_bgcolor="#f9fafb",
    )
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Avg: Rs.%{x:,.0f}<extra></extra>")
    return fig


@st.cache_data(ttl=60)
def fetch_meta():
    for _ in range(10):          # wait up to 10 s for the API thread to be ready
        try:
            r = requests.get(f"{API_URL}/meta", timeout=2)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(1)
    return None


def run_frontend():
    st.set_page_config(
        page_title="Laptop Price Predictor",
        page_icon="💻",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
        .main-header{font-size:2.4rem;font-weight:700;color:#1e3a5f;text-align:center;margin-bottom:.2rem}
        .sub-header{font-size:1rem;color:#5a6a7e;text-align:center;margin-bottom:2rem}
        .price-card{background:linear-gradient(135deg,#1e3a5f 0%,#2563eb 100%);border-radius:16px;
                    padding:2rem;text-align:center;color:white}
        .price-label{font-size:.9rem;opacity:.85;letter-spacing:.05em}
        .price-value{font-size:2.8rem;font-weight:800}
        .price-range{font-size:.95rem;opacity:.8;margin-top:.4rem}
        .metric-card{background:#f0f4ff;border-radius:10px;padding:1rem 1.5rem;text-align:center}
        .metric-title{font-size:.75rem;color:#6b7280;text-transform:uppercase}
        .metric-value{font-size:1.6rem;font-weight:700;color:#1e3a5f}
        .stButton>button{background-color:#2563eb;color:white;border:none;border-radius:8px;
                         padding:.6rem 2.5rem;font-size:1rem;font-weight:600;width:100%}
        .stButton>button:hover{background-color:#1d4ed8}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">💻 Laptop Price Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Powered by XGBoost · FastAPI · Streamlit</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading model metadata..."):
        meta = fetch_meta()

    if meta is None:
        st.error("Backend API did not respond. Please restart the app.")
        st.stop()

    brands      = meta["brands"]
    cpu_tiers   = meta["cpu_tiers"]
    ram_types   = meta["ram_types"]
    res_classes = meta["res_classes"]
    os_options  = meta["os_options"]
    metrics     = meta["model_metrics"]

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Laptop Specs")
        st.markdown("---")

        brand = st.selectbox("Brand", brands)

        st.markdown("#### Processor")
        cpu_tier = st.selectbox("CPU Tier", cpu_tiers)
        cpu_gen  = st.slider("CPU Generation", 1, 14, 12)

        st.markdown("#### Memory")
        ram_gb   = st.select_slider("RAM (GB)", [4, 8, 16, 32, 64], value=8)
        ram_type = st.selectbox("RAM Type", ram_types)
        rom_gb   = st.select_slider("Storage (GB)", [64, 128, 256, 512, 1024, 2048], value=512)
        is_ssd   = st.radio("Storage Type", ["SSD", "HDD"], horizontal=True)

        st.markdown("#### Display")
        display_size     = st.select_slider("Display Size (inches)",
                                            [11.6, 13.3, 14.0, 15.6, 16.0, 17.3], value=15.6)
        resolution_class = st.selectbox("Resolution", res_classes)

        st.markdown("#### GPU")
        dedicated_gpu = st.toggle("Dedicated GPU", value=False)
        gpu_vram_gb   = 0
        if dedicated_gpu:
            gpu_vram_gb = st.select_slider("GPU VRAM (GB)", [2, 4, 6, 8, 12, 16], value=4)

        st.markdown("#### OS & Extras")
        os_clean    = st.selectbox("Operating System", os_options)
        spec_rating = st.slider("Spec Rating", 0.0, 100.0, 69.0, step=0.5)
        warranty    = st.number_input("Warranty (years)", min_value=0, max_value=5, value=1)

        st.markdown("---")
        predict_btn = st.button("Predict Price", use_container_width=True)

    # ── Main Area ─────────────────────────────────────────────────────────────
    col_main, col_info = st.columns([3, 2], gap="large")

    with col_info:
        st.markdown("### Model Performance")
        m1, m2, m3 = st.columns(3)
        for col, title, val in [
            (m1, "R² Score", f"{metrics['r2']:.2f}"),
            (m2, "MAE",      f"Rs.{int(metrics['mae']):,}"),
            (m3, "RMSE",     f"Rs.{int(metrics['rmse']):,}"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-title">{title}</div>'
                    f'<div class="metric-value">{val}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(brand_bar_chart(), use_container_width=True)

        with st.expander("What affects laptop price?"):
            st.markdown("""
| Feature | Impact |
|---|---|
| Brand (Apple, MSI) | Very High |
| CPU Tier (i9, Ryzen 9) | Very High |
| Dedicated GPU | High |
| RAM Size | High |
| Storage Size | Medium |
| Display Size | Medium |
| CPU Generation | Moderate |
| Resolution | Moderate |
""")

    with col_main:
        if predict_btn:
            payload = {
                "brand": brand, "ram_gb": float(ram_gb), "rom_gb": float(rom_gb),
                "gpu_vram_gb": float(gpu_vram_gb), "dedicated_gpu": int(dedicated_gpu),
                "cpu_gen": float(cpu_gen), "cpu_tier": cpu_tier,
                "display_size": float(display_size), "spec_rating": float(spec_rating),
                "is_ssd": 1 if is_ssd == "SSD" else 0, "warranty": int(warranty),
                "ram_type": ram_type, "resolution_class": resolution_class,
                "os_clean": os_clean,
            }
            with st.spinner("Predicting..."):
                try:
                    r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                    r.raise_for_status()
                    result = r.json()
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

            price   = result["predicted_price"]
            price_r = result["predicted_price_rounded"]
            low     = result["price_range_low"]
            high    = result["price_range_high"]

            st.markdown(
                f'<div class="price-card">'
                f'<div class="price-label">ESTIMATED PRICE</div>'
                f'<div class="price-value">{format_inr(price_r)}</div>'
                f'<div class="price-range">Likely range: {format_inr(low)} – {format_inr(high)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(price), use_container_width=True)

            st.markdown("### Configured Specs")
            spec_df = pd.DataFrame([
                ("Brand",        brand),
                ("CPU",          f"{cpu_tier} (Gen {cpu_gen})"),
                ("RAM",          f"{ram_gb} GB {ram_type}"),
                ("Storage",      f"{rom_gb} GB {'SSD' if is_ssd == 'SSD' else 'HDD'}"),
                ("GPU",          f"{gpu_vram_gb} GB VRAM (Dedicated)" if dedicated_gpu else "Integrated"),
                ("Display",      f"{display_size}\" {resolution_class}"),
                ("OS",           os_clean),
                ("Spec Rating",  f"{spec_rating}/100"),
                ("Warranty",     f"{warranty} year(s)"),
            ], columns=["Feature", "Value"])
            st.dataframe(spec_df, use_container_width=True, hide_index=True)

        else:
            st.info("Configure specs in the sidebar and click **Predict Price**.", icon="💡")
            st.markdown("### Brand Price Overview")
            st.plotly_chart(brand_bar_chart(), use_container_width=True)
            st.markdown("""
#### How it works
1. Select specs in the sidebar
2. Click **Predict Price**
3. View the price card, gauge, and ±10% confidence range

The model is trained on **893 real Indian e-commerce laptop listings**
using **XGBoost regression** with log-transformed prices.
""")

    st.markdown("---")
    st.markdown(
        "<center style='color:#9ca3af;font-size:.8rem;'>"
        "Laptop Price Predictor · XGBoost + FastAPI + Streamlit</center>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_model():
    """Train model if not already saved."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        with st.spinner("Training model for the first time... (~30 seconds)"):
            train_model()
        st.success("Model trained and saved!")
        st.cache_data.clear()


def _ensure_api_thread():
    """Start FastAPI in a daemon thread (once per Streamlit process)."""
    if "api_started" not in st.session_state:
        t = threading.Thread(target=_start_api, daemon=True, name="fastapi")
        t.start()
        st.session_state["api_started"] = True


# Streamlit calls the whole script on every interaction.
# We guard side-effects with session_state so they only run once.
_ensure_model()
_ensure_api_thread()
run_frontend()
