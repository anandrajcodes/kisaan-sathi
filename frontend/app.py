import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# Add backend ml path for explainability import
BACKEND_ML_PATH = Path(__file__).resolve().parents[1] / "backend" / "ml"
if str(BACKEND_ML_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_ML_PATH))

try:
    from explain_prediction import explain_prediction
except Exception:
    explain_prediction = None

# ---------------- CONFIGURATION ----------------
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Kissaan AI | Soil Intelligence Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Crop emoji & styling dictionary
CROP_METADATA = {
    "rice": {"emoji": "🌾", "category": "Cereal", "color": "#10b981"},
    "maize": {"emoji": "🌽", "category": "Cereal", "color": "#f59e0b"},
    "chickpea": {"emoji": "🌰", "category": "Pulse", "color": "#8b5cf6"},
    "kidneybeans": {"emoji": "🫘", "category": "Pulse", "color": "#ef4444"},
    "pigeonpeas": {"emoji": "🫛", "category": "Pulse", "color": "#14b8a6"},
    "mothbeans": {"emoji": "🫘", "category": "Pulse", "color": "#d97706"},
    "mungbean": {"emoji": "🌱", "category": "Pulse", "color": "#84cc16"},
    "blackgram": {"emoji": "🫘", "category": "Pulse", "color": "#475569"},
    "lentil": {"emoji": "🍲", "category": "Pulse", "color": "#f97316"},
    "pomegranate": {"emoji": "🍎", "category": "Fruit", "color": "#e11d48"},
    "banana": {"emoji": "🍌", "category": "Fruit", "color": "#eab308"},
    "mango": {"emoji": "🥭", "category": "Fruit", "color": "#f59e0b"},
    "grapes": {"emoji": "🍇", "category": "Fruit", "color": "#7c3aed"},
    "watermelon": {"emoji": "🍉", "category": "Fruit", "color": "#ec4899"},
    "muskmelon": {"emoji": "🍈", "category": "Fruit", "color": "#10b981"},
    "apple": {"emoji": "🍎", "category": "Fruit", "color": "#dc2626"},
    "orange": {"emoji": "🍊", "category": "Fruit", "color": "#ea580c"},
    "papaya": {"emoji": "🍈", "category": "Fruit", "color": "#f59e0b"},
    "coconut": {"emoji": "🥥", "category": "Plantation", "color": "#059669"},
    "cotton": {"emoji": "☁️", "category": "Cash Crop", "color": "#0ea5e9"},
    "jute": {"emoji": "🌿", "category": "Fiber", "color": "#15803d"},
    "coffee": {"emoji": "☕", "category": "Plantation", "color": "#78350f"},
}

PRESETS = {
    "Paddy / Wetland Rice": {
        "N": 80, "P": 48, "K": 40, "ph": 6.4,
        "temperature": 24.0, "humidity": 82.0, "rainfall": 236.0,
        "desc": "Heavy rainfall, high humidity, warm tropical climate"
    },
    "Hybrid Maize": {
        "N": 78, "P": 48, "K": 20, "ph": 6.2,
        "temperature": 22.5, "humidity": 65.0, "rainfall": 85.0,
        "desc": "Moderate rainfall, sunny warm conditions, well-drained loam"
    },
    "Plantation Coffee": {
        "N": 101, "P": 29, "K": 30, "ph": 6.8,
        "temperature": 25.5, "humidity": 59.0, "rainfall": 158.0,
        "desc": "Elevated hill tracts, moderate rain, rich organic soil"
    },
    "Bt Cotton (Black Soil)": {
        "N": 118, "P": 46, "K": 20, "ph": 6.9,
        "temperature": 24.0, "humidity": 80.0, "rainfall": 80.0,
        "desc": "Deep fertile black soil, warm temperatures, sunny dry harvest"
    },
    "Apple Orchard": {
        "N": 20, "P": 134, "K": 199, "ph": 5.9,
        "temperature": 22.0, "humidity": 92.0, "rainfall": 112.0,
        "desc": "Temperate climate, high potassium and phosphorus reserve"
    }
}

# ---------------- PREMIUM HIGH-CONTRAST & ACCESSIBLE CSS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

    :root {
        --primary-forest: #0d351b;
        --deep-emerald: #134629;
        --vibrant-green: #10b981;
        --accent-lime: #34d399;
        --gold-amber: #d97706;
        --text-slate: #0f172a;
    }

    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Enforce solid contrast for base text across all pages */
    .stApp {
        background-color: #f7faf8 !important;
        background-image: radial-gradient(circle at 12% 16%, #eef6f0 0%, #f7faf8 55%, #edf4ef 100%) !important;
        color: #0f172a !important;
    }

    /* All markdown and body paragraphs */
    .stApp p, .stApp label, .stApp li, .stMarkdown p {
        color: #1e293b !important;
    }

    /* Typography Headings */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .outfit-font {
        font-family: 'Outfit', sans-serif !important;
        color: #0d351b !important;
        letter-spacing: -0.02em;
        font-weight: 700 !important;
    }

    /* Captions & small diagnostic notes */
    [data-testid="stCaptionContainer"] p, .stCaption, small {
        color: #475569 !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
    }

    /* Streamlit native container with border (Cards) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border: 1.5px solid #dbe5df !important;
        border-radius: 18px !important;
        box-shadow: 0 4px 18px rgba(13, 46, 27, 0.05) !important;
        padding: 20px 24px !important;
        margin-bottom: 20px !important;
        transition: all 0.25s ease !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(16, 185, 129, 0.5) !important;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.12) !important;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1280px;
    }

    /* Glassmorphic Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #091e12 0%, #0e2a1b 45%, #08170e 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.25);
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] * {
        color: #ecfdf5 !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #d1fae5 !important;
    }

    /* Sidebar Navigation Pills */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.06) !important;
        padding: 10px 16px !important;
        border-radius: 12px !important;
        margin-bottom: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(16, 185, 129, 0.2) !important;
        border-color: rgba(52, 211, 153, 0.5) !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.4), rgba(5, 150, 105, 0.55)) !important;
        border: 1.5px solid rgba(52, 211, 153, 0.9) !important;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.25) !important;
    }

    /* Status dot animation */
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #10b981;
        margin-right: 6px;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1.5px solid #dbe5df !important;
        padding: 18px 22px !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 14px rgba(13, 46, 27, 0.04) !important;
        transition: all 0.25s ease !important;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.14) !important;
        border-color: rgba(52, 211, 153, 0.5) !important;
    }

    [data-testid="stMetricLabel"] p {
        color: #475569 !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricValue"] div {
        color: #065f46 !important;
        font-weight: 800 !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.65rem !important;
    }

    /* Regular Buttons (Presets, Hubs, Downloads, Tabs) */
    div.stButton > button {
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 12px !important;
        min-height: 42px !important;
        font-size: 0.92rem !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }

    div.stButton > button,
    div.stButton > button p,
    div.stButton > button span,
    div.stButton > button div {
        color: #0d351b !important;
        font-weight: 700 !important;
    }

    div.stButton > button:hover {
        background: #ecfdf5 !important;
        border-color: #10b981 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.18) !important;
    }

    div.stButton > button:hover p,
    div.stButton > button:hover span,
    div.stButton > button:hover div {
        color: #047857 !important;
    }

    /* Primary Submit Buttons (Predict / Confirm) */
    div.stFormSubmitButton > button {
        background: linear-gradient(135deg, #0d5c2f 0%, #16934c 100%) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        min-height: 50px !important;
        font-size: 1.05rem !important;
        box-shadow: 0 8px 24px rgba(16, 114, 58, 0.32) !important;
        transition: all 0.25s ease !important;
        width: 100% !important;
    }

    div.stFormSubmitButton > button,
    div.stFormSubmitButton > button p,
    div.stFormSubmitButton > button span,
    div.stFormSubmitButton > button div {
        color: #ffffff !important;
        font-weight: 800 !important;
        letter-spacing: 0.02em !important;
    }

    div.stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, #094020 0%, #12793e 100%) !important;
        box-shadow: 0 12px 30px rgba(16, 114, 58, 0.45) !important;
        transform: translateY(-2px) !important;
    }

    /* Download Buttons */
    div[data-testid="stDownloadButton"] > button {
        background: #ffffff !important;
        border: 1.5px solid #10b981 !important;
        border-radius: 12px !important;
        min-height: 44px !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.12) !important;
        width: 100% !important;
    }

    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stDownloadButton"] > button p,
    div[data-testid="stDownloadButton"] > button span,
    div[data-testid="stDownloadButton"] > button div {
        color: #065f46 !important;
        font-weight: 800 !important;
    }

    div[data-testid="stDownloadButton"] > button:hover {
        background: #ecfdf5 !important;
        border-color: #059669 !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.2) !important;
    }

    /* Sliders */
    div[data-testid="stSlider"] {
        padding-top: 6px;
        padding-bottom: 6px;
    }

    div[data-testid="stSlider"] label p {
        font-weight: 700 !important;
        color: #0f291e !important;
        font-size: 0.94rem !important;
    }

    div[data-testid="stSlider"] [data-testid="stThumbValue"] {
        color: #065f46 !important;
        font-weight: 800 !important;
        font-family: 'Outfit', sans-serif !important;
    }

    /* Tabs */
    div[data-testid="stTabs"] button[role="tab"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        padding: 10px 22px !important;
        border-radius: 10px 10px 0 0 !important;
        color: #475569 !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stTabs"] button[role="tab"] p {
        color: #475569 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #0d351b !important;
        border-bottom: 3.5px solid #10b981 !important;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] p {
        color: #0d351b !important;
        font-weight: 800 !important;
    }

    /* Text inputs and number inputs */
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        color: #0f172a !important;
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #64748b !important;
    }

    /* Hero Banner (Always white text on dark gradient) */
    .hero-banner {
        background: linear-gradient(135deg, #092617 0%, #12472b 50%, #1e633d 100%) !important;
        padding: 38px 36px;
        border-radius: 24px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 16px 40px -10px rgba(9, 38, 23, 0.4);
        margin-bottom: 28px;
    }

    .hero-banner h1,
    .hero-banner h2,
    .hero-banner h3 {
        color: #ffffff !important;
    }

    .hero-banner p {
        color: #d1fae5 !important;
        font-size: 1.05rem !important;
        line-height: 1.6;
    }

    .hero-banner span {
        color: #6ee7b7 !important;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.14) !important;
        backdrop-filter: blur(8px);
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        color: #6ee7b7 !important;
        border: 1px solid rgba(110, 231, 183, 0.35);
        margin-bottom: 14px;
    }

    /* Prediction Result Hero Card */
    .prediction-hero-card {
        background: linear-gradient(135deg, #0a2919 0%, #155533 60%, #1d7345 100%) !important;
        border: 1px solid rgba(52, 211, 153, 0.4);
        padding: 32px;
        border-radius: 24px;
        text-align: center;
        box-shadow: 0 20px 48px -10px rgba(10, 41, 25, 0.35);
        margin-bottom: 24px;
    }

    .prediction-hero-card h1,
    .prediction-hero-card h2,
    .prediction-hero-card h3 {
        color: #ffffff !important;
    }

    .prediction-hero-card p {
        color: #d1fae5 !important;
    }

    .prediction-hero-card span {
        color: #6ee7b7 !important;
    }

    .crop-emoji-display {
        font-size: 4.8rem;
        line-height: 1;
        margin-bottom: 12px;
        display: inline-block;
        filter: drop-shadow(0 8px 16px rgba(0,0,0,0.25));
    }

    /* Candidate Cards */
    .candidate-card {
        background: #ffffff !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 14px rgba(13, 46, 27, 0.04);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .feature-tag {
        display: inline-block;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 4px 10px;
        border-radius: 8px;
        margin-right: 6px;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 14px !important;
        border-width: 1px !important;
    }

    /* Code blocks in Model Performance */
    [data-testid="stCodeBlock"] pre {
        background: #0f172a !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }

    [data-testid="stCodeBlock"] code {
        color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- API CLIENT & HEALTH CHECK ----------------
@st.cache_data(ttl=10)
def check_backend_health():
    try:
        res = requests.get(f"{API_URL}/health", timeout=3)
        return res.status_code == 200
    except Exception:
        return False


def call_api(endpoint, payload):
    try:
        response = requests.post(
            f"{API_URL}/{endpoint}",
            json=payload,
            timeout=15,
        )
        if response.ok:
            return response.json()

        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text

        st.error(f"API Error ({response.status_code}): {detail}")
    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ Unable to reach FastAPI backend. "
            "Please ensure the backend is running (`uvicorn main:app --port 8000`)."
        )
    except Exception as exc:
        st.error(f"API Request Failed: {exc}")
    return None


def call_api_get(endpoint, params=None):
    try:
        response = requests.get(
            f"{API_URL}/{endpoint}",
            params=params,
            timeout=15,
        )
        if response.ok:
            return response.json()

        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text

        st.error(f"API Error ({response.status_code}): {detail}")
    except requests.exceptions.ConnectionError:
        return None
    except Exception as exc:
        st.error(f"API GET Request Failed: {exc}")
    return None


def fetch_weather_auto(district=None, lat=None, lon=None):
    """
    Fetch live weather telemetry via FastAPI backend (or fallback to Open-Meteo direct API).
    """
    # 1. Try FastAPI backend first
    if check_backend_health():
        if district:
            res = call_api_get("weather/district", params={"query": district})
            if res:
                return res
        elif lat is not None and lon is not None:
            res = call_api_get("weather/coordinates", params={"lat": lat, "lon": lon})
            if res:
                return res

    # 2. Resilient fallback to Open-Meteo public endpoints
    try:
        if district:
            g_url = "https://geocoding-api.open-meteo.com/v1/search"
            g_res = requests.get(
                g_url,
                params={"name": district, "count": 1, "language": "en", "format": "json"},
                headers={"User-Agent": "KissaanAI-Frontend/0.4.0"},
                timeout=6,
            ).json()
            results = g_res.get("results", [])
            if not results:
                st.warning(f"No location found matching '{district}'.")
                return None
            top = results[0]
            lat = top["latitude"]
            lon = top["longitude"]
            parts = [top.get("name", ""), top.get("admin1", ""), top.get("country", "")]
            loc_name = ", ".join([p for p in parts if p])
        else:
            loc_name = f"Coordinates ({lat:.3f}, {lon:.3f})"

        w_url = "https://api.open-meteo.com/v1/forecast"
        w_params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
            "daily": "precipitation_sum,temperature_2m_mean",
            "past_days": 30,
            "forecast_days": 7,
            "timezone": "auto",
        }
        w_res = requests.get(
            w_url,
            params=w_params,
            headers={"User-Agent": "KissaanAI-Frontend/0.4.0"},
            timeout=7,
        ).json()
        curr = w_res.get("current", {})
        daily = w_res.get("daily", {})

        p_list = [p for p in daily.get("precipitation_sum", []) if p is not None]
        total_precip = round(sum(p_list), 1) if p_list else 0.0

        m_temps = [t for t in daily.get("temperature_2m_mean", []) if t is not None]
        m_temp = round(sum(m_temps) / len(m_temps), 1) if m_temps else curr.get("temperature_2m", 25.0)

        wmo_codes = {
            0: ("Clear Sky", "☀️"), 1: ("Mainly Clear", "🌤️"), 2: ("Partly Cloudy", "⛅"),
            3: ("Overcast", "☁️"), 45: ("Foggy", "🌫️"), 51: ("Light Drizzle", "🌦️"),
            61: ("Slight Rain", "🌧️"), 63: ("Moderate Rain", "🌧️"), 65: ("Heavy Rain", "🌧️"),
            80: ("Rain Showers", "🌦️"), 95: ("Thunderstorm", "⛈️"),
        }
        cond, icon = wmo_codes.get(curr.get("weather_code", 0), ("Fair Weather", "🌤️"))

        return {
            "location_name": loc_name,
            "latitude": round(float(lat), 4),
            "longitude": round(float(lon), 4),
            "temperature": round(float(curr.get("temperature_2m", 25.0)), 1),
            "mean_temperature_30d": m_temp,
            "humidity": round(float(curr.get("relative_humidity_2m", 65.0)), 1),
            "rainfall": total_precip,
            "condition": cond,
            "icon": icon,
            "source": "Open-Meteo Direct API",
        }
    except Exception as exc:
        st.error(f"Weather lookup error: {exc}")
        return None


# ---------------- PDF GENERATION ENGINE ----------------
def generate_soil_report(input_values, predicted_crop, top_candidates, location_info="Custom Farm Location"):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0d351b"),
        alignment=0,
    )
    h2_style = ParagraphStyle(
        "ReportH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1b5e20"),
        spaceBefore=14,
        spaceAfter=8,
    )
    normal_style = ParagraphStyle(
        "ReportNormal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2a3b30"),
    )

    story = []
    story.append(Paragraph("🌱 Kissaan AI — Soil Intelligence", title_style))
    story.append(Paragraph("Comprehensive Soil Suitability & Crop Diagnostic Report", h2_style))
    story.append(
        Paragraph(
            f"<b>Generated on:</b> {datetime.now().strftime('%d %B %Y at %I:%M %p')} &nbsp;|&nbsp; "
            f"<b>Field Location:</b> {location_info} &nbsp;|&nbsp; "
            f"<b>Model:</b> Random Forest Classifier (200 Estimators)",
            normal_style,
        )
    )
    story.append(Spacer(1, 14))

    # 1. Inputs Table
    story.append(Paragraph("1. Soil & Environmental Parameters Analyzed", h2_style))
    input_rows = [["Parameter", "Field Value", "Standard Diagnostic Range"]]
    ref_ranges = {
        "Nitrogen (N)": "0 – 140 kg/ha equiv.",
        "Phosphorus (P)": "5 – 145 kg/ha equiv.",
        "Potassium (K)": "5 – 205 kg/ha equiv.",
        "Soil pH": "0 – 14 (6.0 - 7.5 optimal)",
        "Temperature (°C)": "-10°C to 50°C",
        "Humidity (%)": "0% – 100%",
        "Rainfall (mm)": "0 – 5000 mm",
    }
    for key, value in input_values.items():
        ref = ref_ranges.get(key, "Custom input")
        input_rows.append([str(key), str(value), ref])

    input_table = Table(input_rows, colWidths=[200, 140, 180])
    input_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#124629")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d5e2d8")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8faf8"), colors.white]),
            ("PADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(input_table)
    story.append(Spacer(1, 14))

    # 2. Predicted Crop
    story.append(Paragraph("2. Primary Crop Recommendation", h2_style))
    emoji = CROP_METADATA.get(predicted_crop.lower(), {}).get("emoji", "🌱")
    story.append(
        Paragraph(
            f"<b>Recommended Crop:</b> {emoji} <font size='13' color='#1b5e20'><b>{predicted_crop.title()}</b></font><br/>"
            f"The ensemble machine learning model identified <b>{predicted_crop.title()}</b> as possessing the highest agro-climatic "
            f"suitability score for your field's soil nutrient concentrations and atmospheric profile.",
            normal_style,
        )
    )
    story.append(Spacer(1, 12))

    # 3. Top Candidates
    story.append(Paragraph("3. Top 3 Alternative Suitability Candidates", h2_style))
    candidate_rows = [["Rank", "Crop Name", "Suitability Probability"]]
    for idx, cand in enumerate(top_candidates):
        c_emoji = CROP_METADATA.get(cand["crop"].lower(), {}).get("emoji", "🌱")
        score = float(cand["model_score"]) * 100
        candidate_rows.append([f"#{idx+1}", f"{c_emoji} {cand['crop'].title()}", f"{score:.2f}%"])

    cand_table = Table(candidate_rows, colWidths=[60, 240, 220])
    cand_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b6e3a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d5e2d8")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8faf8"), colors.white]),
            ("PADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(cand_table)
    story.append(Spacer(1, 14))

    # 4. Agronomic Notices
    story.append(Paragraph("4. Agronomic Standards & Safety Advisory", h2_style))
    story.append(
        Paragraph(
            "<b>Important Limitations:</b><br/>"
            "• <b>Dataset Constraints:</b> Suitability scores reflect pattern matching against 2,200 agricultural samples.<br/>"
            "• <b>Fertilizer Protocol:</b> The Government of India's Soil Testing Methods Manual (2011) and ICAR guidelines "
            "require that fertilizer dosages are calibrated to target yields, soil-test extractants, and regional field trials.<br/>"
            "• <b>Field Validation:</b> Always consult with a local Krishi Vigyan Kendra (KVK) or agronomist before sowing.",
            normal_style,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ---------------- SIDEBAR NAVIGATION & PRESETS ----------------
is_backend_up = check_backend_health()

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 16px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 2.2rem;">🌱</span>
                <div>
                    <h2 style="margin: 0; font-size: 1.45rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">KISSAAN AI</h2>
                    <span style="font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #34d399; font-weight: 700;">Soil Intelligence</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Live backend status indicator
    if is_backend_up:
        st.markdown(
            """
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(52, 211, 153, 0.35); padding: 7px 12px; border-radius: 10px; margin-bottom: 20px; display: flex; align-items: center;">
                <span class="status-dot"></span>
                <span style="font-size: 0.82rem; font-weight: 600; color: #6ee7b7 !important;">FastAPI Online (Port 8000)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(248, 113, 113, 0.4); padding: 7px 12px; border-radius: 10px; margin-bottom: 20px; display: flex; align-items: center;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: #ef4444; margin-right: 8px;"></span>
                <span style="font-size: 0.82rem; font-weight: 600; color: #fca5a5 !important;">Backend Offline (Run uvicorn)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<p style='font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: #6ee7b7 !important; font-weight: 700; margin-bottom: 6px;'>Navigation</p>", unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🌱 Crop Recommendation",
            "🧪 Soil Health",
            "🌿 Fertilizer Guidance",
            "📊 Model Performance",
            "ℹ️ About Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.divider()

    # Quick Agro-Insight Card
    st.markdown(
        """
        <div style="background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); padding: 14px; border-radius: 14px; margin-top: 10px;">
            <p style="font-size: 0.8rem; font-weight: 700; color: #6ee7b7 !important; margin: 0 0 6px 0;">💡 Agronomy Tip</p>
            <p style="font-size: 0.78rem; color: #cbd5e1 !important; margin: 0; line-height: 1.45;">
                Soil pH directly dictates nutrient bioavailability. Below 5.5, phosphorus locks into insoluble iron/aluminum compounds.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="padding-top: 16px; font-size: 0.74rem; color: #64748b !important; text-align: center;">
            Kissaan AI Platform · v0.4.0<br/>
            Engineered with Random Forest & SHAP
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------- DASHBOARD PAGE ----------------
if page == "🏠 Dashboard":
    st.markdown(
        """
        <div class="hero-banner">
            <span class="hero-badge">✨ Next-Generation Precision Agriculture</span>
            <h1>Smarter Soil Health & AI Crop Intelligence</h1>
            <p>
                Analyze soil chemistry, uncover optimal crop varieties, evaluate nutrient balances, 
                and receive transparent machine learning explanations for data-driven farming decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Live Metric Cards Grid
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="🌲 ML Algorithm", value="Random Forest", help="200 ensemble decision trees trained on balanced multi-class records")
    with m2:
        st.metric(label="🎯 Model Accuracy", value="99.32%", delta="+94.8% vs Baseline", help="Stratified 5-fold cross-validated evaluation on test records")
    with m3:
        st.metric(label="🌾 Supported Crops", value="22 Varieties", help="Cereals, pulses, fruits, plantation, and cash crops")
    with m4:
        st.metric(label="🧪 Predictor Features", value="7 Soil & Weather", help="N, P, K, pH, Temperature, Humidity, and Rainfall")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Interactive Soil pH Spectrum Chart
    with st.container(border=True):
        st.markdown("### 🌈 Soil pH & Crop Suitability Landscape")
        st.caption("Explore how soil pH influences nutrient availability and which crops thrive in specific pH zones.")

        ph_data = pd.DataFrame([
            {"Crop": "Tea / Coffee", "Min_pH": 4.5, "Max_pH": 6.0, "Category": "Acid Loving"},
            {"Crop": "Rice / Paddy", "Min_pH": 5.5, "Max_pH": 7.2, "Category": "Slightly Acidic to Neutral"},
            {"Crop": "Maize / Corn", "Min_pH": 6.0, "Max_pH": 7.5, "Category": "Optimal Neutral"},
            {"Crop": "Chickpea / Pulses", "Min_pH": 6.0, "Max_pH": 7.8, "Category": "Optimal Neutral"},
            {"Crop": "Cotton", "Min_pH": 6.5, "Max_pH": 8.0, "Category": "Neutral to Alkaline"},
            {"Crop": "Pomegranate", "Min_pH": 6.5, "Max_pH": 8.5, "Category": "Alkaline Tolerant"},
        ])

        fig_ph = go.Figure()

        # pH Zones colored background
        fig_ph.add_vrect(x0=0, x1=5.5, fillcolor="rgba(239, 68, 68, 0.12)", line_width=0, annotation_text="Strongly Acidic", annotation_position="top left")
        fig_ph.add_vrect(x0=5.5, x1=6.5, fillcolor="rgba(245, 158, 11, 0.12)", line_width=0, annotation_text="Moderately Acidic", annotation_position="top left")
        fig_ph.add_vrect(x0=6.5, x1=7.5, fillcolor="rgba(16, 185, 129, 0.15)", line_width=0, annotation_text="Optimal Neutral", annotation_position="top left")
        fig_ph.add_vrect(x0=7.5, x1=8.5, fillcolor="rgba(6, 182, 212, 0.12)", line_width=0, annotation_text="Moderately Alkaline", annotation_position="top left")
        fig_ph.add_vrect(x0=8.5, x1=14.0, fillcolor="rgba(139, 92, 246, 0.12)", line_width=0, annotation_text="Strongly Alkaline", annotation_position="top left")

        for _, row in ph_data.iterrows():
            fig_ph.add_trace(go.Bar(
                y=[row["Crop"]],
                x=[row["Max_pH"] - row["Min_pH"]],
                base=[row["Min_pH"]],
                orientation="h",
                name=row["Crop"],
                marker=dict(
                    color="rgba(16, 185, 129, 0.8)",
                    line=dict(color="#065f46", width=1.5),
                ),
                hovertemplate="<b>%{y}</b><br>Suitable pH: " + str(row["Min_pH"]) + " – " + str(row["Max_pH"]) + "<extra></extra>",
            ))

        fig_ph.update_layout(
            xaxis=dict(title="Soil pH Scale (0 = Acidic, 7 = Neutral, 14 = Alkaline)", range=[3.5, 10.0], tickmode="linear", tick0=4.0, dtick=1.0),
            yaxis=dict(title=""),
            height=280,
            margin=dict(l=20, r=20, t=30, b=30),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ph, use_container_width=True)

    # Feature Cards Row
    st.markdown("### 🚀 Platform Modules")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div style="background: #ffffff; border: 1.5px solid #dbe5df; border-top: 4px solid #10b981; border-radius: 18px; padding: 22px; min-height: 220px; box-shadow: 0 4px 14px rgba(13, 46, 27, 0.04);">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">🌱</div>
                <h3 style="color: #065f46; margin: 0 0 8px 0; font-size: 1.25rem;">AI Crop Predictor</h3>
                <p style="color: #334155; font-size: 0.92rem; line-height: 1.55; margin-bottom: 14px;">
                    Enter your soil NPK levels and regional weather to identify the top 3 high-yield crop matches with SHAP interpretability.
                </p>
                <span class="feature-tag" style="background: #dcfce7; color: #166534;">22 Crops</span>
                <span class="feature-tag" style="background: #dcfce7; color: #166534;">Top-3 Scoring</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="background: #ffffff; border: 1.5px solid #dbe5df; border-top: 4px solid #0284c7; border-radius: 18px; padding: 22px; min-height: 220px; box-shadow: 0 4px 14px rgba(13, 46, 27, 0.04);">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">🧪</div>
                <h3 style="color: #0369a1; margin: 0 0 8px 0; font-size: 1.25rem;">Soil Diagnostics</h3>
                <p style="color: #334155; font-size: 0.92rem; line-height: 1.55; margin-bottom: 14px;">
                    Interactive pH gauge meter and nutrient balance charts to diagnose acidity, alkalinity, and nutrient imbalances.
                </p>
                <span class="feature-tag" style="background: #e0f2fe; color: #075985;">Visual Gauge</span>
                <span class="feature-tag" style="background: #e0f2fe; color: #075985;">NPK Ratios</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div style="background: #ffffff; border: 1.5px solid #dbe5df; border-top: 4px solid #d97706; border-radius: 18px; padding: 22px; min-height: 220px; box-shadow: 0 4px 14px rgba(13, 46, 27, 0.04);">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">🌿</div>
                <h3 style="color: #92400e; margin: 0 0 8px 0; font-size: 1.25rem;">Fertilizer Guidance</h3>
                <p style="color: #334155; font-size: 0.92rem; line-height: 1.55; margin-bottom: 14px;">
                    Adheres to Indian Council of Agricultural Research (ICAR) guidelines. Avoids arbitrary dosage prescriptions without soil test calibration.
                </p>
                <span class="feature-tag" style="background: #fef3c7; color: #92400e;">ICAR Standard</span>
                <span class="feature-tag" style="background: #fef3c7; color: #92400e;">STCR Method</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------- CROP RECOMMENDATION PAGE ----------------
elif page == "🌱 Crop Recommendation":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="color: #0b3d20; margin: 0 0 6px 0; font-weight: 800;">🌱 AI Crop Suitability Studio</h2>
            <p style="color: #476251; margin: 0; font-size: 0.98rem;">
                Adjust your soil chemistry and local climatic conditions or select a quick field preset to predict optimal crops.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Session state initialization
    defaults = {
        "n_val": 90, "p_val": 42, "k_val": 43, "ph_val": 6.5,
        "temp_val": 25.0, "hum_val": 80.0, "rain_val": 100.0,
        "weather_location": "Custom Farm Location",
        "weather_synced": False,
        "weather_data": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ---------------- SATELLITE & STATION WEATHER AUTO-FILL ----------------
    with st.container(border=True):
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.85rem;">🛰️</span>
                    <div>
                        <h3 style="margin: 0; color: #0c4a6e !important; font-size: 1.25rem;">Live Agro-Weather & Climate Auto-Fill</h3>
                        <span style="font-size: 0.76rem; color: #0284c7 !important; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;">Powered by Open-Meteo Meteorological Satellite & Station Grid</span>
                    </div>
                </div>
                <span style="background: rgba(14, 165, 233, 0.12); color: #0369a1 !important; border: 1.5px solid rgba(56, 189, 248, 0.4); padding: 5px 14px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700;">
                    ⚡ Zero Guesswork Weather
                </span>
            </div>
            <p style="margin: 0 0 14px 0; color: #334155 !important; font-size: 0.9rem; line-height: 1.5;">
                Avoid manual guessing. Search any agricultural district or specify GPS coordinates to automatically sync real-time ambient temperature, humidity, and 30-day cumulative precipitation.
            </p>
            """,
            unsafe_allow_html=True,
        )

        tab_w_district, tab_w_gps = st.tabs(["🔍 Search District or City", "📍 GPS Coordinates (Latitude / Longitude)"])

        with tab_w_district:
            w_col1, w_col2 = st.columns([3, 1])
            with w_col1:
                search_query = st.text_input(
                    "Enter District or City Name",
                    placeholder="e.g., Ludhiana, Nashik, Guntur, Varanasi, Patna, Burdwan, Nagpur...",
                    label_visibility="collapsed",
                    key="district_text_input",
                )
            with w_col2:
                search_btn = st.button("🌤️ Sync Live Weather", key="btn_sync_district", use_container_width=True)

            if search_btn and search_query.strip():
                with st.spinner(f"🛰️ Querying Open-Meteo satellite & station data for '{search_query}'..."):
                    w_result = fetch_weather_auto(district=search_query.strip())
                    if w_result:
                        st.session_state.temp_val = w_result["temperature"]
                        st.session_state.hum_val = w_result["humidity"]
                        st.session_state.rain_val = min(3000.0, max(0.0, w_result["rainfall"]))
                        st.session_state.weather_location = w_result["location_name"]
                        st.session_state.weather_synced = True
                        st.session_state.weather_data = w_result
                        st.rerun()

            # Quick Agricultural District Chips (2 Rows of 4 columns for full legible names)
            st.markdown("<p style='font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 800; color: #0284c7 !important; margin: 12px 0 8px 0;'>⚡ Quick District Hubs:</p>", unsafe_allow_html=True)
            hubs = [
                ("🌾 Ludhiana (PB)", "Ludhiana"),
                ("🍇 Nashik (MH)", "Nashik"),
                ("🌶️ Guntur (AP)", "Guntur"),
                ("🍚 Burdwan (WB)", "Burdwan"),
                ("🌽 Patna (BR)", "Patna"),
                ("🥭 Varanasi (UP)", "Varanasi"),
                ("🍎 Srinagar (JK)", "Srinagar"),
                ("🌴 Coimbatore (TN)", "Coimbatore"),
            ]
            hub_row1 = st.columns(4)
            for i, (label, query) in enumerate(hubs[:4]):
                with hub_row1[i]:
                    if st.button(label, key=f"btn_hub_{i}", use_container_width=True):
                        with st.spinner(f"🛰️ Fetching weather for {query}..."):
                            w_res = fetch_weather_auto(district=query)
                            if w_res:
                                st.session_state.temp_val = w_res["temperature"]
                                st.session_state.hum_val = w_res["humidity"]
                                st.session_state.rain_val = min(3000.0, max(0.0, w_res["rainfall"]))
                                st.session_state.weather_location = w_res["location_name"]
                                st.session_state.weather_synced = True
                                st.session_state.weather_data = w_res
                                st.rerun()

            hub_row2 = st.columns(4)
            for j, (label, query) in enumerate(hubs[4:], start=4):
                with hub_row2[j - 4]:
                    if st.button(label, key=f"btn_hub_{j}", use_container_width=True):
                        with st.spinner(f"🛰️ Fetching weather for {query}..."):
                            w_res = fetch_weather_auto(district=query)
                            if w_res:
                                st.session_state.temp_val = w_res["temperature"]
                                st.session_state.hum_val = w_res["humidity"]
                                st.session_state.rain_val = min(3000.0, max(0.0, w_res["rainfall"]))
                                st.session_state.weather_location = w_res["location_name"]
                                st.session_state.weather_synced = True
                                st.session_state.weather_data = w_res
                                st.rerun()

        with tab_w_gps:
            gps_c1, gps_c2, gps_c3 = st.columns([2, 2, 1.5])
            with gps_c1:
                in_lat = st.number_input("Latitude (°N/°S)", min_value=-90.0, max_value=90.0, value=25.317, step=0.001, format="%.4f", key="gps_lat")
            with gps_c2:
                in_lon = st.number_input("Longitude (°E/°W)", min_value=-180.0, max_value=180.0, value=83.010, step=0.001, format="%.4f", key="gps_lon")
            with gps_c3:
                st.markdown("<div style='height: 27px;'></div>", unsafe_allow_html=True)
                gps_btn = st.button("📡 Sync GPS Weather", key="btn_sync_gps", use_container_width=True)

            if gps_btn:
                with st.spinner(f"🛰️ Querying Open-Meteo for coordinates ({in_lat:.3f}, {in_lon:.3f})..."):
                    w_res = fetch_weather_auto(lat=in_lat, lon=in_lon)
                    if w_res:
                        st.session_state.temp_val = w_res["temperature"]
                        st.session_state.hum_val = w_res["humidity"]
                        st.session_state.rain_val = min(3000.0, max(0.0, w_res["rainfall"]))
                        st.session_state.weather_location = w_res["location_name"]
                        st.session_state.weather_synced = True
                        st.session_state.weather_data = w_res
                        st.rerun()

        # Live Synced Weather Banner (if active)
        if st.session_state.get("weather_synced") and st.session_state.get("weather_data"):
            w = st.session_state.weather_data
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, rgba(14, 165, 233, 0.12) 0%, rgba(16, 185, 129, 0.12) 100%); border: 1.5px solid rgba(56, 189, 248, 0.5); border-radius: 14px; padding: 16px 20px; margin-top: 14px;">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                        <div>
                            <span style="font-size: 0.74rem; text-transform: uppercase; font-weight: 800; color: #0284c7 !important; letter-spacing: 0.06em;">✅ LIVE TELEMETRY ACTIVE</span>
                            <h4 style="margin: 2px 0 0 0; color: #0c4a6e !important; font-size: 1.2rem;">{w.get('icon', '🌤️')} {st.session_state.weather_location}</h4>
                            <span style="font-size: 0.82rem; color: #334155 !important;">Conditions: <b style="color: #0f172a !important;">{w.get('condition', 'Clear')}</b> &nbsp;|&nbsp; 30-Day Mean Temp: <b style="color: #0f172a !important;">{w.get('mean_temperature_30d', 25.0)}°C</b></span>
                        </div>
                        <div style="display: flex; gap: 20px; align-items: center;">
                            <div style="text-align: center; border-right: 1.5px solid rgba(56, 189, 248, 0.35); padding-right: 18px;">
                                <span style="font-size: 0.74rem; color: #475569 !important; font-weight: 700;">TEMPERATURE</span>
                                <div style="font-size: 1.45rem; font-weight: 800; color: #0284c7 !important; font-family: 'Outfit';">{st.session_state.temp_val}°C</div>
                            </div>
                            <div style="text-align: center; border-right: 1.5px solid rgba(56, 189, 248, 0.35); padding-right: 18px;">
                                <span style="font-size: 0.74rem; color: #475569 !important; font-weight: 700;">HUMIDITY</span>
                                <div style="font-size: 1.45rem; font-weight: 800; color: #059669 !important; font-family: 'Outfit';">{st.session_state.hum_val}%</div>
                            </div>
                            <div style="text-align: center; padding-right: 8px;">
                                <span style="font-size: 0.74rem; color: #475569 !important; font-weight: 700;">PRECIPITATION (30d)</span>
                                <div style="font-size: 1.45rem; font-weight: 800; color: #0284c7 !important; font-family: 'Outfit';">{st.session_state.rain_val} mm</div>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_clear_l, col_clear_r = st.columns([4, 1])
            with col_clear_r:
                if st.button("✕ Reset Weather Sync", key="btn_clear_weather_sync", use_container_width=True):
                    st.session_state.weather_synced = False
                    st.session_state.weather_data = None
                    st.session_state.weather_location = "Custom Farm Location"
                    st.session_state.temp_val = 25.0
                    st.session_state.hum_val = 80.0
                    st.session_state.rain_val = 100.0
                    st.rerun()

    # Preset selector chips
    st.markdown("<p style='font-weight: 800; color: #0d351b !important; font-size: 0.95rem; margin-top: 14px; margin-bottom: 8px;'>⚡ Quick Field Presets (Archetypes):</p>", unsafe_allow_html=True)
    p_cols = st.columns(len(PRESETS) + 1)

    preset_display = {
        "Paddy / Wetland Rice": "🌾 Paddy Rice",
        "Hybrid Maize": "🌽 Hybrid Maize",
        "Plantation Coffee": "☕ Plantation Coffee",
        "Bt Cotton (Black Soil)": "☁️ Bt Cotton",
        "Apple Orchard": "🍎 Apple Orchard",
    }

    for idx, (p_name, p_data) in enumerate(PRESETS.items()):
        with p_cols[idx]:
            btn_lbl = preset_display.get(p_name, p_name)
            if st.button(btn_lbl, key=f"btn_p_{idx}", help=p_data["desc"], use_container_width=True):
                st.session_state.n_val = p_data["N"]
                st.session_state.p_val = p_data["P"]
                st.session_state.k_val = p_data["K"]
                st.session_state.ph_val = p_data["ph"]
                st.session_state.temp_val = p_data["temperature"]
                st.session_state.hum_val = p_data["humidity"]
                st.session_state.rain_val = p_data["rainfall"]
                st.session_state.weather_synced = False
                st.session_state.weather_location = f"{p_name} Archetype"
                st.session_state.weather_data = None
                st.rerun()

    with p_cols[-1]:
        if st.button("🔄 Reset Inputs", key="btn_reset", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    with st.form("crop_prediction_form"):
        col_soil, col_climate = st.columns(2)

        with col_soil:
            st.markdown(
                """
                <div style="background: rgba(16, 185, 129, 0.1); border: 1.5px solid rgba(16, 185, 129, 0.4); padding: 16px 20px; border-radius: 16px; margin-bottom: 16px;">
                    <h4 style="margin: 0; color: #065f46 !important; font-weight: 800;">🧪 Soil Chemistry Studio</h4>
                    <p style="margin: 2px 0 0 0; font-size: 0.85rem; color: #334155 !important;">Adjust primary soil nutrients and acidity</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            nitrogen = st.slider(
                "Nitrogen (N) [Ratio / Content]",
                min_value=0, max_value=140,
                value=int(st.session_state.n_val),
                help="Promotes leaf development and chlorophyll synthesis.",
            )
            phosphorus = st.slider(
                "Phosphorus (P) [Ratio / Content]",
                min_value=5, max_value=145,
                value=int(st.session_state.p_val),
                help="Vital for root expansion, flowering, and seed set.",
            )
            potassium = st.slider(
                "Potassium (K) [Ratio / Content]",
                min_value=5, max_value=205,
                value=int(st.session_state.k_val),
                help="Improves plant immunity, water regulation, and stress resilience.",
            )
            ph = st.slider(
                "Soil Acidity / Alkalinity (pH)",
                min_value=0.0, max_value=14.0,
                value=float(st.session_state.ph_val),
                step=0.1,
                help="Optimal soil range for most arable crops is 6.0 to 7.5.",
            )

            # Live N:P:K Ratio indicator
            npk_sum = max(1, nitrogen + phosphorus + potassium)
            n_pct = (nitrogen / npk_sum) * 100
            p_pct = (phosphorus / npk_sum) * 100
            k_pct = (potassium / npk_sum) * 100
            st.caption(f"📊 Nutrient Proportion: **N:** {n_pct:.1f}% | **P:** {p_pct:.1f}% | **K:** {k_pct:.1f}%")

        with col_climate:
            st.markdown(
                """
                <div style="background: rgba(14, 165, 233, 0.1); border: 1.5px solid rgba(56, 189, 248, 0.4); padding: 16px 20px; border-radius: 16px; margin-bottom: 16px;">
                    <h4 style="margin: 0; color: #0369a1 !important; font-weight: 800;">⛅ Climate & Weather Studio</h4>
                    <p style="margin: 2px 0 0 0; font-size: 0.85rem; color: #334155 !important;">Simulate seasonal temperature, moisture, and rainfall</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.session_state.get("weather_synced"):
                w_info = st.session_state.get("weather_data") or {}
                w_icon = w_info.get("icon", "🛰️")
                w_cond = w_info.get("condition", "Live Telemetry")
                st.caption(f"🛰️ **Live Weather Active:** {w_icon} {st.session_state.weather_location} ({w_cond})")

            temperature = st.slider(
                "Average Temperature (°C)",
                min_value=-10.0, max_value=50.0,
                value=float(st.session_state.temp_val),
                step=0.5,
                help="Mean ambient growing temperature.",
            )
            humidity = st.slider(
                "Relative Humidity (%)",
                min_value=0.0, max_value=100.0,
                value=float(st.session_state.hum_val),
                step=1.0,
                help="Atmospheric moisture content.",
            )
            rainfall = st.slider(
                "Seasonal Rainfall (mm)",
                min_value=0.0, max_value=3000.0,
                value=float(st.session_state.rain_val),
                step=10.0,
                help="Total expected seasonal precipitation.",
            )

            # Weather interpretation
            if rainfall > 200 and humidity > 75:
                climate_desc = "🌊 Heavy Monsoon / Wetland Conditions"
            elif rainfall < 60 and temperature > 28:
                climate_desc = "☀️ Arid / Low Moisture Conditions"
            elif temperature < 18:
                climate_desc = "❄️ Temperate / Cool Conditions"
            else:
                climate_desc = "🌤️ Moderate Tropical / Sub-Tropical"
            st.caption(f"🌤️ Microclimate Profile: **{climate_desc}**")

        submitted = st.form_submit_button("⚡ Predict Optimal Crop Variety")

    if submitted:
        with st.spinner("🤖 Analyzing multi-dimensional agro-climatic feature vectors..."):
            result = call_api(
                "predict-crop",
                {
                    "N": nitrogen,
                    "P": phosphorus,
                    "K": potassium,
                    "temperature": temperature,
                    "humidity": humidity,
                    "ph": ph,
                    "rainfall": rainfall,
                },
            )

        if result:
            predicted_crop = result.get("predicted_crop", "Unknown")
            candidates = result.get("top_3_candidates", [])
            meta = CROP_METADATA.get(predicted_crop.lower(), {"emoji": "🌱", "category": "Crop", "color": "#10b981"})

            # Result Hero Display
            st.markdown(
                f"""
                <div class="prediction-hero-card">
                    <span class="hero-badge" style="background: rgba(255,255,255,0.18);">
                        🎯 HIGH CONFIDENCE ML RECOMMENDATION
                    </span>
                    <div class="crop-emoji-display">{meta['emoji']}</div>
                    <h1 style="font-size: 2.8rem; font-weight: 800; margin: 4px 0 8px 0; color: #ffffff;">
                        {predicted_crop.title()}
                    </h1>
                    <p style="font-size: 1.05rem; color: #d1fae5; margin: 0 auto; max-width: 600px;">
                        Category: <b>{meta['category']}</b> &nbsp;|&nbsp; 
                        Matches your field's soil nutrient matrix and prevailing weather factors.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Results Details: Radar Chart + Top 3 Candidates
            res_left, res_right = st.columns([1.1, 0.9])

            with res_left:
                with st.container(border=True):
                    st.markdown("#### 🕸️ Agro-Climatic Signature (Radar Chart)")
                    st.caption("Normalized field profile across the 7 input dimensions.")

                    # Normalize features for radar chart (0-100 scale)
                    r_values = [
                        min(100, (nitrogen / 140) * 100),
                        min(100, (phosphorus / 145) * 100),
                        min(100, (potassium / 205) * 100),
                        min(100, (ph / 14) * 100),
                        min(100, ((temperature + 10) / 60) * 100),
                        min(100, humidity),
                        min(100, (rainfall / 300) * 100),
                    ]
                    categories = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Temperature", "Humidity", "Rainfall"]

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=r_values + [r_values[0]],
                        theta=categories + [categories[0]],
                        fill='toself',
                        fillcolor='rgba(16, 185, 129, 0.35)',
                        line=dict(color='#10b981', width=2),
                        name='Field Signature'
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
                            bgcolor="rgba(0,0,0,0)"
                        ),
                        height=300,
                        margin=dict(l=30, r=30, t=20, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        showlegend=False
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

            with res_right:
                with st.container(border=True):
                    st.markdown("#### 🏆 Top 3 Candidate Crops")
                    st.caption("Alternative crop options ranked by model probability.")

                    medals = ["🥇 Primary Match", "🥈 Second Choice", "🥉 Third Alternative"]
                    for i, cand in enumerate(candidates):
                        c_name = cand["crop"].title()
                        c_score = float(cand["model_score"])
                        c_meta = CROP_METADATA.get(cand["crop"].lower(), {"emoji": "🌱"})

                        st.markdown(
                            f"""
                            <div class="candidate-card">
                                <div style="display: flex; align-items: center; gap: 12px;">
                                    <span style="font-size: 1.8rem;">{c_meta['emoji']}</span>
                                    <div>
                                        <div style="font-weight: 800; color: #0c331d !important; font-size: 1.05rem;">{c_name}</div>
                                        <span style="font-size: 0.78rem; color: #047857 !important; font-weight: 700;">{medals[i]}</span>
                                    </div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-weight: 800; font-size: 1.25rem; color: #059669 !important; font-family: 'Outfit';">
                                        {c_score * 100:.1f}%
                                    </div>
                                    <span style="font-size: 0.74rem; color: #475569 !important; font-weight: 600;">Confidence</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.progress(max(0.0, min(1.0, c_score)))

            # SHAP Model Explainability
            with st.container(border=True):
                st.markdown("### 🔍 Model Explainability & Feature Drivers")
                st.caption("How specific soil and weather inputs drove the AI model's recommendation.")

                if explain_prediction:
                    try:
                        exp_inputs = {
                            "N": nitrogen, "P": phosphorus, "K": potassium,
                            "temperature": temperature, "humidity": humidity,
                            "ph": ph, "rainfall": rainfall,
                        }
                        shap_results = explain_prediction(exp_inputs)

                        feat_labels = {
                            "N": "Nitrogen", "P": "Phosphorus", "K": "Potassium",
                            "temperature": "Temperature", "humidity": "Humidity",
                            "ph": "Soil pH", "rainfall": "Rainfall",
                        }

                        df_shap = pd.DataFrame(shap_results)
                        df_shap["Feature_Name"] = df_shap["feature"].map(feat_labels)
                        df_shap["Color"] = df_shap["shap_value"].apply(
                            lambda v: "#10b981" if v >= 0 else "#ef4444"
                        )

                        fig_shap = px.bar(
                            df_shap,
                            x="shap_value",
                            y="Feature_Name",
                            orientation="h",
                            color="Color",
                            color_discrete_map="identity",
                            labels={"shap_value": "SHAP Contribution Value", "Feature_Name": "Agro-Feature"},
                        )
                        fig_shap.update_layout(
                            height=240,
                            margin=dict(l=20, r=20, t=10, b=20),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
                            yaxis=dict(categoryorder="total ascending"),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)

                        pos_feats = [r["feature"] for r in shap_results if r["shap_value"] > 0]
                        if pos_feats:
                            pos_str = ", ".join([feat_labels.get(f, f) for f in pos_feats[:3]])
                            st.success(f"🌾 **Key Supporting Factors:** The measured values for **{pos_str}** were the strongest positive drivers favoring **{predicted_crop.title()}**.")
                    except Exception as e:
                        st.info(f"Model explanation notice: {e}")

            # Download PDF Report Card
            with st.container(border=True):
                col_dl_text, col_dl_btn = st.columns([2.5, 1])
                with col_dl_text:
                    st.markdown("#### 📄 Download Certified Soil & Crop Analysis Report")
                    st.write("Generate a formatted PDF summary including input parameters, crop predictions, and agronomic advisories for offline records.")
                with col_dl_btn:
                    report_inputs = {
                        "Nitrogen (N)": nitrogen,
                        "Phosphorus (P)": phosphorus,
                        "Potassium (K)": potassium,
                        "Soil pH": ph,
                        "Temperature (°C)": temperature,
                        "Humidity (%)": humidity,
                        "Rainfall (mm)": rainfall,
                    }
                    pdf_data = generate_soil_report(
                        report_inputs,
                        predicted_crop,
                        candidates,
                        location_info=st.session_state.get("weather_location", "Custom Farm Location"),
                    )
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_data,
                        file_name=f"Kissaan_AI_{predicted_crop.lower()}_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )


# ---------------- SOIL HEALTH PAGE ----------------
elif page == "🧪 Soil Health":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="color: #0b3d20; margin: 0 0 6px 0; font-weight: 800;">🧪 Soil Health & Chemistry Diagnostics</h2>
            <p style="color: #476251; margin: 0; font-size: 0.98rem;">
                Assess laboratory soil test readings, determine acidity/alkalinity classifications, and review nutrient balances.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_ph, tab_npk = st.tabs(["⚡ Soil pH Diagnostic Gauge", "📊 NPK Nutrient Analysis"])

    with tab_ph:
        with st.container(border=True):
            col_ph_in, col_ph_gauge = st.columns([1, 1.2])

            with col_ph_in:
                st.markdown("#### ⚡ Soil pH Diagnostic Input")
                st.write("Adjust to match the pH reading from your laboratory or field digital pH pen.")
                test_ph = st.slider("Soil pH Value", min_value=0.0, max_value=14.0, value=6.5, step=0.1, key="gauge_ph")

                if st.button("Diagnose Soil pH"):
                    res_ph = call_api("assess-soil-ph", {"ph": test_ph})
                    if res_ph:
                        status = res_ph.get("assessment", "Normal")
                        st.metric("pH Assessment", status)
                        st.info(res_ph.get("interpretation", ""))
                        if res_ph.get("notice"):
                            st.caption(f"ℹ️ {res_ph['notice']}")

            with col_ph_gauge:
                # Interactive Semi-Circular Gauge Meter using Plotly
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=test_ph,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Soil pH Gauge", 'font': {'size': 20, 'family': 'Outfit'}},
                    gauge={
                        'axis': {'range': [0, 14], 'tickwidth': 1, 'tickcolor': "#475569"},
                        'bar': {'color': "#0d351b", 'thickness': 0.25},
                        'bgcolor': "white",
                        'borderwidth': 1,
                        'bordercolor': "#cbd5e1",
                        'steps': [
                            {'range': [0, 5.5], 'color': '#fca5a5'},       # Strongly Acidic
                            {'range': [5.5, 6.5], 'color': '#fde68a'},     # Moderately Acidic
                            {'range': [6.5, 7.5], 'color': '#86efac'},     # Optimal Neutral
                            {'range': [7.5, 8.5], 'color': '#bae6fd'},     # Moderately Alkaline
                            {'range': [8.5, 14.0], 'color': '#ddd6fe'},    # Strongly Alkaline
                        ],
                        'threshold': {
                            'line': {'color': "#059669", 'width': 4},
                            'thickness': 0.8,
                            'value': 7.0
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=260,
                    margin=dict(l=20, r=20, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Corrective agronomic advice card with high contrast colors
            if test_ph < 5.5:
                box_style = "background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444;"
                p_color = "#991b1b"
                rec = "🌱 **Acidic Soil Remediation:** Consider applying Agricultural Limestone ($CaCO_3$) or Dolomite to raise soil pH and prevent Aluminum toxicity."
            elif test_ph > 8.0:
                box_style = "background: rgba(245, 158, 11, 0.12); border-left: 4px solid #f59e0b;"
                p_color = "#92400e"
                rec = "🌾 **Alkaline Soil Remediation:** Consider applying Agricultural Gypsum ($CaSO_4$) or Elemental Sulfur to lower pH and reclaim sodic/alkaline conditions."
            else:
                box_style = "background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981;"
                p_color = "#065f46"
                rec = "✅ **Optimal Soil pH:** Most arable crops achieve maximum nutrient bioavailability in this range."

            st.markdown(
                f"""
                <div style="{box_style} padding: 14px 18px; border-radius: 10px; margin-top: 14px;">
                    <p style="margin: 0; color: {p_color} !important; font-size: 0.94rem; font-weight: 600;">{rec}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_npk:
        with st.container(border=True):
            st.markdown("#### 🧪 NPK Macronutrient Proportion Studio")
            st.write("Record and evaluate your nitrogen, phosphorus, and potassium measurements.")

            npk_c1, npk_c2, npk_c3 = st.columns(3)
            with npk_c1:
                val_n = st.slider("Nitrogen (N)", 0.0, 140.0, 90.0, 1.0, key="npk_n")
            with npk_c2:
                val_p = st.slider("Phosphorus (P)", 5.0, 145.0, 42.0, 1.0, key="npk_p")
            with npk_c3:
                val_k = st.slider("Potassium (K)", 5.0, 205.0, 43.0, 1.0, key="npk_k")

            if st.button("Analyze NPK Ratios"):
                res_npk = call_api("assess-soil-npk", {"n": val_n, "p": val_p, "k": val_k})
                if res_npk:
                    n_col, p_col, k_col = st.columns(3)
                    with n_col:
                        st.metric("Nitrogen (N)", f"{val_n:.1f}")
                    with p_col:
                        st.metric("Phosphorus (P)", f"{val_p:.1f}")
                    with k_col:
                        st.metric("Potassium (K)", f"{val_k:.1f}")

                    # Interactive NPK Donut Chart
                    fig_donut = px.pie(
                        values=[val_n, val_p, val_k],
                        names=["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"],
                        color_discrete_sequence=["#10b981", "#f59e0b", "#3b82f6"],
                        hole=0.45,
                        title="Macronutrient Proportion Breakdown",
                    )
                    fig_donut.update_layout(
                        height=280,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)

                    if res_npk.get("notice"):
                        st.info(f"📋 Notice: {res_npk['notice']}")


# ---------------- FERTILIZER GUIDANCE PAGE ----------------
elif page == "🌿 Fertilizer Guidance":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="color: #0b3d20; margin: 0 0 6px 0; font-weight: 800;">🌿 Precision Fertilizer Guidance</h2>
            <p style="color: #476251; margin: 0; font-size: 0.98rem;">
                Safeguarding soil health through validated scientific protocols and ICAR agronomic standards.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        with st.form("fert_form"):
            crop_options = sorted(list(CROP_METADATA.keys()))
            selected_crop = st.selectbox(
                "Select Target Crop Variety",
                options=crop_options,
                format_func=lambda c: f"{CROP_METADATA.get(c, {}).get('emoji', '🌱')} {c.title()} ({CROP_METADATA.get(c, {}).get('category', 'Crop')})",
            )

            f_c1, f_c2, f_c3 = st.columns(3)
            with f_c1:
                fn = st.slider("Nitrogen (N)", 0.0, 140.0, 90.0, 1.0)
            with f_c2:
                fp = st.slider("Phosphorus (P)", 5.0, 145.0, 42.0, 1.0)
            with f_c3:
                fk = st.slider("Potassium (K)", 5.0, 205.0, 43.0, 1.0)

            f_submit = st.form_submit_button("Check Fertilizer Protocol Status")

        if f_submit:
            res_fert = call_api(
                "recommend-fertilizer",
                {
                    "crop": selected_crop,
                    "nitrogen": fn,
                    "phosphorus": fp,
                    "potassium": fk,
                },
            )
            if res_fert:
                st.markdown(
                    f"""
                    <div style="background: rgba(245, 158, 11, 0.12); border-left: 4px solid #f59e0b; padding: 18px; border-radius: 12px; margin-top: 14px;">
                        <h4 style="color: #92400e !important; margin: 0 0 6px 0; font-weight: 800;">⚠️ Scientific Dosage Safeguard Active</h4>
                        <p style="color: #78350f !important; font-size: 0.94rem; margin: 0; line-height: 1.5; font-weight: 500;">
                            Status: <b>{res_fert.get('status', 'Reference data required')}</b><br/>
                            {res_fert.get('notice', '')}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Scientific Education on STCR (Soil Test Crop Response)
    with st.container(border=True):
        st.markdown("### 📚 Why Blanket Fertilizer Recommendations are Unsafe")
        st.markdown(
            """
            Under the **Indian Council of Agricultural Research (ICAR)** framework:
            1. **Soil-Test Crop Response (STCR) Equations:**
               Fertilizer quantity is determined by:
               $$\\text{Fertilizer Dose} = \\frac{\\text{Target Yield} \\times \\text{Nutrient Requirement} - (\\text{Soil Test Value} \\times \\text{Soil Efficiency})}{\\text{Fertilizer Efficiency}}$$
            2. **Risk of Arbitrary Dosage:** Over-application of Nitrogen triggers nitrate leaching into groundwater and vegetative lodging; over-application of Potassium induces Magnesium deficiencies.
            3. **Safe Path Forward:** Collect a composite soil sample (0–15 cm depth) and submit it to your nearest Soil Testing Laboratory (STL) for calibrated regional dosage cards.
            """
        )


# ---------------- MODEL PERFORMANCE PAGE ----------------
elif page == "📊 Model Performance":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="color: #0b3d20; margin: 0 0 6px 0; font-weight: 800;">📊 Model Evaluation & Validation Metrics</h2>
            <p style="color: #476251; margin: 0; font-size: 0.98rem;">
                Transparent audit of the Random Forest multi-class crop classifier on unseen held-out test data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric(label="Overall Accuracy", value="99.32%", delta="+94.8% over baseline")
    with p2:
        st.metric(label="Macro Precision", value="0.99", help="Average precision across all 22 crop classes")
    with p3:
        st.metric(label="Macro Recall", value="0.99", help="Average recall across all 22 crop classes")
    with p4:
        st.metric(label="Macro F1-Score", value="0.99", help="Harmonic mean of precision and recall")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    project_root = Path(__file__).resolve().parents[1]
    eval_dir = project_root / "docs" / "model_evaluation"
    matrix_path = eval_dir / "confusion_matrix.png"
    report_path = eval_dir / "classification_report.txt"

    col_mat, col_rep = st.columns([1.2, 1])

    with col_mat:
        with st.container(border=True):
            st.markdown("#### 🎯 Multi-Class Confusion Matrix")
            if matrix_path.exists():
                st.image(str(matrix_path), caption="22-Class Confusion Matrix on 440 Held-Out Test Records", use_container_width=True)
            else:
                st.warning("Confusion matrix image not found.")

    with col_rep:
        with st.container(border=True):
            st.markdown("#### 📋 Detailed Classification Report")
            if report_path.exists():
                rep_text = report_path.read_text(encoding="utf-8")
                st.code(rep_text, language="text")
                st.download_button(
                    "📥 Download Classification Report (.txt)",
                    data=rep_text,
                    file_name="classification_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            else:
                st.warning("Classification report text not found.")


# ---------------- ABOUT PROJECT PAGE ----------------
elif page == "ℹ️ About Project":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="color: #0b3d20; margin: 0 0 6px 0; font-weight: 800;">ℹ️ About Kissaan AI</h2>
            <p style="color: #476251; margin: 0; font-size: 0.98rem;">
                AI-Driven Soil Health Assessment & Agro-Climatic Crop Suitability Platform
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            """
            ### 🌿 Mission & Vision
            **Kissaan AI** empowers farmers and agricultural consultants with data-driven insights. 
            By synthesizing multi-parameter soil test metrics ($N, P, K, \\text{pH}$) and climatic factors (Temperature, Humidity, Rainfall), 
            the system delivers probabilistic crop suitability rankings while maintaining strict adherence to scientific safety standards.

            ---

            ### 🏗️ Technical Architecture
            - **Machine Learning Core:** Random Forest Classifier (200 trees, stratified split, balanced weights)
            - **Model Explainability:** SHAP (SHapley Additive exPlanations) for transparent feature attribution
            - **API Microservice:** FastAPI asynchronous backend with Pydantic v2 validation schemas
            - **Frontend Interface:** Streamlit Enterprise UI with custom CSS glassmorphism and Plotly vector charts
            - **Report Engine:** ReportLab automated PDF generator for field-ready soil analysis sheets

            ---

            ### 🛡️ Ethical AI & Agricultural Safeguards
            - **No Blind Fertilizer Dosage:** The platform explicitly refuses to prescribe blanket chemical fertilizer amounts without localized soil laboratory calibration.
            - **Dataset Transparency:** Predictions reflect the statistical distribution of the 2,200 records in `Crop_recommendation.csv`.
            - **Advisory Role:** Designed as an intelligent decision-support tool, not an absolute replacement for local agronomists and Krishi Vigyan Kendra (KVK) guidance.
            """
        )
